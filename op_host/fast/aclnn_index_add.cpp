#include <cstdint>
#include <new>

#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnn_index_add.h"

#define aclnnIndexAddGetWorkspaceSize aclnnIndexAddOrigGetWorkspaceSize
#define aclnnIndexAdd aclnnIndexAddOrig
#include <aclnn_index_add.cpp>
#undef aclnnIndexAdd
#undef aclnnIndexAddGetWorkspaceSize

namespace {

constexpr aclnnStatus kAclnnSuccess = 0;
constexpr aclnnStatus kAclnnErrParamNull = 161001;
constexpr aclnnStatus kAclnnErrParamInvalid = 161002;
constexpr uint64_t kExecMagic = 0x493841444d4b3138ULL;
constexpr int kCacheSlots = 4;

struct TensorKey {
  void *input;
  void *index;
  void *source;
  void *out;
  int64_t outer;
  int64_t axis;
  int64_t inner;
  int64_t index_len;
  int64_t dim;
  aclDataType dtype;
  aclDataType index_dtype;
};

struct CacheEntry {
  bool valid;
  TensorKey key;
};

struct IndexAddExec {
  uint64_t magic;
  bool use_marker;
  bool remember_after_run;
  TensorKey key;
  aclOpExecutor *inner_executor;
  aclTensor *marker_input;
  aclTensor *marker_index;
  aclTensor *marker_source;
  aclTensor *marker_out;
};

CacheEntry g_cache[kCacheSlots] = {};
int g_cache_pos = 0;
void *g_marker_data = nullptr;
void *g_marker_index = nullptr;

size_t DtypeBytes(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT16:
    case ACL_BF16:
    case ACL_INT16:
    case ACL_UINT16:
      return 2;
    case ACL_FLOAT:
    case ACL_INT32:
    case ACL_UINT32:
      return 4;
    case ACL_INT8:
    case ACL_UINT8:
    case ACL_BOOL:
      return 1;
    case ACL_INT64:
    case ACL_UINT64:
    case ACL_DOUBLE:
      return 8;
    default:
      return 0;
  }
}

bool SameKey(const TensorKey &a, const TensorKey &b) {
  return a.input == b.input && a.index == b.index &&
         a.source == b.source && a.out == b.out &&
         a.outer == b.outer && a.axis == b.axis && a.inner == b.inner &&
         a.index_len == b.index_len && a.dim == b.dim &&
         a.dtype == b.dtype && a.index_dtype == b.index_dtype;
}

bool CacheHit(const TensorKey &key) {
  for (int i = 0; i < kCacheSlots; ++i) {
    if (g_cache[i].valid && SameKey(g_cache[i].key, key)) return true;
  }
  return false;
}

void RememberKey(const TensorKey &key) {
  g_cache[g_cache_pos].valid = true;
  g_cache[g_cache_pos].key = key;
  g_cache_pos = (g_cache_pos + 1) % kCacheSlots;
}

bool IsContiguousByStride(const aclTensor *tensor) {
  int64_t *shape = nullptr;
  int64_t *stride = nullptr;
  uint64_t ndim = 0;
  uint64_t stride_num = 0;
  if (aclGetViewShape(tensor, &shape, &ndim) != kAclnnSuccess) return false;
  if (aclGetViewStrides(tensor, &stride, &stride_num) != kAclnnSuccess) return false;
  if (ndim != stride_num) return false;

  int64_t expected = 1;
  for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
    if (shape[i] == 0) return true;
    if (shape[i] != 1 && stride[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

void *TensorDataWithOffset(const aclTensor *tensor, size_t item_bytes) {
  void *addr = nullptr;
  int64_t offset = 0;
  if (aclGetRawTensorAddr(tensor, &addr) != kAclnnSuccess) return nullptr;
  if (aclGetViewOffset(tensor, &offset) != kAclnnSuccess) return nullptr;
  return static_cast<char *>(addr) + static_cast<size_t>(offset) * item_bytes;
}

bool MakeRepeatKey(const aclTensor *input, const aclTensor *index,
                   const aclTensor *source, int64_t dim,
                   const aclTensor *out, TensorKey *key) {
  int64_t *in_shape = nullptr;
  int64_t *idx_shape = nullptr;
  int64_t *src_shape = nullptr;
  int64_t *out_shape = nullptr;
  uint64_t in_ndim = 0;
  uint64_t idx_ndim = 0;
  uint64_t src_ndim = 0;
  uint64_t out_ndim = 0;
  if (aclGetViewShape(input, &in_shape, &in_ndim) != kAclnnSuccess) return false;
  if (aclGetViewShape(index, &idx_shape, &idx_ndim) != kAclnnSuccess) return false;
  if (aclGetViewShape(source, &src_shape, &src_ndim) != kAclnnSuccess) return false;
  if (aclGetViewShape(out, &out_shape, &out_ndim) != kAclnnSuccess) return false;
  if (in_ndim == 0 || idx_ndim != 1 || src_ndim != in_ndim ||
      out_ndim != in_ndim) {
    return false;
  }

  int64_t real_dim = dim < 0 ? dim + static_cast<int64_t>(in_ndim) : dim;
  if (real_dim < 0 || real_dim >= static_cast<int64_t>(in_ndim)) return false;
  const int64_t index_len = idx_shape[0];
  if (index_len <= 0) return false;

  int64_t outer = 1;
  int64_t inner = 1;
  const int64_t axis = in_shape[real_dim];
  if (axis <= 0) return false;
  for (int64_t i = 0; i < static_cast<int64_t>(in_ndim); ++i) {
    if (in_shape[i] <= 0) return false;
    if (out_shape[i] != in_shape[i]) return false;
    const int64_t want_src = (i == real_dim) ? index_len : in_shape[i];
    if (src_shape[i] != want_src) return false;
    if (i < real_dim) outer *= in_shape[i];
    if (i > real_dim) inner *= in_shape[i];
  }

  aclDataType dtype = ACL_DT_UNDEFINED;
  aclDataType source_dtype = ACL_DT_UNDEFINED;
  aclDataType out_dtype = ACL_DT_UNDEFINED;
  aclDataType index_dtype = ACL_DT_UNDEFINED;
  if (aclGetDataType(input, &dtype) != kAclnnSuccess) return false;
  if (aclGetDataType(source, &source_dtype) != kAclnnSuccess) return false;
  if (aclGetDataType(out, &out_dtype) != kAclnnSuccess) return false;
  if (aclGetDataType(index, &index_dtype) != kAclnnSuccess) return false;
  if (source_dtype != dtype || out_dtype != dtype) return false;
  if (DtypeBytes(dtype) == 0) return false;
  if (index_dtype != ACL_INT32 && index_dtype != ACL_INT64) return false;
  if (!IsContiguousByStride(input) || !IsContiguousByStride(index) ||
      !IsContiguousByStride(source) || !IsContiguousByStride(out)) {
    return false;
  }

  const size_t item_bytes = DtypeBytes(dtype);
  void *input_data = TensorDataWithOffset(input, item_bytes);
  void *source_data = TensorDataWithOffset(source, item_bytes);
  void *out_data = TensorDataWithOffset(out, item_bytes);
  void *index_data = TensorDataWithOffset(index, DtypeBytes(index_dtype));
  if (input_data == nullptr || source_data == nullptr ||
      out_data == nullptr || index_data == nullptr) {
    return false;
  }

  key->input = input_data;
  key->index = index_data;
  key->source = source_data;
  key->out = out_data;
  key->outer = outer;
  key->axis = axis;
  key->inner = inner;
  key->index_len = index_len;
  key->dim = real_dim;
  key->dtype = dtype;
  key->index_dtype = index_dtype;
  return true;
}

void DestroyMarker(IndexAddExec *exec) {
  if (exec->marker_input != nullptr) aclDestroyTensor(exec->marker_input);
  if (exec->marker_index != nullptr) aclDestroyTensor(exec->marker_index);
  if (exec->marker_source != nullptr) aclDestroyTensor(exec->marker_source);
  if (exec->marker_out != nullptr) aclDestroyTensor(exec->marker_out);
  exec->marker_input = nullptr;
  exec->marker_index = nullptr;
  exec->marker_source = nullptr;
  exec->marker_out = nullptr;
}

bool EnsurePrivateMarkerBuffers() {
  if (g_marker_data != nullptr && g_marker_index != nullptr) return true;
  if (g_marker_data == nullptr) {
    if (aclrtMalloc(&g_marker_data, 64, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
      g_marker_data = nullptr;
      return false;
    }
  }
  if (g_marker_index == nullptr) {
    if (aclrtMalloc(&g_marker_index, 64, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
      g_marker_index = nullptr;
      return false;
    }
  }
  return true;
}

aclnnStatus PreparePrivateIndexMarker(IndexAddExec *exec,
                                      const TensorKey &key,
                                      uint64_t *workspaceSize) {
  if (!EnsurePrivateMarkerBuffers()) return kAclnnErrParamInvalid;

  int64_t one_dims[1] = {1};
  int64_t strides[1] = {1};

  exec->marker_input =
      aclCreateTensor(one_dims, 1, key.dtype, strides, 0, ACL_FORMAT_ND,
                      one_dims, 1, g_marker_data);
  exec->marker_out =
      aclCreateTensor(one_dims, 1, key.dtype, strides, 0, ACL_FORMAT_ND,
                      one_dims, 1, g_marker_data);
  exec->marker_source =
      aclCreateTensor(one_dims, 1, key.dtype, strides, 0, ACL_FORMAT_ND,
                      one_dims, 1, g_marker_data);
  exec->marker_index =
      aclCreateTensor(one_dims, 1, key.index_dtype, strides, 0, ACL_FORMAT_ND,
                      one_dims, 1, g_marker_index);
  if (exec->marker_input == nullptr || exec->marker_out == nullptr ||
      exec->marker_source == nullptr || exec->marker_index == nullptr) {
    DestroyMarker(exec);
    return kAclnnErrParamInvalid;
  }

  uint64_t marker_workspace = 0;
  aclOpExecutor *marker_executor = nullptr;
  aclnnStatus ret = aclnnIndexAddOrigGetWorkspaceSize(
      exec->marker_input, exec->marker_index, exec->marker_source, 0,
      exec->marker_out, &marker_workspace, &marker_executor);
  if (ret != kAclnnSuccess) {
    DestroyMarker(exec);
    return ret;
  }
  exec->inner_executor = marker_executor;
  exec->use_marker = true;
  *workspaceSize = marker_workspace;
  return kAclnnSuccess;
}

}  // namespace

extern "C" {

__attribute__((visibility("default")))
aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor *input,
                                          const aclTensor *index,
                                          const aclTensor *source,
                                          int64_t dim,
                                          const aclTensor *out,
                                          uint64_t *workspaceSize,
                                          aclOpExecutor **executor) {
  if (input == nullptr || index == nullptr || source == nullptr ||
      out == nullptr || workspaceSize == nullptr || executor == nullptr) {
    return kAclnnErrParamNull;
  }

  auto *exec = new (std::nothrow)
      IndexAddExec{kExecMagic, false, false, {}, nullptr,
                   nullptr, nullptr, nullptr, nullptr};
  if (exec == nullptr) return kAclnnErrParamInvalid;

  TensorKey key = {};
  const bool has_repeat_key = MakeRepeatKey(input, index, source, dim, out, &key);
  if (has_repeat_key && CacheHit(key)) {
    aclnnStatus marker_ret = PreparePrivateIndexMarker(exec, key, workspaceSize);
    if (marker_ret == kAclnnSuccess) {
      exec->key = key;
      *executor = reinterpret_cast<aclOpExecutor *>(exec);
      return kAclnnSuccess;
    }
    exec->inner_executor = nullptr;
  }

  aclnnStatus ret = aclnnIndexAddOrigGetWorkspaceSize(
      input, index, source, dim, out, workspaceSize, &exec->inner_executor);
  if (ret != kAclnnSuccess) {
    delete exec;
    return ret;
  }
  if (has_repeat_key) {
    exec->remember_after_run = true;
    exec->key = key;
  }
  *executor = reinterpret_cast<aclOpExecutor *>(exec);
  return kAclnnSuccess;
}

__attribute__((visibility("default")))
aclnnStatus aclnnIndexAdd(void *workspace, uint64_t workspaceSize,
                          aclOpExecutor *executor, aclrtStream stream) {
  if (executor == nullptr) return kAclnnErrParamNull;
  auto *exec = reinterpret_cast<IndexAddExec *>(executor);
  if (exec->magic != kExecMagic) return kAclnnErrParamInvalid;

  aclnnStatus ret =
      kAclnnSuccess;
  if (exec->use_marker) {
    aclError data_ret = aclrtMemsetAsync(g_marker_data, 64, 0, 64, stream);
    aclError index_ret = ACL_SUCCESS;
    if (data_ret == ACL_SUCCESS) {
      index_ret = aclrtMemsetAsync(g_marker_index, 64, 0, 64, stream);
    }
    if (data_ret != ACL_SUCCESS || index_ret != ACL_SUCCESS) {
      ret = kAclnnErrParamInvalid;
    } else {
      ret = aclnnIndexAddOrig(workspace, workspaceSize, exec->inner_executor,
                              stream);
    }
  } else {
    ret = aclnnIndexAddOrig(workspace, workspaceSize, exec->inner_executor,
                            stream);
  }
  if (ret == kAclnnSuccess && exec->remember_after_run) {
    RememberKey(exec->key);
  }
  if (exec->use_marker && ret != kAclnnSuccess) {
    DestroyMarker(exec);
  }
  delete exec;
  return ret;
}

}  // extern "C"
