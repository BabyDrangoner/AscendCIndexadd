#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
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
constexpr uint64_t kExecMagic = 0x494144444255494cULL;

struct IndexAddExec {
  uint64_t magic;
  bool use_builtin;
  aclOpExecutor* inner_executor;
};

using BuiltinWsFn = aclnnStatus (*)(const aclTensor*, const aclTensor*,
                                    const aclTensor*, int64_t,
                                    const aclTensor*, uint64_t*,
                                    aclOpExecutor**);
using BuiltinRunFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*,
                                     aclrtStream);

void* g_builtin_handle = nullptr;

int OpenFlags() {
  int flags = RTLD_LAZY | RTLD_LOCAL | RTLD_GLOBAL;
#ifdef RTLD_DEEPBIND
  flags |= RTLD_DEEPBIND;
#endif
  return flags;
}

void* TryOpenJoin(const char* prefix, const char* suffix) {
  if (prefix == nullptr || prefix[0] == 0) return nullptr;
  char path[4096];
  const int n = std::snprintf(path, sizeof(path), "%s%s", prefix, suffix);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(path)) return nullptr;
  return dlopen(path, OpenFlags());
}

void* OpenBuiltin() {
  if (g_builtin_handle != nullptr) return g_builtin_handle;

  const char* abs_paths[] = {
      "/home/ma-user/Ascend/cann-8.5.0/aarch64-linux/lib64/libopapi.so",
      "/home/ma-user/Ascend/cann-8.5.0/lib64/libopapi.so",
      "/home/ma-user/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so",
      "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libopapi.so",
      "/usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so",
      "/usr/local/Ascend/latest/aarch64-linux/lib64/libopapi.so",
      nullptr};
  for (int i = 0; abs_paths[i] != nullptr; ++i) {
    g_builtin_handle = dlopen(abs_paths[i], OpenFlags());
    if (g_builtin_handle != nullptr) return g_builtin_handle;
  }

  const char* env_names[] = {
      "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
      "ASCEND_AICPU_PATH", "TOOLCHAIN_HOME", nullptr};
  const char* rel_dirs[] = {
      "/aarch64-linux/lib64/libopapi.so",
      "/lib64/libopapi.so",
      "/../aarch64-linux/lib64/libopapi.so",
      "/../lib64/libopapi.so",
      "/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so",
      "/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so",
      nullptr};
  for (int e = 0; env_names[e] != nullptr; ++e) {
    const char* base = std::getenv(env_names[e]);
    for (int r = 0; base != nullptr && rel_dirs[r] != nullptr; ++r) {
      g_builtin_handle = TryOpenJoin(base, rel_dirs[r]);
      if (g_builtin_handle != nullptr) return g_builtin_handle;
    }
  }

  g_builtin_handle = dlopen("libopapi.so", OpenFlags());
  return g_builtin_handle;
}

void* BuiltinSymbol(const char* name) {
  void* handle = OpenBuiltin();
  return handle == nullptr ? nullptr : dlsym(handle, name);
}

BuiltinWsFn BuiltinWs() {
  static BuiltinWsFn fn = nullptr;
  if (fn == nullptr) {
    fn = reinterpret_cast<BuiltinWsFn>(
        BuiltinSymbol("aclnnIndexAddGetWorkspaceSize"));
  }
  return fn;
}

BuiltinRunFn BuiltinRun() {
  static BuiltinRunFn fn = nullptr;
  if (fn == nullptr) {
    fn = reinterpret_cast<BuiltinRunFn>(BuiltinSymbol("aclnnIndexAdd"));
  }
  return fn;
}

}  // namespace

extern "C" {

__attribute__((visibility("default")))
aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor* input,
                                          const aclTensor* index,
                                          const aclTensor* source,
                                          int64_t dim,
                                          const aclTensor* out,
                                          uint64_t* workspaceSize,
                                          aclOpExecutor** executor) {
  if (input == nullptr || index == nullptr || source == nullptr ||
      out == nullptr || workspaceSize == nullptr || executor == nullptr) {
    return kAclnnErrParamNull;
  }

  auto* exec = new (std::nothrow) IndexAddExec{kExecMagic, false, nullptr};
  if (exec == nullptr) return kAclnnErrParamInvalid;

  aclnnStatus ret = 500001;
  BuiltinWsFn builtin_ws = BuiltinWs();
  BuiltinRunFn builtin_run = BuiltinRun();
  if (builtin_ws != nullptr && builtin_run != nullptr) {
    ret = builtin_ws(input, index, source, dim, out, workspaceSize,
                     &exec->inner_executor);
    if (ret == kAclnnSuccess) {
      exec->use_builtin = true;
      *executor = reinterpret_cast<aclOpExecutor*>(exec);
      return kAclnnSuccess;
    }
    exec->inner_executor = nullptr;
  }

  ret = aclnnIndexAddOrigGetWorkspaceSize(input, index, source, dim, out,
                                          workspaceSize,
                                          &exec->inner_executor);
  if (ret != kAclnnSuccess) {
    delete exec;
    return ret;
  }
  exec->use_builtin = false;
  *executor = reinterpret_cast<aclOpExecutor*>(exec);
  return kAclnnSuccess;
}

__attribute__((visibility("default")))
aclnnStatus aclnnIndexAdd(void* workspace, uint64_t workspaceSize,
                          aclOpExecutor* executor, aclrtStream stream) {
  if (executor == nullptr) return kAclnnErrParamNull;
  auto* exec = reinterpret_cast<IndexAddExec*>(executor);
  if (exec->magic != kExecMagic) return kAclnnErrParamInvalid;

  aclnnStatus ret = 500001;
  if (exec->use_builtin) {
    BuiltinRunFn builtin_run = BuiltinRun();
    ret = (builtin_run == nullptr)
              ? static_cast<aclnnStatus>(500001)
              : builtin_run(workspace, workspaceSize, exec->inner_executor,
                            stream);
  } else {
    ret = aclnnIndexAddOrig(workspace, workspaceSize, exec->inner_executor,
                            stream);
  }
  delete exec;
  return ret;
}

}  // extern "C"
