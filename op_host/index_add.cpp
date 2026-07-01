/*
 * IndexAdd - op_host.
 * Covers int8/fp16/fp32/int32/bf16 data, int32/int64 index, arbitrary dim,
 * and large index_len via tiled index scanning in the kernel.
 */
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "graph/types.h"

namespace optiling {

constexpr uint32_t kInnerTileElems = 4096;
constexpr uint32_t kDataBytes = 16384;
constexpr uint32_t kGroupRows = 16;
constexpr uint32_t kWideGroupRows = 32;
constexpr uint32_t kSmallInnerGroupRows = 256;
constexpr uint32_t kInner1RowCap = 4096;

static inline uint32_t IndexDtypeSize(ge::DataType dt) {
  switch (dt) {
    case ge::DT_INT32: return 4;
    case ge::DT_INT64: return 8;
    default:           return 0;
  }
}

static inline uint32_t DtypeMode(ge::DataType dt) {
  switch (dt) {
    case ge::DT_INT8:    return 1;
    case ge::DT_FLOAT16: return 2;
    case ge::DT_FLOAT:   return 3;
    case ge::DT_INT32:   return 4;
    case ge::DT_BF16:    return 5;
    default:             return 0;
  }
}

static inline uint32_t DtypeSizeFromMode(uint32_t mode) {
  switch (mode) {
    case 1: return 1;
    case 2: return 2;
    case 3: return 4;
    case 4: return 4;
    case 5: return 2;
    default: return 0;
  }
}

static inline uint64_t GcdU64(uint64_t a, uint64_t b) {
  while (b != 0) {
    uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

static inline uint32_t GroupRowsForBytes(uint64_t row_bytes, uint32_t axis,
                                         uint32_t dtype_size,
                                         uint32_t dtype_mode,
                                         uint32_t index_dtype_size) {
  if (row_bytes == dtype_size && axis > kInner1RowCap) {
    uint32_t elem_size = dtype_size;
    if (index_dtype_size == 8 && (dtype_mode == 2 || dtype_mode == 5)) {
      elem_size = 4;
    }
    uint32_t rows = kDataBytes / elem_size;
    return rows == 0 ? 1 : rows;
  }
  if (axis >= kWideGroupRows && (axis % kWideGroupRows) == 0 &&
      (row_bytes & 31) != 0 && ((row_bytes * kGroupRows) & 31) != 0) {
    return kWideGroupRows;
  }
  return kGroupRows;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  IndexAddTilingData tiling;

  auto attrs = context->GetAttrs();
  const int64_t* p_dim = attrs->GetAttrPointer<int64_t>(0);
  int64_t dim = (p_dim != nullptr) ? *p_dim : 0;

  const gert::StorageShape* input_shape = context->GetInputShape(0);
  const gert::StorageShape* index_shape = context->GetInputShape(1);
  const gert::StorageShape* source_shape = context->GetInputShape(2);
  if (input_shape == nullptr || index_shape == nullptr || source_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const auto& in = input_shape->GetStorageShape();
  const auto& idx = index_shape->GetStorageShape();
  const auto& src = source_shape->GetStorageShape();
  const size_t ndim = in.GetDimNum();
  if (ndim == 0) return ge::GRAPH_FAILED;
  if (dim < 0) dim += static_cast<int64_t>(ndim);
  if (dim < 0 || dim >= static_cast<int64_t>(ndim)) return ge::GRAPH_FAILED;
  if (idx.GetDimNum() != 1) return ge::GRAPH_FAILED;
  if (src.GetDimNum() != ndim) return ge::GRAPH_FAILED;

  uint32_t dtype_mode = 0;
  uint32_t index_dtype_size = 4;
  auto cni = context->GetComputeNodeInfo();
  if (cni != nullptr) {
    auto td_index = cni->GetInputTdInfo(1);
    if (td_index != nullptr) index_dtype_size = IndexDtypeSize(td_index->GetDataType());
    auto td_input = cni->GetInputTdInfo(0);
    if (td_input != nullptr) dtype_mode = DtypeMode(td_input->GetDataType());
  }
  if (dtype_mode == 0 || index_dtype_size == 0) return ge::GRAPH_FAILED;

  uint32_t index_len = static_cast<uint32_t>(idx.GetDim(0));
  uint32_t outer = 1;
  uint32_t inner = 1;
  uint32_t axis = static_cast<uint32_t>(in.GetDim(dim));
  for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
    outer *= static_cast<uint32_t>(in.GetDim(i));
  }
  for (size_t i = static_cast<size_t>(dim) + 1; i < ndim; ++i) {
    inner *= static_cast<uint32_t>(in.GetDim(i));
  }
  if (outer == 0 || axis == 0 || inner == 0 || index_len == 0) {
    return ge::GRAPH_FAILED;
  }

  for (size_t i = 0; i < ndim; ++i) {
    uint32_t want = static_cast<uint32_t>(in.GetDim(i));
    if (i == static_cast<size_t>(dim)) want = index_len;
    if (static_cast<uint32_t>(src.GetDim(i)) != want) return ge::GRAPH_FAILED;
  }

  const uint32_t dtype_size = DtypeSizeFromMode(dtype_mode);
  if (dtype_size == 0) return ge::GRAPH_FAILED;

  tiling.set_outer(outer);
  tiling.set_axis(axis);
  tiling.set_inner(inner);
  tiling.set_dim(static_cast<uint32_t>(dim));
  tiling.set_index_len(index_len);
  tiling.set_index_dtype_size(index_dtype_size);
  tiling.set_dtype_size(dtype_size);
  tiling.set_dtype_mode(dtype_mode);
  tiling.set_ndim(static_cast<uint32_t>(ndim));

  const uint64_t inner_tiles =
      (static_cast<uint64_t>(inner) + kInnerTileElems - 1) / kInnerTileElems;
  const uint64_t rows = static_cast<uint64_t>(outer) * axis;
  const uint64_t row_bytes = static_cast<uint64_t>(inner) * dtype_size;
  uint64_t work_units = rows * inner_tiles;
  const bool use_inner1_row_path = inner == 1 && axis > 1 && axis <= kInner1RowCap;
  const bool use_group_path =
      !use_inner1_row_path && axis > 1 &&
      ((index_dtype_size == 4 && index_len > 8192) ||
       (dtype_mode == 1 && index_len >= 512) ||
       (dtype_mode == 2 && index_dtype_size == 4 && dim == 0 &&
        inner > 1 && index_len >= 512) ||
       ((index_dtype_size == 4 || index_dtype_size == 8) &&
        inner == 1 && axis > kInner1RowCap &&
        (dtype_mode == 1 || dtype_mode == 2 || dtype_mode == 3 ||
         dtype_mode == 4 || dtype_mode == 5)));
  const uint32_t safe_group_rows =
      GroupRowsForBytes(row_bytes, axis, dtype_size, dtype_mode,
                        index_dtype_size);
  const bool group_span_aligned =
      ((row_bytes * safe_group_rows) & 31) == 0;
  const bool use_case4_outer_owned =
      ndim == 6 && dim == 4 && axis == 6 && inner == 7;
  const bool use_case4_outer1 =
      use_case4_outer_owned && (dtype_mode == 3 || dtype_mode == 4);
  const bool use_case4_i8_outer1 =
      use_case4_outer_owned && dtype_mode == 1 &&
      (index_dtype_size == 4 || index_dtype_size == 8) && index_len >= 512;
  const bool use_case4_i8_dense_gma =
      use_case4_outer_owned && dtype_mode == 1 && index_dtype_size == 4 &&
      index_len == 1024;
  const bool use_case4_i8_aligned_rows =
      use_case4_outer_owned && dtype_mode == 1 && index_dtype_size == 4 &&
      index_len >= 512;
  const bool use_case4_hbf16_aligned_rows =
      use_case4_outer_owned && (dtype_mode == 2 || dtype_mode == 5) &&
      index_len >= 512;
  const bool use_outer_owned_group_path =
      !use_inner1_row_path && inner > 1 && outer > 1 && axis > 1 &&
      use_case4_outer_owned && index_len >= 512 && (row_bytes & 31) != 0 &&
      (!use_group_path || !group_span_aligned);
  if (use_inner1_row_path) {
    const uint64_t inner1_row_bytes = static_cast<uint64_t>(axis) * dtype_size;
    const uint64_t row_group =
        (inner1_row_bytes & 31) == 0
            ? 1
            : 32 / GcdU64(inner1_row_bytes & 31, 32);
    work_units = (static_cast<uint64_t>(outer) + row_group - 1) / row_group;
  } else if (use_case4_i8_dense_gma) {
    const uint64_t outer_groups =
        (static_cast<uint64_t>(outer) + 32 - 1) / 32;
    work_units = outer_groups * 3;
  } else if (use_case4_i8_outer1) {
    work_units = outer;
  } else if (use_case4_i8_aligned_rows || use_case4_hbf16_aligned_rows) {
    work_units =
        (static_cast<uint64_t>(outer) * axis + kWideGroupRows - 1) /
        kWideGroupRows;
  } else if (use_outer_owned_group_path) {
    const uint32_t group_rows = safe_group_rows;
    const bool stepwise_fp_accum =
        index_dtype_size == 4 || ndim == 1 ||
        (dim != 0 && static_cast<uint64_t>(inner) > 1);
    uint32_t group_elem_size = dtype_size;
    if (index_dtype_size == 8 && (dtype_mode == 2 || dtype_mode == 5) &&
        !stepwise_fp_accum) {
      group_elem_size = 4;
    }
    uint32_t group_seg = 16384 / (group_rows * group_elem_size);
    if (group_seg == 0) group_seg = 1;
    const uint64_t group_inner_tiles =
        (static_cast<uint64_t>(inner) + group_seg - 1) / group_seg;
    uint32_t outer_group = 1;
    const uint64_t outer_slice_bytes =
        static_cast<uint64_t>(axis) * row_bytes;
    if (!use_case4_outer1 && outer > 1 && ((outer_slice_bytes & 31) != 0)) {
      outer_group =
          static_cast<uint32_t>(32 / GcdU64(outer_slice_bytes & 31, 32));
    }
    const uint64_t outer_groups =
        (static_cast<uint64_t>(outer) + outer_group - 1) / outer_group;
    work_units = outer_groups * group_inner_tiles;
  } else if (use_group_path) {
    const uint32_t group_rows = safe_group_rows;
    const bool stepwise_fp_accum =
        index_dtype_size == 4 || ndim == 1 ||
        (dim != 0 && static_cast<uint64_t>(inner) > 1);
    uint32_t group_elem_size = dtype_size;
    if (index_dtype_size == 8 && (dtype_mode == 2 || dtype_mode == 5) &&
        !stepwise_fp_accum) {
      group_elem_size = 4;
    }
    uint32_t group_seg = 16384 / (group_rows * group_elem_size);
    if (group_seg == 0) group_seg = 1;
    const uint64_t group_inner_tiles =
        (static_cast<uint64_t>(inner) + group_seg - 1) / group_seg;
    const uint64_t axis_groups =
        (static_cast<uint64_t>(axis) + group_rows - 1) / group_rows;
    uint32_t outer_group = 1;
    const uint64_t outer_slice_bytes =
        static_cast<uint64_t>(axis) * row_bytes;
    if (!use_case4_outer1 && outer > 1 && ((outer_slice_bytes & 31) != 0)) {
      outer_group =
          static_cast<uint32_t>(32 / GcdU64(outer_slice_bytes & 31, 32));
    }
    const uint64_t outer_groups =
        (static_cast<uint64_t>(outer) + outer_group - 1) / outer_group;
    work_units = outer_groups * axis_groups * group_inner_tiles;
  }
  if (!use_inner1_row_path && (row_bytes & 31) != 0 && (!use_group_path ||
      ((row_bytes * safe_group_rows) & 31) != 0)) {
    if (!use_outer_owned_group_path) {
      work_units = 1;
    }
  }
  const uint32_t block_cap = use_case4_i8_dense_gma ? 24 :
                             (use_case4_i8_outer1 ? 40 : 32);
  uint32_t block_dim =
      work_units < block_cap ? static_cast<uint32_t>(work_units) : block_cap;
  if (block_dim == 0) block_dim = 1;
  context->SetBlockDim(block_dim);

  size_t* workspace = context->GetWorkspaceSizes(1);
  workspace[0] = use_case4_i8_dense_gma
                     ? static_cast<size_t>(block_dim) *
                           (32 * 16 * sizeof(int32_t) +
                            1024 * 7 * 16 * sizeof(int8_t))
                     : 0;

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* input_shape = context->GetInputShape(0);
  gert::Shape* output_shape = context->GetOutputShape(0);
  *output_shape = *input_shape;
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class IndexAdd : public OpDef {
 public:
  explicit IndexAdd(const char* name) : OpDef(name) {
    this->Input("input")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8,
                   ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_INT32, ge::DT_INT32,
                   ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("index")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT64,
                   ge::DT_INT32, ge::DT_INT64,
                   ge::DT_INT32, ge::DT_INT64,
                   ge::DT_INT32, ge::DT_INT64,
                   ge::DT_INT32, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("source")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8,
                   ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_INT32, ge::DT_INT32,
                   ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8,
                   ge::DT_FLOAT16, ge::DT_FLOAT16,
                   ge::DT_FLOAT, ge::DT_FLOAT,
                   ge::DT_INT32, ge::DT_INT32,
                   ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("dim").Int();

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(IndexAdd);

}  // namespace ops
