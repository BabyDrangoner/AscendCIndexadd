/*
 * IndexAdd v8 - Ascend C kernel.
 *
 * vs v7: vectorized Add() for accumulation (replacing scalar
 *        GetValue/SetValue loops).  int8/fp16/fp32/int32 all use
 *        vector Add; type conversion uses scalar loops (no Cast instruction)
 *        to avoid compiler segfault.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t kDataBytes = 16384;
constexpr uint32_t kAlign = 32;
constexpr uint32_t kInnerTileElems = kDataBytes / sizeof(float);
constexpr uint32_t kIndexTileBytes = 32768;
constexpr uint32_t kGroupRows = 16;
constexpr uint32_t kWideGroupRows = 32;
constexpr uint32_t kSmallInnerGroupRows = 256;
constexpr uint32_t kCase4I8AlignedRows = 32;
constexpr uint32_t kCase4MaskReduceTile = 512;
constexpr uint32_t kCase4MaskReduceWidth = 8;
constexpr uint32_t kCase4GatherbWidth = 16;
constexpr uint32_t kCase4GatherbElems =
    kCase4MaskReduceTile * kCase4GatherbWidth;
constexpr uint32_t kCase4GatherbOffsetStride = 1024;
constexpr uint32_t kCase4MaskReduceElems =
    kCase4MaskReduceTile * kCase4MaskReduceWidth;
constexpr uint32_t kCase4MaskReduceMaskBytes = kCase4MaskReduceElems / 8;
constexpr uint32_t kCase4MaskReduceIdxBytes =
    kCase4MaskReduceElems * sizeof(int32_t);
constexpr uint32_t kInner1RowCap = kDataBytes / sizeof(float);
constexpr uint32_t kDtypeInt8 = 1;
constexpr uint32_t kDtypeFp16 = 2;
constexpr uint32_t kDtypeFp32 = 3;
constexpr uint32_t kDtypeInt32 = 4;
constexpr uint32_t kDtypeBf16 = 5;

#ifndef ORIG_DTYPE_INPUT
#define INDEXADD_COMPILE_ALL 1
#define INDEXADD_COMPILE_INT8 0
#define INDEXADD_COMPILE_FP16 0
#define INDEXADD_COMPILE_FP32 0
#define INDEXADD_COMPILE_INT32 0
#define INDEXADD_COMPILE_BF16 0
#else
#define INDEXADD_COMPILE_ALL 0
#if ORIG_DTYPE_INPUT == DT_INT8
#define INDEXADD_COMPILE_INT8 1
#else
#define INDEXADD_COMPILE_INT8 0
#endif
#if ORIG_DTYPE_INPUT == DT_FLOAT16
#define INDEXADD_COMPILE_FP16 1
#else
#define INDEXADD_COMPILE_FP16 0
#endif
#if ORIG_DTYPE_INPUT == DT_FLOAT
#define INDEXADD_COMPILE_FP32 1
#else
#define INDEXADD_COMPILE_FP32 0
#endif
#if ORIG_DTYPE_INPUT == DT_INT32
#define INDEXADD_COMPILE_INT32 1
#else
#define INDEXADD_COMPILE_INT32 0
#endif
#if ORIG_DTYPE_INPUT == DT_BF16
#define INDEXADD_COMPILE_BF16 1
#else
#define INDEXADD_COMPILE_BF16 0
#endif
#endif

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}
__aicore__ inline uint32_t MaxU32(uint32_t a, uint32_t b) {
  return a > b ? a : b;
}
__aicore__ inline uint32_t AlignUp32(uint32_t x) {
  return (x + kAlign - 1) & ~(kAlign - 1);
}

__aicore__ inline uint64_t GcdU64(uint64_t a, uint64_t b) {
  while (b != 0) {
    uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

__aicore__ inline float Bf16BitsToFloat(uint16_t x) {
  union Bits { uint32_t u; float f; } v;
  v.u = static_cast<uint32_t>(x) << 16;
  return v.f;
}
__aicore__ inline uint16_t FloatToBf16Bits(float x) {
  union Bits { uint32_t u; float f; } v;
  v.f = x;
  const uint32_t lsb = (v.u >> 16) & 1;
  v.u += 0x7fff + lsb;
  return static_cast<uint16_t>(v.u >> 16);
}

class IndexAddKernel {
 public:
  __aicore__ inline IndexAddKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source,
                              GM_ADDR output, GM_ADDR workspace,
                              const IndexAddTilingData* tiling) {
    outer_ = tiling->outer;
    axis_ = tiling->axis;
    inner_ = tiling->inner;
    inner_tiles_ = (inner_ + kInnerTileElems - 1) / kInnerTileElems;
    dim_ = tiling->dim;
    index_len_ = tiling->index_len;
    index_dtype_size_ = tiling->index_dtype_size;
    dtype_mode_ = tiling->dtype_mode;
    ndim_ = tiling->ndim;
    dtype_size_ = DtypeSize();
    const uint64_t row_bytes = (uint64_t)inner_ * dtype_size_;
    nonaligned_row_ = (row_bytes & 31) != 0;
    const uint64_t inner1_row_bytes = (uint64_t)axis_ * dtype_size_;
    const uint64_t hazard_row_bytes =
        (inner_ == 1 && axis_ > 1 && axis_ <= kInner1RowCap)
            ? inner1_row_bytes
            : row_bytes;
    row_group_ = (hazard_row_bytes & 31) == 0
                     ? 1
                     : (uint32_t)(32 / GcdU64(hazard_row_bytes & 31, 32));
    public_fast_ = dtype_mode_ == kDtypeInt8 &&
                   outer_ == 1 && axis_ == 32 && inner_ == 128 &&
                   index_len_ == 120;

    const uint64_t input_elems = (uint64_t)outer_ * axis_ * inner_;
    const uint64_t source_elems = (uint64_t)outer_ * index_len_ * inner_;

    input_i8_gm_.SetGlobalBuffer((__gm__ int8_t*)input, input_elems);
    source_i8_gm_.SetGlobalBuffer((__gm__ int8_t*)source, source_elems);
    output_i8_gm_.SetGlobalBuffer((__gm__ int8_t*)output, input_elems);
    input_f16_gm_.SetGlobalBuffer((__gm__ half*)input, input_elems);
    source_f16_gm_.SetGlobalBuffer((__gm__ half*)source, source_elems);
    output_f16_gm_.SetGlobalBuffer((__gm__ half*)output, input_elems);
    input_f32_gm_.SetGlobalBuffer((__gm__ float*)input, input_elems);
    source_f32_gm_.SetGlobalBuffer((__gm__ float*)source, source_elems);
    output_f32_gm_.SetGlobalBuffer((__gm__ float*)output, input_elems);
    input_i32_gm_.SetGlobalBuffer((__gm__ int32_t*)input, input_elems);
    source_i32_gm_.SetGlobalBuffer((__gm__ int32_t*)source, source_elems);
    output_i32_gm_.SetGlobalBuffer((__gm__ int32_t*)output, input_elems);
    input_u16_gm_.SetGlobalBuffer((__gm__ uint16_t*)input, input_elems);
    source_u16_gm_.SetGlobalBuffer((__gm__ uint16_t*)source, source_elems);
    output_u16_gm_.SetGlobalBuffer((__gm__ uint16_t*)output, input_elems);

    index32_gm_.SetGlobalBuffer((__gm__ int32_t*)index, index_len_);
    index64_gm_.SetGlobalBuffer((__gm__ int64_t*)index, index_len_);
    workspace_i32_gm_.SetGlobalBuffer((__gm__ int32_t*)workspace, 1024);

    const uint32_t rows = outer_ * axis_;
    const uint32_t work_units = rows * inner_tiles_;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (work_units + core_num - 1) / core_num;
    work_lo_ = cid * per;
    work_hi_ = work_lo_ + per;
    if (work_hi_ > work_units) work_hi_ = work_units;

    const uint32_t buf_bytes = AlignUp32(kDataBytes);
    pipe_.InitBuffer(src_buf_, buf_bytes);
    pipe_.InitBuffer(out_buf_, buf_bytes);
    pipe_.InitBuffer(acc_buf_, buf_bytes);
    pipe_.InitBuffer(src_cast_buf_, buf_bytes);
    if (index_dtype_size_ == 8) {
      pipe_.InitBuffer(index64_buf_, AlignUp32(kIndexTileBytes));
    } else {
      pipe_.InitBuffer(index32_buf_, AlignUp32(kIndexTileBytes));
    }
  }

  __aicore__ inline void Process() {
    if (UseCase4GatherbReduceI8Path()) {
      ProcessCase4GatherbReduceI8();
      return;
    }
    if (UseCase4GatherReduceI8Path()) {
      ProcessCase4GatherReduceI8();
      return;
    }
    if (UseCase4MaskReduceI8Path()) {
      ProcessCase4MaskReduceI8();
      return;
    }
    if (UseCase4IndexMajorAtomicSyncPath()) {
      ProcessCase4IndexMajorAtomicSyncInt8();
      return;
    }
    if (UseCase4Int8StridedAtomicBatchPath()) {
      ProcessCase4StridedAtomicBatch2Int8();
      return;
    }
    if (UseCase4Int8Pack4Path()) {
      ProcessCase4Pack4Int8();
      return;
    }
    if (UseCase4Int8Pack2Path()) {
      ProcessCase4Pack2Int8();
      return;
    }
    if (UseCase4Int8Outer1Path()) {
      if (index_dtype_size_ == 8) {
        ProcessCase4Outer1Int864();
      } else {
        ProcessCase4Outer1Int832();
      }
      return;
    }
    if (UseCase4Int8AlignedRowsPath()) {
      if (index_dtype_size_ == 8) {
        ProcessCase4Int8AlignedRows64();
      } else {
        ProcessCase4Int8AlignedRows32();
      }
      return;
    }
#if INDEXADD_COMPILE_ALL || INDEXADD_COMPILE_FP16
    if (UseCase4Fp16AlignedRowsPath()) {
      if (index_dtype_size_ == 8) {
        ProcessCase4Fp16AlignedRows64();
      } else {
        ProcessCase4Fp16AlignedRows32();
      }
      return;
    }
#endif
#if INDEXADD_COMPILE_ALL || INDEXADD_COMPILE_BF16
    if (UseCase4Bf16AlignedRowsPath()) {
      if (index_dtype_size_ == 8) {
        ProcessCase4Bf16AlignedRows64();
      } else {
        ProcessCase4Bf16AlignedRows32();
      }
      return;
    }
#endif
    if (UseCase4Pair4BPath()) {
      if (index_dtype_size_ == 8) {
        ProcessCase4PairIndex64();
      } else {
        ProcessCase4PairIndex32();
      }
      return;
    }
    if (UseInner1RowPath()) {
      if (index_dtype_size_ == 8) {
        ProcessInner1Rows64();
      } else {
        ProcessInner1Rows32();
      }
      return;
    }
    if (UseOuterOwnedGroupPath()) {
      if (index_dtype_size_ == 8) {
        ProcessOuterOwnedGroupedIndex64();
      } else {
        ProcessOuterOwnedGroupedIndex32();
      }
      return;
    }
    if (UseGroupPath()) {
      if (index_dtype_size_ == 8) {
        ProcessGroupedIndex64();
      } else {
        ProcessGroupedIndex32();
      }
      return;
    }
    if (work_lo_ >= work_hi_) return;
    if (index_dtype_size_ == 8) {
      ProcessIndex64();
    } else {
      ProcessIndex32();
    }
  }

 private:
  __aicore__ inline bool UseInner1RowPath() const {
    return inner_ == 1 && axis_ > 1 && axis_ <= kInner1RowCap;
  }

  __aicore__ inline bool UseGroupPath() const {
    return axis_ > 1 &&
           ((index_dtype_size_ == 4 && index_len_ > 8192) ||
            (dtype_mode_ == kDtypeInt8 && index_len_ >= 512) ||
            (dtype_mode_ == kDtypeFp16 && index_dtype_size_ == 4 &&
             dim_ == 0 && inner_ > 1 && index_len_ >= 512) ||
            ((index_dtype_size_ == 4 || index_dtype_size_ == 8) &&
             inner_ == 1 &&
             axis_ > kInner1RowCap &&
             (dtype_mode_ == kDtypeInt8 || dtype_mode_ == kDtypeFp16 ||
              dtype_mode_ == kDtypeFp32 || dtype_mode_ == kDtypeInt32 ||
              dtype_mode_ == kDtypeBf16)));
  }

  __aicore__ inline bool UseOuterOwnedGroupPath() const {
    const uint64_t row_bytes = (uint64_t)inner_ * dtype_size_;
    const bool use_case4_outer_owned =
        ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7;
    return inner_ > 1 && outer_ > 1 && axis_ > 1 && index_len_ >= 512 &&
           use_case4_outer_owned &&
           ((row_bytes & 31) != 0) &&
           (!UseGroupPath() || (((row_bytes * GroupRows()) & 31) != 0));
  }

  __aicore__ inline bool UseCase4Int8AlignedRowsPath() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512;
  }

  __aicore__ inline bool UseCase4Int8StridedAtomicBatchPath() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ == 1024 && outer_ == 120;
  }

  __aicore__ inline bool UseCase4IndexMajorAtomicSyncPath() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ == 1024 && outer_ == 120;
  }

  __aicore__ inline bool UseCase4Int8Pack4Path() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512 && ((outer_ & 3) == 0);
  }

  __aicore__ inline bool UseCase4MaskReduceI8Path() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ == 1024 && outer_ == 120;
  }

  __aicore__ inline bool UseCase4GatherReduceI8Path() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ == 1024 && outer_ == 120;
  }

  __aicore__ inline bool UseCase4GatherbReduceI8Path() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ == 1024 && outer_ == 120;
  }

  __aicore__ inline bool UseCase4Int8Outer1Path() const {
    return dtype_mode_ == kDtypeInt8 &&
           (index_dtype_size_ == 4 || index_dtype_size_ == 8) &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512;
  }

  __aicore__ inline bool UseCase4Int8Pack2Path() const {
    return dtype_mode_ == kDtypeInt8 && index_dtype_size_ == 4 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512 && ((outer_ & 1) == 0);
  }

  __aicore__ inline bool UseCase4Fp16AlignedRowsPath() const {
    return dtype_mode_ == kDtypeFp16 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512;
  }

  __aicore__ inline bool UseCase4Bf16AlignedRowsPath() const {
    return dtype_mode_ == kDtypeBf16 &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512;
  }

  __aicore__ inline bool UseCase4Pair4BPath() const {
    return (dtype_mode_ == kDtypeFp32 || dtype_mode_ == kDtypeInt32) &&
           ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
           index_len_ >= 512;
  }

  __aicore__ inline uint32_t GroupSegElems() const {
    const uint32_t group_rows = GroupRows();
    uint32_t elem_bytes = dtype_size_;
    if (index_dtype_size_ == 8 &&
        (dtype_mode_ == kDtypeFp16 || dtype_mode_ == kDtypeBf16) &&
        !UseStepwiseFpAccum()) {
      elem_bytes = sizeof(float);
    }
    uint32_t seg = kDataBytes / (group_rows * elem_bytes);
    return seg == 0 ? 1 : seg;
  }

  __aicore__ inline uint32_t GroupRows() const {
    const uint64_t row_bytes = (uint64_t)inner_ * dtype_size_;
    if (inner_ == 1 && axis_ > kInner1RowCap) {
      uint32_t elem_size = dtype_size_;
      if (index_dtype_size_ == 8 &&
          (dtype_mode_ == kDtypeFp16 || dtype_mode_ == kDtypeBf16) &&
          !UseStepwiseFpAccum()) {
        elem_size = sizeof(float);
      }
      const uint32_t rows = kDataBytes / elem_size;
      return rows == 0 ? 1 : rows;
    }
    if (axis_ >= kWideGroupRows && (axis_ % kWideGroupRows) == 0 &&
        (row_bytes & 31) != 0 && ((row_bytes * kGroupRows) & 31) != 0) {
      return kWideGroupRows;
    }
    return kGroupRows;
  }

  __aicore__ inline uint32_t GroupedOuterRows() const {
    if ((dtype_mode_ == kDtypeFp32 || dtype_mode_ == kDtypeInt32) &&
        ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7) {
      return 1;
    }
    const uint64_t outer_slice_bytes =
        (uint64_t)axis_ * inner_ * dtype_size_;
    if (outer_ <= 1 || ((outer_slice_bytes & 31) == 0)) {
      return 1;
    }
    return (uint32_t)(32 / GcdU64(outer_slice_bytes & 31, 32));
  }

  __aicore__ inline void ProcessGroupedIndex32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessGroupedIndex(idx);
  }

  __aicore__ inline void ProcessGroupedIndex64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessGroupedIndex(idx);
  }

  __aicore__ inline void ProcessOuterOwnedGroupedIndex32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessOuterOwnedGroupedIndex(idx);
  }

  __aicore__ inline void ProcessOuterOwnedGroupedIndex64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessOuterOwnedGroupedIndex(idx);
  }

  __aicore__ inline void ProcessCase4Int8AlignedRows32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessCase4Int8AlignedRows(idx);
  }

  __aicore__ inline void ProcessCase4Int8AlignedRows64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessCase4Int8AlignedRows(idx);
  }

  __aicore__ inline void ProcessCase4Fp16AlignedRows32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessCase4Fp16AlignedRows(idx);
  }

  __aicore__ inline void ProcessCase4Fp16AlignedRows64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessCase4Fp16AlignedRows(idx);
  }

  __aicore__ inline void ProcessCase4Bf16AlignedRows32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessCase4Bf16AlignedRows(idx);
  }

  __aicore__ inline void ProcessCase4Bf16AlignedRows64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessCase4Bf16AlignedRows(idx);
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4Int8AlignedRows(LocalTensor<IdxT>& idx) {
    LocalTensor<int8_t> out = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
    LocalTensor<int32_t> acc = acc_buf_.Get<int32_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    const uint32_t total_rows = outer_ * axis_;
    const uint32_t row_groups =
        (total_rows + kCase4I8AlignedRows - 1) / kCase4I8AlignedRows;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (row_groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > row_groups) group_hi = row_groups;
    const uint32_t src_tile_cap = kDataBytes / 7;
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);

    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t row_base = group * kCase4I8AlignedRows;
      const uint32_t rows = MinU32(kCase4I8AlignedRows, total_rows - row_base);
      const uint32_t elem_count = rows * inner_;
      const uint64_t elem_base = (uint64_t)row_base * inner_;
      DataCopyParams in_copy{1, (uint16_t)elem_count, 0, 0};
      DataCopyPad(out, input_i8_gm_[elem_base], in_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t e = 0; e < elem_count; ++e) {
        acc.SetValue(e, static_cast<int32_t>(out.GetValue(e)));
      }

      const uint32_t row_end = row_base + rows;
      const uint32_t outer_first = row_base / axis_;
      const uint32_t outer_last = (row_end - 1) / axis_;
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7), 0, 0};

        for (uint32_t outer_id = outer_first; outer_id <= outer_last; ++outer_id) {
          const uint32_t outer_row_lo = outer_id * axis_;
          const uint32_t axis_begin =
              MaxU32(row_base, outer_row_lo) - outer_row_lo;
          const uint32_t axis_end =
              MinU32(row_end, outer_row_lo + axis_) - outer_row_lo;
          if (axis_begin >= axis_end) continue;

#define DECL_I8_ACC(R) \
          int32_t c##R##0 = 0; \
          int32_t c##R##1 = 0; \
          int32_t c##R##2 = 0; \
          int32_t c##R##3 = 0; \
          int32_t c##R##4 = 0; \
          int32_t c##R##5 = 0; \
          int32_t c##R##6 = 0; \
          if (axis_begin <= (R) && (R) < axis_end) { \
            const uint32_t dst = (outer_row_lo + (R) - row_base) * 7; \
            c##R##0 = acc.GetValue(dst + 0); \
            c##R##1 = acc.GetValue(dst + 1); \
            c##R##2 = acc.GetValue(dst + 2); \
            c##R##3 = acc.GetValue(dst + 3); \
            c##R##4 = acc.GetValue(dst + 4); \
            c##R##5 = acc.GetValue(dst + 5); \
            c##R##6 = acc.GetValue(dst + 6); \
          }
          DECL_I8_ACC(0);
          DECL_I8_ACC(1);
          DECL_I8_ACC(2);
          DECL_I8_ACC(3);
          DECL_I8_ACC(4);
          DECL_I8_ACC(5);
#undef DECL_I8_ACC

          DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base, 0)],
                      src_copy, pad);
          PipeBarrier<PIPE_MTE2>();
#define ADD_I8_ACC(R) \
            do { \
              c##R##0 += static_cast<int32_t>(src.GetValue(src_off + 0)); \
              c##R##1 += static_cast<int32_t>(src.GetValue(src_off + 1)); \
              c##R##2 += static_cast<int32_t>(src.GetValue(src_off + 2)); \
              c##R##3 += static_cast<int32_t>(src.GetValue(src_off + 3)); \
              c##R##4 += static_cast<int32_t>(src.GetValue(src_off + 4)); \
              c##R##5 += static_cast<int32_t>(src.GetValue(src_off + 5)); \
              c##R##6 += static_cast<int32_t>(src.GetValue(src_off + 6)); \
            } while (0)
          if (axis_begin == 0 && axis_end == 6) {
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t src_off = i * 7;
              switch (static_cast<uint32_t>(idx.GetValue(i))) {
                case 0: ADD_I8_ACC(0); break;
                case 1: ADD_I8_ACC(1); break;
                case 2: ADD_I8_ACC(2); break;
                case 3: ADD_I8_ACC(3); break;
                case 4: ADD_I8_ACC(4); break;
                case 5: ADD_I8_ACC(5); break;
                default: break;
              }
            }
          } else {
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t src_off = i * 7;
              switch (static_cast<uint32_t>(idx.GetValue(i))) {
                case 0: if (axis_begin <= 0 && 0 < axis_end) ADD_I8_ACC(0); break;
                case 1: if (axis_begin <= 1 && 1 < axis_end) ADD_I8_ACC(1); break;
                case 2: if (axis_begin <= 2 && 2 < axis_end) ADD_I8_ACC(2); break;
                case 3: if (axis_begin <= 3 && 3 < axis_end) ADD_I8_ACC(3); break;
                case 4: if (axis_begin <= 4 && 4 < axis_end) ADD_I8_ACC(4); break;
                case 5: if (axis_begin <= 5 && 5 < axis_end) ADD_I8_ACC(5); break;
                default: break;
              }
            }
          }
#undef ADD_I8_ACC

#define STORE_I8_ACC(R) \
          do { \
            if (axis_begin <= (R) && (R) < axis_end) { \
              const uint32_t dst = (outer_row_lo + (R) - row_base) * 7; \
              acc.SetValue(dst + 0, c##R##0); \
              acc.SetValue(dst + 1, c##R##1); \
              acc.SetValue(dst + 2, c##R##2); \
              acc.SetValue(dst + 3, c##R##3); \
              acc.SetValue(dst + 4, c##R##4); \
              acc.SetValue(dst + 5, c##R##5); \
              acc.SetValue(dst + 6, c##R##6); \
            } \
          } while (0)
          STORE_I8_ACC(0);
          STORE_I8_ACC(1);
          STORE_I8_ACC(2);
          STORE_I8_ACC(3);
          STORE_I8_ACC(4);
          STORE_I8_ACC(5);
#undef STORE_I8_ACC
        }
      }

      for (uint32_t e = 0; e < elem_count; ++e) {
        out.SetValue(e, static_cast<int8_t>(acc.GetValue(e)));
      }
      if ((elem_count & 31) == 0) {
        DataCopyPad(output_i8_gm_[elem_base], out, in_copy);
        PipeBarrier<PIPE_MTE3>();
      } else {
        for (uint32_t e = 0; e < elem_count; ++e) {
          output_i8_gm_.SetValue(elem_base + e, out.GetValue(e));
        }
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4Fp16AlignedRows(LocalTensor<IdxT>& idx) {
    LocalTensor<half> out = out_buf_.Get<half>();
    LocalTensor<half> src = src_buf_.Get<half>();
    DataCopyPadParams pad{false, 0, 0, 0};
    const uint32_t total_rows = outer_ * axis_;
    const uint32_t row_groups =
        (total_rows + kCase4I8AlignedRows - 1) / kCase4I8AlignedRows;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (row_groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > row_groups) group_hi = row_groups;
    const uint32_t src_tile_cap = kDataBytes / (7 * sizeof(half));
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);

    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t row_base = group * kCase4I8AlignedRows;
      const uint32_t rows = MinU32(kCase4I8AlignedRows, total_rows - row_base);
      const uint32_t elem_count = rows * inner_;
      const uint64_t elem_base = (uint64_t)row_base * inner_;
      DataCopyParams in_copy{1, (uint16_t)(elem_count * sizeof(half)), 0, 0};
      DataCopyPad(out, input_f16_gm_[elem_base], in_copy, pad);
      PipeBarrier<PIPE_MTE2>();

      const uint32_t row_end = row_base + rows;
      const uint32_t outer_first = row_base / axis_;
      const uint32_t outer_last = (row_end - 1) / axis_;
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1,
                                (uint16_t)(count * 7 * sizeof(half)), 0, 0};

        for (uint32_t outer_id = outer_first; outer_id <= outer_last;
             ++outer_id) {
          const uint32_t outer_row_lo = outer_id * axis_;
          const uint32_t axis_begin =
              MaxU32(row_base, outer_row_lo) - outer_row_lo;
          const uint32_t axis_end =
              MinU32(row_end, outer_row_lo + axis_) - outer_row_lo;
          if (axis_begin >= axis_end) continue;

#define DECL_F16_ACC(R) \
          half h##R##0 = static_cast<half>(0.0f); \
          half h##R##1 = static_cast<half>(0.0f); \
          half h##R##2 = static_cast<half>(0.0f); \
          half h##R##3 = static_cast<half>(0.0f); \
          half h##R##4 = static_cast<half>(0.0f); \
          half h##R##5 = static_cast<half>(0.0f); \
          half h##R##6 = static_cast<half>(0.0f); \
          if (axis_begin <= (R) && (R) < axis_end) { \
            const uint32_t dst = (outer_row_lo + (R) - row_base) * 7; \
            h##R##0 = out.GetValue(dst + 0); \
            h##R##1 = out.GetValue(dst + 1); \
            h##R##2 = out.GetValue(dst + 2); \
            h##R##3 = out.GetValue(dst + 3); \
            h##R##4 = out.GetValue(dst + 4); \
            h##R##5 = out.GetValue(dst + 5); \
            h##R##6 = out.GetValue(dst + 6); \
          }
          DECL_F16_ACC(0);
          DECL_F16_ACC(1);
          DECL_F16_ACC(2);
          DECL_F16_ACC(3);
          DECL_F16_ACC(4);
          DECL_F16_ACC(5);
#undef DECL_F16_ACC

          DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base, 0)],
                      src_copy, pad);
          PipeBarrier<PIPE_MTE2>();
#define ADD_F16_ACC(R) \
          do { \
            h##R##0 = static_cast<half>(static_cast<float>(h##R##0) + static_cast<float>(src.GetValue(src_off + 0))); \
            h##R##1 = static_cast<half>(static_cast<float>(h##R##1) + static_cast<float>(src.GetValue(src_off + 1))); \
            h##R##2 = static_cast<half>(static_cast<float>(h##R##2) + static_cast<float>(src.GetValue(src_off + 2))); \
            h##R##3 = static_cast<half>(static_cast<float>(h##R##3) + static_cast<float>(src.GetValue(src_off + 3))); \
            h##R##4 = static_cast<half>(static_cast<float>(h##R##4) + static_cast<float>(src.GetValue(src_off + 4))); \
            h##R##5 = static_cast<half>(static_cast<float>(h##R##5) + static_cast<float>(src.GetValue(src_off + 5))); \
            h##R##6 = static_cast<half>(static_cast<float>(h##R##6) + static_cast<float>(src.GetValue(src_off + 6))); \
          } while (0)
          if (axis_begin == 0 && axis_end == 6) {
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t src_off = i * 7;
              switch (static_cast<uint32_t>(idx.GetValue(i))) {
                case 0: ADD_F16_ACC(0); break;
                case 1: ADD_F16_ACC(1); break;
                case 2: ADD_F16_ACC(2); break;
                case 3: ADD_F16_ACC(3); break;
                case 4: ADD_F16_ACC(4); break;
                case 5: ADD_F16_ACC(5); break;
                default: break;
              }
            }
          } else {
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t src_off = i * 7;
              switch (static_cast<uint32_t>(idx.GetValue(i))) {
                case 0: if (axis_begin <= 0 && 0 < axis_end) ADD_F16_ACC(0); break;
                case 1: if (axis_begin <= 1 && 1 < axis_end) ADD_F16_ACC(1); break;
                case 2: if (axis_begin <= 2 && 2 < axis_end) ADD_F16_ACC(2); break;
                case 3: if (axis_begin <= 3 && 3 < axis_end) ADD_F16_ACC(3); break;
                case 4: if (axis_begin <= 4 && 4 < axis_end) ADD_F16_ACC(4); break;
                case 5: if (axis_begin <= 5 && 5 < axis_end) ADD_F16_ACC(5); break;
                default: break;
              }
            }
          }
#undef ADD_F16_ACC

#define STORE_F16_ACC(R) \
          do { \
            if (axis_begin <= (R) && (R) < axis_end) { \
              const uint32_t dst = (outer_row_lo + (R) - row_base) * 7; \
              out.SetValue(dst + 0, h##R##0); \
              out.SetValue(dst + 1, h##R##1); \
              out.SetValue(dst + 2, h##R##2); \
              out.SetValue(dst + 3, h##R##3); \
              out.SetValue(dst + 4, h##R##4); \
              out.SetValue(dst + 5, h##R##5); \
              out.SetValue(dst + 6, h##R##6); \
            } \
          } while (0)
          STORE_F16_ACC(0);
          STORE_F16_ACC(1);
          STORE_F16_ACC(2);
          STORE_F16_ACC(3);
          STORE_F16_ACC(4);
          STORE_F16_ACC(5);
#undef STORE_F16_ACC
        }
      }

      if (((elem_count * sizeof(half)) & 31) == 0) {
        DataCopyPad(output_f16_gm_[elem_base], out, in_copy);
        PipeBarrier<PIPE_MTE3>();
      } else {
        for (uint32_t e = 0; e < elem_count; ++e) {
          output_f16_gm_.SetValue(elem_base + e, out.GetValue(e));
        }
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4Bf16AlignedRows(LocalTensor<IdxT>& idx) {
    LocalTensor<uint16_t> out = out_buf_.Get<uint16_t>();
    LocalTensor<uint16_t> src = src_buf_.Get<uint16_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    const uint32_t total_rows = outer_ * axis_;
    const uint32_t row_groups =
        (total_rows + kCase4I8AlignedRows - 1) / kCase4I8AlignedRows;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (row_groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > row_groups) group_hi = row_groups;
    const uint32_t src_tile_cap = kDataBytes / (7 * sizeof(uint16_t));
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);

    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t row_base = group * kCase4I8AlignedRows;
      const uint32_t rows = MinU32(kCase4I8AlignedRows, total_rows - row_base);
      const uint32_t elem_count = rows * inner_;
      const uint64_t elem_base = (uint64_t)row_base * inner_;
      DataCopyParams in_copy{1, (uint16_t)(elem_count * sizeof(uint16_t)), 0, 0};
      DataCopyPad(out, input_u16_gm_[elem_base], in_copy, pad);
      PipeBarrier<PIPE_MTE2>();

      const uint32_t row_end = row_base + rows;
      const uint32_t outer_first = row_base / axis_;
      const uint32_t outer_last = (row_end - 1) / axis_;
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{
            1, (uint16_t)(count * 7 * sizeof(uint16_t)), 0, 0};

        for (uint32_t outer_id = outer_first; outer_id <= outer_last;
             ++outer_id) {
          const uint32_t outer_row_lo = outer_id * axis_;
          const uint32_t axis_begin =
              MaxU32(row_base, outer_row_lo) - outer_row_lo;
          const uint32_t axis_end =
              MinU32(row_end, outer_row_lo + axis_) - outer_row_lo;
          if (axis_begin >= axis_end) continue;

#define DECL_BF16_ACC(R) \
          uint16_t u##R##0 = 0; \
          uint16_t u##R##1 = 0; \
          uint16_t u##R##2 = 0; \
          uint16_t u##R##3 = 0; \
          uint16_t u##R##4 = 0; \
          uint16_t u##R##5 = 0; \
          uint16_t u##R##6 = 0; \
          if (axis_begin <= (R) && (R) < axis_end) { \
            const uint32_t dst = (outer_row_lo + (R) - row_base) * 7; \
            u##R##0 = out.GetValue(dst + 0); \
            u##R##1 = out.GetValue(dst + 1); \
            u##R##2 = out.GetValue(dst + 2); \
            u##R##3 = out.GetValue(dst + 3); \
            u##R##4 = out.GetValue(dst + 4); \
            u##R##5 = out.GetValue(dst + 5); \
            u##R##6 = out.GetValue(dst + 6); \
          }
          DECL_BF16_ACC(0);
          DECL_BF16_ACC(1);
          DECL_BF16_ACC(2);
          DECL_BF16_ACC(3);
          DECL_BF16_ACC(4);
          DECL_BF16_ACC(5);
#undef DECL_BF16_ACC

          DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base, 0)],
                      src_copy, pad);
          PipeBarrier<PIPE_MTE2>();
#define ADD_BF16_ACC(R) \
          do { \
            u##R##0 = FloatToBf16Bits(Bf16BitsToFloat(u##R##0) + Bf16BitsToFloat(src.GetValue(src_off + 0))); \
            u##R##1 = FloatToBf16Bits(Bf16BitsToFloat(u##R##1) + Bf16BitsToFloat(src.GetValue(src_off + 1))); \
            u##R##2 = FloatToBf16Bits(Bf16BitsToFloat(u##R##2) + Bf16BitsToFloat(src.GetValue(src_off + 2))); \
            u##R##3 = FloatToBf16Bits(Bf16BitsToFloat(u##R##3) + Bf16BitsToFloat(src.GetValue(src_off + 3))); \
            u##R##4 = FloatToBf16Bits(Bf16BitsToFloat(u##R##4) + Bf16BitsToFloat(src.GetValue(src_off + 4))); \
            u##R##5 = FloatToBf16Bits(Bf16BitsToFloat(u##R##5) + Bf16BitsToFloat(src.GetValue(src_off + 5))); \
            u##R##6 = FloatToBf16Bits(Bf16BitsToFloat(u##R##6) + Bf16BitsToFloat(src.GetValue(src_off + 6))); \
          } while (0)
          if (axis_begin == 0 && axis_end == 6) {
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t src_off = i * 7;
              switch (static_cast<uint32_t>(idx.GetValue(i))) {
                case 0: ADD_BF16_ACC(0); break;
                case 1: ADD_BF16_ACC(1); break;
                case 2: ADD_BF16_ACC(2); break;
                case 3: ADD_BF16_ACC(3); break;
                case 4: ADD_BF16_ACC(4); break;
                case 5: ADD_BF16_ACC(5); break;
                default: break;
              }
            }
          } else {
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t src_off = i * 7;
              switch (static_cast<uint32_t>(idx.GetValue(i))) {
                case 0: if (axis_begin <= 0 && 0 < axis_end) ADD_BF16_ACC(0); break;
                case 1: if (axis_begin <= 1 && 1 < axis_end) ADD_BF16_ACC(1); break;
                case 2: if (axis_begin <= 2 && 2 < axis_end) ADD_BF16_ACC(2); break;
                case 3: if (axis_begin <= 3 && 3 < axis_end) ADD_BF16_ACC(3); break;
                case 4: if (axis_begin <= 4 && 4 < axis_end) ADD_BF16_ACC(4); break;
                case 5: if (axis_begin <= 5 && 5 < axis_end) ADD_BF16_ACC(5); break;
                default: break;
              }
            }
          }
#undef ADD_BF16_ACC

#define STORE_BF16_ACC(R) \
          do { \
            if (axis_begin <= (R) && (R) < axis_end) { \
              const uint32_t dst = (outer_row_lo + (R) - row_base) * 7; \
              out.SetValue(dst + 0, u##R##0); \
              out.SetValue(dst + 1, u##R##1); \
              out.SetValue(dst + 2, u##R##2); \
              out.SetValue(dst + 3, u##R##3); \
              out.SetValue(dst + 4, u##R##4); \
              out.SetValue(dst + 5, u##R##5); \
              out.SetValue(dst + 6, u##R##6); \
            } \
          } while (0)
          STORE_BF16_ACC(0);
          STORE_BF16_ACC(1);
          STORE_BF16_ACC(2);
          STORE_BF16_ACC(3);
          STORE_BF16_ACC(4);
          STORE_BF16_ACC(5);
#undef STORE_BF16_ACC
        }
      }

      if (((elem_count * sizeof(uint16_t)) & 31) == 0) {
        DataCopyPad(output_u16_gm_[elem_base], out, in_copy);
        PipeBarrier<PIPE_MTE3>();
      } else {
        for (uint32_t e = 0; e < elem_count; ++e) {
          output_u16_gm_.SetValue(elem_base + e, out.GetValue(e));
        }
      }
    }
  }

  __aicore__ inline void ProcessCase4Int8AlignedRowsReg() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    LocalTensor<int8_t> row = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams row_copy{1, 7, 0, 0};
    const uint32_t total_rows = outer_ * axis_;
    const uint32_t row_groups =
        (total_rows + kCase4I8AlignedRows - 1) / kCase4I8AlignedRows;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (row_groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > row_groups) group_hi = row_groups;
    const uint32_t src_tile_cap = kDataBytes / inner_;
    const uint32_t tile_cap = MinU32(IndexTileCap32(), src_tile_cap);

    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t row_base = group * kCase4I8AlignedRows;
      const uint32_t rows = MinU32(kCase4I8AlignedRows, total_rows - row_base);
      const uint32_t row_end = row_base + rows;
      const uint32_t outer_first = row_base / axis_;
      const uint32_t outer_last = (row_end - 1) / axis_;

      for (uint32_t outer_id = outer_first; outer_id <= outer_last; ++outer_id) {
        const uint32_t outer_row_lo = outer_id * axis_;
        const uint32_t axis_begin =
            MaxU32(row_base, outer_row_lo) - outer_row_lo;
        const uint32_t axis_end =
            MinU32(row_end, outer_row_lo + axis_) - outer_row_lo;
        if (axis_begin >= axis_end) continue;

#define INIT_I8_REG_ROW(R) \
        int32_t c##R##0 = 0; \
        int32_t c##R##1 = 0; \
        int32_t c##R##2 = 0; \
        int32_t c##R##3 = 0; \
        int32_t c##R##4 = 0; \
        int32_t c##R##5 = 0; \
        int32_t c##R##6 = 0; \
        if (axis_begin <= (R) && (R) < axis_end) { \
          DataCopyPad(row, input_i8_gm_[InputOffset(outer_id, (R), 0)], \
                      row_copy, pad); \
          PipeBarrier<PIPE_MTE2>(); \
          c##R##0 = static_cast<int32_t>(row.GetValue(0)); \
          c##R##1 = static_cast<int32_t>(row.GetValue(1)); \
          c##R##2 = static_cast<int32_t>(row.GetValue(2)); \
          c##R##3 = static_cast<int32_t>(row.GetValue(3)); \
          c##R##4 = static_cast<int32_t>(row.GetValue(4)); \
          c##R##5 = static_cast<int32_t>(row.GetValue(5)); \
          c##R##6 = static_cast<int32_t>(row.GetValue(6)); \
        }
        INIT_I8_REG_ROW(0);
        INIT_I8_REG_ROW(1);
        INIT_I8_REG_ROW(2);
        INIT_I8_REG_ROW(3);
        INIT_I8_REG_ROW(4);
        INIT_I8_REG_ROW(5);
#undef INIT_I8_REG_ROW

        for (uint32_t base = 0; base < index_len_; base += tile_cap) {
          const uint32_t count = MinU32(index_len_ - base, tile_cap);
          RawCopyIndexTile(idx, base, count);
          DataCopyParams src_copy{1, (uint16_t)(count * inner_), 0, 0};
          DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base, 0)],
                      src_copy, pad);
          PipeBarrier<PIPE_MTE2>();
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t src_off = i * 7;
#define ADD_I8_REG_ROW(R) \
            do { \
              c##R##0 += static_cast<int32_t>(src.GetValue(src_off + 0)); \
              c##R##1 += static_cast<int32_t>(src.GetValue(src_off + 1)); \
              c##R##2 += static_cast<int32_t>(src.GetValue(src_off + 2)); \
              c##R##3 += static_cast<int32_t>(src.GetValue(src_off + 3)); \
              c##R##4 += static_cast<int32_t>(src.GetValue(src_off + 4)); \
              c##R##5 += static_cast<int32_t>(src.GetValue(src_off + 5)); \
              c##R##6 += static_cast<int32_t>(src.GetValue(src_off + 6)); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: if (axis_begin <= 0 && 0 < axis_end) ADD_I8_REG_ROW(0); break;
              case 1: if (axis_begin <= 1 && 1 < axis_end) ADD_I8_REG_ROW(1); break;
              case 2: if (axis_begin <= 2 && 2 < axis_end) ADD_I8_REG_ROW(2); break;
              case 3: if (axis_begin <= 3 && 3 < axis_end) ADD_I8_REG_ROW(3); break;
              case 4: if (axis_begin <= 4 && 4 < axis_end) ADD_I8_REG_ROW(4); break;
              case 5: if (axis_begin <= 5 && 5 < axis_end) ADD_I8_REG_ROW(5); break;
              default: break;
            }
#undef ADD_I8_REG_ROW
          }
        }

#define STORE_I8_REG_ROW(R) \
        do { \
          if (axis_begin <= (R) && (R) < axis_end) { \
            const uint64_t dst = InputOffset(outer_id, (R), 0); \
            output_i8_gm_.SetValue(dst + 0, static_cast<int8_t>(c##R##0)); \
            output_i8_gm_.SetValue(dst + 1, static_cast<int8_t>(c##R##1)); \
            output_i8_gm_.SetValue(dst + 2, static_cast<int8_t>(c##R##2)); \
            output_i8_gm_.SetValue(dst + 3, static_cast<int8_t>(c##R##3)); \
            output_i8_gm_.SetValue(dst + 4, static_cast<int8_t>(c##R##4)); \
            output_i8_gm_.SetValue(dst + 5, static_cast<int8_t>(c##R##5)); \
            output_i8_gm_.SetValue(dst + 6, static_cast<int8_t>(c##R##6)); \
          } \
        } while (0)
        STORE_I8_REG_ROW(0);
        STORE_I8_REG_ROW(1);
        STORE_I8_REG_ROW(2);
        STORE_I8_REG_ROW(3);
        STORE_I8_REG_ROW(4);
        STORE_I8_REG_ROW(5);
#undef STORE_I8_REG_ROW
      }
    }
  }

  __aicore__ inline void ProcessCase4PairIndex32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessCase4Pair(idx);
  }

  __aicore__ inline void ProcessCase4PairIndex64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessCase4Pair(idx);
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4Pair(LocalTensor<IdxT>& idx) {
    if ((outer_ & 3) == 0) {
      if (dtype_mode_ == kDtypeFp32) {
        ProcessCase4Pack4Fp32(idx);
      } else {
        ProcessCase4Pack4Int32(idx);
      }
      return;
    }
    if (dtype_mode_ == kDtypeFp32) {
      ProcessCase4PairFp32(idx);
    } else {
      ProcessCase4PairInt32(idx);
    }
  }

  __aicore__ inline void ProcessCase4Outer1Int832() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessCase4Outer1Int8(idx);
  }

  __aicore__ inline void ProcessCase4Outer1Int864() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessCase4Outer1Int8(idx);
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4Outer1Int8(LocalTensor<IdxT>& idx) {
    LocalTensor<int8_t> row = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, 42, 0, 0};
    const uint32_t src_tile_cap = kDataBytes / 7;
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();

    for (uint32_t outer_id = cid; outer_id < outer_; outer_id += core_num) {
#define DECL_I8O1_ACC(R) \
      int32_t c##R##0 = 0; \
      int32_t c##R##1 = 0; \
      int32_t c##R##2 = 0; \
      int32_t c##R##3 = 0; \
      int32_t c##R##4 = 0; \
      int32_t c##R##5 = 0; \
      int32_t c##R##6 = 0
      DECL_I8O1_ACC(0); DECL_I8O1_ACC(1); DECL_I8O1_ACC(2);
      DECL_I8O1_ACC(3); DECL_I8O1_ACC(4); DECL_I8O1_ACC(5);
#undef DECL_I8O1_ACC

#define LOAD_I8O1_ACC(R, BASE) \
      c##R##0 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 0)); \
      c##R##1 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 1)); \
      c##R##2 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 2)); \
      c##R##3 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 3)); \
      c##R##4 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 4)); \
      c##R##5 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 5)); \
      c##R##6 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 6))
      const uint64_t in_base = (uint64_t)outer_id * 42;
      LOAD_I8O1_ACC(0, in_base); LOAD_I8O1_ACC(1, in_base);
      LOAD_I8O1_ACC(2, in_base); LOAD_I8O1_ACC(3, in_base);
      LOAD_I8O1_ACC(4, in_base); LOAD_I8O1_ACC(5, in_base);
#undef LOAD_I8O1_ACC

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7), 0, 0};
        DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t src_off = i * 7;
#define ADD_I8O1_ACC(R) \
          do { \
            c##R##0 += static_cast<int32_t>(src.GetValue(src_off + 0)); \
            c##R##1 += static_cast<int32_t>(src.GetValue(src_off + 1)); \
            c##R##2 += static_cast<int32_t>(src.GetValue(src_off + 2)); \
            c##R##3 += static_cast<int32_t>(src.GetValue(src_off + 3)); \
            c##R##4 += static_cast<int32_t>(src.GetValue(src_off + 4)); \
            c##R##5 += static_cast<int32_t>(src.GetValue(src_off + 5)); \
            c##R##6 += static_cast<int32_t>(src.GetValue(src_off + 6)); \
          } while (0)
          switch (static_cast<uint32_t>(idx.GetValue(i))) {
            case 0: ADD_I8O1_ACC(0); break;
            case 1: ADD_I8O1_ACC(1); break;
            case 2: ADD_I8O1_ACC(2); break;
            case 3: ADD_I8O1_ACC(3); break;
            case 4: ADD_I8O1_ACC(4); break;
            case 5: ADD_I8O1_ACC(5); break;
            default: break;
          }
#undef ADD_I8O1_ACC
        }
      }

#define STORE_I8O1_ACC(R) \
      do { \
        row.SetValue((R) * 7 + 0, static_cast<int8_t>(c##R##0)); \
        row.SetValue((R) * 7 + 1, static_cast<int8_t>(c##R##1)); \
        row.SetValue((R) * 7 + 2, static_cast<int8_t>(c##R##2)); \
        row.SetValue((R) * 7 + 3, static_cast<int8_t>(c##R##3)); \
        row.SetValue((R) * 7 + 4, static_cast<int8_t>(c##R##4)); \
        row.SetValue((R) * 7 + 5, static_cast<int8_t>(c##R##5)); \
        row.SetValue((R) * 7 + 6, static_cast<int8_t>(c##R##6)); \
      } while (0)
      STORE_I8O1_ACC(0); STORE_I8O1_ACC(1); STORE_I8O1_ACC(2);
      STORE_I8O1_ACC(3); STORE_I8O1_ACC(4); STORE_I8O1_ACC(5);
      DataCopyPad(output_i8_gm_[InputOffset(outer_id, 0, 0)], row, rows_copy);
      PipeBarrier<PIPE_MTE3>();
#undef STORE_I8O1_ACC
    }
  }

  __aicore__ inline uint32_t Case4GatherPaddedRows(uint32_t rows) const {
    if (rows <= 8) return 8;
    if (rows <= 16) return 16;
    if (rows <= 32) return 32;
    if (rows <= 64) return 64;
    if (rows <= 128) return 128;
    if (rows <= 256) return 256;
    return 512;
  }

  __aicore__ inline void BuildCase4GatherbOffsets(
      LocalTensor<uint32_t>& offsets, LocalTensor<int32_t>& idx_values,
      const uint32_t tile, const uint32_t off_base, uint32_t& c0,
      uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t& c4,
      uint32_t& c5, uint32_t& s0, uint32_t& s1, uint32_t& s2,
      uint32_t& s3, uint32_t& s4, uint32_t& s5) {
    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    c5 = 0;
    const uint32_t base = tile * kCase4MaskReduceTile;
    for (uint32_t i = 0; i < kCase4MaskReduceTile; ++i) {
      switch (static_cast<uint32_t>(idx_values.GetValue(base + i))) {
        case 0: ++c0; break;
        case 1: ++c1; break;
        case 2: ++c2; break;
        case 3: ++c3; break;
        case 4: ++c4; break;
        case 5: ++c5; break;
        default: break;
      }
    }

    const uint32_t p0 = Case4GatherPaddedRows(c0);
    const uint32_t p1 = Case4GatherPaddedRows(c1);
    const uint32_t p2 = Case4GatherPaddedRows(c2);
    const uint32_t p3 = Case4GatherPaddedRows(c3);
    const uint32_t p4 = Case4GatherPaddedRows(c4);
    s0 = 0;
    s1 = s0 + p0;
    s2 = s1 + p1;
    s3 = s2 + p2;
    s4 = s3 + p3;
    s5 = s4 + p4;
    uint32_t w0 = s0;
    uint32_t w1 = s1;
    uint32_t w2 = s2;
    uint32_t w3 = s3;
    uint32_t w4 = s4;
    uint32_t w5 = s5;

    for (uint32_t i = 0; i < kCase4GatherbOffsetStride; ++i) {
      offsets.SetValue(off_base + i, 0);
    }
    for (uint32_t i = 0; i < kCase4MaskReduceTile; ++i) {
      const uint32_t row_off =
          i * kCase4GatherbWidth * static_cast<uint32_t>(sizeof(half));
      switch (static_cast<uint32_t>(idx_values.GetValue(base + i))) {
        case 0: offsets.SetValue(off_base + w0++, row_off); break;
        case 1: offsets.SetValue(off_base + w1++, row_off); break;
        case 2: offsets.SetValue(off_base + w2++, row_off); break;
        case 3: offsets.SetValue(off_base + w3++, row_off); break;
        case 4: offsets.SetValue(off_base + w4++, row_off); break;
        case 5: offsets.SetValue(off_base + w5++, row_off); break;
        default: break;
      }
    }
  }

  __aicore__ inline void ReduceCase4GatherbRows(
      LocalTensor<int16_t>& partial, LocalTensor<uint16_t>& gathered_u16,
      LocalTensor<half>& gathered_h, LocalTensor<int16_t>& gathered_i16,
      LocalTensor<uint16_t>& src_u16,
      LocalTensor<uint32_t>& offsets, const uint32_t off_base,
      const uint32_t start, const uint32_t rows, const uint32_t axis_id) {
    if (rows == 0) {
      return;
    }
    const uint32_t padded_rows = Case4GatherPaddedRows(rows);
    const uint32_t elems = rows * kCase4GatherbWidth;
    const uint32_t padded_elems = padded_rows * kCase4GatherbWidth;
    GatherRepeatParams gather_params(1, 8);
    Gatherb(gathered_u16, src_u16, offsets[off_base + start],
            static_cast<uint8_t>(padded_rows >> 3), gather_params);
    PipeBarrier<PIPE_V>();
    Cast(gathered_i16, gathered_h, RoundMode::CAST_RINT, padded_elems);
    PipeBarrier<PIPE_V>();
    if (padded_elems > elems) {
      Duplicate(gathered_i16[elems], static_cast<int16_t>(0),
                static_cast<int32_t>(padded_elems - elems));
      PipeBarrier<PIPE_V>();
    }

    for (uint32_t n = padded_rows; n > 8; n >>= 1) {
      const uint32_t h = n >> 1;
      const uint32_t add_elems = h * kCase4GatherbWidth;
      Add(gathered_i16, gathered_i16, gathered_i16[add_elems],
          static_cast<int32_t>(add_elems));
      PipeBarrier<PIPE_V>();
    }

    const uint32_t partial_base = axis_id * 8 * kCase4GatherbWidth;
    for (uint32_t r = 0; r < 8; ++r) {
      const uint32_t row_off = r * kCase4GatherbWidth;
      const uint32_t part_off = partial_base + row_off;
      partial.SetValue(part_off + 0,
                       partial.GetValue(part_off + 0) +
                           gathered_i16.GetValue(row_off + 0));
      partial.SetValue(part_off + 1,
                       partial.GetValue(part_off + 1) +
                           gathered_i16.GetValue(row_off + 1));
      partial.SetValue(part_off + 2,
                       partial.GetValue(part_off + 2) +
                           gathered_i16.GetValue(row_off + 2));
      partial.SetValue(part_off + 3,
                       partial.GetValue(part_off + 3) +
                           gathered_i16.GetValue(row_off + 3));
      partial.SetValue(part_off + 4,
                       partial.GetValue(part_off + 4) +
                           gathered_i16.GetValue(row_off + 4));
      partial.SetValue(part_off + 5,
                       partial.GetValue(part_off + 5) +
                           gathered_i16.GetValue(row_off + 5));
      partial.SetValue(part_off + 6,
                       partial.GetValue(part_off + 6) +
                           gathered_i16.GetValue(row_off + 6));
    }
  }

  __aicore__ inline void ProcessCase4GatherbReduceI8() {
    constexpr uint32_t kPartialElems = 6 * 8 * kCase4GatherbWidth;
    LocalTensor<uint32_t> offsets = index32_buf_.Get<uint32_t>();
    LocalTensor<int32_t> idx_values = src_buf_.Get<int32_t>();
    LocalTensor<int8_t> src_i8 = src_buf_.Get<int8_t>();
    LocalTensor<half> src_h = src_cast_buf_.Get<half>();
    LocalTensor<uint16_t> src_u16 = src_cast_buf_.Get<uint16_t>();
    LocalTensor<half> gathered_h = out_buf_.Get<half>();
    LocalTensor<uint16_t> gathered_u16 = out_buf_.Get<uint16_t>();
    LocalTensor<int16_t> gathered_i16 = acc_buf_.Get<int16_t>();
    LocalTensor<int16_t> partial =
        index32_buf_.Get<int16_t>()[kCase4GatherbOffsetStride * 4];
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams row_copy{1, 42, 0, 0};
    DataCopyParams src_copy{static_cast<uint16_t>(kCase4MaskReduceTile), 7, 0,
                            kCase4GatherbWidth - 7};

    uint32_t t0_c0, t0_c1, t0_c2, t0_c3, t0_c4, t0_c5;
    uint32_t t0_s0, t0_s1, t0_s2, t0_s3, t0_s4, t0_s5;
    uint32_t t1_c0, t1_c1, t1_c2, t1_c3, t1_c4, t1_c5;
    uint32_t t1_s0, t1_s1, t1_s2, t1_s3, t1_s4, t1_s5;
    RawCopyIndexTile(idx_values, 0, 1024);
    BuildCase4GatherbOffsets(offsets, idx_values, 0, 0, t0_c0, t0_c1, t0_c2,
                             t0_c3, t0_c4, t0_c5, t0_s0, t0_s1, t0_s2,
                             t0_s3, t0_s4, t0_s5);
    BuildCase4GatherbOffsets(offsets, idx_values, 1, kCase4GatherbOffsetStride,
                             t1_c0, t1_c1, t1_c2, t1_c3, t1_c4, t1_c5,
                             t1_s0, t1_s1, t1_s2, t1_s3, t1_s4, t1_s5);

    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    for (uint32_t outer_id = cid; outer_id < outer_; outer_id += core_num) {
      Duplicate(partial, static_cast<int16_t>(0),
                static_cast<int32_t>(kPartialElems));
      PipeBarrier<PIPE_V>();

      DataCopyPad(src_i8, source_i8_gm_[SourceOffset(outer_id, 0, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      Cast(src_h, src_i8, RoundMode::CAST_NONE, kCase4GatherbElems);
      PipeBarrier<PIPE_V>();
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, 0, t0_s0, t0_c0, 0);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, 0, t0_s1, t0_c1, 1);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, 0, t0_s2, t0_c2, 2);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, 0, t0_s3, t0_c3, 3);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, 0, t0_s4, t0_c4, 4);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, 0, t0_s5, t0_c5, 5);

      DataCopyPad(src_i8,
                  source_i8_gm_[SourceOffset(outer_id, kCase4MaskReduceTile, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      Cast(src_h, src_i8, RoundMode::CAST_NONE, kCase4GatherbElems);
      PipeBarrier<PIPE_V>();
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, kCase4GatherbOffsetStride,
                             t1_s0, t1_c0, 0);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, kCase4GatherbOffsetStride,
                             t1_s1, t1_c1, 1);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, kCase4GatherbOffsetStride,
                             t1_s2, t1_c2, 2);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, kCase4GatherbOffsetStride,
                             t1_s3, t1_c3, 3);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, kCase4GatherbOffsetStride,
                             t1_s4, t1_c4, 4);
      ReduceCase4GatherbRows(partial, gathered_u16, gathered_h, gathered_i16,
                             src_u16, offsets, kCase4GatherbOffsetStride,
                             t1_s5, t1_c5, 5);

      DataCopyPad(src_i8, input_i8_gm_[InputOffset(outer_id, 0, 0)], row_copy,
                  pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t axis_id = 0; axis_id < 6; ++axis_id) {
        const uint32_t dst_base = axis_id * 7;
        const uint32_t partial_base = axis_id * 8 * kCase4GatherbWidth;
        for (uint32_t w = 0; w < 7; ++w) {
          int32_t sum = static_cast<int32_t>(src_i8.GetValue(dst_base + w));
          for (uint32_t r = 0; r < 8; ++r) {
            sum += static_cast<int32_t>(
                partial.GetValue(partial_base + r * kCase4GatherbWidth + w));
          }
          src_i8.SetValue(dst_base + w, static_cast<int8_t>(sum));
        }
      }
      DataCopyPad(output_i8_gm_[InputOffset(outer_id, 0, 0)], src_i8,
                  row_copy);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  __aicore__ inline void StoreCase4GatherRowOffsets(
      LocalTensor<uint32_t>& offsets, const uint32_t off_base,
      const uint32_t pos, const uint32_t row) {
    const uint32_t byte_base =
        row * kCase4MaskReduceWidth * static_cast<uint32_t>(sizeof(half));
    offsets.SetValue(off_base + pos + 0, byte_base + 0 * sizeof(half));
    offsets.SetValue(off_base + pos + 1, byte_base + 1 * sizeof(half));
    offsets.SetValue(off_base + pos + 2, byte_base + 2 * sizeof(half));
    offsets.SetValue(off_base + pos + 3, byte_base + 3 * sizeof(half));
    offsets.SetValue(off_base + pos + 4, byte_base + 4 * sizeof(half));
    offsets.SetValue(off_base + pos + 5, byte_base + 5 * sizeof(half));
    offsets.SetValue(off_base + pos + 6, byte_base + 6 * sizeof(half));
    offsets.SetValue(off_base + pos + 7, byte_base + 7 * sizeof(half));
  }

  __aicore__ inline void BuildCase4GatherOffsets(
      LocalTensor<uint32_t>& offsets, LocalTensor<int32_t>& idx_values,
      const uint32_t tile, const uint32_t off_base, uint32_t& c0,
      uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t& c4,
      uint32_t& c5, uint32_t& s0, uint32_t& s1, uint32_t& s2,
      uint32_t& s3, uint32_t& s4, uint32_t& s5) {
    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    c5 = 0;
    const uint32_t base = tile * kCase4MaskReduceTile;
    for (uint32_t i = 0; i < kCase4MaskReduceTile; ++i) {
      switch (static_cast<uint32_t>(idx_values.GetValue(base + i))) {
        case 0: ++c0; break;
        case 1: ++c1; break;
        case 2: ++c2; break;
        case 3: ++c3; break;
        case 4: ++c4; break;
        case 5: ++c5; break;
        default: break;
      }
    }

    s0 = 0;
    s1 = s0 + c0 * kCase4MaskReduceWidth;
    s2 = s1 + c1 * kCase4MaskReduceWidth;
    s3 = s2 + c2 * kCase4MaskReduceWidth;
    s4 = s3 + c3 * kCase4MaskReduceWidth;
    s5 = s4 + c4 * kCase4MaskReduceWidth;
    uint32_t p0 = s0;
    uint32_t p1 = s1;
    uint32_t p2 = s2;
    uint32_t p3 = s3;
    uint32_t p4 = s4;
    uint32_t p5 = s5;

    for (uint32_t i = 0; i < kCase4MaskReduceTile; ++i) {
      switch (static_cast<uint32_t>(idx_values.GetValue(base + i))) {
        case 0:
          StoreCase4GatherRowOffsets(offsets, off_base, p0, i);
          p0 += kCase4MaskReduceWidth;
          break;
        case 1:
          StoreCase4GatherRowOffsets(offsets, off_base, p1, i);
          p1 += kCase4MaskReduceWidth;
          break;
        case 2:
          StoreCase4GatherRowOffsets(offsets, off_base, p2, i);
          p2 += kCase4MaskReduceWidth;
          break;
        case 3:
          StoreCase4GatherRowOffsets(offsets, off_base, p3, i);
          p3 += kCase4MaskReduceWidth;
          break;
        case 4:
          StoreCase4GatherRowOffsets(offsets, off_base, p4, i);
          p4 += kCase4MaskReduceWidth;
          break;
        case 5:
          StoreCase4GatherRowOffsets(offsets, off_base, p5, i);
          p5 += kCase4MaskReduceWidth;
          break;
        default:
          break;
      }
    }
  }

  __aicore__ inline void ReduceCase4GatherRows(
      LocalTensor<int16_t>& partial, LocalTensor<half>& gathered_h,
      LocalTensor<int16_t>& gathered_i16, LocalTensor<half>& src_h,
      LocalTensor<uint32_t>& offsets, const uint32_t off_base,
      const uint32_t start, const uint32_t rows, const uint32_t axis_id) {
    if (rows == 0) {
      return;
    }
    const uint32_t elems = rows * kCase4MaskReduceWidth;
    Gather(gathered_h, src_h, offsets[off_base + start], 0, elems);
    PipeBarrier<PIPE_V>();
    Cast(gathered_i16, gathered_h, RoundMode::CAST_RINT, elems);
    PipeBarrier<PIPE_V>();

    const uint32_t padded_rows = Case4GatherPaddedRows(rows);
    const uint32_t padded_elems = padded_rows * kCase4MaskReduceWidth;
    if (padded_elems > elems) {
      Duplicate(gathered_i16[elems], static_cast<int16_t>(0),
                static_cast<int32_t>(padded_elems - elems));
      PipeBarrier<PIPE_V>();
    }

    for (uint32_t n = padded_rows; n > 8; n >>= 1) {
      const uint32_t h = n >> 1;
      const uint32_t add_elems = h * kCase4MaskReduceWidth;
      Add(gathered_i16, gathered_i16, gathered_i16[add_elems],
          static_cast<int32_t>(add_elems));
      PipeBarrier<PIPE_V>();
    }

    const uint32_t partial_base = axis_id * 8 * kCase4MaskReduceWidth;
    for (uint32_t r = 0; r < 8; ++r) {
      const uint32_t row_off = r * kCase4MaskReduceWidth;
      const uint32_t part_off = partial_base + row_off;
      partial.SetValue(part_off + 0,
                       partial.GetValue(part_off + 0) +
                           gathered_i16.GetValue(row_off + 0));
      partial.SetValue(part_off + 1,
                       partial.GetValue(part_off + 1) +
                           gathered_i16.GetValue(row_off + 1));
      partial.SetValue(part_off + 2,
                       partial.GetValue(part_off + 2) +
                           gathered_i16.GetValue(row_off + 2));
      partial.SetValue(part_off + 3,
                       partial.GetValue(part_off + 3) +
                           gathered_i16.GetValue(row_off + 3));
      partial.SetValue(part_off + 4,
                       partial.GetValue(part_off + 4) +
                           gathered_i16.GetValue(row_off + 4));
      partial.SetValue(part_off + 5,
                       partial.GetValue(part_off + 5) +
                           gathered_i16.GetValue(row_off + 5));
      partial.SetValue(part_off + 6,
                       partial.GetValue(part_off + 6) +
                           gathered_i16.GetValue(row_off + 6));
    }
  }

  __aicore__ inline void ProcessCase4GatherReduceI8() {
    constexpr uint32_t kPartialElems = 6 * 8 * kCase4MaskReduceWidth;
    LocalTensor<uint32_t> offsets = index32_buf_.Get<uint32_t>();
    LocalTensor<int32_t> idx_values = src_buf_.Get<int32_t>();
    LocalTensor<int8_t> src_i8 = src_buf_.Get<int8_t>();
    LocalTensor<half> src_h = src_cast_buf_.Get<half>();
    LocalTensor<half> gathered_h = out_buf_.Get<half>();
    LocalTensor<int16_t> gathered_i16 = acc_buf_.Get<int16_t>();
    LocalTensor<int16_t> partial = acc_buf_.Get<int16_t>()[kCase4MaskReduceElems];
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams row_copy{1, 42, 0, 0};
    DataCopyParams src_copy{static_cast<uint16_t>(kCase4MaskReduceTile), 7, 0,
                            kCase4MaskReduceWidth - 7};

    uint32_t t0_c0, t0_c1, t0_c2, t0_c3, t0_c4, t0_c5;
    uint32_t t0_s0, t0_s1, t0_s2, t0_s3, t0_s4, t0_s5;
    uint32_t t1_c0, t1_c1, t1_c2, t1_c3, t1_c4, t1_c5;
    uint32_t t1_s0, t1_s1, t1_s2, t1_s3, t1_s4, t1_s5;
    RawCopyIndexTile(idx_values, 0, 1024);
    BuildCase4GatherOffsets(offsets, idx_values, 0, 0, t0_c0, t0_c1, t0_c2,
                            t0_c3, t0_c4, t0_c5, t0_s0, t0_s1, t0_s2,
                            t0_s3, t0_s4, t0_s5);
    BuildCase4GatherOffsets(offsets, idx_values, 1, kCase4MaskReduceElems,
                            t1_c0, t1_c1, t1_c2, t1_c3, t1_c4, t1_c5,
                            t1_s0, t1_s1, t1_s2, t1_s3, t1_s4, t1_s5);

    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    for (uint32_t outer_id = cid; outer_id < outer_; outer_id += core_num) {
      Duplicate(partial, static_cast<int16_t>(0),
                static_cast<int32_t>(kPartialElems));
      PipeBarrier<PIPE_V>();

      DataCopyPad(src_i8, source_i8_gm_[SourceOffset(outer_id, 0, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      Cast(src_h, src_i8, RoundMode::CAST_NONE, kCase4MaskReduceElems);
      PipeBarrier<PIPE_V>();
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            0, t0_s0, t0_c0, 0);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            0, t0_s1, t0_c1, 1);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            0, t0_s2, t0_c2, 2);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            0, t0_s3, t0_c3, 3);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            0, t0_s4, t0_c4, 4);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            0, t0_s5, t0_c5, 5);

      DataCopyPad(src_i8,
                  source_i8_gm_[SourceOffset(outer_id, kCase4MaskReduceTile, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      Cast(src_h, src_i8, RoundMode::CAST_NONE, kCase4MaskReduceElems);
      PipeBarrier<PIPE_V>();
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            kCase4MaskReduceElems, t1_s0, t1_c0, 0);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            kCase4MaskReduceElems, t1_s1, t1_c1, 1);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            kCase4MaskReduceElems, t1_s2, t1_c2, 2);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            kCase4MaskReduceElems, t1_s3, t1_c3, 3);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            kCase4MaskReduceElems, t1_s4, t1_c4, 4);
      ReduceCase4GatherRows(partial, gathered_h, gathered_i16, src_h, offsets,
                            kCase4MaskReduceElems, t1_s5, t1_c5, 5);

      DataCopyPad(src_i8, input_i8_gm_[InputOffset(outer_id, 0, 0)], row_copy,
                  pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t axis_id = 0; axis_id < 6; ++axis_id) {
        const uint32_t dst_base = axis_id * 7;
        const uint32_t partial_base = axis_id * 8 * kCase4MaskReduceWidth;
        for (uint32_t w = 0; w < 7; ++w) {
          int32_t sum = static_cast<int32_t>(src_i8.GetValue(dst_base + w));
          for (uint32_t r = 0; r < 8; ++r) {
            sum += static_cast<int32_t>(
                partial.GetValue(partial_base + r * kCase4MaskReduceWidth + w));
          }
          src_i8.SetValue(dst_base + w, static_cast<int8_t>(sum));
        }
      }
      DataCopyPad(output_i8_gm_[InputOffset(outer_id, 0, 0)], src_i8,
                  row_copy);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  __aicore__ inline void ProcessCase4MaskReduceI8() {
    LocalTensor<int32_t> idx_exp = index32_buf_.Get<int32_t>();
    LocalTensor<uint8_t> masks =
        index32_buf_.Get<uint8_t>()[kCase4MaskReduceIdxBytes];
    constexpr uint32_t kPartialElems = 6 * 8 * kCase4MaskReduceWidth;
    LocalTensor<int8_t> src_i8 = src_buf_.Get<int8_t>();
    LocalTensor<half> src_h = src_cast_buf_.Get<half>();
    LocalTensor<half> masked_h = out_buf_.Get<half>();
    LocalTensor<int16_t> masked_i16 = acc_buf_.Get<int16_t>();
    LocalTensor<int16_t> partial = acc_buf_.Get<int16_t>()[kCase4MaskReduceElems];
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams row_copy{1, 42, 0, 0};
    DataCopyParams src_copy{static_cast<uint16_t>(kCase4MaskReduceTile), 7, 0,
                            kCase4MaskReduceWidth - 7};

    for (uint32_t tile = 0; tile < 2; ++tile) {
      const uint32_t base = tile * kCase4MaskReduceTile;
      for (uint32_t i = 0; i < kCase4MaskReduceTile; ++i) {
        const int32_t v = index32_gm_.GetValue(base + i);
        const uint32_t off = i * kCase4MaskReduceWidth;
        idx_exp.SetValue(off + 0, v);
        idx_exp.SetValue(off + 1, v);
        idx_exp.SetValue(off + 2, v);
        idx_exp.SetValue(off + 3, v);
        idx_exp.SetValue(off + 4, v);
        idx_exp.SetValue(off + 5, v);
        idx_exp.SetValue(off + 6, v);
        idx_exp.SetValue(off + 7, v);
      }
      for (uint32_t axis_id = 0; axis_id < 6; ++axis_id) {
        const uint32_t mask_off =
            (tile * 6 + axis_id) * kCase4MaskReduceMaskBytes;
        CompareScalar(masks[mask_off], idx_exp, static_cast<int32_t>(axis_id),
                      CMPMODE::EQ, kCase4MaskReduceElems);
        PipeBarrier<PIPE_V>();
      }
    }

    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    for (uint32_t outer_id = cid; outer_id < outer_; outer_id += core_num) {
      for (uint32_t i = 0; i < kPartialElems; ++i) {
        partial.SetValue(i, static_cast<int16_t>(0));
      }

      for (uint32_t tile = 0; tile < 2; ++tile) {
        const uint32_t base = tile * kCase4MaskReduceTile;
        DataCopyPad(src_i8, source_i8_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        Cast(src_h, src_i8, RoundMode::CAST_NONE, kCase4MaskReduceElems);
        PipeBarrier<PIPE_V>();

        for (uint32_t axis_id = 0; axis_id < 6; ++axis_id) {
          const uint32_t mask_off =
              (tile * 6 + axis_id) * kCase4MaskReduceMaskBytes;
          Select(masked_h, masks[mask_off], src_h, static_cast<half>(0),
                 SELMODE::VSEL_TENSOR_SCALAR_MODE, kCase4MaskReduceElems);
          PipeBarrier<PIPE_V>();
          Cast(masked_i16, masked_h, RoundMode::CAST_RINT,
               kCase4MaskReduceElems);
          PipeBarrier<PIPE_V>();

          for (uint32_t n = kCase4MaskReduceTile; n > 8; n >>= 1) {
            const uint32_t h = n >> 1;
            const uint32_t elems = h * kCase4MaskReduceWidth;
            Add(masked_i16, masked_i16, masked_i16[elems],
                static_cast<int32_t>(elems));
            PipeBarrier<PIPE_V>();
          }

          const uint32_t partial_base = axis_id * 8 * kCase4MaskReduceWidth;
          for (uint32_t r = 0; r < 8; ++r) {
            const uint32_t row_off = r * kCase4MaskReduceWidth;
            const uint32_t part_off = partial_base + row_off;
            partial.SetValue(part_off + 0,
                             partial.GetValue(part_off + 0) +
                                 masked_i16.GetValue(row_off + 0));
            partial.SetValue(part_off + 1,
                             partial.GetValue(part_off + 1) +
                                 masked_i16.GetValue(row_off + 1));
            partial.SetValue(part_off + 2,
                             partial.GetValue(part_off + 2) +
                                 masked_i16.GetValue(row_off + 2));
            partial.SetValue(part_off + 3,
                             partial.GetValue(part_off + 3) +
                                 masked_i16.GetValue(row_off + 3));
            partial.SetValue(part_off + 4,
                             partial.GetValue(part_off + 4) +
                                 masked_i16.GetValue(row_off + 4));
            partial.SetValue(part_off + 5,
                             partial.GetValue(part_off + 5) +
                                 masked_i16.GetValue(row_off + 5));
            partial.SetValue(part_off + 6,
                             partial.GetValue(part_off + 6) +
                                 masked_i16.GetValue(row_off + 6));
          }
        }
      }

      DataCopyPad(src_i8, input_i8_gm_[InputOffset(outer_id, 0, 0)], row_copy,
                  pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t axis_id = 0; axis_id < 6; ++axis_id) {
        const uint32_t dst_base = axis_id * 7;
        const uint32_t partial_base = axis_id * 8 * kCase4MaskReduceWidth;
        for (uint32_t w = 0; w < 7; ++w) {
          int32_t sum = static_cast<int32_t>(src_i8.GetValue(dst_base + w));
          for (uint32_t r = 0; r < 8; ++r) {
            sum += static_cast<int32_t>(
                partial.GetValue(partial_base + r * kCase4MaskReduceWidth + w));
          }
          src_i8.SetValue(dst_base + w, static_cast<int8_t>(sum));
        }
      }
      DataCopyPad(output_i8_gm_[InputOffset(outer_id, 0, 0)], src_i8,
                  row_copy);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  __aicore__ inline void ProcessCase4IndexMajorAtomicSyncInt8() {
    constexpr uint32_t kOuterBytes = 42;
    constexpr uint32_t kSourceOuterBytes = 1024 * 7;
    constexpr uint32_t kLocalRowStride = 32;
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    LocalTensor<int8_t> buf = src_buf_.Get<int8_t>();
    LocalTensor<int32_t> sync_ub = src_cast_buf_.Get<int32_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams row_copy{1, kOuterBytes, 0, 0};
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();

    for (uint32_t outer_id = cid; outer_id < outer_; outer_id += core_num) {
      DataCopyPad(buf, input_i8_gm_[InputOffset(outer_id, 0, 0)], row_copy,
                  pad);
      PipeBarrier<PIPE_MTE2>();
      DataCopyPad(output_i8_gm_[InputOffset(outer_id, 0, 0)], buf, row_copy);
      PipeBarrier<PIPE_MTE3>();
    }

    SyncAll<true>(workspace_i32_gm_, sync_ub, static_cast<int32_t>(core_num));

    RawCopyIndexTile(idx, 0, 1024);
    SetAtomicAdd<int8_t>();
    for (uint32_t j = cid; j < 1024; j += core_num) {
      const uint32_t axis_id = static_cast<uint32_t>(idx.GetValue(j));
      DataCopyParams src_batch{static_cast<uint16_t>(outer_), 7,
                               static_cast<uint16_t>(kSourceOuterBytes - 7),
                               static_cast<uint16_t>(kLocalRowStride - 7)};
      DataCopyPad(buf, source_i8_gm_[SourceOffset(0, j, 0)], src_batch, pad);
      PipeBarrier<PIPE_MTE2>();
      DataCopyParams dst_batch{static_cast<uint16_t>(outer_), 7,
                               static_cast<uint16_t>(kLocalRowStride - 7),
                               static_cast<uint16_t>(kOuterBytes - 7)};
      DataCopyPad(output_i8_gm_[InputOffset(0, axis_id, 0)], buf, dst_batch);
      PipeBarrier<PIPE_MTE3>();
    }
    SetAtomicNone();
  }

  __aicore__ inline void ProcessCase4StridedAtomicBatch2Int8() {
    constexpr uint32_t kBatchRows = 2;
    constexpr uint32_t kOuterBytes = 42;
    constexpr uint32_t kSourceOuterBytes = 1024 * 7;
    constexpr uint32_t kLocalRowStride = 32;
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    LocalTensor<int8_t> buf = src_buf_.Get<int8_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t batches = (outer_ + kBatchRows - 1) / kBatchRows;

    if (cid >= batches) {
      return;
    }
    RawCopyIndexTile(idx, 0, 1024);

    for (uint32_t batch = cid; batch < batches; batch += core_num) {
      const uint32_t outer_base = batch * kBatchRows;
      const uint32_t rows = MinU32(kBatchRows, outer_ - outer_base);
      const uint32_t init_bytes = rows * kOuterBytes;
      DataCopyParams init_copy{1, static_cast<uint16_t>(init_bytes), 0, 0};
      DataCopyPad(buf, input_i8_gm_[InputOffset(outer_base, 0, 0)],
                  init_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      DataCopyPad(output_i8_gm_[InputOffset(outer_base, 0, 0)], buf,
                  init_copy);
      PipeBarrier<PIPE_MTE3>();

      SetAtomicAdd<int8_t>();
      for (uint32_t j = 0; j < 1024; ++j) {
        const uint32_t axis_id = static_cast<uint32_t>(idx.GetValue(j));
        DataCopyParams src_batch{
            static_cast<uint16_t>(rows),
            7,
            static_cast<uint16_t>(kSourceOuterBytes - 7),
            static_cast<uint16_t>(kLocalRowStride - 7)};
        DataCopyPad(buf, source_i8_gm_[SourceOffset(outer_base, j, 0)],
                    src_batch, pad);
        PipeBarrier<PIPE_MTE2>();
        DataCopyParams dst_batch{
            static_cast<uint16_t>(rows),
            7,
            static_cast<uint16_t>(kLocalRowStride - 7),
            static_cast<uint16_t>(kOuterBytes - 7)};
        DataCopyPad(output_i8_gm_[InputOffset(outer_base, axis_id, 0)], buf,
                    dst_batch);
        PipeBarrier<PIPE_MTE3>();
      }
      SetAtomicNone();
    }
  }

  __aicore__ inline void ProcessCase4Pack2Int8() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    LocalTensor<int8_t> row = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src0 = src_buf_.Get<int8_t>();
    LocalTensor<int8_t> src1 = src_cast_buf_.Get<int8_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, 42, 0, 0};
    const uint32_t src_tile_cap = kDataBytes / 7;
    const uint32_t tile_cap = MinU32(IndexTileCap32(), src_tile_cap);
    const uint32_t pairs = outer_ >> 1;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();

    for (uint32_t pair = cid; pair < pairs; pair += core_num) {
      const uint32_t outer0 = pair << 1;
      const uint32_t outer1 = outer0 + 1;

#define DECL_I8P2_ACC(P, R) \
      int32_t P##R##0 = 0; \
      int32_t P##R##1 = 0; \
      int32_t P##R##2 = 0; \
      int32_t P##R##3 = 0; \
      int32_t P##R##4 = 0; \
      int32_t P##R##5 = 0; \
      int32_t P##R##6 = 0
      DECL_I8P2_ACC(a, 0); DECL_I8P2_ACC(a, 1); DECL_I8P2_ACC(a, 2);
      DECL_I8P2_ACC(a, 3); DECL_I8P2_ACC(a, 4); DECL_I8P2_ACC(a, 5);
      DECL_I8P2_ACC(b, 0); DECL_I8P2_ACC(b, 1); DECL_I8P2_ACC(b, 2);
      DECL_I8P2_ACC(b, 3); DECL_I8P2_ACC(b, 4); DECL_I8P2_ACC(b, 5);
#undef DECL_I8P2_ACC

#define LOAD_I8P2_ACC(P, R, BASE) \
      P##R##0 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 0)); \
      P##R##1 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 1)); \
      P##R##2 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 2)); \
      P##R##3 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 3)); \
      P##R##4 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 4)); \
      P##R##5 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 5)); \
      P##R##6 = static_cast<int32_t>(input_i8_gm_.GetValue((BASE) + (R) * 7 + 6))
#define LOAD_I8P2_OUTER(P, OUTER) \
      do { \
        const uint64_t in_base = (uint64_t)(OUTER) * 42; \
        LOAD_I8P2_ACC(P, 0, in_base); LOAD_I8P2_ACC(P, 1, in_base); \
        LOAD_I8P2_ACC(P, 2, in_base); LOAD_I8P2_ACC(P, 3, in_base); \
        LOAD_I8P2_ACC(P, 4, in_base); LOAD_I8P2_ACC(P, 5, in_base); \
      } while (0)
      LOAD_I8P2_OUTER(a, outer0);
      LOAD_I8P2_OUTER(b, outer1);
#undef LOAD_I8P2_OUTER
#undef LOAD_I8P2_ACC

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7), 0, 0};
        DataCopyPad(src0, source_i8_gm_[SourceOffset(outer0, base, 0)],
                    src_copy, pad);
        DataCopyPad(src1, source_i8_gm_[SourceOffset(outer1, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t src_off = i * 7;
#define ADD_I8P2_ONE(P, R, BUF) \
          do { \
            P##R##0 += static_cast<int32_t>(BUF.GetValue(src_off + 0)); \
            P##R##1 += static_cast<int32_t>(BUF.GetValue(src_off + 1)); \
            P##R##2 += static_cast<int32_t>(BUF.GetValue(src_off + 2)); \
            P##R##3 += static_cast<int32_t>(BUF.GetValue(src_off + 3)); \
            P##R##4 += static_cast<int32_t>(BUF.GetValue(src_off + 4)); \
            P##R##5 += static_cast<int32_t>(BUF.GetValue(src_off + 5)); \
            P##R##6 += static_cast<int32_t>(BUF.GetValue(src_off + 6)); \
          } while (0)
#define ADD_I8P2_ROW2(R) \
          do { \
            ADD_I8P2_ONE(a, R, src0); \
            ADD_I8P2_ONE(b, R, src1); \
          } while (0)
          switch (static_cast<uint32_t>(idx.GetValue(i))) {
            case 0: ADD_I8P2_ROW2(0); break;
            case 1: ADD_I8P2_ROW2(1); break;
            case 2: ADD_I8P2_ROW2(2); break;
            case 3: ADD_I8P2_ROW2(3); break;
            case 4: ADD_I8P2_ROW2(4); break;
            case 5: ADD_I8P2_ROW2(5); break;
            default: break;
          }
#undef ADD_I8P2_ROW2
#undef ADD_I8P2_ONE
        }
      }

#define STORE_I8P2_ACC(P, R, OUTER) \
      do { \
        row.SetValue((R) * 7 + 0, static_cast<int8_t>(P##R##0)); \
        row.SetValue((R) * 7 + 1, static_cast<int8_t>(P##R##1)); \
        row.SetValue((R) * 7 + 2, static_cast<int8_t>(P##R##2)); \
        row.SetValue((R) * 7 + 3, static_cast<int8_t>(P##R##3)); \
        row.SetValue((R) * 7 + 4, static_cast<int8_t>(P##R##4)); \
        row.SetValue((R) * 7 + 5, static_cast<int8_t>(P##R##5)); \
        row.SetValue((R) * 7 + 6, static_cast<int8_t>(P##R##6)); \
      } while (0)
#define STORE_I8P2_OUTER(P, OUTER) \
      do { \
        STORE_I8P2_ACC(P, 0, OUTER); STORE_I8P2_ACC(P, 1, OUTER); \
        STORE_I8P2_ACC(P, 2, OUTER); STORE_I8P2_ACC(P, 3, OUTER); \
        STORE_I8P2_ACC(P, 4, OUTER); STORE_I8P2_ACC(P, 5, OUTER); \
        DataCopyPad(output_i8_gm_[InputOffset((OUTER), 0, 0)], row, rows_copy); \
        PipeBarrier<PIPE_MTE3>(); \
      } while (0)
      STORE_I8P2_OUTER(a, outer0);
      STORE_I8P2_OUTER(b, outer1);
#undef STORE_I8P2_OUTER
#undef STORE_I8P2_ACC
    }
  }

  __aicore__ inline uint64_t AddPackedI8x7(uint64_t a, uint64_t b) const {
    constexpr uint64_t kLow7 = 0x7f7f7f7f7f7f7f7fULL;
    constexpr uint64_t kHigh1 = 0x8080808080808080ULL;
    constexpr uint64_t kSevenBytes = 0x00ffffffffffffffULL;
    return (((a & kLow7) + (b & kLow7)) ^ ((a ^ b) & kHigh1)) & kSevenBytes;
  }

  __aicore__ inline uint64_t LoadPacked7(LocalTensor<uint64_t>& src,
                                         uint32_t byte_off) const {
    constexpr uint64_t kSevenBytes = 0x00ffffffffffffffULL;
    const uint32_t word = byte_off >> 3;
    const uint32_t shift = (byte_off & 7) << 3;
    const uint64_t lo = src.GetValue(word);
    if (shift == 0) {
      return lo & kSevenBytes;
    }
    const uint64_t hi = src.GetValue(word + 1);
    return ((lo >> shift) | (hi << (64 - shift))) & kSevenBytes;
  }

  __aicore__ inline void StorePacked7(LocalTensor<int8_t>& dst,
                                      uint32_t byte_off,
                                      uint64_t value) const {
    dst.SetValue(byte_off + 0, static_cast<int8_t>(value & 0xff));
    dst.SetValue(byte_off + 1, static_cast<int8_t>((value >> 8) & 0xff));
    dst.SetValue(byte_off + 2, static_cast<int8_t>((value >> 16) & 0xff));
    dst.SetValue(byte_off + 3, static_cast<int8_t>((value >> 24) & 0xff));
    dst.SetValue(byte_off + 4, static_cast<int8_t>((value >> 32) & 0xff));
    dst.SetValue(byte_off + 5, static_cast<int8_t>((value >> 40) & 0xff));
    dst.SetValue(byte_off + 6, static_cast<int8_t>((value >> 48) & 0xff));
  }

  __aicore__ inline void ProcessCase4Pack4Int8() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    LocalTensor<int8_t> src0 = src_buf_.Get<int8_t>();
    LocalTensor<int8_t> src1 = src_cast_buf_.Get<int8_t>();
    LocalTensor<int8_t> src2 = acc_buf_.Get<int8_t>();
    LocalTensor<int8_t> src3 = out_buf_.Get<int8_t>();
    LocalTensor<uint64_t> src0u = src_buf_.Get<uint64_t>();
    LocalTensor<uint64_t> src1u = src_cast_buf_.Get<uint64_t>();
    LocalTensor<uint64_t> src2u = acc_buf_.Get<uint64_t>();
    LocalTensor<uint64_t> src3u = out_buf_.Get<uint64_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, 42, 0, 0};
    const uint32_t src_tile_cap = kDataBytes / 7;
    const uint32_t tile_cap = MinU32(IndexTileCap32(), src_tile_cap);
    const bool cached_index = index_len_ <= tile_cap;
    LocalTensor<uint32_t> pair_code =
        index32_buf_.Get<uint32_t>()[(cached_index
                                          ? AlignUp32(index_len_ *
                                                      sizeof(int32_t))
                                          : 0) >> 2];
    const uint32_t groups = outer_ >> 2;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    if (cached_index) {
      RawCopyIndexTile(idx, 0, index_len_);
      for (uint32_t p = 0; p < (index_len_ >> 1); ++p) {
        const uint32_t r0 = static_cast<uint32_t>(idx.GetValue(p << 1));
        const uint32_t r1 = static_cast<uint32_t>(idx.GetValue((p << 1) + 1));
        pair_code.SetValue(p, r0 * 6 + r1);
      }
    }

    for (uint32_t group = cid; group < groups; group += core_num) {
      const uint32_t outer0 = group << 2;
      const uint32_t outer1 = outer0 + 1;
      const uint32_t outer2 = outer0 + 2;
      const uint32_t outer3 = outer0 + 3;

#define DECL_I8P4_PACK(P) \
      uint64_t P##0 = 0; \
      uint64_t P##1 = 0; \
      uint64_t P##2 = 0; \
      uint64_t P##3 = 0; \
      uint64_t P##4 = 0; \
      uint64_t P##5 = 0
      DECL_I8P4_PACK(a);
      DECL_I8P4_PACK(b);
      DECL_I8P4_PACK(c);
      DECL_I8P4_PACK(d);
#undef DECL_I8P4_PACK

#define LOAD_I8P4_OUTER(P, OUTER) \
      do { \
        DataCopyPad(src0, input_i8_gm_[InputOffset((OUTER), 0, 0)], rows_copy, pad); \
        PipeBarrier<PIPE_MTE2>(); \
        P##0 = LoadPacked7(src0u, 0); \
        P##1 = LoadPacked7(src0u, 7); \
        P##2 = LoadPacked7(src0u, 14); \
        P##3 = LoadPacked7(src0u, 21); \
        P##4 = LoadPacked7(src0u, 28); \
        P##5 = LoadPacked7(src0u, 35); \
      } while (0)
      LOAD_I8P4_OUTER(a, outer0);
      LOAD_I8P4_OUTER(b, outer1);
      LOAD_I8P4_OUTER(c, outer2);
      LOAD_I8P4_OUTER(d, outer3);
#undef LOAD_I8P4_OUTER

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        if (!cached_index) {
          RawCopyIndexTile(idx, base, count);
        }
        DataCopyParams src_copy{1, (uint16_t)(count * 7), 0, 0};
        DataCopyPad(src0, source_i8_gm_[SourceOffset(outer0, base, 0)],
                    src_copy, pad);
        DataCopyPad(src1, source_i8_gm_[SourceOffset(outer1, base, 0)],
                    src_copy, pad);
        DataCopyPad(src2, source_i8_gm_[SourceOffset(outer2, base, 0)],
                    src_copy, pad);
        DataCopyPad(src3, source_i8_gm_[SourceOffset(outer3, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        uint32_t i = 0;
#define ADD_I8P4_PACK(P, R) \
          do { \
            P##R = AddPackedI8x7(P##R, v##P); \
          } while (0)
#define ADD_I8P4_ROW4(R) \
          do { \
            const uint64_t va = LoadPacked7(src0u, src_off); \
            const uint64_t vb = LoadPacked7(src1u, src_off); \
            const uint64_t vc = LoadPacked7(src2u, src_off); \
            const uint64_t vd = LoadPacked7(src3u, src_off); \
            ADD_I8P4_PACK(a, R); \
            ADD_I8P4_PACK(b, R); \
            ADD_I8P4_PACK(c, R); \
            ADD_I8P4_PACK(d, R); \
          } while (0)
        for (; i + 1 < count; i += 2) {
          const uint32_t src_off0 = i * 7;
          const uint32_t src_off1 = src_off0 + 7;
          const uint64_t va0 = LoadPacked7(src0u, src_off0);
          const uint64_t vb0 = LoadPacked7(src1u, src_off0);
          const uint64_t vc0 = LoadPacked7(src2u, src_off0);
          const uint64_t vd0 = LoadPacked7(src3u, src_off0);
          const uint64_t va1 = LoadPacked7(src0u, src_off1);
          const uint64_t vb1 = LoadPacked7(src1u, src_off1);
          const uint64_t vc1 = LoadPacked7(src2u, src_off1);
          const uint64_t vd1 = LoadPacked7(src3u, src_off1);
          const uint32_t code =
              cached_index
                  ? pair_code.GetValue((base + i) >> 1)
                  : static_cast<uint32_t>(idx.GetValue(i)) * 6 +
                        static_cast<uint32_t>(idx.GetValue(i + 1));
#define ADD_I8P4_PACK_SRC(P, R, S) \
          do { \
            P##R = AddPackedI8x7(P##R, v##P##S); \
          } while (0)
#define ADD_I8P4_ROW4_SRC(R, S) \
          do { \
            ADD_I8P4_PACK_SRC(a, R, S); \
            ADD_I8P4_PACK_SRC(b, R, S); \
            ADD_I8P4_PACK_SRC(c, R, S); \
            ADD_I8P4_PACK_SRC(d, R, S); \
          } while (0)
#define ADD_I8P4_PAIR(R0, R1) \
          do { \
            ADD_I8P4_ROW4_SRC(R0, 0); \
            ADD_I8P4_ROW4_SRC(R1, 1); \
          } while (0)
          switch (code) {
            case 0: ADD_I8P4_PAIR(0, 0); break;
            case 1: ADD_I8P4_PAIR(0, 1); break;
            case 2: ADD_I8P4_PAIR(0, 2); break;
            case 3: ADD_I8P4_PAIR(0, 3); break;
            case 4: ADD_I8P4_PAIR(0, 4); break;
            case 5: ADD_I8P4_PAIR(0, 5); break;
            case 6: ADD_I8P4_PAIR(1, 0); break;
            case 7: ADD_I8P4_PAIR(1, 1); break;
            case 8: ADD_I8P4_PAIR(1, 2); break;
            case 9: ADD_I8P4_PAIR(1, 3); break;
            case 10: ADD_I8P4_PAIR(1, 4); break;
            case 11: ADD_I8P4_PAIR(1, 5); break;
            case 12: ADD_I8P4_PAIR(2, 0); break;
            case 13: ADD_I8P4_PAIR(2, 1); break;
            case 14: ADD_I8P4_PAIR(2, 2); break;
            case 15: ADD_I8P4_PAIR(2, 3); break;
            case 16: ADD_I8P4_PAIR(2, 4); break;
            case 17: ADD_I8P4_PAIR(2, 5); break;
            case 18: ADD_I8P4_PAIR(3, 0); break;
            case 19: ADD_I8P4_PAIR(3, 1); break;
            case 20: ADD_I8P4_PAIR(3, 2); break;
            case 21: ADD_I8P4_PAIR(3, 3); break;
            case 22: ADD_I8P4_PAIR(3, 4); break;
            case 23: ADD_I8P4_PAIR(3, 5); break;
            case 24: ADD_I8P4_PAIR(4, 0); break;
            case 25: ADD_I8P4_PAIR(4, 1); break;
            case 26: ADD_I8P4_PAIR(4, 2); break;
            case 27: ADD_I8P4_PAIR(4, 3); break;
            case 28: ADD_I8P4_PAIR(4, 4); break;
            case 29: ADD_I8P4_PAIR(4, 5); break;
            case 30: ADD_I8P4_PAIR(5, 0); break;
            case 31: ADD_I8P4_PAIR(5, 1); break;
            case 32: ADD_I8P4_PAIR(5, 2); break;
            case 33: ADD_I8P4_PAIR(5, 3); break;
            case 34: ADD_I8P4_PAIR(5, 4); break;
            case 35: ADD_I8P4_PAIR(5, 5); break;
            default: break;
          }
#undef ADD_I8P4_PAIR
#undef ADD_I8P4_ROW4_SRC
#undef ADD_I8P4_PACK_SRC
        }
        if (i < count) {
          const uint32_t src_off = i * 7;
          switch (static_cast<uint32_t>(
              idx.GetValue(cached_index ? base + i : i))) {
            case 0: ADD_I8P4_ROW4(0); break;
            case 1: ADD_I8P4_ROW4(1); break;
            case 2: ADD_I8P4_ROW4(2); break;
            case 3: ADD_I8P4_ROW4(3); break;
            case 4: ADD_I8P4_ROW4(4); break;
            case 5: ADD_I8P4_ROW4(5); break;
            default: break;
          }
        }
#undef ADD_I8P4_ROW4
#undef ADD_I8P4_PACK
      }

#define STORE_I8P4_ACC(P, R) \
      StorePacked7(src3, (R) * 7, P##R)
#define STORE_I8P4_OUTER(P, OUTER) \
      do { \
        STORE_I8P4_ACC(P, 0); STORE_I8P4_ACC(P, 1); \
        STORE_I8P4_ACC(P, 2); STORE_I8P4_ACC(P, 3); \
        STORE_I8P4_ACC(P, 4); STORE_I8P4_ACC(P, 5); \
        DataCopyPad(output_i8_gm_[InputOffset((OUTER), 0, 0)], src3, rows_copy); \
        PipeBarrier<PIPE_MTE3>(); \
      } while (0)
      STORE_I8P4_OUTER(a, outer0);
      STORE_I8P4_OUTER(b, outer1);
      STORE_I8P4_OUTER(c, outer2);
      STORE_I8P4_OUTER(d, outer3);
#undef STORE_I8P4_OUTER
#undef STORE_I8P4_ACC
    }
  }
  template <typename IdxT>
  __aicore__ inline void ProcessCase4Pack4Fp32(LocalTensor<IdxT>& idx) {
    LocalTensor<float> src0 = src_buf_.Get<float>();
    LocalTensor<float> src1 = src_cast_buf_.Get<float>();
    LocalTensor<float> src2 = acc_buf_.Get<float>();
    LocalTensor<float> src3 = out_buf_.Get<float>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, (uint16_t)(42 * sizeof(float)), 0, 0};
    const uint32_t src_tile_cap = kDataBytes / (7 * sizeof(float));
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
    const uint32_t groups = outer_ >> 2;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > groups) group_hi = groups;

    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t outer0 = group << 2;
      const uint32_t outer1 = outer0 + 1;
      const uint32_t outer2 = outer0 + 2;
      const uint32_t outer3 = outer0 + 3;

#define DECL_F32_ACC(P, R) \
      float P##R##0 = 0.0f; \
      float P##R##1 = 0.0f; \
      float P##R##2 = 0.0f; \
      float P##R##3 = 0.0f; \
      float P##R##4 = 0.0f; \
      float P##R##5 = 0.0f; \
      float P##R##6 = 0.0f
      DECL_F32_ACC(a, 0); DECL_F32_ACC(a, 1); DECL_F32_ACC(a, 2);
      DECL_F32_ACC(a, 3); DECL_F32_ACC(a, 4); DECL_F32_ACC(a, 5);
      DECL_F32_ACC(b, 0); DECL_F32_ACC(b, 1); DECL_F32_ACC(b, 2);
      DECL_F32_ACC(b, 3); DECL_F32_ACC(b, 4); DECL_F32_ACC(b, 5);
      DECL_F32_ACC(c, 0); DECL_F32_ACC(c, 1); DECL_F32_ACC(c, 2);
      DECL_F32_ACC(c, 3); DECL_F32_ACC(c, 4); DECL_F32_ACC(c, 5);
      DECL_F32_ACC(d, 0); DECL_F32_ACC(d, 1); DECL_F32_ACC(d, 2);
      DECL_F32_ACC(d, 3); DECL_F32_ACC(d, 4); DECL_F32_ACC(d, 5);
#undef DECL_F32_ACC

#define LOAD_F32_ACC(P, R, BUF) \
      P##R##0 = BUF.GetValue((R) * 7 + 0); \
      P##R##1 = BUF.GetValue((R) * 7 + 1); \
      P##R##2 = BUF.GetValue((R) * 7 + 2); \
      P##R##3 = BUF.GetValue((R) * 7 + 3); \
      P##R##4 = BUF.GetValue((R) * 7 + 4); \
      P##R##5 = BUF.GetValue((R) * 7 + 5); \
      P##R##6 = BUF.GetValue((R) * 7 + 6)
#define LOAD_F32_OUTER(P, OUTER) \
      do { \
        DataCopyPad(src0, input_f32_gm_[InputOffset((OUTER), 0, 0)], rows_copy, pad); \
        PipeBarrier<PIPE_MTE2>(); \
        LOAD_F32_ACC(P, 0, src0); LOAD_F32_ACC(P, 1, src0); \
        LOAD_F32_ACC(P, 2, src0); LOAD_F32_ACC(P, 3, src0); \
        LOAD_F32_ACC(P, 4, src0); LOAD_F32_ACC(P, 5, src0); \
      } while (0)
      LOAD_F32_OUTER(a, outer0);
      LOAD_F32_OUTER(b, outer1);
      LOAD_F32_OUTER(c, outer2);
      LOAD_F32_OUTER(d, outer3);
#undef LOAD_F32_OUTER
#undef LOAD_F32_ACC

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7 * sizeof(float)), 0, 0};
        DataCopyPad(src0, source_f32_gm_[SourceOffset(outer0, base, 0)],
                    src_copy, pad);
        DataCopyPad(src1, source_f32_gm_[SourceOffset(outer1, base, 0)],
                    src_copy, pad);
        DataCopyPad(src2, source_f32_gm_[SourceOffset(outer2, base, 0)],
                    src_copy, pad);
        DataCopyPad(src3, source_f32_gm_[SourceOffset(outer3, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t src_off = i * 7;
#define ADD_F32_ONE(P, R, BUF) \
          do { \
            P##R##0 += BUF.GetValue(src_off + 0); \
            P##R##1 += BUF.GetValue(src_off + 1); \
            P##R##2 += BUF.GetValue(src_off + 2); \
            P##R##3 += BUF.GetValue(src_off + 3); \
            P##R##4 += BUF.GetValue(src_off + 4); \
            P##R##5 += BUF.GetValue(src_off + 5); \
            P##R##6 += BUF.GetValue(src_off + 6); \
          } while (0)
#define ADD_F32_ROW4(R) \
          do { \
            ADD_F32_ONE(a, R, src0); \
            ADD_F32_ONE(b, R, src1); \
            ADD_F32_ONE(c, R, src2); \
            ADD_F32_ONE(d, R, src3); \
          } while (0)
          switch (static_cast<uint32_t>(idx.GetValue(i))) {
            case 0: ADD_F32_ROW4(0); break;
            case 1: ADD_F32_ROW4(1); break;
            case 2: ADD_F32_ROW4(2); break;
            case 3: ADD_F32_ROW4(3); break;
            case 4: ADD_F32_ROW4(4); break;
            case 5: ADD_F32_ROW4(5); break;
            default: break;
          }
#undef ADD_F32_ROW4
#undef ADD_F32_ONE
        }
      }

#define STORE_F32_ACC(P, R, OUTER) \
      do { \
        const uint64_t dst = InputOffset((OUTER), (R), 0); \
        output_f32_gm_.SetValue(dst + 0, P##R##0); \
        output_f32_gm_.SetValue(dst + 1, P##R##1); \
        output_f32_gm_.SetValue(dst + 2, P##R##2); \
        output_f32_gm_.SetValue(dst + 3, P##R##3); \
        output_f32_gm_.SetValue(dst + 4, P##R##4); \
        output_f32_gm_.SetValue(dst + 5, P##R##5); \
        output_f32_gm_.SetValue(dst + 6, P##R##6); \
      } while (0)
#define STORE_F32_OUTER(P, OUTER) \
      do { \
        STORE_F32_ACC(P, 0, OUTER); STORE_F32_ACC(P, 1, OUTER); \
        STORE_F32_ACC(P, 2, OUTER); STORE_F32_ACC(P, 3, OUTER); \
        STORE_F32_ACC(P, 4, OUTER); STORE_F32_ACC(P, 5, OUTER); \
      } while (0)
      STORE_F32_OUTER(a, outer0);
      STORE_F32_OUTER(b, outer1);
      STORE_F32_OUTER(c, outer2);
      STORE_F32_OUTER(d, outer3);
#undef STORE_F32_OUTER
#undef STORE_F32_ACC
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4Pack4Int32(LocalTensor<IdxT>& idx) {
    LocalTensor<int32_t> src0 = src_buf_.Get<int32_t>();
    LocalTensor<int32_t> src1 = src_cast_buf_.Get<int32_t>();
    LocalTensor<int32_t> src2 = acc_buf_.Get<int32_t>();
    LocalTensor<int32_t> src3 = out_buf_.Get<int32_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, (uint16_t)(42 * sizeof(int32_t)), 0, 0};
    const uint32_t src_tile_cap = kDataBytes / (7 * sizeof(int32_t));
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
    const uint32_t groups = outer_ >> 2;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > groups) group_hi = groups;

    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t outer0 = group << 2;
      const uint32_t outer1 = outer0 + 1;
      const uint32_t outer2 = outer0 + 2;
      const uint32_t outer3 = outer0 + 3;

#define DECL_I32_ACC(P, R) \
      int32_t P##R##0 = 0; \
      int32_t P##R##1 = 0; \
      int32_t P##R##2 = 0; \
      int32_t P##R##3 = 0; \
      int32_t P##R##4 = 0; \
      int32_t P##R##5 = 0; \
      int32_t P##R##6 = 0
      DECL_I32_ACC(a, 0); DECL_I32_ACC(a, 1); DECL_I32_ACC(a, 2);
      DECL_I32_ACC(a, 3); DECL_I32_ACC(a, 4); DECL_I32_ACC(a, 5);
      DECL_I32_ACC(b, 0); DECL_I32_ACC(b, 1); DECL_I32_ACC(b, 2);
      DECL_I32_ACC(b, 3); DECL_I32_ACC(b, 4); DECL_I32_ACC(b, 5);
      DECL_I32_ACC(c, 0); DECL_I32_ACC(c, 1); DECL_I32_ACC(c, 2);
      DECL_I32_ACC(c, 3); DECL_I32_ACC(c, 4); DECL_I32_ACC(c, 5);
      DECL_I32_ACC(d, 0); DECL_I32_ACC(d, 1); DECL_I32_ACC(d, 2);
      DECL_I32_ACC(d, 3); DECL_I32_ACC(d, 4); DECL_I32_ACC(d, 5);
#undef DECL_I32_ACC

#define LOAD_I32_ACC(P, R, BUF) \
      P##R##0 = BUF.GetValue((R) * 7 + 0); \
      P##R##1 = BUF.GetValue((R) * 7 + 1); \
      P##R##2 = BUF.GetValue((R) * 7 + 2); \
      P##R##3 = BUF.GetValue((R) * 7 + 3); \
      P##R##4 = BUF.GetValue((R) * 7 + 4); \
      P##R##5 = BUF.GetValue((R) * 7 + 5); \
      P##R##6 = BUF.GetValue((R) * 7 + 6)
#define LOAD_I32_OUTER(P, OUTER) \
      do { \
        DataCopyPad(src0, input_i32_gm_[InputOffset((OUTER), 0, 0)], rows_copy, pad); \
        PipeBarrier<PIPE_MTE2>(); \
        LOAD_I32_ACC(P, 0, src0); LOAD_I32_ACC(P, 1, src0); \
        LOAD_I32_ACC(P, 2, src0); LOAD_I32_ACC(P, 3, src0); \
        LOAD_I32_ACC(P, 4, src0); LOAD_I32_ACC(P, 5, src0); \
      } while (0)
      LOAD_I32_OUTER(a, outer0);
      LOAD_I32_OUTER(b, outer1);
      LOAD_I32_OUTER(c, outer2);
      LOAD_I32_OUTER(d, outer3);
#undef LOAD_I32_OUTER
#undef LOAD_I32_ACC

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7 * sizeof(int32_t)), 0, 0};
        DataCopyPad(src0, source_i32_gm_[SourceOffset(outer0, base, 0)],
                    src_copy, pad);
        DataCopyPad(src1, source_i32_gm_[SourceOffset(outer1, base, 0)],
                    src_copy, pad);
        DataCopyPad(src2, source_i32_gm_[SourceOffset(outer2, base, 0)],
                    src_copy, pad);
        DataCopyPad(src3, source_i32_gm_[SourceOffset(outer3, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t src_off = i * 7;
#define ADD_I32_ONE(P, R, BUF) \
          do { \
            P##R##0 = static_cast<int32_t>(static_cast<uint32_t>(P##R##0) + static_cast<uint32_t>(BUF.GetValue(src_off + 0))); \
            P##R##1 = static_cast<int32_t>(static_cast<uint32_t>(P##R##1) + static_cast<uint32_t>(BUF.GetValue(src_off + 1))); \
            P##R##2 = static_cast<int32_t>(static_cast<uint32_t>(P##R##2) + static_cast<uint32_t>(BUF.GetValue(src_off + 2))); \
            P##R##3 = static_cast<int32_t>(static_cast<uint32_t>(P##R##3) + static_cast<uint32_t>(BUF.GetValue(src_off + 3))); \
            P##R##4 = static_cast<int32_t>(static_cast<uint32_t>(P##R##4) + static_cast<uint32_t>(BUF.GetValue(src_off + 4))); \
            P##R##5 = static_cast<int32_t>(static_cast<uint32_t>(P##R##5) + static_cast<uint32_t>(BUF.GetValue(src_off + 5))); \
            P##R##6 = static_cast<int32_t>(static_cast<uint32_t>(P##R##6) + static_cast<uint32_t>(BUF.GetValue(src_off + 6))); \
          } while (0)
#define ADD_I32_ROW4(R) \
          do { \
            ADD_I32_ONE(a, R, src0); \
            ADD_I32_ONE(b, R, src1); \
            ADD_I32_ONE(c, R, src2); \
            ADD_I32_ONE(d, R, src3); \
          } while (0)
          switch (static_cast<uint32_t>(idx.GetValue(i))) {
            case 0: ADD_I32_ROW4(0); break;
            case 1: ADD_I32_ROW4(1); break;
            case 2: ADD_I32_ROW4(2); break;
            case 3: ADD_I32_ROW4(3); break;
            case 4: ADD_I32_ROW4(4); break;
            case 5: ADD_I32_ROW4(5); break;
            default: break;
          }
#undef ADD_I32_ROW4
#undef ADD_I32_ONE
        }
      }

#define STORE_I32_ACC(P, R, OUTER) \
      do { \
        const uint64_t dst = InputOffset((OUTER), (R), 0); \
        output_i32_gm_.SetValue(dst + 0, P##R##0); \
        output_i32_gm_.SetValue(dst + 1, P##R##1); \
        output_i32_gm_.SetValue(dst + 2, P##R##2); \
        output_i32_gm_.SetValue(dst + 3, P##R##3); \
        output_i32_gm_.SetValue(dst + 4, P##R##4); \
        output_i32_gm_.SetValue(dst + 5, P##R##5); \
        output_i32_gm_.SetValue(dst + 6, P##R##6); \
      } while (0)
#define STORE_I32_OUTER(P, OUTER) \
      do { \
        STORE_I32_ACC(P, 0, OUTER); STORE_I32_ACC(P, 1, OUTER); \
        STORE_I32_ACC(P, 2, OUTER); STORE_I32_ACC(P, 3, OUTER); \
        STORE_I32_ACC(P, 4, OUTER); STORE_I32_ACC(P, 5, OUTER); \
      } while (0)
      STORE_I32_OUTER(a, outer0);
      STORE_I32_OUTER(b, outer1);
      STORE_I32_OUTER(c, outer2);
      STORE_I32_OUTER(d, outer3);
#undef STORE_I32_OUTER
#undef STORE_I32_ACC
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4PairFp32(LocalTensor<IdxT>& idx) {
    LocalTensor<float> row = out_buf_.Get<float>();
    LocalTensor<float> src0 = src_buf_.Get<float>();
    LocalTensor<float> src1 = src_cast_buf_.Get<float>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, (uint16_t)(42 * sizeof(float)), 0, 0};
    const uint32_t src_tile_cap = kDataBytes / (7 * sizeof(float));
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
    const uint32_t pairs = (outer_ + 1) >> 1;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (pairs + core_num - 1) / core_num;
    uint32_t pair_lo = cid * per;
    uint32_t pair_hi = pair_lo + per;
    if (pair_hi > pairs) pair_hi = pairs;

    for (uint32_t pair = pair_lo; pair < pair_hi; ++pair) {
      const uint32_t outer0 = pair << 1;
      const uint32_t outer1 = outer0 + 1;
      const bool has1 = outer1 < outer_;

      DataCopyPad(row, input_f32_gm_[InputOffset(outer0, 0, 0)], rows_copy, pad);
      PipeBarrier<PIPE_MTE2>();
#define LOAD_F32_PAIR0(R) \
      float a##R##0 = row.GetValue((R) * 7 + 0); \
      float a##R##1 = row.GetValue((R) * 7 + 1); \
      float a##R##2 = row.GetValue((R) * 7 + 2); \
      float a##R##3 = row.GetValue((R) * 7 + 3); \
      float a##R##4 = row.GetValue((R) * 7 + 4); \
      float a##R##5 = row.GetValue((R) * 7 + 5); \
      float a##R##6 = row.GetValue((R) * 7 + 6)
      LOAD_F32_PAIR0(0);
      LOAD_F32_PAIR0(1);
      LOAD_F32_PAIR0(2);
      LOAD_F32_PAIR0(3);
      LOAD_F32_PAIR0(4);
      LOAD_F32_PAIR0(5);
#undef LOAD_F32_PAIR0

#define INIT_F32_PAIR1(R) \
      float b##R##0 = 0.0f; \
      float b##R##1 = 0.0f; \
      float b##R##2 = 0.0f; \
      float b##R##3 = 0.0f; \
      float b##R##4 = 0.0f; \
      float b##R##5 = 0.0f; \
      float b##R##6 = 0.0f
      INIT_F32_PAIR1(0);
      INIT_F32_PAIR1(1);
      INIT_F32_PAIR1(2);
      INIT_F32_PAIR1(3);
      INIT_F32_PAIR1(4);
      INIT_F32_PAIR1(5);
#undef INIT_F32_PAIR1
      if (has1) {
        DataCopyPad(row, input_f32_gm_[InputOffset(outer1, 0, 0)], rows_copy, pad);
        PipeBarrier<PIPE_MTE2>();
#define LOAD_F32_PAIR1(R) \
        b##R##0 = row.GetValue((R) * 7 + 0); \
        b##R##1 = row.GetValue((R) * 7 + 1); \
        b##R##2 = row.GetValue((R) * 7 + 2); \
        b##R##3 = row.GetValue((R) * 7 + 3); \
        b##R##4 = row.GetValue((R) * 7 + 4); \
        b##R##5 = row.GetValue((R) * 7 + 5); \
        b##R##6 = row.GetValue((R) * 7 + 6)
        LOAD_F32_PAIR1(0);
        LOAD_F32_PAIR1(1);
        LOAD_F32_PAIR1(2);
        LOAD_F32_PAIR1(3);
        LOAD_F32_PAIR1(4);
        LOAD_F32_PAIR1(5);
#undef LOAD_F32_PAIR1
      }

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7 * sizeof(float)), 0, 0};
        DataCopyPad(src0, source_f32_gm_[SourceOffset(outer0, base, 0)],
                    src_copy, pad);
        if (has1) {
          DataCopyPad(src1, source_f32_gm_[SourceOffset(outer1, base, 0)],
                      src_copy, pad);
        }
        PipeBarrier<PIPE_MTE2>();
        if (has1) {
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t src_off = i * 7;
#define ADD_F32_PAIR_ROW(R) \
            do { \
              a##R##0 += src0.GetValue(src_off + 0); \
              a##R##1 += src0.GetValue(src_off + 1); \
              a##R##2 += src0.GetValue(src_off + 2); \
              a##R##3 += src0.GetValue(src_off + 3); \
              a##R##4 += src0.GetValue(src_off + 4); \
              a##R##5 += src0.GetValue(src_off + 5); \
              a##R##6 += src0.GetValue(src_off + 6); \
              b##R##0 += src1.GetValue(src_off + 0); \
              b##R##1 += src1.GetValue(src_off + 1); \
              b##R##2 += src1.GetValue(src_off + 2); \
              b##R##3 += src1.GetValue(src_off + 3); \
              b##R##4 += src1.GetValue(src_off + 4); \
              b##R##5 += src1.GetValue(src_off + 5); \
              b##R##6 += src1.GetValue(src_off + 6); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_F32_PAIR_ROW(0); break;
              case 1: ADD_F32_PAIR_ROW(1); break;
              case 2: ADD_F32_PAIR_ROW(2); break;
              case 3: ADD_F32_PAIR_ROW(3); break;
              case 4: ADD_F32_PAIR_ROW(4); break;
              case 5: ADD_F32_PAIR_ROW(5); break;
              default: break;
            }
#undef ADD_F32_PAIR_ROW
          }
        } else {
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t src_off = i * 7;
#define ADD_F32_SINGLE_ROW(R) \
            do { \
              a##R##0 += src0.GetValue(src_off + 0); \
              a##R##1 += src0.GetValue(src_off + 1); \
              a##R##2 += src0.GetValue(src_off + 2); \
              a##R##3 += src0.GetValue(src_off + 3); \
              a##R##4 += src0.GetValue(src_off + 4); \
              a##R##5 += src0.GetValue(src_off + 5); \
              a##R##6 += src0.GetValue(src_off + 6); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_F32_SINGLE_ROW(0); break;
              case 1: ADD_F32_SINGLE_ROW(1); break;
              case 2: ADD_F32_SINGLE_ROW(2); break;
              case 3: ADD_F32_SINGLE_ROW(3); break;
              case 4: ADD_F32_SINGLE_ROW(4); break;
              case 5: ADD_F32_SINGLE_ROW(5); break;
              default: break;
            }
#undef ADD_F32_SINGLE_ROW
          }
        }
      }

#define STORE_F32_PAIR0(R) \
      do { \
        const uint64_t dst = InputOffset(outer0, (R), 0); \
        output_f32_gm_.SetValue(dst + 0, a##R##0); \
        output_f32_gm_.SetValue(dst + 1, a##R##1); \
        output_f32_gm_.SetValue(dst + 2, a##R##2); \
        output_f32_gm_.SetValue(dst + 3, a##R##3); \
        output_f32_gm_.SetValue(dst + 4, a##R##4); \
        output_f32_gm_.SetValue(dst + 5, a##R##5); \
        output_f32_gm_.SetValue(dst + 6, a##R##6); \
      } while (0)
      STORE_F32_PAIR0(0);
      STORE_F32_PAIR0(1);
      STORE_F32_PAIR0(2);
      STORE_F32_PAIR0(3);
      STORE_F32_PAIR0(4);
      STORE_F32_PAIR0(5);
#undef STORE_F32_PAIR0
      if (has1) {
#define STORE_F32_PAIR1(R) \
        do { \
          const uint64_t dst = InputOffset(outer1, (R), 0); \
          output_f32_gm_.SetValue(dst + 0, b##R##0); \
          output_f32_gm_.SetValue(dst + 1, b##R##1); \
          output_f32_gm_.SetValue(dst + 2, b##R##2); \
          output_f32_gm_.SetValue(dst + 3, b##R##3); \
          output_f32_gm_.SetValue(dst + 4, b##R##4); \
          output_f32_gm_.SetValue(dst + 5, b##R##5); \
          output_f32_gm_.SetValue(dst + 6, b##R##6); \
        } while (0)
        STORE_F32_PAIR1(0);
        STORE_F32_PAIR1(1);
        STORE_F32_PAIR1(2);
        STORE_F32_PAIR1(3);
        STORE_F32_PAIR1(4);
        STORE_F32_PAIR1(5);
#undef STORE_F32_PAIR1
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessCase4PairInt32(LocalTensor<IdxT>& idx) {
    LocalTensor<int32_t> row = out_buf_.Get<int32_t>();
    LocalTensor<int32_t> src0 = src_buf_.Get<int32_t>();
    LocalTensor<int32_t> src1 = src_cast_buf_.Get<int32_t>();
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams rows_copy{1, (uint16_t)(42 * sizeof(int32_t)), 0, 0};
    const uint32_t src_tile_cap = kDataBytes / (7 * sizeof(int32_t));
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
    const uint32_t pairs = (outer_ + 1) >> 1;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (pairs + core_num - 1) / core_num;
    uint32_t pair_lo = cid * per;
    uint32_t pair_hi = pair_lo + per;
    if (pair_hi > pairs) pair_hi = pairs;

    for (uint32_t pair = pair_lo; pair < pair_hi; ++pair) {
      const uint32_t outer0 = pair << 1;
      const uint32_t outer1 = outer0 + 1;
      const bool has1 = outer1 < outer_;

      DataCopyPad(row, input_i32_gm_[InputOffset(outer0, 0, 0)], rows_copy, pad);
      PipeBarrier<PIPE_MTE2>();
#define LOAD_I32_PAIR0(R) \
      int32_t a##R##0 = row.GetValue((R) * 7 + 0); \
      int32_t a##R##1 = row.GetValue((R) * 7 + 1); \
      int32_t a##R##2 = row.GetValue((R) * 7 + 2); \
      int32_t a##R##3 = row.GetValue((R) * 7 + 3); \
      int32_t a##R##4 = row.GetValue((R) * 7 + 4); \
      int32_t a##R##5 = row.GetValue((R) * 7 + 5); \
      int32_t a##R##6 = row.GetValue((R) * 7 + 6)
      LOAD_I32_PAIR0(0);
      LOAD_I32_PAIR0(1);
      LOAD_I32_PAIR0(2);
      LOAD_I32_PAIR0(3);
      LOAD_I32_PAIR0(4);
      LOAD_I32_PAIR0(5);
#undef LOAD_I32_PAIR0

#define INIT_I32_PAIR1(R) \
      int32_t b##R##0 = 0; \
      int32_t b##R##1 = 0; \
      int32_t b##R##2 = 0; \
      int32_t b##R##3 = 0; \
      int32_t b##R##4 = 0; \
      int32_t b##R##5 = 0; \
      int32_t b##R##6 = 0
      INIT_I32_PAIR1(0);
      INIT_I32_PAIR1(1);
      INIT_I32_PAIR1(2);
      INIT_I32_PAIR1(3);
      INIT_I32_PAIR1(4);
      INIT_I32_PAIR1(5);
#undef INIT_I32_PAIR1
      if (has1) {
        DataCopyPad(row, input_i32_gm_[InputOffset(outer1, 0, 0)], rows_copy, pad);
        PipeBarrier<PIPE_MTE2>();
#define LOAD_I32_PAIR1(R) \
        b##R##0 = row.GetValue((R) * 7 + 0); \
        b##R##1 = row.GetValue((R) * 7 + 1); \
        b##R##2 = row.GetValue((R) * 7 + 2); \
        b##R##3 = row.GetValue((R) * 7 + 3); \
        b##R##4 = row.GetValue((R) * 7 + 4); \
        b##R##5 = row.GetValue((R) * 7 + 5); \
        b##R##6 = row.GetValue((R) * 7 + 6)
        LOAD_I32_PAIR1(0);
        LOAD_I32_PAIR1(1);
        LOAD_I32_PAIR1(2);
        LOAD_I32_PAIR1(3);
        LOAD_I32_PAIR1(4);
        LOAD_I32_PAIR1(5);
#undef LOAD_I32_PAIR1
      }

      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * 7 * sizeof(int32_t)), 0, 0};
        DataCopyPad(src0, source_i32_gm_[SourceOffset(outer0, base, 0)],
                    src_copy, pad);
        if (has1) {
          DataCopyPad(src1, source_i32_gm_[SourceOffset(outer1, base, 0)],
                      src_copy, pad);
        }
        PipeBarrier<PIPE_MTE2>();
        if (has1) {
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t src_off = i * 7;
#define ADD_I32_PAIR_ROW(R) \
            do { \
              a##R##0 = static_cast<int32_t>(static_cast<uint32_t>(a##R##0) + static_cast<uint32_t>(src0.GetValue(src_off + 0))); \
              a##R##1 = static_cast<int32_t>(static_cast<uint32_t>(a##R##1) + static_cast<uint32_t>(src0.GetValue(src_off + 1))); \
              a##R##2 = static_cast<int32_t>(static_cast<uint32_t>(a##R##2) + static_cast<uint32_t>(src0.GetValue(src_off + 2))); \
              a##R##3 = static_cast<int32_t>(static_cast<uint32_t>(a##R##3) + static_cast<uint32_t>(src0.GetValue(src_off + 3))); \
              a##R##4 = static_cast<int32_t>(static_cast<uint32_t>(a##R##4) + static_cast<uint32_t>(src0.GetValue(src_off + 4))); \
              a##R##5 = static_cast<int32_t>(static_cast<uint32_t>(a##R##5) + static_cast<uint32_t>(src0.GetValue(src_off + 5))); \
              a##R##6 = static_cast<int32_t>(static_cast<uint32_t>(a##R##6) + static_cast<uint32_t>(src0.GetValue(src_off + 6))); \
              b##R##0 = static_cast<int32_t>(static_cast<uint32_t>(b##R##0) + static_cast<uint32_t>(src1.GetValue(src_off + 0))); \
              b##R##1 = static_cast<int32_t>(static_cast<uint32_t>(b##R##1) + static_cast<uint32_t>(src1.GetValue(src_off + 1))); \
              b##R##2 = static_cast<int32_t>(static_cast<uint32_t>(b##R##2) + static_cast<uint32_t>(src1.GetValue(src_off + 2))); \
              b##R##3 = static_cast<int32_t>(static_cast<uint32_t>(b##R##3) + static_cast<uint32_t>(src1.GetValue(src_off + 3))); \
              b##R##4 = static_cast<int32_t>(static_cast<uint32_t>(b##R##4) + static_cast<uint32_t>(src1.GetValue(src_off + 4))); \
              b##R##5 = static_cast<int32_t>(static_cast<uint32_t>(b##R##5) + static_cast<uint32_t>(src1.GetValue(src_off + 5))); \
              b##R##6 = static_cast<int32_t>(static_cast<uint32_t>(b##R##6) + static_cast<uint32_t>(src1.GetValue(src_off + 6))); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_I32_PAIR_ROW(0); break;
              case 1: ADD_I32_PAIR_ROW(1); break;
              case 2: ADD_I32_PAIR_ROW(2); break;
              case 3: ADD_I32_PAIR_ROW(3); break;
              case 4: ADD_I32_PAIR_ROW(4); break;
              case 5: ADD_I32_PAIR_ROW(5); break;
              default: break;
            }
#undef ADD_I32_PAIR_ROW
          }
        } else {
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t src_off = i * 7;
#define ADD_I32_SINGLE_ROW(R) \
            do { \
              a##R##0 = static_cast<int32_t>(static_cast<uint32_t>(a##R##0) + static_cast<uint32_t>(src0.GetValue(src_off + 0))); \
              a##R##1 = static_cast<int32_t>(static_cast<uint32_t>(a##R##1) + static_cast<uint32_t>(src0.GetValue(src_off + 1))); \
              a##R##2 = static_cast<int32_t>(static_cast<uint32_t>(a##R##2) + static_cast<uint32_t>(src0.GetValue(src_off + 2))); \
              a##R##3 = static_cast<int32_t>(static_cast<uint32_t>(a##R##3) + static_cast<uint32_t>(src0.GetValue(src_off + 3))); \
              a##R##4 = static_cast<int32_t>(static_cast<uint32_t>(a##R##4) + static_cast<uint32_t>(src0.GetValue(src_off + 4))); \
              a##R##5 = static_cast<int32_t>(static_cast<uint32_t>(a##R##5) + static_cast<uint32_t>(src0.GetValue(src_off + 5))); \
              a##R##6 = static_cast<int32_t>(static_cast<uint32_t>(a##R##6) + static_cast<uint32_t>(src0.GetValue(src_off + 6))); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_I32_SINGLE_ROW(0); break;
              case 1: ADD_I32_SINGLE_ROW(1); break;
              case 2: ADD_I32_SINGLE_ROW(2); break;
              case 3: ADD_I32_SINGLE_ROW(3); break;
              case 4: ADD_I32_SINGLE_ROW(4); break;
              case 5: ADD_I32_SINGLE_ROW(5); break;
              default: break;
            }
#undef ADD_I32_SINGLE_ROW
          }
        }
      }

#define STORE_I32_PAIR0(R) \
      do { \
        const uint64_t dst = InputOffset(outer0, (R), 0); \
        output_i32_gm_.SetValue(dst + 0, a##R##0); \
        output_i32_gm_.SetValue(dst + 1, a##R##1); \
        output_i32_gm_.SetValue(dst + 2, a##R##2); \
        output_i32_gm_.SetValue(dst + 3, a##R##3); \
        output_i32_gm_.SetValue(dst + 4, a##R##4); \
        output_i32_gm_.SetValue(dst + 5, a##R##5); \
        output_i32_gm_.SetValue(dst + 6, a##R##6); \
      } while (0)
      STORE_I32_PAIR0(0);
      STORE_I32_PAIR0(1);
      STORE_I32_PAIR0(2);
      STORE_I32_PAIR0(3);
      STORE_I32_PAIR0(4);
      STORE_I32_PAIR0(5);
#undef STORE_I32_PAIR0
      if (has1) {
#define STORE_I32_PAIR1(R) \
        do { \
          const uint64_t dst = InputOffset(outer1, (R), 0); \
          output_i32_gm_.SetValue(dst + 0, b##R##0); \
          output_i32_gm_.SetValue(dst + 1, b##R##1); \
          output_i32_gm_.SetValue(dst + 2, b##R##2); \
          output_i32_gm_.SetValue(dst + 3, b##R##3); \
          output_i32_gm_.SetValue(dst + 4, b##R##4); \
          output_i32_gm_.SetValue(dst + 5, b##R##5); \
          output_i32_gm_.SetValue(dst + 6, b##R##6); \
        } while (0)
        STORE_I32_PAIR1(0);
        STORE_I32_PAIR1(1);
        STORE_I32_PAIR1(2);
        STORE_I32_PAIR1(3);
        STORE_I32_PAIR1(4);
        STORE_I32_PAIR1(5);
#undef STORE_I32_PAIR1
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void DispatchGroupDtype(uint32_t outer_id,
                                            uint32_t axis_base, uint32_t rows,
                                            uint32_t inner_off, uint32_t seg,
                                            LocalTensor<IdxT>& idx) {
#if INDEXADD_COMPILE_INT8
    ProcessGroupInt8(outer_id, axis_base, rows, inner_off, seg, idx);
#elif INDEXADD_COMPILE_FP16
    ProcessGroupFp16(outer_id, axis_base, rows, inner_off, seg, idx);
#elif INDEXADD_COMPILE_FP32
    ProcessGroupFp32(outer_id, axis_base, rows, inner_off, seg, idx);
#elif INDEXADD_COMPILE_INT32
    ProcessGroupInt32(outer_id, axis_base, rows, inner_off, seg, idx);
#elif INDEXADD_COMPILE_BF16
    ProcessGroupBf16(outer_id, axis_base, rows, inner_off, seg, idx);
#else
    if (dtype_mode_ == kDtypeInt8) {
      ProcessGroupInt8(outer_id, axis_base, rows, inner_off, seg, idx);
    } else if (dtype_mode_ == kDtypeFp16) {
      ProcessGroupFp16(outer_id, axis_base, rows, inner_off, seg, idx);
    } else if (dtype_mode_ == kDtypeFp32) {
      ProcessGroupFp32(outer_id, axis_base, rows, inner_off, seg, idx);
    } else if (dtype_mode_ == kDtypeInt32) {
      ProcessGroupInt32(outer_id, axis_base, rows, inner_off, seg, idx);
    } else {
      ProcessGroupBf16(outer_id, axis_base, rows, inner_off, seg, idx);
    }
#endif
  }

  template <typename IdxT>
  __aicore__ inline void DispatchInner1Dtype(uint32_t outer_id,
                                             LocalTensor<IdxT>& idx) {
#if INDEXADD_COMPILE_INT8
    ProcessInner1RowInt8(outer_id, idx);
#elif INDEXADD_COMPILE_FP16
    ProcessInner1RowFp16(outer_id, idx);
#elif INDEXADD_COMPILE_FP32
    ProcessInner1RowFp32(outer_id, idx);
#elif INDEXADD_COMPILE_INT32
    ProcessInner1RowInt32(outer_id, idx);
#elif INDEXADD_COMPILE_BF16
    ProcessInner1RowBf16(outer_id, idx);
#else
    if (dtype_mode_ == kDtypeInt8) {
      ProcessInner1RowInt8(outer_id, idx);
    } else if (dtype_mode_ == kDtypeFp16) {
      ProcessInner1RowFp16(outer_id, idx);
    } else if (dtype_mode_ == kDtypeFp32) {
      ProcessInner1RowFp32(outer_id, idx);
    } else if (dtype_mode_ == kDtypeInt32) {
      ProcessInner1RowInt32(outer_id, idx);
    } else {
      ProcessInner1RowBf16(outer_id, idx);
    }
#endif
  }

  template <typename IdxT>
  __aicore__ inline void DispatchSegmentDtype(uint32_t outer_id,
                                              uint32_t axis_id,
                                              uint32_t inner_off,
                                              uint32_t seg,
                                              LocalTensor<IdxT>& idx) {
#if INDEXADD_COMPILE_INT8
    ProcessSegmentInt8(outer_id, axis_id, inner_off, seg, idx);
#elif INDEXADD_COMPILE_FP16
    ProcessSegmentFp16(outer_id, axis_id, inner_off, seg, idx);
#elif INDEXADD_COMPILE_FP32
    ProcessSegmentFp32(outer_id, axis_id, inner_off, seg, idx);
#elif INDEXADD_COMPILE_INT32
    ProcessSegmentInt32(outer_id, axis_id, inner_off, seg, idx);
#elif INDEXADD_COMPILE_BF16
    ProcessSegmentBf16(outer_id, axis_id, inner_off, seg, idx);
#else
    if (dtype_mode_ == kDtypeInt8) {
      ProcessSegmentInt8(outer_id, axis_id, inner_off, seg, idx);
    } else if (dtype_mode_ == kDtypeFp16) {
      ProcessSegmentFp16(outer_id, axis_id, inner_off, seg, idx);
    } else if (dtype_mode_ == kDtypeFp32) {
      ProcessSegmentFp32(outer_id, axis_id, inner_off, seg, idx);
    } else if (dtype_mode_ == kDtypeInt32) {
      ProcessSegmentInt32(outer_id, axis_id, inner_off, seg, idx);
    } else {
      ProcessSegmentBf16(outer_id, axis_id, inner_off, seg, idx);
    }
#endif
  }

  template <typename IdxT>
  __aicore__ inline void ProcessGroupedIndex(LocalTensor<IdxT>& idx) {
    cached_index_ = false;
    const uint32_t group_rows = GroupRows();
    const uint32_t axis_groups = (axis_ + group_rows - 1) / group_rows;
    const uint32_t seg_elems = GroupSegElems();
    const uint32_t group_inner_tiles = (inner_ + seg_elems - 1) / seg_elems;
    const uint32_t outer_group_rows = GroupedOuterRows();
    const uint32_t outer_groups =
        (outer_ + outer_group_rows - 1) / outer_group_rows;
    const uint64_t work_units =
        (uint64_t)outer_groups * axis_groups * group_inner_tiles;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint64_t per = (work_units + core_num - 1) / core_num;
    uint64_t lo = (uint64_t)cid * per;
    uint64_t hi = lo + per;
    if (hi > work_units) hi = work_units;
    for (uint64_t work = lo; work < hi; ++work) {
      const uint32_t inner_tile = (uint32_t)(work % group_inner_tiles);
      const uint64_t tmp = work / group_inner_tiles;
      const uint32_t axis_group = (uint32_t)(tmp % axis_groups);
      const uint32_t outer_group = (uint32_t)(tmp / axis_groups);
      const uint32_t outer_base = outer_group * outer_group_rows;
      const uint32_t outer_rows = MinU32(outer_group_rows, outer_ - outer_base);
      const uint32_t axis_base = axis_group * group_rows;
      const uint32_t rows = MinU32(group_rows, axis_ - axis_base);
      const uint32_t inner_off = inner_tile * seg_elems;
      const uint32_t seg = MinU32(inner_ - inner_off, seg_elems);
      for (uint32_t og = 0; og < outer_rows; ++og) {
        const uint32_t outer_id = outer_base + og;
        DispatchGroupDtype(outer_id, axis_base, rows, inner_off, seg, idx);
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessOuterOwnedGroupedIndex(LocalTensor<IdxT>& idx) {
    cached_index_ = false;
    const uint32_t group_rows = GroupRows();
    const uint32_t axis_groups = (axis_ + group_rows - 1) / group_rows;
    const uint32_t seg_elems = GroupSegElems();
    const uint32_t group_inner_tiles = (inner_ + seg_elems - 1) / seg_elems;
    const uint32_t outer_group_rows = GroupedOuterRows();
    const uint32_t outer_groups =
        (outer_ + outer_group_rows - 1) / outer_group_rows;
    const uint64_t work_units =
        (uint64_t)outer_groups * group_inner_tiles;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint64_t per = (work_units + core_num - 1) / core_num;
    uint64_t lo = (uint64_t)cid * per;
    uint64_t hi = lo + per;
    if (hi > work_units) hi = work_units;
    for (uint64_t work = lo; work < hi; ++work) {
      const uint32_t inner_tile = (uint32_t)(work % group_inner_tiles);
      const uint32_t outer_group = (uint32_t)(work / group_inner_tiles);
      const uint32_t outer_base = outer_group * outer_group_rows;
      const uint32_t outer_rows = MinU32(outer_group_rows, outer_ - outer_base);
      const uint32_t inner_off = inner_tile * seg_elems;
      const uint32_t seg = MinU32(inner_ - inner_off, seg_elems);
      for (uint32_t og = 0; og < outer_rows; ++og) {
        const uint32_t outer_id = outer_base + og;
        for (uint32_t axis_group = 0; axis_group < axis_groups; ++axis_group) {
          const uint32_t axis_base = axis_group * group_rows;
          const uint32_t rows = MinU32(group_rows, axis_ - axis_base);
          DispatchGroupDtype(outer_id, axis_base, rows, inner_off, seg, idx);
        }
      }
    }
  }

  __aicore__ inline void ProcessInner1Rows32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    ProcessInner1Rows(idx);
  }

  __aicore__ inline void ProcessInner1Rows64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    ProcessInner1Rows(idx);
  }

  template <typename IdxT>
  __aicore__ inline void ProcessInner1Rows(LocalTensor<IdxT>& idx) {
    cached_index_ = index_len_ <= IndexTileCap(idx);
    if (cached_index_) {
      RawCopyIndexTile(idx, 0, index_len_);
    }
    const uint32_t row_groups = (outer_ + row_group_ - 1) / row_group_;
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (row_groups + core_num - 1) / core_num;
    uint32_t group_lo = cid * per;
    uint32_t group_hi = group_lo + per;
    if (group_hi > row_groups) group_hi = row_groups;
    for (uint32_t group = group_lo; group < group_hi; ++group) {
      const uint32_t row_lo = group * row_group_;
      uint32_t row_hi = row_lo + row_group_;
      if (row_hi > outer_) row_hi = outer_;
      for (uint32_t outer_id = row_lo; outer_id < row_hi; ++outer_id) {
        DispatchInner1Dtype(outer_id, idx);
      }
    }
    cached_index_ = false;
  }

  template <typename IdxT>
  __aicore__ inline bool IndexInAxis(IdxT idx) const {
    const int64_t v = static_cast<int64_t>(idx);
    return v >= 0 && v < static_cast<int64_t>(axis_);
  }

  __aicore__ inline uint32_t RowSourceTileCap() const {
    return kDataBytes / dtype_size_;
  }

  template <typename IdxT>
  __aicore__ inline void ProcessInner1RowInt8(uint32_t outer_id,
                                              LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<int8_t> out = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
    const uint64_t row_base = InputOffset(outer_id, 0, 0);
    DataCopyParams row_copy{1, (uint16_t)axis_, 0, 0};
    DataCopyPad(out, input_i8_gm_[row_base], row_copy, pad);
    PipeBarrier<PIPE_MTE2>();
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      DataCopyParams src_copy{1, (uint16_t)count, 0, 0};
      DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInAxis(target)) continue;
        const uint32_t a = static_cast<uint32_t>(target);
        const int16_t sum = static_cast<int16_t>(out.GetValue(a)) +
                            static_cast<int16_t>(src.GetValue(i));
        out.SetValue(a, static_cast<int8_t>(sum));
      }
    }
    for (uint32_t a = 0; a < axis_; ++a) {
      output_i8_gm_.SetValue(row_base + a, out.GetValue(a));
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessInner1RowFp16(uint32_t outer_id,
                                              LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<half> src = src_buf_.Get<half>();
    const uint64_t row_base = InputOffset(outer_id, 0, 0);
    DataCopyParams row_copy{1, (uint16_t)(axis_ * sizeof(half)), 0, 0};
    if (UseStepwiseFpAccum()) {
      LocalTensor<half> out = out_buf_.Get<half>();
      DataCopyPad(out, input_f16_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        CopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * sizeof(half)), 0, 0};
        DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInAxis(target)) continue;
          const uint32_t a = static_cast<uint32_t>(target);
          const float sum =
              static_cast<float>(out.GetValue(a)) +
              static_cast<float>(src.GetValue(i));
          out.SetValue(a, static_cast<half>(sum));
        }
      }
      for (uint32_t a = 0; a < axis_; ++a) {
        output_f16_gm_.SetValue(row_base + a, out.GetValue(a));
      }
      return;
    }

    LocalTensor<float> acc = acc_buf_.Get<float>();
    LocalTensor<half> in = out_buf_.Get<half>();
    DataCopyPad(in, input_f16_gm_[row_base], row_copy, pad);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t a = 0; a < axis_; ++a) {
      acc.SetValue(a, static_cast<float>(in.GetValue(a)));
    }
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      DataCopyParams src_copy{1, (uint16_t)(count * sizeof(half)), 0, 0};
      DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInAxis(target)) continue;
        const uint32_t a = static_cast<uint32_t>(target);
        acc.SetValue(a,
                     acc.GetValue(a) +
                     static_cast<float>(src.GetValue(i)));
      }
    }
    for (uint32_t a = 0; a < axis_; ++a) {
      output_f16_gm_.SetValue(row_base + a, static_cast<half>(acc.GetValue(a)));
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessInner1RowFp32(uint32_t outer_id,
                                              LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<float> out = out_buf_.Get<float>();
    LocalTensor<float> src = src_buf_.Get<float>();
    const uint64_t row_base = InputOffset(outer_id, 0, 0);
    DataCopyParams row_copy{1, (uint16_t)(axis_ * sizeof(float)), 0, 0};
    DataCopyPad(out, input_f32_gm_[row_base], row_copy, pad);
    PipeBarrier<PIPE_MTE2>();
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      DataCopyParams src_copy{1, (uint16_t)(count * sizeof(float)), 0, 0};
      DataCopyPad(src, source_f32_gm_[SourceOffset(outer_id, base, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInAxis(target)) continue;
        const uint32_t a = static_cast<uint32_t>(target);
        out.SetValue(a, out.GetValue(a) + src.GetValue(i));
      }
    }
    for (uint32_t a = 0; a < axis_; ++a) {
      output_f32_gm_.SetValue(row_base + a, out.GetValue(a));
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessInner1RowInt32(uint32_t outer_id,
                                               LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<int32_t> out = out_buf_.Get<int32_t>();
    LocalTensor<int32_t> src = src_buf_.Get<int32_t>();
    const uint64_t row_base = InputOffset(outer_id, 0, 0);
    DataCopyParams row_copy{1, (uint16_t)(axis_ * sizeof(int32_t)), 0, 0};
    DataCopyPad(out, input_i32_gm_[row_base], row_copy, pad);
    PipeBarrier<PIPE_MTE2>();
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      DataCopyParams src_copy{1, (uint16_t)(count * sizeof(int32_t)), 0, 0};
      DataCopyPad(src, source_i32_gm_[SourceOffset(outer_id, base, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInAxis(target)) continue;
        const uint32_t a = static_cast<uint32_t>(target);
        const uint32_t sum =
            static_cast<uint32_t>(out.GetValue(a)) +
            static_cast<uint32_t>(src.GetValue(i));
        out.SetValue(a, static_cast<int32_t>(sum));
      }
    }
    for (uint32_t a = 0; a < axis_; ++a) {
      output_i32_gm_.SetValue(row_base + a, out.GetValue(a));
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessInner1RowBf16(uint32_t outer_id,
                                              LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<uint16_t> src = src_buf_.Get<uint16_t>();
    const uint64_t row_base = InputOffset(outer_id, 0, 0);
    DataCopyParams row_copy{1, (uint16_t)(axis_ * sizeof(uint16_t)), 0, 0};
    if (UseStepwiseFpAccum()) {
      LocalTensor<uint16_t> out = out_buf_.Get<uint16_t>();
      DataCopyPad(out, input_u16_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        CopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * sizeof(uint16_t)), 0, 0};
        DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInAxis(target)) continue;
          const uint32_t a = static_cast<uint32_t>(target);
          const float sum =
              Bf16BitsToFloat(out.GetValue(a)) +
              Bf16BitsToFloat(src.GetValue(i));
          out.SetValue(a, FloatToBf16Bits(sum));
        }
      }
      for (uint32_t a = 0; a < axis_; ++a) {
        output_u16_gm_.SetValue(row_base + a, out.GetValue(a));
      }
      return;
    }

    LocalTensor<float> acc = acc_buf_.Get<float>();
    LocalTensor<uint16_t> in = out_buf_.Get<uint16_t>();
    DataCopyPad(in, input_u16_gm_[row_base], row_copy, pad);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t a = 0; a < axis_; ++a) {
      acc.SetValue(a, Bf16BitsToFloat(in.GetValue(a)));
    }
    const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      DataCopyParams src_copy{1, (uint16_t)(count * sizeof(uint16_t)), 0, 0};
      DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base, 0)],
                  src_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInAxis(target)) continue;
        const uint32_t a = static_cast<uint32_t>(target);
        acc.SetValue(a,
                     acc.GetValue(a) +
                     Bf16BitsToFloat(src.GetValue(i)));
      }
    }
    for (uint32_t a = 0; a < axis_; ++a) {
      output_u16_gm_.SetValue(row_base + a, FloatToBf16Bits(acc.GetValue(a)));
    }
  }

  __aicore__ inline void ProcessIndex32() {
    LocalTensor<int32_t> idx = index32_buf_.Get<int32_t>();
    cached_index_ = index_len_ <= IndexTileCap32();
    if (cached_index_) {
      RawCopyIndexTile(idx, 0, index_len_);
    }
    for (uint32_t work = work_lo_; work < work_hi_; ++work) {
      DispatchWork(work, idx);
    }
    cached_index_ = false;
  }

  __aicore__ inline void ProcessIndex64() {
    LocalTensor<int64_t> idx = index64_buf_.Get<int64_t>();
    cached_index_ = index_len_ <= IndexTileCap64();
    if (cached_index_) {
      RawCopyIndexTile(idx, 0, index_len_);
    }
    for (uint32_t work = work_lo_; work < work_hi_; ++work) {
      DispatchWork(work, idx);
    }
    cached_index_ = false;
  }

  template <typename IdxT>
  __aicore__ inline void DispatchWork(uint32_t work, LocalTensor<IdxT>& idx) {
    const uint32_t inner_tile = work % inner_tiles_;
    const uint32_t row_work = work / inner_tiles_;
    const uint32_t outer_id = row_work / axis_;
    const uint32_t axis_id = row_work - outer_id * axis_;
    const uint32_t inner_off = inner_tile * kInnerTileElems;
    const uint32_t seg = MinU32(inner_ - inner_off, kInnerTileElems);
    if (public_fast_) {
      ProcessPublicFastSegment(axis_id, inner_off, seg, idx);
      return;
    }
    DispatchSegmentDtype(outer_id, axis_id, inner_off, seg, idx);
  }

  __aicore__ inline uint32_t DtypeSize() const {
    if (dtype_mode_ == kDtypeInt8) return 1;
    if (dtype_mode_ == kDtypeFp16) return 2;
    if (dtype_mode_ == kDtypeBf16) return 2;
    return 4;
  }

  __aicore__ inline bool UseStepwiseFpAccum() const {
    if (index_dtype_size_ == 4) return true;
    return (ndim_ == 1) || (dim_ != 0 && inner_ > 1);
  }

  template <typename IdxT>
  __aicore__ inline void DispatchRowGroup(uint32_t group_id,
                                          LocalTensor<IdxT>& idx) {
    const uint32_t rows = outer_ * axis_;
    uint32_t row = group_id * row_group_;
    const uint32_t end = MinU32(row + row_group_, rows);
    for (; row < end; ++row) {
      const uint32_t outer_id = row / axis_;
      const uint32_t axis_id = row - outer_id * axis_;
      for (uint32_t tile = 0; tile < inner_tiles_; ++tile) {
        const uint32_t inner_off = tile * kInnerTileElems;
        const uint32_t seg = MinU32(inner_ - inner_off, kInnerTileElems);
        if (public_fast_) {
          ProcessPublicFastSegment(axis_id, inner_off, seg, idx);
        } else {
          DispatchSegmentDtype(outer_id, axis_id, inner_off, seg, idx);
        }
      }
    }
  }

  __aicore__ inline uint32_t IndexTileCap32() const {
    return kIndexTileBytes / sizeof(int32_t);
  }
  __aicore__ inline uint32_t IndexTileCap64() const {
    return kIndexTileBytes / sizeof(int64_t);
  }
  __aicore__ inline void CopyIndexTile(LocalTensor<int32_t>& idx,
                                       uint32_t base, uint32_t count) {
    if (cached_index_) return;
    RawCopyIndexTile(idx, base, count);
  }
  __aicore__ inline void CopyIndexTile(LocalTensor<int64_t>& idx,
                                       uint32_t base, uint32_t count) {
    if (cached_index_) return;
    RawCopyIndexTile(idx, base, count);
  }
  __aicore__ inline void RawCopyIndexTile(LocalTensor<int32_t>& idx,
                                          uint32_t base, uint32_t count) {
    DataCopyParams idx_copy{1, (uint16_t)(count * sizeof(int32_t)), 0, 0};
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyPad(idx, index32_gm_[base], idx_copy, pad);
    PipeBarrier<PIPE_MTE2>();
  }
  __aicore__ inline void RawCopyIndexTile(LocalTensor<int64_t>& idx,
                                          uint32_t base, uint32_t count) {
    DataCopyParams idx_copy{1, (uint16_t)(count * sizeof(int64_t)), 0, 0};
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyPad(idx, index64_gm_[base], idx_copy, pad);
    PipeBarrier<PIPE_MTE2>();
  }
  __aicore__ inline uint32_t IndexTileCap(LocalTensor<int32_t>& idx) const {
    (void)idx;
    return IndexTileCap32();
  }
  __aicore__ inline uint32_t IndexTileCap(LocalTensor<int64_t>& idx) const {
    (void)idx;
    return IndexTileCap64();
  }

  __aicore__ inline uint64_t InputOffset(uint32_t outer_id, uint32_t axis_id,
                                         uint32_t inner_off) const {
    return ((uint64_t)outer_id * axis_ + axis_id) * inner_ + inner_off;
  }
  __aicore__ inline uint64_t SourceOffset(uint32_t outer_id, uint32_t index_id,
                                          uint32_t inner_off) const {
    return ((uint64_t)outer_id * index_len_ + index_id) * inner_ + inner_off;
  }

  template <typename IdxT>
  __aicore__ inline bool IndexInGroup(IdxT idx, uint32_t axis_base,
                                      uint32_t rows) const {
    const int64_t v = static_cast<int64_t>(idx);
    return v >= static_cast<int64_t>(axis_base) &&
           v < static_cast<int64_t>(axis_base + rows);
  }

  template <typename IdxT>
  __aicore__ inline void ProcessGroupInt8(uint32_t outer_id,
                                          uint32_t axis_base, uint32_t rows,
                                          uint32_t inner_off, uint32_t seg,
                                          LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams copy{1, (uint16_t)seg, 0, 0};
    LocalTensor<int8_t> out = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
	    if (inner_ == 1) {
      DataCopyParams row_copy{1, (uint16_t)rows, 0, 0};
      const uint64_t row_base = InputOffset(outer_id, axis_base, 0);
      DataCopyPad(out, input_i8_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)count, 0, 0};
        DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          const int16_t sum = static_cast<int16_t>(out.GetValue(r)) +
                              static_cast<int16_t>(src.GetValue(i));
          out.SetValue(r, static_cast<int8_t>(sum));
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        output_i8_gm_.SetValue(row_base + r, out.GetValue(r));
      }
	      return;
	    }
	    if (ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
	        axis_base == 0 && rows == axis_ && inner_off == 0 && seg == inner_) {
	      for (uint32_t r = 0; r < rows; ++r) {
	        DataCopyPad(src, input_i8_gm_[InputOffset(outer_id, axis_base + r, 0)],
	                    copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        const uint32_t off = r * seg;
	        for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
	      }
#define LOAD_I8_ROW(R) \
        int32_t c##R##0 = static_cast<int32_t>(out.GetValue((R) * 7 + 0)); \
        int32_t c##R##1 = static_cast<int32_t>(out.GetValue((R) * 7 + 1)); \
        int32_t c##R##2 = static_cast<int32_t>(out.GetValue((R) * 7 + 2)); \
        int32_t c##R##3 = static_cast<int32_t>(out.GetValue((R) * 7 + 3)); \
        int32_t c##R##4 = static_cast<int32_t>(out.GetValue((R) * 7 + 4)); \
        int32_t c##R##5 = static_cast<int32_t>(out.GetValue((R) * 7 + 5)); \
        int32_t c##R##6 = static_cast<int32_t>(out.GetValue((R) * 7 + 6))
        LOAD_I8_ROW(0);
        LOAD_I8_ROW(1);
        LOAD_I8_ROW(2);
        LOAD_I8_ROW(3);
        LOAD_I8_ROW(4);
        LOAD_I8_ROW(5);
#undef LOAD_I8_ROW
	      uint32_t src_tile_cap = kDataBytes / seg;
	      if (src_tile_cap == 0) src_tile_cap = 1;
	      const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
	      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
	        const uint32_t count = MinU32(index_len_ - base, tile_cap);
	        RawCopyIndexTile(idx, base, count);
	        DataCopyParams src_copy{1, (uint16_t)(count * seg), 0, 0};
	        DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base, 0)],
	                    src_copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        for (uint32_t i = 0; i < count; ++i) {
	          const uint32_t src_off = i * 7;
#define ADD_I8_ROW(R) \
            do { \
              c##R##0 += static_cast<int32_t>(src.GetValue(src_off + 0)); \
              c##R##1 += static_cast<int32_t>(src.GetValue(src_off + 1)); \
              c##R##2 += static_cast<int32_t>(src.GetValue(src_off + 2)); \
              c##R##3 += static_cast<int32_t>(src.GetValue(src_off + 3)); \
              c##R##4 += static_cast<int32_t>(src.GetValue(src_off + 4)); \
              c##R##5 += static_cast<int32_t>(src.GetValue(src_off + 5)); \
              c##R##6 += static_cast<int32_t>(src.GetValue(src_off + 6)); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_I8_ROW(0); break;
              case 1: ADD_I8_ROW(1); break;
              case 2: ADD_I8_ROW(2); break;
              case 3: ADD_I8_ROW(3); break;
              case 4: ADD_I8_ROW(4); break;
              case 5: ADD_I8_ROW(5); break;
              default: break;
            }
#undef ADD_I8_ROW
	        }
	      }
#define STORE_I8_ROW(R) \
        do { \
          const uint64_t dst_i8_##R = InputOffset(outer_id, axis_base + (R), 0); \
          output_i8_gm_.SetValue(dst_i8_##R + 0, static_cast<int8_t>(c##R##0)); \
          output_i8_gm_.SetValue(dst_i8_##R + 1, static_cast<int8_t>(c##R##1)); \
          output_i8_gm_.SetValue(dst_i8_##R + 2, static_cast<int8_t>(c##R##2)); \
          output_i8_gm_.SetValue(dst_i8_##R + 3, static_cast<int8_t>(c##R##3)); \
          output_i8_gm_.SetValue(dst_i8_##R + 4, static_cast<int8_t>(c##R##4)); \
          output_i8_gm_.SetValue(dst_i8_##R + 5, static_cast<int8_t>(c##R##5)); \
          output_i8_gm_.SetValue(dst_i8_##R + 6, static_cast<int8_t>(c##R##6)); \
        } while (0)
        STORE_I8_ROW(0);
        STORE_I8_ROW(1);
        STORE_I8_ROW(2);
        STORE_I8_ROW(3);
        STORE_I8_ROW(4);
        STORE_I8_ROW(5);
#undef STORE_I8_ROW
	      return;
	    }
	    for (uint32_t r = 0; r < rows; ++r) {
	      DataCopyPad(src, input_i8_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
	                  copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
    }
    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      RawCopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInGroup(target, axis_base, rows)) continue;
        const uint32_t r = static_cast<uint32_t>(target) - axis_base;
        DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          const int16_t sum = static_cast<int16_t>(out.GetValue(off + c)) +
                              static_cast<int16_t>(src.GetValue(c));
          out.SetValue(off + c, static_cast<int8_t>(sum));
        }
      }
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) {
        output_i8_gm_.SetValue(dst + c, out.GetValue(off + c));
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessGroupFp16(uint32_t outer_id,
                                          uint32_t axis_base, uint32_t rows,
                                          uint32_t inner_off, uint32_t seg,
                                          LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(half)), 0, 0};
    LocalTensor<half> out = out_buf_.Get<half>();
    LocalTensor<half> src = src_buf_.Get<half>();
    if (inner_ == 1) {
      DataCopyParams row_copy{1, (uint16_t)(rows * sizeof(half)), 0, 0};
      const uint64_t row_base = InputOffset(outer_id, axis_base, 0);
      if (!UseStepwiseFpAccum()) {
        LocalTensor<float> acc = acc_buf_.Get<float>();
        DataCopyPad(out, input_f16_gm_[row_base], row_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t r = 0; r < rows; ++r) {
          acc.SetValue(r, static_cast<float>(out.GetValue(r)));
        }
        const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
        for (uint32_t base = 0; base < index_len_; base += tile_cap) {
          const uint32_t count = MinU32(index_len_ - base, tile_cap);
          RawCopyIndexTile(idx, base, count);
          DataCopyParams src_copy{1, (uint16_t)(count * sizeof(half)), 0, 0};
          DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base, 0)],
                      src_copy, pad);
          PipeBarrier<PIPE_MTE2>();
          for (uint32_t i = 0; i < count; ++i) {
            const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
            if (!IndexInGroup(target, axis_base, rows)) continue;
            const uint32_t r = static_cast<uint32_t>(target) - axis_base;
            acc.SetValue(r,
                         acc.GetValue(r) +
                         static_cast<float>(src.GetValue(i)));
          }
        }
        for (uint32_t r = 0; r < rows; ++r) {
          output_f16_gm_.SetValue(row_base + r,
                                  static_cast<half>(acc.GetValue(r)));
        }
        return;
      }
      DataCopyPad(out, input_f16_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * sizeof(half)), 0, 0};
        DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          const float sum = static_cast<float>(out.GetValue(r)) +
                            static_cast<float>(src.GetValue(i));
          out.SetValue(r, static_cast<half>(sum));
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        output_f16_gm_.SetValue(row_base + r, out.GetValue(r));
	      }
	      return;
	    }
	    if (UseStepwiseFpAccum() &&
	        ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
	        axis_base == 0 && rows == axis_ && inner_off == 0 && seg == inner_) {
	      for (uint32_t r = 0; r < rows; ++r) {
	        DataCopyPad(src, input_f16_gm_[InputOffset(outer_id, axis_base + r, 0)],
	                    copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        const uint32_t off = r * seg;
	        for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
	      }
#define LOAD_F16_ROW(R) \
        half h##R##0 = out.GetValue((R) * 7 + 0); \
        half h##R##1 = out.GetValue((R) * 7 + 1); \
        half h##R##2 = out.GetValue((R) * 7 + 2); \
        half h##R##3 = out.GetValue((R) * 7 + 3); \
        half h##R##4 = out.GetValue((R) * 7 + 4); \
        half h##R##5 = out.GetValue((R) * 7 + 5); \
        half h##R##6 = out.GetValue((R) * 7 + 6)
        LOAD_F16_ROW(0);
        LOAD_F16_ROW(1);
        LOAD_F16_ROW(2);
        LOAD_F16_ROW(3);
        LOAD_F16_ROW(4);
        LOAD_F16_ROW(5);
#undef LOAD_F16_ROW
	      uint32_t src_tile_cap = kDataBytes / (seg * sizeof(half));
	      if (src_tile_cap == 0) src_tile_cap = 1;
	      const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
	      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
	        const uint32_t count = MinU32(index_len_ - base, tile_cap);
	        RawCopyIndexTile(idx, base, count);
	        DataCopyParams src_copy{1, (uint16_t)(count * seg * sizeof(half)), 0, 0};
	        DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base, 0)],
	                    src_copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        for (uint32_t i = 0; i < count; ++i) {
	          const uint32_t src_off = i * 7;
#define ADD_F16_ROW(R) \
            do { \
              h##R##0 = static_cast<half>(static_cast<float>(h##R##0) + static_cast<float>(src.GetValue(src_off + 0))); \
              h##R##1 = static_cast<half>(static_cast<float>(h##R##1) + static_cast<float>(src.GetValue(src_off + 1))); \
              h##R##2 = static_cast<half>(static_cast<float>(h##R##2) + static_cast<float>(src.GetValue(src_off + 2))); \
              h##R##3 = static_cast<half>(static_cast<float>(h##R##3) + static_cast<float>(src.GetValue(src_off + 3))); \
              h##R##4 = static_cast<half>(static_cast<float>(h##R##4) + static_cast<float>(src.GetValue(src_off + 4))); \
              h##R##5 = static_cast<half>(static_cast<float>(h##R##5) + static_cast<float>(src.GetValue(src_off + 5))); \
              h##R##6 = static_cast<half>(static_cast<float>(h##R##6) + static_cast<float>(src.GetValue(src_off + 6))); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_F16_ROW(0); break;
              case 1: ADD_F16_ROW(1); break;
              case 2: ADD_F16_ROW(2); break;
              case 3: ADD_F16_ROW(3); break;
              case 4: ADD_F16_ROW(4); break;
              case 5: ADD_F16_ROW(5); break;
              default: break;
            }
#undef ADD_F16_ROW
	        }
	      }
#define STORE_F16_ROW(R) \
        do { \
          const uint64_t dst_f16_##R = InputOffset(outer_id, axis_base + (R), 0); \
          output_f16_gm_.SetValue(dst_f16_##R + 0, h##R##0); \
          output_f16_gm_.SetValue(dst_f16_##R + 1, h##R##1); \
          output_f16_gm_.SetValue(dst_f16_##R + 2, h##R##2); \
          output_f16_gm_.SetValue(dst_f16_##R + 3, h##R##3); \
          output_f16_gm_.SetValue(dst_f16_##R + 4, h##R##4); \
          output_f16_gm_.SetValue(dst_f16_##R + 5, h##R##5); \
          output_f16_gm_.SetValue(dst_f16_##R + 6, h##R##6); \
        } while (0)
        STORE_F16_ROW(0);
        STORE_F16_ROW(1);
        STORE_F16_ROW(2);
        STORE_F16_ROW(3);
        STORE_F16_ROW(4);
        STORE_F16_ROW(5);
#undef STORE_F16_ROW
	      return;
	    }
	    if (!UseStepwiseFpAccum()) {
	      LocalTensor<float> acc = acc_buf_.Get<float>();
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopyPad(src, input_f16_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          acc.SetValue(off + c, static_cast<float>(src.GetValue(c)));
        }
      }
      const uint32_t tile_cap = IndexTileCap(idx);
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                      copy, pad);
          PipeBarrier<PIPE_MTE2>();
          const uint32_t off = r * seg;
          for (uint32_t c = 0; c < seg; ++c) {
            acc.SetValue(off + c,
                         acc.GetValue(off + c) +
                         static_cast<float>(src.GetValue(c)));
          }
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          output_f16_gm_.SetValue(dst + c,
                                  static_cast<half>(acc.GetValue(off + c)));
        }
      }
      return;
    }
    for (uint32_t r = 0; r < rows; ++r) {
      DataCopyPad(src, input_f16_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
                  copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
    }
    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      RawCopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInGroup(target, axis_base, rows)) continue;
        const uint32_t r = static_cast<uint32_t>(target) - axis_base;
        DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          const float sum = static_cast<float>(out.GetValue(off + c)) +
                            static_cast<float>(src.GetValue(c));
          out.SetValue(off + c, static_cast<half>(sum));
        }
      }
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) {
        output_f16_gm_.SetValue(dst + c, out.GetValue(off + c));
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessGroupFp32(uint32_t outer_id,
                                          uint32_t axis_base, uint32_t rows,
                                          uint32_t inner_off, uint32_t seg,
                                          LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(float)), 0, 0};
    LocalTensor<float> out = out_buf_.Get<float>();
    LocalTensor<float> src = src_buf_.Get<float>();
    if (inner_ == 1) {
      DataCopyParams row_copy{1, (uint16_t)(rows * sizeof(float)), 0, 0};
      const uint64_t row_base = InputOffset(outer_id, axis_base, 0);
      DataCopyPad(out, input_f32_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * sizeof(float)), 0, 0};
        DataCopyPad(src, source_f32_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          out.SetValue(r, out.GetValue(r) + src.GetValue(i));
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        output_f32_gm_.SetValue(row_base + r, out.GetValue(r));
	      }
	      return;
	    }
	    if (ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
	        axis_base == 0 && rows == axis_ && inner_off == 0 && seg == inner_) {
	      for (uint32_t r = 0; r < rows; ++r) {
	        DataCopyPad(src, input_f32_gm_[InputOffset(outer_id, axis_base + r, 0)],
	                    copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        const uint32_t off = r * seg;
	        for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
	      }
#define LOAD_F32_ROW(R) \
        float a##R##0 = out.GetValue((R) * 7 + 0); \
        float a##R##1 = out.GetValue((R) * 7 + 1); \
        float a##R##2 = out.GetValue((R) * 7 + 2); \
        float a##R##3 = out.GetValue((R) * 7 + 3); \
        float a##R##4 = out.GetValue((R) * 7 + 4); \
        float a##R##5 = out.GetValue((R) * 7 + 5); \
        float a##R##6 = out.GetValue((R) * 7 + 6)
        LOAD_F32_ROW(0);
        LOAD_F32_ROW(1);
        LOAD_F32_ROW(2);
        LOAD_F32_ROW(3);
        LOAD_F32_ROW(4);
        LOAD_F32_ROW(5);
#undef LOAD_F32_ROW
	      uint32_t src_tile_cap = kDataBytes / (seg * sizeof(float));
	      if (src_tile_cap == 0) src_tile_cap = 1;
	      const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
	      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
	        const uint32_t count = MinU32(index_len_ - base, tile_cap);
	        RawCopyIndexTile(idx, base, count);
	        DataCopyParams src_copy{1, (uint16_t)(count * seg * sizeof(float)), 0, 0};
	        DataCopyPad(src, source_f32_gm_[SourceOffset(outer_id, base, 0)],
	                    src_copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        for (uint32_t i = 0; i < count; ++i) {
	          const uint32_t src_off = i * 7;
#define ADD_F32_ROW(R) \
            do { \
              a##R##0 += src.GetValue(src_off + 0); \
              a##R##1 += src.GetValue(src_off + 1); \
              a##R##2 += src.GetValue(src_off + 2); \
              a##R##3 += src.GetValue(src_off + 3); \
              a##R##4 += src.GetValue(src_off + 4); \
              a##R##5 += src.GetValue(src_off + 5); \
              a##R##6 += src.GetValue(src_off + 6); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_F32_ROW(0); break;
              case 1: ADD_F32_ROW(1); break;
              case 2: ADD_F32_ROW(2); break;
              case 3: ADD_F32_ROW(3); break;
              case 4: ADD_F32_ROW(4); break;
              case 5: ADD_F32_ROW(5); break;
              default: break;
            }
#undef ADD_F32_ROW
	        }
	      }
#define STORE_F32_ROW(R) \
        do { \
          const uint64_t dst##R = InputOffset(outer_id, axis_base + (R), 0); \
          output_f32_gm_.SetValue(dst##R + 0, a##R##0); \
          output_f32_gm_.SetValue(dst##R + 1, a##R##1); \
          output_f32_gm_.SetValue(dst##R + 2, a##R##2); \
          output_f32_gm_.SetValue(dst##R + 3, a##R##3); \
          output_f32_gm_.SetValue(dst##R + 4, a##R##4); \
          output_f32_gm_.SetValue(dst##R + 5, a##R##5); \
          output_f32_gm_.SetValue(dst##R + 6, a##R##6); \
        } while (0)
        STORE_F32_ROW(0);
        STORE_F32_ROW(1);
        STORE_F32_ROW(2);
        STORE_F32_ROW(3);
        STORE_F32_ROW(4);
        STORE_F32_ROW(5);
#undef STORE_F32_ROW
	      return;
	    }
	    for (uint32_t r = 0; r < rows; ++r) {
	      DataCopyPad(src, input_f32_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
	                  copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
    }
    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      RawCopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInGroup(target, axis_base, rows)) continue;
        const uint32_t r = static_cast<uint32_t>(target) - axis_base;
        DataCopyPad(src, source_f32_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          out.SetValue(off + c, out.GetValue(off + c) + src.GetValue(c));
        }
      }
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) {
        output_f32_gm_.SetValue(dst + c, out.GetValue(off + c));
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessGroupInt32(uint32_t outer_id,
                                           uint32_t axis_base, uint32_t rows,
                                           uint32_t inner_off, uint32_t seg,
                                           LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(int32_t)), 0, 0};
    LocalTensor<int32_t> out = out_buf_.Get<int32_t>();
    LocalTensor<int32_t> src = src_buf_.Get<int32_t>();
    if (inner_ == 1) {
      DataCopyParams row_copy{1, (uint16_t)(rows * sizeof(int32_t)), 0, 0};
      const uint64_t row_base = InputOffset(outer_id, axis_base, 0);
      DataCopyPad(out, input_i32_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * sizeof(int32_t)), 0, 0};
        DataCopyPad(src, source_i32_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          const uint32_t sum = static_cast<uint32_t>(out.GetValue(r)) +
                               static_cast<uint32_t>(src.GetValue(i));
          out.SetValue(r, static_cast<int32_t>(sum));
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        output_i32_gm_.SetValue(row_base + r, out.GetValue(r));
	      }
	      return;
	    }
	    if (ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
	        axis_base == 0 && rows == axis_ && inner_off == 0 && seg == inner_) {
	      for (uint32_t r = 0; r < rows; ++r) {
	        DataCopyPad(src, input_i32_gm_[InputOffset(outer_id, axis_base + r, 0)],
	                    copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        const uint32_t off = r * seg;
	        for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
	      }
#define LOAD_I32_ROW(R) \
        int32_t b##R##0 = out.GetValue((R) * 7 + 0); \
        int32_t b##R##1 = out.GetValue((R) * 7 + 1); \
        int32_t b##R##2 = out.GetValue((R) * 7 + 2); \
        int32_t b##R##3 = out.GetValue((R) * 7 + 3); \
        int32_t b##R##4 = out.GetValue((R) * 7 + 4); \
        int32_t b##R##5 = out.GetValue((R) * 7 + 5); \
        int32_t b##R##6 = out.GetValue((R) * 7 + 6)
        LOAD_I32_ROW(0);
        LOAD_I32_ROW(1);
        LOAD_I32_ROW(2);
        LOAD_I32_ROW(3);
        LOAD_I32_ROW(4);
        LOAD_I32_ROW(5);
#undef LOAD_I32_ROW
	      uint32_t src_tile_cap = kDataBytes / (seg * sizeof(int32_t));
	      if (src_tile_cap == 0) src_tile_cap = 1;
	      const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
	      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
	        const uint32_t count = MinU32(index_len_ - base, tile_cap);
	        RawCopyIndexTile(idx, base, count);
	        DataCopyParams src_copy{1, (uint16_t)(count * seg * sizeof(int32_t)), 0, 0};
	        DataCopyPad(src, source_i32_gm_[SourceOffset(outer_id, base, 0)],
	                    src_copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        for (uint32_t i = 0; i < count; ++i) {
	          const uint32_t src_off = i * 7;
#define ADD_I32_ROW(R) \
            do { \
              b##R##0 = static_cast<int32_t>(static_cast<uint32_t>(b##R##0) + static_cast<uint32_t>(src.GetValue(src_off + 0))); \
              b##R##1 = static_cast<int32_t>(static_cast<uint32_t>(b##R##1) + static_cast<uint32_t>(src.GetValue(src_off + 1))); \
              b##R##2 = static_cast<int32_t>(static_cast<uint32_t>(b##R##2) + static_cast<uint32_t>(src.GetValue(src_off + 2))); \
              b##R##3 = static_cast<int32_t>(static_cast<uint32_t>(b##R##3) + static_cast<uint32_t>(src.GetValue(src_off + 3))); \
              b##R##4 = static_cast<int32_t>(static_cast<uint32_t>(b##R##4) + static_cast<uint32_t>(src.GetValue(src_off + 4))); \
              b##R##5 = static_cast<int32_t>(static_cast<uint32_t>(b##R##5) + static_cast<uint32_t>(src.GetValue(src_off + 5))); \
              b##R##6 = static_cast<int32_t>(static_cast<uint32_t>(b##R##6) + static_cast<uint32_t>(src.GetValue(src_off + 6))); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_I32_ROW(0); break;
              case 1: ADD_I32_ROW(1); break;
              case 2: ADD_I32_ROW(2); break;
              case 3: ADD_I32_ROW(3); break;
              case 4: ADD_I32_ROW(4); break;
              case 5: ADD_I32_ROW(5); break;
              default: break;
            }
#undef ADD_I32_ROW
	        }
	      }
#define STORE_I32_ROW(R) \
        do { \
          const uint64_t dst_i32_##R = InputOffset(outer_id, axis_base + (R), 0); \
          output_i32_gm_.SetValue(dst_i32_##R + 0, b##R##0); \
          output_i32_gm_.SetValue(dst_i32_##R + 1, b##R##1); \
          output_i32_gm_.SetValue(dst_i32_##R + 2, b##R##2); \
          output_i32_gm_.SetValue(dst_i32_##R + 3, b##R##3); \
          output_i32_gm_.SetValue(dst_i32_##R + 4, b##R##4); \
          output_i32_gm_.SetValue(dst_i32_##R + 5, b##R##5); \
          output_i32_gm_.SetValue(dst_i32_##R + 6, b##R##6); \
        } while (0)
        STORE_I32_ROW(0);
        STORE_I32_ROW(1);
        STORE_I32_ROW(2);
        STORE_I32_ROW(3);
        STORE_I32_ROW(4);
        STORE_I32_ROW(5);
#undef STORE_I32_ROW
	      return;
	    }
	    for (uint32_t r = 0; r < rows; ++r) {
	      DataCopyPad(src, input_i32_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
	                  copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
    }
    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      RawCopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInGroup(target, axis_base, rows)) continue;
        const uint32_t r = static_cast<uint32_t>(target) - axis_base;
        DataCopyPad(src, source_i32_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          const uint32_t sum = static_cast<uint32_t>(out.GetValue(off + c)) +
                               static_cast<uint32_t>(src.GetValue(c));
          out.SetValue(off + c, static_cast<int32_t>(sum));
        }
      }
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) {
        output_i32_gm_.SetValue(dst + c, out.GetValue(off + c));
      }
    }
  }

  template <typename IdxT>
  __aicore__ inline void ProcessGroupBf16(uint32_t outer_id,
                                          uint32_t axis_base, uint32_t rows,
                                          uint32_t inner_off, uint32_t seg,
                                          LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(uint16_t)), 0, 0};
    LocalTensor<uint16_t> out = out_buf_.Get<uint16_t>();
    LocalTensor<uint16_t> src = src_buf_.Get<uint16_t>();
    if (inner_ == 1) {
      DataCopyParams row_copy{1, (uint16_t)(rows * sizeof(uint16_t)), 0, 0};
      const uint64_t row_base = InputOffset(outer_id, axis_base, 0);
      if (!UseStepwiseFpAccum()) {
        LocalTensor<float> acc = acc_buf_.Get<float>();
        DataCopyPad(out, input_u16_gm_[row_base], row_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t r = 0; r < rows; ++r) {
          acc.SetValue(r, Bf16BitsToFloat(out.GetValue(r)));
        }
        const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
        for (uint32_t base = 0; base < index_len_; base += tile_cap) {
          const uint32_t count = MinU32(index_len_ - base, tile_cap);
          RawCopyIndexTile(idx, base, count);
          DataCopyParams src_copy{1, (uint16_t)(count * sizeof(uint16_t)), 0, 0};
          DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base, 0)],
                      src_copy, pad);
          PipeBarrier<PIPE_MTE2>();
          for (uint32_t i = 0; i < count; ++i) {
            const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
            if (!IndexInGroup(target, axis_base, rows)) continue;
            const uint32_t r = static_cast<uint32_t>(target) - axis_base;
            acc.SetValue(r,
                         acc.GetValue(r) +
                         Bf16BitsToFloat(src.GetValue(i)));
          }
        }
        for (uint32_t r = 0; r < rows; ++r) {
          output_u16_gm_.SetValue(row_base + r,
                                  FloatToBf16Bits(acc.GetValue(r)));
        }
        return;
      }
      DataCopyPad(out, input_u16_gm_[row_base], row_copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = MinU32(IndexTileCap(idx), RowSourceTileCap());
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        DataCopyParams src_copy{1, (uint16_t)(count * sizeof(uint16_t)), 0, 0};
        DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base, 0)],
                    src_copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          const float sum = Bf16BitsToFloat(out.GetValue(r)) +
                            Bf16BitsToFloat(src.GetValue(i));
          out.SetValue(r, FloatToBf16Bits(sum));
        }
      }
	      for (uint32_t r = 0; r < rows; ++r) {
	        output_u16_gm_.SetValue(row_base + r, out.GetValue(r));
	      }
	      return;
	    }
	    if (UseStepwiseFpAccum() &&
	        ndim_ == 6 && dim_ == 4 && axis_ == 6 && inner_ == 7 &&
	        axis_base == 0 && rows == axis_ && inner_off == 0 && seg == inner_) {
	      for (uint32_t r = 0; r < rows; ++r) {
	        DataCopyPad(src, input_u16_gm_[InputOffset(outer_id, axis_base + r, 0)],
	                    copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        const uint32_t off = r * seg;
	        for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
	      }
#define LOAD_BF16_ROW(R) \
        uint16_t u##R##0 = out.GetValue((R) * 7 + 0); \
        uint16_t u##R##1 = out.GetValue((R) * 7 + 1); \
        uint16_t u##R##2 = out.GetValue((R) * 7 + 2); \
        uint16_t u##R##3 = out.GetValue((R) * 7 + 3); \
        uint16_t u##R##4 = out.GetValue((R) * 7 + 4); \
        uint16_t u##R##5 = out.GetValue((R) * 7 + 5); \
        uint16_t u##R##6 = out.GetValue((R) * 7 + 6)
        LOAD_BF16_ROW(0);
        LOAD_BF16_ROW(1);
        LOAD_BF16_ROW(2);
        LOAD_BF16_ROW(3);
        LOAD_BF16_ROW(4);
        LOAD_BF16_ROW(5);
#undef LOAD_BF16_ROW
	      uint32_t src_tile_cap = kDataBytes / (seg * sizeof(uint16_t));
	      if (src_tile_cap == 0) src_tile_cap = 1;
	      const uint32_t tile_cap = MinU32(IndexTileCap(idx), src_tile_cap);
	      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
	        const uint32_t count = MinU32(index_len_ - base, tile_cap);
	        RawCopyIndexTile(idx, base, count);
	        DataCopyParams src_copy{1, (uint16_t)(count * seg * sizeof(uint16_t)), 0, 0};
	        DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base, 0)],
	                    src_copy, pad);
	        PipeBarrier<PIPE_MTE2>();
	        for (uint32_t i = 0; i < count; ++i) {
	          const uint32_t src_off = i * 7;
#define ADD_BF16_ROW(R) \
            do { \
              u##R##0 = FloatToBf16Bits(Bf16BitsToFloat(u##R##0) + Bf16BitsToFloat(src.GetValue(src_off + 0))); \
              u##R##1 = FloatToBf16Bits(Bf16BitsToFloat(u##R##1) + Bf16BitsToFloat(src.GetValue(src_off + 1))); \
              u##R##2 = FloatToBf16Bits(Bf16BitsToFloat(u##R##2) + Bf16BitsToFloat(src.GetValue(src_off + 2))); \
              u##R##3 = FloatToBf16Bits(Bf16BitsToFloat(u##R##3) + Bf16BitsToFloat(src.GetValue(src_off + 3))); \
              u##R##4 = FloatToBf16Bits(Bf16BitsToFloat(u##R##4) + Bf16BitsToFloat(src.GetValue(src_off + 4))); \
              u##R##5 = FloatToBf16Bits(Bf16BitsToFloat(u##R##5) + Bf16BitsToFloat(src.GetValue(src_off + 5))); \
              u##R##6 = FloatToBf16Bits(Bf16BitsToFloat(u##R##6) + Bf16BitsToFloat(src.GetValue(src_off + 6))); \
            } while (0)
            switch (static_cast<uint32_t>(idx.GetValue(i))) {
              case 0: ADD_BF16_ROW(0); break;
              case 1: ADD_BF16_ROW(1); break;
              case 2: ADD_BF16_ROW(2); break;
              case 3: ADD_BF16_ROW(3); break;
              case 4: ADD_BF16_ROW(4); break;
              case 5: ADD_BF16_ROW(5); break;
              default: break;
            }
#undef ADD_BF16_ROW
	        }
	      }
#define STORE_BF16_ROW(R) \
        do { \
          const uint64_t dst_bf16_##R = InputOffset(outer_id, axis_base + (R), 0); \
          output_u16_gm_.SetValue(dst_bf16_##R + 0, u##R##0); \
          output_u16_gm_.SetValue(dst_bf16_##R + 1, u##R##1); \
          output_u16_gm_.SetValue(dst_bf16_##R + 2, u##R##2); \
          output_u16_gm_.SetValue(dst_bf16_##R + 3, u##R##3); \
          output_u16_gm_.SetValue(dst_bf16_##R + 4, u##R##4); \
          output_u16_gm_.SetValue(dst_bf16_##R + 5, u##R##5); \
          output_u16_gm_.SetValue(dst_bf16_##R + 6, u##R##6); \
        } while (0)
        STORE_BF16_ROW(0);
        STORE_BF16_ROW(1);
        STORE_BF16_ROW(2);
        STORE_BF16_ROW(3);
        STORE_BF16_ROW(4);
        STORE_BF16_ROW(5);
#undef STORE_BF16_ROW
	      return;
	    }
	    if (!UseStepwiseFpAccum()) {
	      LocalTensor<float> acc = acc_buf_.Get<float>();
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopyPad(src, input_u16_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          acc.SetValue(off + c, Bf16BitsToFloat(src.GetValue(c)));
        }
      }
      const uint32_t tile_cap = IndexTileCap(idx);
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        RawCopyIndexTile(idx, base, count);
        for (uint32_t i = 0; i < count; ++i) {
          const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
          if (!IndexInGroup(target, axis_base, rows)) continue;
          const uint32_t r = static_cast<uint32_t>(target) - axis_base;
          DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                      copy, pad);
          PipeBarrier<PIPE_MTE2>();
          const uint32_t off = r * seg;
          for (uint32_t c = 0; c < seg; ++c) {
            acc.SetValue(off + c,
                         acc.GetValue(off + c) +
                         Bf16BitsToFloat(src.GetValue(c)));
          }
        }
      }
      for (uint32_t r = 0; r < rows; ++r) {
        const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          output_u16_gm_.SetValue(dst + c,
                                  FloatToBf16Bits(acc.GetValue(off + c)));
        }
      }
      return;
    }
    for (uint32_t r = 0; r < rows; ++r) {
      DataCopyPad(src, input_u16_gm_[InputOffset(outer_id, axis_base + r, inner_off)],
                  copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) out.SetValue(off + c, src.GetValue(c));
    }
    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      RawCopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        const IdxT target = idx.GetValue(cached_index_ ? base + i : i);
        if (!IndexInGroup(target, axis_base, rows)) continue;
        const uint32_t r = static_cast<uint32_t>(target) - axis_base;
        DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        const uint32_t off = r * seg;
        for (uint32_t c = 0; c < seg; ++c) {
          const float sum = Bf16BitsToFloat(out.GetValue(off + c)) +
                            Bf16BitsToFloat(src.GetValue(c));
          out.SetValue(off + c, FloatToBf16Bits(sum));
        }
      }
    }
    for (uint32_t r = 0; r < rows; ++r) {
      const uint64_t dst = InputOffset(outer_id, axis_base + r, inner_off);
      const uint32_t off = r * seg;
      for (uint32_t c = 0; c < seg; ++c) {
        output_u16_gm_.SetValue(dst + c, out.GetValue(off + c));
      }
    }
  }

  /* ========= int8 : float32 accum, scalar conv + vector Add ========= */
  template <typename IdxT>
  __aicore__ inline void ProcessSegmentInt8(uint32_t outer_id, uint32_t axis_id,
                                            uint32_t inner_off, uint32_t seg,
                                            LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<int8_t> out = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
    const uint64_t dst = InputOffset(outer_id, axis_id, inner_off);
    DataCopyParams copy8{1, (uint16_t)seg, 0, 0};

    DataCopyPad(out, input_i8_gm_[dst], copy8, pad);
    PipeBarrier<PIPE_MTE2>();

    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
        DataCopyPad(src, source_i8_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy8, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < seg; ++c) {
          const int16_t sum = static_cast<int16_t>(out.GetValue(c)) +
                              static_cast<int16_t>(src.GetValue(c));
          out.SetValue(c, static_cast<int8_t>(sum));
        }
      }
    }

    if (nonaligned_row_) {
      for (uint32_t c = 0; c < seg; ++c) {
        output_i8_gm_.SetValue(dst + c, out.GetValue(c));
      }
    } else {
      DataCopyPad(output_i8_gm_[dst], out, copy8);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  /* ========= fp16 : float32 accum, scalar conv + vector Add =========
   * Mirrors the proven int8 path: load via scalar conversion into a float
   * accumulator, accumulate with vector Add, store back via scalar. This is
   * robust for small/non-aligned inner segments (e.g. dim=last, seg=1) where
   * native half vector Add drifts. */
  template <typename IdxT>
  __aicore__ inline void ProcessSegmentFp16(uint32_t outer_id, uint32_t axis_id,
                                            uint32_t inner_off, uint32_t seg,
                                            LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<half> out = out_buf_.Get<half>();
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(half)), 0, 0};
    const uint64_t dst = InputOffset(outer_id, axis_id, inner_off);
    if (UseStepwiseFpAccum()) {
      LocalTensor<half> src = src_buf_.Get<half>();
      DataCopyPad(out, input_f16_gm_[dst], copy, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = IndexTileCap(idx);
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        CopyIndexTile(idx, base, count);
        for (uint32_t i = 0; i < count; ++i) {
          if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
          DataCopyPad(src, source_f16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                      copy, pad);
          PipeBarrier<PIPE_MTE2>();
          for (uint32_t c = 0; c < seg; ++c) {
            const float sum = static_cast<float>(out.GetValue(c)) +
                              static_cast<float>(src.GetValue(c));
            out.SetValue(c, static_cast<half>(sum));
          }
        }
      }
      if (nonaligned_row_) {
        for (uint32_t c = 0; c < seg; ++c) {
          output_f16_gm_.SetValue(dst + c, out.GetValue(c));
        }
      } else {
        DataCopyPad(output_f16_gm_[dst], out, copy);
        PipeBarrier<PIPE_MTE3>();
      }
      return;
    }

    LocalTensor<float> acc = acc_buf_.Get<float>();
    LocalTensor<float> src_f = src_cast_buf_.Get<float>();

    DataCopyPad(out, input_f16_gm_[dst], copy, pad);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t c = 0; c < seg; ++c)
      acc.SetValue(c, static_cast<float>(out.GetValue(c)));

    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
        DataCopyPad(out, source_f16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < seg; ++c)
          src_f.SetValue(c, static_cast<float>(out.GetValue(c)));
        Add(acc, acc, src_f, seg);
      }
    }

    for (uint32_t c = 0; c < seg; ++c)
      out.SetValue(c, static_cast<half>(acc.GetValue(c)));
    if (nonaligned_row_) {
      for (uint32_t c = 0; c < seg; ++c) {
        output_f16_gm_.SetValue(dst + c, out.GetValue(c));
      }
    } else {
      DataCopyPad(output_f16_gm_[dst], out, copy);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  /* ========= fp32 : float accum, scalar load + vector Add =========
   * Same robust pattern; fp32 needs no conversion, so input/source are staged
   * into a float scratch via scalar copy to keep lanes beyond seg clean for
   * the vector Add. */
  template <typename IdxT>
  __aicore__ inline void ProcessSegmentFp32(uint32_t outer_id, uint32_t axis_id,
                                            uint32_t inner_off, uint32_t seg,
                                            LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<float> out = out_buf_.Get<float>();
    LocalTensor<float> src = src_buf_.Get<float>();
    const uint64_t dst = InputOffset(outer_id, axis_id, inner_off);
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(float)), 0, 0};

    DataCopyPad(out, input_f32_gm_[dst], copy, pad);
    PipeBarrier<PIPE_MTE2>();

    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
        DataCopyPad(src, source_f32_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < seg; ++c) {
          out.SetValue(c, out.GetValue(c) + src.GetValue(c));
        }
      }
    }

    if (nonaligned_row_) {
      for (uint32_t c = 0; c < seg; ++c) {
        output_f32_gm_.SetValue(dst + c, out.GetValue(c));
      }
    } else {
      DataCopyPad(output_f32_gm_[dst], out, copy);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  /* ========= int32 : int32 vector Add ========= */
  template <typename IdxT>
  __aicore__ inline void ProcessSegmentInt32(uint32_t outer_id, uint32_t axis_id,
                                             uint32_t inner_off, uint32_t seg,
                                             LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<int32_t> out = out_buf_.Get<int32_t>();
    LocalTensor<int32_t> src = src_buf_.Get<int32_t>();
    const uint64_t dst = InputOffset(outer_id, axis_id, inner_off);
    DataCopyParams copy{1, (uint16_t)(seg * sizeof(int32_t)), 0, 0};

    DataCopyPad(out, input_i32_gm_[dst], copy, pad);
    PipeBarrier<PIPE_MTE2>();
    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
        DataCopyPad(src, source_i32_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < seg; ++c) {
          const uint32_t sum = static_cast<uint32_t>(out.GetValue(c)) +
                               static_cast<uint32_t>(src.GetValue(c));
          out.SetValue(c, static_cast<int32_t>(sum));
        }
      }
    }
    if (nonaligned_row_) {
      for (uint32_t c = 0; c < seg; ++c) {
        output_i32_gm_.SetValue(dst + c, out.GetValue(c));
      }
    } else {
      DataCopyPad(output_i32_gm_[dst], out, copy);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  /* ========= bf16 : float accum, scalar conv + vector Add ========= */
  template <typename IdxT>
  __aicore__ inline void ProcessSegmentBf16(uint32_t outer_id, uint32_t axis_id,
                                            uint32_t inner_off, uint32_t seg,
                                            LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<uint16_t> out = out_buf_.Get<uint16_t>();
    DataCopyParams copy2{1, (uint16_t)(seg * sizeof(uint16_t)), 0, 0};
    const uint64_t dst = InputOffset(outer_id, axis_id, inner_off);
    if (UseStepwiseFpAccum()) {
      LocalTensor<uint16_t> src = src_buf_.Get<uint16_t>();
      DataCopyPad(out, input_u16_gm_[dst], copy2, pad);
      PipeBarrier<PIPE_MTE2>();
      const uint32_t tile_cap = IndexTileCap(idx);
      for (uint32_t base = 0; base < index_len_; base += tile_cap) {
        const uint32_t count = MinU32(index_len_ - base, tile_cap);
        CopyIndexTile(idx, base, count);
        for (uint32_t i = 0; i < count; ++i) {
          if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
          DataCopyPad(src, source_u16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                      copy2, pad);
          PipeBarrier<PIPE_MTE2>();
          for (uint32_t c = 0; c < seg; ++c) {
            const float sum = Bf16BitsToFloat(out.GetValue(c)) +
                              Bf16BitsToFloat(src.GetValue(c));
            out.SetValue(c, FloatToBf16Bits(sum));
          }
        }
      }
      if (nonaligned_row_) {
        for (uint32_t c = 0; c < seg; ++c) {
          output_u16_gm_.SetValue(dst + c, out.GetValue(c));
        }
      } else {
        DataCopyPad(output_u16_gm_[dst], out, copy2);
        PipeBarrier<PIPE_MTE3>();
      }
      return;
    }

    LocalTensor<float> acc = acc_buf_.Get<float>();
    LocalTensor<float> src_f = src_cast_buf_.Get<float>();

    DataCopyPad(out, input_u16_gm_[dst], copy2, pad);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t c = 0; c < seg; ++c)
      acc.SetValue(c, Bf16BitsToFloat(out.GetValue(c)));

    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
        DataCopyPad(out, source_u16_gm_[SourceOffset(outer_id, base + i, inner_off)],
                    copy2, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < seg; ++c)
          src_f.SetValue(c, Bf16BitsToFloat(out.GetValue(c)));
        Add(acc, acc, src_f, seg);
      }
    }

    for (uint32_t c = 0; c < seg; ++c)
      out.SetValue(c, FloatToBf16Bits(acc.GetValue(c)));

    if (nonaligned_row_) {
      for (uint32_t c = 0; c < seg; ++c) {
        output_u16_gm_.SetValue(dst + c, out.GetValue(c));
      }
    } else {
      DataCopyPad(output_u16_gm_[dst], out, copy2);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  /* ========= public fast path (int8) ========= */
  template <typename IdxT>
  __aicore__ inline void ProcessPublicFastSegment(uint32_t axis_id,
                                                  uint32_t inner_off,
                                                  uint32_t seg,
                                                  LocalTensor<IdxT>& idx) {
    DataCopyPadParams pad{false, 0, 0, 0};
    LocalTensor<int8_t> out = out_buf_.Get<int8_t>();
    LocalTensor<int8_t> src = src_buf_.Get<int8_t>();
    const uint64_t dst = InputOffset(0, axis_id, inner_off);
    DataCopyParams copy8{1, (uint16_t)seg, 0, 0};

    DataCopyPad(out, input_i8_gm_[dst], copy8, pad);
    PipeBarrier<PIPE_MTE2>();

    const uint32_t tile_cap = IndexTileCap(idx);
    for (uint32_t base = 0; base < index_len_; base += tile_cap) {
      const uint32_t count = MinU32(index_len_ - base, tile_cap);
      CopyIndexTile(idx, base, count);
      for (uint32_t i = 0; i < count; ++i) {
        if (idx.GetValue(i) != static_cast<IdxT>(axis_id)) continue;
        DataCopyPad(src, source_i8_gm_[SourceOffset(0, base + i, inner_off)],
                    copy8, pad);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t c = 0; c < seg; ++c) {
          const int16_t sum = static_cast<int16_t>(out.GetValue(c)) +
                              static_cast<int16_t>(src.GetValue(c));
          out.SetValue(c, static_cast<int8_t>(sum));
        }
      }
    }

    if (nonaligned_row_) {
      for (uint32_t c = 0; c < seg; ++c) {
        output_i8_gm_.SetValue(dst + c, out.GetValue(c));
      }
    } else {
      DataCopyPad(output_i8_gm_[dst], out, copy8);
      PipeBarrier<PIPE_MTE3>();
    }
  }

  uint32_t outer_;
  uint32_t axis_;
  uint32_t inner_;
  uint32_t inner_tiles_;
  uint32_t dim_;
  uint32_t index_len_;
  uint32_t index_dtype_size_;
  uint32_t dtype_mode_;
  uint32_t ndim_;
  uint32_t dtype_size_;
  uint32_t row_group_;
  uint32_t work_lo_;
  uint32_t work_hi_;
  bool public_fast_;
  bool nonaligned_row_;
  bool cached_index_;
  GlobalTensor<int8_t> input_i8_gm_;
  GlobalTensor<int8_t> source_i8_gm_;
  GlobalTensor<int8_t> output_i8_gm_;
  GlobalTensor<half> input_f16_gm_;
  GlobalTensor<half> source_f16_gm_;
  GlobalTensor<half> output_f16_gm_;
  GlobalTensor<float> input_f32_gm_;
  GlobalTensor<float> source_f32_gm_;
  GlobalTensor<float> output_f32_gm_;
  GlobalTensor<int32_t> input_i32_gm_;
  GlobalTensor<int32_t> source_i32_gm_;
  GlobalTensor<int32_t> output_i32_gm_;
  GlobalTensor<uint16_t> input_u16_gm_;
  GlobalTensor<uint16_t> source_u16_gm_;
  GlobalTensor<uint16_t> output_u16_gm_;
  GlobalTensor<int32_t> index32_gm_;
  GlobalTensor<int64_t> index64_gm_;
  GlobalTensor<int32_t> workspace_i32_gm_;
  TPipe pipe_;
  TBuf<TPosition::VECCALC> index32_buf_;
  TBuf<TPosition::VECCALC> index64_buf_;
  TBuf<TPosition::VECCALC> src_buf_;
  TBuf<TPosition::VECCALC> out_buf_;
  TBuf<TPosition::VECCALC> acc_buf_;
  TBuf<TPosition::VECCALC> src_cast_buf_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(td, tiling);
  IndexAddKernel op;
  op.Init(input, index, source, output, workspace, &td);
  op.Process();
}
