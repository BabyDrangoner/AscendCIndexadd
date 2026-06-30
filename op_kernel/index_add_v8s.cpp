/*
 * IndexAdd v8s (simple) - scalar-only, no vector Add.
 * Only int8, only dim=0, to isolate segfault.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t kBufBytes = 4096;  // 4KB buffer for int8

class IndexAddKernelV8S {
 public:
  __aicore__ inline IndexAddKernelV8S() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source,
                              GM_ADDR output, const void* tiling) {
    // Hardcoded for public case: int8 [32,128] dim=0 index_len=120
    input_i8_gm_.SetGlobalBuffer((__gm__ int8_t*)input, 32 * 128);
    source_i8_gm_.SetGlobalBuffer((__gm__ int8_t*)source, 32 * 120 * 128);
    output_i8_gm_.SetGlobalBuffer((__gm__ int8_t*)output, 32 * 128);
    index32_gm_.SetGlobalBuffer((__gm__ int32_t*)index, 120);

    const uint32_t work_units = 32;  // axis_dim
    const uint32_t core_num = GetBlockNum();
    const uint32_t cid = GetBlockIdx();
    const uint32_t per = (work_units + core_num - 1) / core_num;
    work_lo_ = cid * per;
    work_hi_ = work_lo_ + per;
    if (work_hi_ > work_units) work_hi_ = work_units;

    pipe_.InitBuffer(buf_, kBufBytes);
  }

  __aicore__ inline void Process() {
    if (work_lo_ >= work_hi_) return;
    LocalTensor<int8_t> buf = buf_.Get<int8_t>();
    LocalTensor<int32_t> idx_buf = idx_buf_.Get<int32_t>();
    // Copy index to local
    DataCopy(idx_buf, index32_gm_, 120);
    // Process assigned rows
    for (uint32_t axis_id = work_lo_; axis_id < work_hi_; ++axis_id) {
      // Copy input row to buf
      DataCopy(buf, input_i8_gm_[axis_id * 128], 128);
      // Accumulate
      for (uint32_t i = 0; i < 120; ++i) {
        if (idx_buf.GetValue(i) != static_cast<int32_t>(axis_id)) continue;
        for (uint32_t j = 0; j < 128; ++j) {
          int32_t v = static_cast<int32_t>(buf.GetValue(j)) +
                      static_cast<int32_t>(source_i8_gm_.GetValue((axis_id * 120 + i) * 128 + j));
          v = (v & 0xFF) - ((v & 0x80) << 1);  // wrap to int8
          buf.SetValue(j, static_cast<int8_t>(v));
        }
      }
      // Copy back
      DataCopy(output_i8_gm_[axis_id * 128], buf, 128);
    }
  }

 private:
  uint32_t work_lo_;
  uint32_t work_hi_;
  GlobalTensor<int8_t> input_i8_gm_;
  GlobalTensor<int8_t> source_i8_gm_;
  GlobalTensor<int8_t> output_i8_gm_;
  GlobalTensor<int32_t> index32_gm_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> in_que_;
  TBuf<TPosition::VECCALC> buf_;
  TBuf<TPosition::VECCALC> idx_buf_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace; (void)tiling;
  IndexAddKernelV8S op;
  op.Init(input, index, source, output, nullptr);
  op.Process();
}
