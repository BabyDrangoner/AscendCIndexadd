/*
 * IndexAdd v8b - minimal passthrough kernel.
 * Just copies input -> output for int8 [32,128] dim=0.
 * No computation, to isolate segfault.
 */
#include "kernel_operator.h"
using namespace AscendC;

class IndexAddKernelV8B {
 public:
  __aicore__ inline IndexAddKernelV8B() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source,
                              GM_ADDR output, const void* tiling) {
    (void)index; (void)source; (void)tiling;
    input_gm_.SetGlobalBuffer((__gm__ int8_t*)input, 32 * 128);
    output_gm_.SetGlobalBuffer((__gm__ int8_t*)output, 32 * 128);
    pipe_.InitBuffer(buf_, 128);
  }

  __aicore__ inline void Process() {
    LocalTensor<int8_t> buf = buf_.Get<int8_t>();
    // Just copy input to output, one row per core
    const uint32_t cid = GetBlockIdx();
    if (cid < 32) {
      DataCopy(buf, input_gm_[cid * 128], 128);
      DataCopy(output_gm_[cid * 128], buf, 128);
    }
  }

 private:
  GlobalTensor<int8_t> input_gm_;
  GlobalTensor<int8_t> output_gm_;
  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> in_que_;
  TBuf<TPosition::VECCALC> buf_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)index; (void)source; (void)workspace; (void)tiling;
  IndexAddKernelV8B op;
  op.Init(input, nullptr, nullptr, output, nullptr);
  op.Process();
}
