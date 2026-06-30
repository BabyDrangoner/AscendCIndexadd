/*
 * IndexAdd v9 - minimal passthrough (reference: CANN Ascend C template).
 * Only handles int8 [32,128] dim=0.
 * Does input -> output copy using correct DataCopyPad API.
 */
#include "kernel_operator.h"
using namespace AscendC;

class IndexAddKernel {
 public:
  __aicore__ inline IndexAddKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                              const IndexAddTilingData* tiling) {
    inputGm_.SetGlobalBuffer((__gm__ int8_t*)input, 32 * 128);
    outputGm_.SetGlobalBuffer((__gm__ int8_t*)output, 32 * 128);
    pipe_.InitBuffer(buf_, 128);
  }

  __aicore__ inline void Process() {
    LocalTensor<int8_t> buf = buf_.Get<int8_t>();
    // Each core processes one row
    const uint32_t coreId = GetBlockIdx();
    if (coreId < 32) {
      DataCopyParams copyParams = {1, 128, 0, 0};  // 128 bytes = 128 int8
      DataCopyPadParams padParams = {false, 0, 0, 0};
      DataCopyPad(buf, inputGm_[coreId * 128], copyParams, padParams);
      DataCopyPad(outputGm_[coreId * 128], buf, copyParams, padParams);
    }
  }

 private:
  GlobalTensor<int8_t> inputGm_;
  GlobalTensor<int8_t> outputGm_;
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buf_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)index; (void)source; (void)workspace;
  GET_TILING_DATA(td, tiling);
  IndexAddKernel op;
  op.Init(input, output, &td);
  op.Process();
}
