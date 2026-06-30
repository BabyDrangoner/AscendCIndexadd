/*
 * IndexAdd v9 - truly minimal (no tiling, no GM).
 * Only does SetValue/GetValue on UB buffer.
 * Goal: verify kernel framework works at all.
 */
#include "kernel_operator.h"
using namespace AscendC;

class IndexAddKernel {
 public:
  __aicore__ inline IndexAddKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source,
                              GM_ADDR output, const void* tiling) {
    (void)input; (void)index; (void)source; (void)output; (void)tiling;
    pipe_.InitBuffer(buf_, 1024);  // 1KB buffer
  }

  __aicore__ inline void Process() {
    LocalTensor<int8_t> buf = buf_.Get<int8_t>();
    // Only do UB-local computation (no GM access)
    for (uint32_t i = 0; i < 1024; ++i) {
      buf.SetValue(i, static_cast<int8_t>(i % 128));
    }
    // Don't write to output (can't access GM)
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buf_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  // Don't call GET_TILING_DATA - just pass nullptr
  IndexAddKernel op;
  op.Init(input, index, source, output, nullptr);
  op.Process();
}
