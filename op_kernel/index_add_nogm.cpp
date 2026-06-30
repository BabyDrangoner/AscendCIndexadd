/*
 * IndexAdd v8d - NO GM access at all.
 * Only does InitBuffer + SetValue on UB.
 * If this segfaults, the issue is with TBuf allocation or kernel framework.
 * If this works, the issue is with GM access (DataCopy or GlobalTensor setup).
 */
#include "kernel_operator.h"
using namespace AscendC;

class IndexAddKernelV8D {
 public:
  __aicore__ inline IndexAddKernelV8D() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source,
                              GM_ADDR output, const void* tiling) {
    (void)input; (void)index; (void)source; (void)output; (void)tiling;
    // Only allocate UB buffer - no GM access
    pipe_.InitBuffer(buf_, 128);
  }

  __aicore__ inline void Process() {
    LocalTensor<int8_t> buf = buf_.Get<int8_t>();
    // Only do SetValue/GetValue on UB - no GM access
    for (uint32_t i = 0; i < 128; ++i) {
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
  (void)input; (void)index; (void)source; (void)output; (void)workspace; (void)tiling;
  IndexAddKernelV8D op;
  op.Init(input, index, source, output, tiling);
  op.Process();
}
