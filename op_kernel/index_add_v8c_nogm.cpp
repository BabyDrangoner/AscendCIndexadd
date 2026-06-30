/*
 * IndexAdd v8c - UB-only kernel (no GM access).
 * Just does SetValue/GetValue on local buffer.
 * If this still segfaults, the issue is with kernel framework.
 * If this works, the issue is with GM access in the kernel.
 */
#include "kernel_operator.h"
using namespace AscendC;

class IndexAddKernelV8C {
 public:
  __aicore__ inline IndexAddKernelV8C() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source,
                              GM_ADDR output, const void* tiling) {
    (void)input; (void)index; (void)source; (void)output; (void)tiling;
    pipe_.InitBuffer(buf_, 128);
  }

  __aicore__ inline void Process() {
    LocalTensor<int8_t> buf = buf_.Get<int8_t>();
    // Just do SetValue/GetValue on local buffer
    for (uint32_t i = 0; i < 128; ++i) {
      buf.SetValue(i, static_cast<int8_t>(i % 128));
    }
    // Copy to output (just to have side effect)
    // Actually, we can't access GM, so just do nothing more
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buf_;
};

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)input; (void)index; (void)source; (void)output; (void)workspace; (void)tiling;
  IndexAddKernelV8C op;
  op.Init(input, index, source, output, tiling);
  op.Process();
}
