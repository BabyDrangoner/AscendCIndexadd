# IndexAdd Versions

## Baseline bd40

- Source: copied from `/Users/xiaolong.xu/workspace/华为算子比赛s9/op/IndexAdd`
- Key host setting: `use_case4_i8_outer1 ? 40 : 32`
- Official package: `IndexAdd_bd40_docker.zip`
- SHA256: `7616c0c99c32fc69f69c1a1b590ab1d7c434b73c6c80cc400aaad8ae94c5f7d6`
- Official result:

```text
Case1: Pass, Result: 36.912
Case2: Pass, Result: 12603.02
Case3: Pass, Result: 60733.108
Case4: Pass, Result: 1823492.76
Case5: Pass, Result: 15615.26
prof_sum: 1912481.06
```

Conclusion: stable, but Case4 dominates. Further work should prioritize Case4 while preserving Case2/3/5.

## v93 pack2-priority

- Change: try `ProcessCase4Pack2Int8()` before the stable `Outer1` case4 int8 path.
- Rationale: the previous dispatch order made `Pack2` unreachable for int32-index case4 with even `outer`; this candidate tests whether processing two outer rows per tile reduces Case4 overhead.
- Official package: `IndexAdd_v93_pack2_priority.zip`
- SHA256: `3237772b1cc4aa7d4bfc907243ca9b136b8a53901eacdb40632691b7c2a9a567`
- Official result:

```text
Case1: Pass, Result: 41.308
Case2: Pass, Result: 12585.728
Case3: Pass, Result: 60731.24
Case4: Pass, Result: 1823524.068
Case5: Pass, Result: 15624.452
prof_sum: 1912506.796
```

Conclusion: worse than bd40 by 25.736us. Pack2 reduces Case2 slightly but does not improve Case4.

## v94 pack4-priority

- Change: enable the existing `ProcessCase4Pack4Int8()` implementation and dispatch it before `Pack2`/`Outer1` when `outer` is divisible by 4.
- Rationale: Case4 likely has `outer=120`, so Pack4 can reuse one index scan across four outer rows and may cut branch/index overhead further than v93.
- Status: build only until v93 official result is captured.

## v95 pack4-datacopy-store

- Change: keep v94 Pack4 dispatch, but write each 42-byte output row through local tensor + `DataCopyPad` instead of scalar GM `SetValue`.
- Rationale: this matches the stable `Outer1`/`Pack2` writeback pattern and should reduce writeback overhead while lowering correctness risk from many non-contiguous scalar GM stores.
- Package: `IndexAdd_v95_pack4_datacopy_store.zip`
- SHA256: `23b0dcbe59a1c30af3b7ef40c6fb85334fd60469200f2cdeca0abbf40f5a556c`
- Status: built; submit after v93 result captured.

## v119 exact-v95-retest

- Change: rebuild the exact v95 source snapshot from commit `56d60db` in an isolated `codex/v95-retest` worktree.
- Rationale: v117 restored the Pack4 entry path but did not reproduce the historical v95 1.729s Case4 band. This retest checks whether the original v95 source/binary shape still performs better on the current platform.
- Package: `IndexAdd_v119_exact_v95_retest.zip`
- SHA256: `205a836ea718b51331a369a83c3bf46f687de8f2b03db7456986212512bc6995`
- Verification: top-level host source matches (`22eb60d19fbc9c6278b481f2cf04a2423149e5b8b384ff7cc132a7d96f46cb7f`); top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match (`b527e6e1c9559e202408e1746713399b7ed9f0eb3ea18b8148512e0d257df6d6`).
- Submit: 2026-06-30 18:54:12, workId `0000019F182A4A9DB01F8D7F53F5AFCA`.
- Official result:

```text
Case1: Pass, Result: 33.368
Case2: Pass, Result: 12613.572
Case3: Pass, Result: 60727.368
Case4: Pass, Result: 1729368.104
Case5: Pass, Result: 15622.308
prof_sum: 1818364.72
```

Conclusion: full pass and reproduces the historical v95 1.818s band. The much slower v117 result came from later source/binary shape changes rather than the Pack4 algorithm itself, so further optimization should continue from this lean v95 branch.

## v120 case4-i16-maskreduce

- Change: add a lean exact-shape int8 Case4 mask-reduce path on top of the v95 source snapshot. The path keeps `Select` in half, casts the masked tile to int16, then performs the tree reduction with int16 `Add` before scalar low-byte int8 writeback.
- Rationale: dav_c220 supports `Select` only for half/float, but supports vector `Add` for int16. This candidate keeps the onehot mask idea while avoiding v118's float expansion and testing whether int16 low-byte accumulation can preserve int8 semantics.
- Package: `IndexAdd_v120_case4_i16_maskreduce.zip`
- SHA256: `9a8c7e2eed8eeadeb14874375b34372e39810c03946f7e95a8f2bdaf9d051013`
- Verification: top-level host source matches (`22eb60d19fbc9c6278b481f2cf04a2423149e5b8b384ff7cc132a7d96f46cb7f`); top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match (`6410d2db10674b315c5cd9f4bc939d3ed07caa9b5a9ca46abe9fa8091577337a`).
- Submit: 2026-06-30 19:03:23, workId `0000019F1832B3C6B015A9B2D31A9490`.
- Official result:

```text
Case1: Pass, Result: 37.76
Case2: Pass, Result: 12565.24
Case3: Pass, Result: 60819.948
Case4: Pass, Result: 1812554.484
Case5: Pass, Result: 15612.864
prof_sum: 1901590.296
```

Conclusion: full pass but falls back into the 1.901s band. The half-select plus int16-reduce path is still dominated by six full bucket scans and cast overhead; stop pursuing onehot mask-reduce for int8 Case4 unless the source layout can be changed to avoid rescanning each bucket.

## v121 pack4-packed-cached

- Change: disable the v120 mask-reduce dispatch and retest Case4 on the lean v95 branch with a packed int8 Pack4 hot loop. Each 7-byte int8 row is loaded into one `uint64_t`, accumulated with byte-wise wraparound, and written back through the existing 42-byte row `DataCopyPad`. The shared int32 index tile is cached once per core when it fits in UB.
- Rationale: profiling points to scalar execution, not GM bandwidth. This candidate reduces the per-index per-four-outer source lane work from 28 scalar byte loads/adds to four packed row loads/adds while preserving the stable v95 outer grouping and output path.
- Package: `IndexAdd_v121_pack4_packed_cached.zip`
- SHA256: `47af783c8f9ad945af7b664a712c7c8884a9bbf2e26fff71e245fb53f05a6f93`
- Verification: top-level host source matches; top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match.
- Submit: 2026-06-30 19:39:58 CST, workId `0000019F185431BBB02149BD9478A0F2`.
- Official result:

```text
Case1: Pass, Result: 33.508
Case2: Pass, Result: 12631.172
Case3: Pass, Result: 60714.872
Case4: Pass, Result: 1729263.24
Case5: Pass, Result: 15605.816
prof_sum: 1818248.608
```

Conclusion: full pass and the best result so far on the lean v95 branch, improving v119 by 116.112us. The gain is still tiny relative to the leaderboard gap, confirming that packed/cache scalar tuning is near its ceiling and the next meaningful attempt needs a cube/matrix-style reduction or another true vectorized segmented reduction.

## v122 case4-cube-lane-prototype

- Change: attempted an exact int8 Case4 cube-lane prototype with AscendC `Matmul` and per-block GM workspace.
- Package: `IndexAdd_v122_cube_lane_proto.zip`
- SHA256: `97b056948f9b9a7884daa39bb719fc62bcc20cc53f3b799381fb35b312eb46bf`
- Submit: 2026-06-30 20:09:11 CST, workId `0000019F186EEB22B015A968C8AC0974`.
- Official result:

```text
Case1: Run failed
Case2: Run failed
Case3: Run failed
Case4: Run failed
Case5: Run failed
```

Conclusion: invalid candidate. The failure affected every case, which points to the `ASCENDC_CUBE_ONLY`/Matmul integration changing the runtime channel or binary shape globally rather than a narrow Case4 arithmetic bug. The v123 dense-GM-A matmul draft is preserved in git stash, but do not submit it until cube registration is isolated from the normal AIV paths.

## v124 kfc-dense-gm-a

- Change: replace the rejected cube-only experiment with a split-core KFC Matmul fast path guarded to exact int8 Case4 (`index=int32`, `ndim=6`, `dim=4`, `axis=6`, `inner=7`, `index_len=1024`). A is read directly from GM as `[outer_group, 1024*7]`, B is a per-tile local onehot matrix in B1, and C is written to user workspace before scalar int8 wraparound storeback.
- Host tiling: exact dense Case4 uses `work_units=ceil(outer/16)*3`, `blockDim<=24` to stay within KFC's 50 AIV flat-block workspace layout, and allocates user workspace for per-block C matrices.
- Package: `IndexAdd_v124_kfc_dense_gma.zip`
- SHA256: `6bba2950e06c1c0d69f738123cbe675fc4db8814b3195d7cbf9c12126cad6bc0`
- Verification: Docker/Lima build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.

Pending conclusion: submit to check whether KFC registration avoids v122's all-case runtime failure and whether Case4 moves from scalar-bound 1.729s toward cube-bound performance.
