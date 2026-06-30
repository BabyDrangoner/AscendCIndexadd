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
- Status: submitted; waiting for official result.

## v120 case4-i16-maskreduce

- Change: add a lean exact-shape int8 Case4 mask-reduce path on top of the v95 source snapshot. The path keeps `Select` in half, casts the masked tile to int16, then performs the tree reduction with int16 `Add` before scalar low-byte int8 writeback.
- Rationale: dav_c220 supports `Select` only for half/float, but supports vector `Add` for int16. This candidate keeps the onehot mask idea while avoiding v118's float expansion and testing whether int16 low-byte accumulation can preserve int8 semantics.
- Package: `IndexAdd_v120_case4_i16_maskreduce.zip`
- SHA256: `9a8c7e2eed8eeadeb14874375b34372e39810c03946f7e95a8f2bdaf9d051013`
- Verification: top-level host source matches (`22eb60d19fbc9c6278b481f2cf04a2423149e5b8b384ff7cc132a7d96f46cb7f`); top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match (`6410d2db10674b315c5cd9f4bc939d3ed07caa9b5a9ca46abe9fa8091577337a`).
- Status: built; submit after v119 exact-v95 retest result is captured.
