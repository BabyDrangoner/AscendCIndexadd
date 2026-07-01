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
- Submit: 2026-06-30 20:37:25 CST, workId `0000019F1888C3ACB015AC3EE5AE3C75`.

Official result:

```text
Case1: Run failed
Case2: Run failed
Case3: Run failed
Case4: Run failed
Case5: Run failed
```

Conclusion: invalid candidate. Because every case failed, including non-Case4 shapes that do not enter the dense arithmetic branch, the problem is the split-core KFC registration/quit path rather than GM-A math.

## v125 kfc-dense-gm-b

- Change: keep the v124 KFC dense-GM-A structure, but move the onehot B matrix from local B1 to per-block GM user workspace. The B matrix is zeroed with 16KB DMA chunks and filled with scalar onehot writes before `SetTensorB` uses a GM tensor.
- Rationale: v124 is faster on paper but may depend on AIV-side local B1 behavior. v125 is a more conservative fallback if v124 fails due to B1/KFC client restrictions.
- Package: `IndexAdd_v125_kfc_dense_gmb.zip`
- SHA256: `bcecff0355d109e41728f016b95a0fa51926bba3a3371b77aa57e5ce567f038c`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet. Wait for the true v124 result first so the platform result is not overwritten.

## v126 kfc-dense-gm-b-m32

- Change: increase the exact int8 Case4 KFC dense Matmul outer group from 16 rows to 32 rows while keeping GM-A/GM-B/C int32 semantics.
- Rationale: Case4 has enough outer parallelism, and the v125 shape issues one KFC Matmul command per 16 outer rows and N tile. M=32 halves the command count and amortizes B zero/fill plus client scheduling overhead without changing the reduction math.
- Package: `IndexAdd_v126_kfc_dense_gmb_m32.zip`
- SHA256: `36c1b682f1bdc77439cde2305785f779201dcd61f3285cce1ba0d3000e6e1dd8`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet. Wait for the true v124 result first so the platform result is not overwritten.

## v127 kfc-dense-gm-b-m64

- Change: increase the exact int8 Case4 KFC dense Matmul outer group to 64 rows.
- Rationale: M=64 halves the KFC command count again versus v126. If cube occupancy remains healthy, this should reduce client scheduling and B-preparation overhead further; if the larger single-core tile hurts cache pressure, v126 should be preferred.
- Package: `IndexAdd_v127_kfc_dense_gmb_m64.zip`
- SHA256: `32963265bf75260f5391f9685763f6cdb0e2b0d7becb38dcd72d3723cfc6266b`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet. Wait for the true v124 result first so the platform result is not overwritten.

## v128 kfc-always-register-m64

- Change: keep the v127 M=64 GM-B dense path, but initialize the dense tiling and execute `REGIST_MATMUL_OBJ` for every runtime shape. Only exact int8 Case4 sends Matmul work; non-dense shapes still register the KFC channel so AIC side can exit cleanly.
- Rationale: v124 failed every case, which points to split-core registration/quit behavior rather than dense arithmetic. This candidate fixes the lifecycle issue before testing the M=64 dense algorithm.
- Package: `IndexAdd_v128_kfc_always_register_m64.zip`
- SHA256: `78b651db31e0811a4e89cc09b32820f65c0a5ec536ceab99f5cec4b7cb136a23`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-06-30 21:10:34 CST, workId `0000019F18A72258B015AC545AD243F4`.

Official result:

```text
Case1: Run failed
Case2: Run failed
Case3: Run failed
Case4: Run failed
Case5: Run failed
```

Conclusion: invalid candidate. Always registering the KFC object did not fix the all-case runtime failure, so the remaining suspects are workspace allocation for fallback shapes and/or CUBE-side AIV buffer initialization before KFC server setup.

## v129 kfc-workspace-all-m64

- Change: keep v128's always-register lifecycle fix, and allocate the per-block GM-B/C user workspace for all runtime shapes instead of only the exact dense Case4 shape.
- Rationale: if v128 still fails non-dense cases, the next likely issue is a null or undersized workspace pointer when `REGIST_MATMUL_OBJ` runs on fallback shapes. This candidate isolates that variable.
- Package: `IndexAdd_v129_kfc_workspace_all_m64.zip`
- SHA256: `2d6b54f01e2488c564469599698c6e1d08311390801be855ff6f758c781fe50a`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet. Submit only if v128 result indicates workspace-related failure.

## v130 kfc-workspace-all-m128

- Change: keep v129's always-register/workspace-all structure and increase the dense Case4 Matmul M tile to 128 rows.
- Rationale: if KFC lifecycle is fixed, M=128 cuts the number of Matmul commands to one quarter of v126 and half of v127/v129. This may reduce client/server scheduling overhead, but could be worse if the larger single-core M tile loses cube efficiency.
- Package: `IndexAdd_v130_kfc_workspace_all_m128.zip`
- SHA256: `e04898ddbb312f44a6499acaae4042a14b3565a1a45eaa6bd42ca755e06ef768`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet. Submit only after v128/v129 establish that KFC path is viable.

## v131 kfc-aiv-only-buffer-init

- Change: keep v130's M=128/workspace-all KFC shape, but initialize VECCALC TBuf buffers only on `ASCEND_IS_AIV`.
- Rationale: KFC CUBE/AIC side should enter the server path with minimal setup. v124/v128 may fail or hang if CUBE-side execution performs AIV-only buffer initialization before `REGIST_MATMUL_OBJ`.
- Package: `IndexAdd_v131_kfc_aiv_only_buffer_init.zip`
- SHA256: `08408c911101b8a6e879ab38b77bef9f48cf392c6654de2c3e6ce27cef07903a`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet; current HiAscend cookie is expired (`401 未登录`).

## v132 kfc-cap24-all

- Change: keep v131's KFC/AIV-only initialization and cap `blockDim` to 24 for every runtime shape.
- Rationale: once KFC is registered for all shapes, fallback cases can no longer use the old 32/40 block caps. KFC flat-block bookkeeping is only safe around 24 AIV blocks; exceeding that can explain why v124/v128 failed every case, not just Case4.
- Package: `IndexAdd_v132_kfc_cap24_all.zip`
- SHA256: `c2db5a882fccd77a6b16dedaf7f9664439654f2176b57d4264a12872dd2403aa`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 10:33:50 CST, workId `IndexAdd_v132_kfc_cap24_all.zip`.
- Monitor: pending as of 2026-07-01 10:34:03 CST; ranking row still points to the older 2026-06-30 21:10:34 submission.
- Pipeline note: HiAscend changed the OBS upload flow used by the submit page. `tools/submit_hiascend.mjs` now posts multipart form data to `/devCenter/contested/enrollment/uploadWorkToObs` and submits with `workId` equal to the zip filename, matching the current frontend.

## v133 kfc-cap24-all-m64

- Change: keep v132's KFC lifecycle/workspace/blockDim fixes, but reduce dense Case4 Matmul M tile from 128 to 64 rows.
- Rationale: M=128 minimizes command count, while M=64 may have better cube/cache efficiency. Both should be kept as official A/B candidates once the cookie is refreshed.
- Package: `IndexAdd_v133_kfc_cap24_all_m64.zip`
- SHA256: `dedf40eafce62568dc465bbf84ce931f3145ce59a8798d931cdc8a995b265c5a`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet; official submit needs a refreshed HiAscend cookie.

## v134 kfc-cap24-all-m32

- Change: keep v132/v133's KFC lifecycle/workspace/blockDim fixes, but use dense Case4 Matmul M tile of 32 rows.
- Rationale: M=32 is the most conservative KFC dense tile in this set. It issues more Matmul commands than M=64/M=128 but may be easier for the KFC scheduler/cache path to execute correctly and consistently.
- Package: `IndexAdd_v134_kfc_cap24_all_m32.zip`
- SHA256: `a6e4459fa03d55493398c596f70865a62406bbe5d80f56a7bdc9e2ad59c5100f`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet; official submit needs a refreshed HiAscend cookie.

## v135 kfc-conditional-m32

- Change: keep the v134 M=32 dense Case4 Matmul tile, but register the KFC Matmul object only for the exact dense Case4 shape. Non-dense shapes bypass `REGIST_MATMUL_OBJ`, restore the previous 32/40 block caps, and allocate zero workspace.
- Rationale: v124 used conditional KFC but still initialized VECCALC buffers on AIC; v132/v134 fixed AIV-only buffer initialization but registered KFC for all shapes. This candidate combines the safer parts: AIC avoids VECCALC setup, and fallback cases avoid idle KFC server registration entirely.
- Package: `IndexAdd_v135_kfc_conditional_m32.zip`
- SHA256: `76d8c0ac83d2c2f3cfbfa6c2712c1d0a3a64ff6435eb99689652eaea26f6c6fb`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted yet; wait for v132's official result before overwriting the platform row.

## v151 kfc-hotcase-outer8192

- Change: keep the v135 conditional KFC M32 dense matmul structure, but narrow the dense dispatch and host workspace/blockDim guard to the inferred hot Case4 shape `outer == 8192`, `dtype=int8`, `index=int32`, `ndim=6`, `dim=4`, `axis=6`, `inner=7`, and `index_len=1024`.
- Rationale: v132 showed KFC could let non-dense cases pass after lifecycle fixes, but dense dispatch still failed Case4 and Case5. The likely remaining risk is over-broad dense matching. This candidate tests the cube/matmul algorithm only on the large hot Case4 while leaving smaller matching shapes on the stable AIV fallback.
- Package: `IndexAdd_v151_kfc_hotcase_outer8192.zip`
- SHA256: `c3ca92e4e6748b760821221ab0bc77c77a526cff3ac381d09ab61b7a5076371e`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 14:23:56 CST, workId `IndexAdd_v151_kfc_hotcase_outer8192.zip`.
- Monitor: pending as of 2026-07-01 14:24 CST; ranking row still points to the older v148 result from 2026-07-01 13:31:43.

## v154 kfc-gmb-flush-hotcase-outer8192

- Change: keep the v151 M32 GM-B dense KFC path, but add `Barrier()` plus `DataCacheCleanAndInvalid(... ENTIRE_DATA_CACHE)` after the AIV-generated onehot B matrix is written to GM workspace and before `SetTensorB/IterateAll`.
- Rationale: the generated B matrix is produced by AIV and consumed by AIC. If v151 fails on Case4, one plausible cause is GM/cache visibility for scalar `SetValue` writes into B. This candidate isolates that hypothesis without changing the dense math.
- Package: `IndexAdd_v154_kfc_gmb_flush_outer8192.zip`
- SHA256: `b148d74e0805dc7a445b1539728db85f9bf7915e7dce6f9a03c7d497e5ea4a66`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: not submitted yet. Wait for the true v151 result before overwriting the platform row.
