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
- Official result:

```text
Case1: Pass, Result: 73.696
Case2: Pass, Result: 16803.192
Case3: Pass, Result: 60730.112
Case4: Run failed!
Case5: Run failed!
```

Conclusion: invalid candidate. Even with AIV-only buffer initialization and a global `blockDim` cap, the KFC registration/matmul shape still fails the exact hot Case4 path and also breaks Case5. Further work should not keep tuning KFC block parameters until the split-core runtime failure is understood in isolation.
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

## v137 case4-vcadd-f32-reduce

- Change: add a non-KFC exact int8 Case4 path for `index=int32`, `axis=6`, `inner=7`, and `index_len=1024`. The path precomputes the shared onehot masks once per core, selects matching rows in half, casts to float for exact int8 accumulation, then uses lane-wise `WholeReduceSum`/`vcadd` over 8-row groups instead of the previous hand-written tree reduction.
- Rationale: v120/v118 proved onehot mask-reduce correctness but spent too much time in six full bucket scans plus manual reductions. This candidate keeps source access contiguous and replaces the reduction core with the hardware reduce primitive while avoiding all KFC registration.
- Package: `IndexAdd_v137_case4_vcadd_f32_reduce.zip`
- SHA256: `b8e607b3e332c79695f097ea2f5f277489430d76f4155ba1a76c191850c37b8e`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 11:03:27 CST, workId `IndexAdd_v137_case4_vcadd_f32_reduce.zip`.
- Official result:

```text
Case1: Run failed
Case2: Run failed
Case3: Run failed
Case4: Run failed
Case5: Run failed
```

Conclusion: invalid candidate. Because all five cases fail, the `WholeReduceSum`/`vcadd` addition likely makes the generated dynamic kernel package fail globally, even though the new branch is exact-shape guarded. Do not submit further candidates using this reduce primitive until it is validated in an isolated minimal operator.

## v138 case4-gather-f32-reduce

- Change: replace the v137 exact int8 Case4 onehot-select path with a gather-reduce path. Each core preclassifies the shared index into `2 x 6` byte-offset lists, casts each 512-row source tile to float once, gathers only matching axis/lane values with vector `Gather<float>`, then reduces the compacted values with `WholeReduceSum`.
- Rationale: v137 still scans all rows for every axis bucket. v138 keeps contiguous source DMA but removes the six full zero-filled scans, testing whether vector gather plus compact reduction can beat both packed scalar and onehot-select reductions.
- Package: `IndexAdd_v138_case4_gather_f32_reduce.zip`
- SHA256: `cb6b8a572652422d26ce757b3095208a1cd6b8d29d88ce41ea79e860ef3c2b2d`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted. v137 showed an all-case runtime failure from the same reduce primitive family, so this package is preserved for source study only and should not overwrite the platform row.

## v121 restore-submit 2026-07-01

- Change: resubmit the known full-pass v121 package after v137 produced an all-case `Run failed` result.
- Rationale: keep the official row out of the `-1` state while new candidates are developed from a stable Pack4/packed baseline.
- Package: `IndexAdd_v121_pack4_packed_cached.zip`
- SHA256: `47af783c8f9ad945af7b664a712c7c8884a9bbf2e26fff71e245fb53f05a6f93`
- Submit: 2026-07-01 11:21:02 CST, workId `IndexAdd_v121_pack4_packed_cached.zip`.
- Official result:

```text
Case1: Pass, Result: 37.392
Case2: Pass, Result: 12593.528
Case3: Pass, Result: 60719.292
Case4: Pass, Result: 1729259.432
Case5: Pass, Result: 15622.868
prof_sum: 1818232.512
```

Conclusion: recovery submit succeeded and confirms the pipeline/package flow is healthy. The `-1` state came from the v137 reduce candidate, not from submit packaging.

## v139 restore-v121-source

- Change: restore `op_host/index_add.cpp` and `op_kernel/index_add.cpp` to the v121 stable source state after the v137/v138 reduce/gather experiments.
- Rationale: v137 failed all cases, so the next safe optimization work should start from the last known full-pass source rather than layering changes on top of high-risk vector-reduce/KFC code.
- Package: `IndexAdd_v139_restore_v121_source.zip`
- SHA256: `d03799f756d3131b1a99cd929824c09f437fd18c23d4608065d99a72c293928a`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: not submitted. v121 was already resubmitted and is pending; keep v139 as the source-aligned recovery package unless the v121 row fails to refresh.

## v140 case4-bucket-pack4

- Change: add an exact int8 Case4 path for `index=int32`, `axis=6`, `inner=7`, and `index_len=1024`. Each core classifies the shared index once into six UB offset buckets, then the Pack4 hot loop accumulates each bucket directly instead of reading `idx` and executing a six-way switch for every source row.
- Rationale: v121 is bottlenecked by scalar control flow. This candidate avoids high-risk KFC/reduce/Gather primitives while removing repeated index loads and branchy dispatch from every outer4 group.
- Package: `IndexAdd_v140_case4_bucket_pack4.zip`
- SHA256: `7fd2e335f2e79f224c0554f470343c884d5e96878e419349c355a06b8ae71d95`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 11:34:34 CST, workId `IndexAdd_v140_case4_bucket_pack4.zip`.
- Official result:

```text
Case1: Pass, Result: 36.164
Case2: Pass, Result: 12676.592
Case3: Pass, Result: 60724.46
Case4: Pass, Result: 1729436.652
Case5: Pass, Result: 15611.492
prof_sum: 1818485.36
```

Conclusion: full pass but worse than the v121 recovery result by 252.848us. Removing the switch/index load is not enough; bucket-table UB reads and non-sequential source access offset the control-flow gain. Do not keep this path as the baseline.

## v141 pack4-shared-loadcalc

- Change: revert the v140 bucketed path and keep the v121 Pack4 source-order hot loop. Within each index iteration, compute the packed 7-byte `word/shift` once and load all four outer source rows with that shared offset calculation instead of calling `LoadPacked7()` four times.
- Rationale: v140 showed that preserving sequential source access matters. This candidate attacks the scalar address/shift overhead inside the proven Pack4 loop while avoiding non-sequential UB reads and high-risk vector reduce primitives.
- Package: `IndexAdd_v141_pack4_shared_loadcalc.zip`
- SHA256: `9b0d486359635326fcf5369324c5ddaea4710313063710420ce659b7d31c4293`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 11:44:59 CST, workId `IndexAdd_v141_pack4_shared_loadcalc.zip`.
- Official result:

```text
Case1: Pass, Result: 33.5
Case2: Pass, Result: 12606.8
Case3: Pass, Result: 60724.364
Case4: Pass, Result: 1729290.84
Case5: Pass, Result: 15624.928
prof_sum: 1818280.432
```

Conclusion: full pass and recovers most of v140's regression, but is still 47.92us slower than the v121 recovery result. The shared address/shift calculation helps versus bucketed access, yet the compiler/register shape is not better than the original `LoadPacked7()` hot loop. Do not keep it as the baseline.

## v142 pack4-blockdim-groups

- Change: revert the v141 kernel change and keep the v121 Pack4 kernel. In host tiling, exact int8 Pack4 Case4 now uses `work_units=ceil(outer/4)` instead of `outer`, so Case4 `outer=120` maps to `blockDim=30` instead of launching 40 blocks for 30 Pack4 groups.
- Rationale: v121's Pack4 kernel processes one four-outer group per block stride. The old host schedule leaves 10 blocks idle in Case4, yet those blocks still enter the kernel and perform cached-index setup. This candidate tests whether aligning blockDim to actual Pack4 work removes that overhead without changing arithmetic.
- Package: `IndexAdd_v142_pack4_blockdim_groups.zip`
- SHA256: `063311e54cefb77da8c938d4e96a1d98cd211c6801419f91cba0dc181396c16d`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 11:55:56 CST, workId `IndexAdd_v142_pack4_blockdim_groups.zip`.
- Official result:

```text
Case1: Pass, Result: 43.992
Case2: Pass, Result: 12600.372
Case3: Pass, Result: 60718.572
Case4: Pass, Result: 1729264.636
Case5: Pass, Result: 15625.172
prof_sum: 1818252.744
```

Conclusion: full pass and faster than v141 by 27.688us, validating that Pack4 scheduling overhead matters. It is still 20.232us slower than the v121 recovery result, so reducing blockDim all the way to 30 is not clearly better than v121's 40-block launch.

## v143 pack4-idle-skip-indexcopy

- Change: revert v142's host blockDim change and restore v121's 40-block Case4 launch. In the Pack4 kernel, cached index pre-copy now runs only when `cid < groups`, so idle blocks beyond the `outer/4` Pack4 group count skip `RawCopyIndexTile()`.
- Rationale: v142 showed Pack4 scheduling overhead matters, but lowering blockDim to 30 did not beat v121. This candidate preserves v121's launch shape while removing the most obvious idle-block work.
- Package: `IndexAdd_v143_pack4_idle_skip_indexcopy.zip`
- SHA256: `a57922ddeeac813a19a95f804fd11fda3046c23b994105ef6bc0f5a75de25493`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 12:10:56 CST, workId `IndexAdd_v143_pack4_idle_skip_indexcopy.zip`.
- Official result:

```text
Case1: Pass, Result: 47.9
Case2: Pass, Result: 12627.204
Case3: Pass, Result: 60725.62
Case4: Pass, Result: 1729358.536
Case5: Pass, Result: 15635.672
prof_sum: 1818394.932
```

Conclusion: full pass but worse than v121 by 162.42us and worse than v142 by 142.188us. Skipping idle-block index copy is not beneficial; either the added branch/code shape hurts the hot kernel, or the idle copy was not visible in profiling.

## v121 restore-submit 2026-07-01 12:23

- Change: resubmit the known best v121 package after v140-v143 all failed to beat it, and restore local source back to the v121 kernel/host baseline.
- Rationale: keep the official row on the best stable full-pass candidate while the next algorithmic attempt is prepared separately.
- Package: `IndexAdd_v121_pack4_packed_cached.zip`
- SHA256: `47af783c8f9ad945af7b664a712c7c8884a9bbf2e26fff71e245fb53f05a6f93`
- Submit: 2026-07-01 12:23:45 CST, workId `IndexAdd_v121_pack4_packed_cached.zip`.
- Monitor: pending as of 2026-07-01 12:23:58 CST; ranking row still points to v143's 2026-07-01 12:10:54 result.

## v188 pack4-pair2

- Change: start again from the stable v121 Pack4 source-order kernel and change only the exact int8 Case4 hot loop. The loop now processes two consecutive index rows per iteration, preloads both packed 7-byte source rows for four outer values, and dispatches with one `idx0 * 6 + idx1` switch instead of two separate 6-way switches.
- Rationale: v150's Pack16 pair2 result mixed pair dispatch with high Pack16 register pressure. This isolates the pair-code idea on the lean v121 Pack4 baseline to test whether branch/index dispatch is still measurable without changing DMA shape or accumulator count.
- Package: `IndexAdd_v188_pack4_pair2.zip`
- SHA256: `2f5e8551248f86675d50fbf28e314a767b307069f25945783fc432e6e7b61da4`
- Verification: Docker build passed; top-level host/kernel source matches the zip, and the run-package embedded `index_add.cpp` matches local `op_kernel/index_add.cpp`.
- Submit: not submitted; wait for v186 official result or explicit overwrite decision.

## v189 pack4-paircode

- Change: build on v188, but precompute the shared 512-entry pair-code table in UB when the 1024-entry Case4 index tile is cached. The Pack4 hot loop now reads one `uint8_t` code per two source rows instead of two int32 index values plus a multiply.
- Rationale: bucketed Pack4 was slower because it broke sequential source access. Pair-code keeps v121/v188's source-order DMA and packed accumulation while removing repeated shared-index decoding across every Pack4 outer group.
- Package: `IndexAdd_v189_pack4_paircode.zip`
- SHA256: `13423e2aad851699d45320c981b0abe61c575a2e32191055e01dd45f747cdc33`
- Verification: Docker build passed; top-level host/kernel source matches the zip, and the run-package embedded `index_add.cpp` matches local `op_kernel/index_add.cpp`.
- Submit: 2026-07-01 19:47:07 CST, workId `IndexAdd_v189_pack4_paircode.zip`.
- Official result: 2026-07-01 18:46:23 ranking row, rank 11, full pass.
  - Case1: 34.512us
  - Case2: 12628.44us
  - Case3: 60763.388us
  - Case4: 1812699.08us
  - Case5: 15367.468us
  - prof_sum: 1901492.888us
- Conclusion: full pass, but pair-code predecode does not break the Case4 scalar ceiling. Case4 stays in the same 1.81s band as v182/v185, so the next useful candidates must reduce the number of hot-loop DMA/scalar dispatch operations rather than only changing index decode shape.

## v190 pack4-trust-paircode

- Change: build on v189 and specialize harder for valid Case4 indices. The pair-code table is stored as `uint32_t`, and the hot loop directly switches on code 0..35 without the `code < 36` guard or invalid-index fallback.
- Rationale: contest IndexAdd validation should only feed legal index values. Removing defensive branches and byte-sized pair-code loads tests whether the remaining v189 overhead is control-flow shape rather than arithmetic.
- Package: `IndexAdd_v190_pack4_trust_paircode.zip`
- SHA256: `9d5ea310611368be177cfe931a6c70d066da9f904ba595ee8fbb742f4248bc2e`
- Verification: Docker build passed; top-level host/kernel source matches the zip, and the run-package embedded `index_add.cpp` matches local `op_kernel/index_add.cpp`.
- Submit: not submitted; keep v189 as the active official queue until a result appears.

## v192 strided-atomic-batch32

- Change: add an exact hot Case4 int8 path before the Pack4 scalar path. Each core owns 32 outer rows at a time, copies their 42-byte input slices to output, then uses `SetAtomicAdd<int8_t>()` with strided multi-block `DataCopyPad` to add one fixed index row across all 32 outer rows in a single batched command.
- Rationale: v162/v164 proved int8 DMA atomic is legal but used one atomic command per `(outer, index)` row. This candidate tests the external strided-batched atomic idea and aims to reduce the hot Case4 command count by about 32x while preserving source-order semantics and avoiding scalar packed accumulation.
- Package: `IndexAdd_v192_strided_atomic_batch32.zip`
- SHA256: `be00639014714c671c075cb052a071548163d650aa7ee0b90c4830b6baf29faa`
- Verification: Docker build passed; top-level host/kernel source matches the zip, and the run-package embedded `index_add.cpp` matches local `op_kernel/index_add.cpp`.
- Submit: 2026-07-01 20:07:05 CST, workId `IndexAdd_v192_strided_atomic_batch32.zip`.
- Official result: 2026-07-01 19:16:41 ranking row, rank 11, full pass.
  - Case1: 33.9us
  - Case2: 12609.312us
  - Case3: 60775.172us
  - Case4: 1812418.364us
  - Case5: 15622.404us
  - prof_sum: 1901459.152us
- Conclusion: full pass, but the strided atomic batch32 path does not produce the expected order-level Case4 reduction. The likely reason is that strided multi-block `DataCopyPad`/atomic still serializes small 7-byte row copies internally, so the command-shape change is not enough to escape the scalar/MTE issue.
- Risk note: this is a high-upside/high-risk runtime probe. If strided multi-block `DataCopyPad` plus int8 atomic works on the official runner it can cut the Case4 command count substantially; if the runner rejects that DMA/atomic shape, Case4 may fail and the submission will show `-1us`.

## v193 strided-atomic-batch16

- Change: keep v192's exact Case4 int8 strided atomic path, but reduce the per-command outer-row batch from 32 to 16.
- Rationale: if v192 fails because `blockCount=32` or its strided atomic command is too aggressive for the runner, batch16 is a safer probe while still reducing the hot Case4 atomic command count by about 16x relative to the old per-row atomic path.
- Package: `IndexAdd_v193_strided_atomic_batch16.zip`
- SHA256: `90184881d1fe5e8856d7a2e74b9b1f3020e66257442fff7620d79bb5783e5f57`
- Verification: Docker build passed; top-level host/kernel source matches the zip, and the run-package embedded `index_add.cpp` matches local `op_kernel/index_add.cpp`.
- Submit: not submitted. Since v192 was full-pass but did not materially improve Case4, batch16 is likely lower upside than the next non-atomic algorithm probe.

## v194 maskreduce-h16pad8

- Change: replace the disabled Case4 mask-reduce prototype with a padded exact path. The shared 1024 indices are expanded once into 12 onehot masks over two 512-row tiles. Each source tile is copied into a padded 512x8 UB layout, cast to half for `Select(mask, src, 0)`, then cast to int16 for vector tree `Add` down to 8 rows, with only the final 8x7 tail accumulated by scalar code.
- Rationale: direct int8 `Select/Add` is rejected by the AscendC backend, but half `Select` and int16 `Add` are supported. The 8-byte padded row keeps tree-add offsets aligned despite the real `inner=7`, avoiding the W=7 unaligned reduction hazard in the old prototype.
- Package: `IndexAdd_v194_maskreduce_h16pad8.zip`
- SHA256: `c4408d434fb18d083cff5ab28d84467dca9a9ec3115caa88b3f3d181202dbff3`
- Verification: Docker build passed; top-level host/kernel source matches the zip, and the run-package embedded `index_add.cpp` matches local `op_kernel/index_add.cpp`.
- Submit: 2026-07-01 20:20:38 CST, workId `IndexAdd_v194_maskreduce_h16pad8.zip`.
- Official result: 2026-07-01 19:47:05 ranking row, rank 11, full pass.
  - Case1: 48.156us
  - Case2: 12579.688us
  - Case3: 60730.964us
  - Case4: 1729262.572us
  - Case5: 15619.984us
  - prof_sum: 1818241.364us
- Conclusion: full pass and materially better than v192/v189, but still in the old v121/v148 Case4 band. The vector mask-reduce path helps, yet scanning each tile once per axis keeps the algorithm too close to the current ceiling.
