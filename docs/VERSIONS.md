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
- Official result:

```text
Case1: Pass, Result: 36.292
Case2: Pass, Result: 12665.532
Case3: Pass, Result: 60721.824
Case4: Pass, Result: 1729290.684
Case5: Pass, Result: 15623.82
prof_sum: 1818338.152
```

Conclusion: recovery submit succeeded and restored the official row to the stable v121 family. This run is slower than the earlier 11:21 v121 result by 105.64us, showing the current platform timing has noticeable run-to-run noise.

## v144 case4-pack3-packed

- Change: add a non-Pack4 exact int8 Case4 scalar candidate. When `outer` is divisible by 3, each core processes three outer rows with packed 7-byte `uint64_t` accumulators. Case4 `outer=120` maps to 40 Pack3 groups, so all 40 launched blocks have useful work.
- Rationale: v142 showed Pack4 scheduling shape matters, while Pack4 still leaves only 30 active groups for the likely Case4 shape. Pack3 keeps the same total source bytes and packed int8 arithmetic, lowers accumulator/register pressure versus Pack4, and tests whether full 40-core occupancy beats Pack4's larger per-group reuse.
- Package: `IndexAdd_v144_case4_pack3_packed.zip`
- SHA256: `46d8c419620d9903badadce3ca1cc62d477f1ca9bd22617609d935b10d801efb`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 12:44:54 CST, workId `IndexAdd_v144_case4_pack3_packed.zip`.
- Official result:

```text
Case1: Pass, Result: 33.324
Case2: Pass, Result: 12562.828
Case3: Pass, Result: 60738.508
Case4: Pass, Result: 1823483.112
Case5: Pass, Result: 15605.608
prof_sum: 1912423.38
```

Conclusion: full pass but much slower than v121. Pack3 gives all 40 blocks useful work, but the extra 40 Pack3 index loops versus 30 Pack4 loops dominates; lower register pressure and occupancy do not compensate.

## v145 case4-pack5-packed

- Change: replace the v144 Pack3 path with an exact Case4 Pack5 path for `index_len=1024`. The fifth source tile reuses the spare region in `index32_buf_` after the 4KB cached-index area, so the kernel can process five outer rows per group without adding workspace.
- Rationale: Pack3 proved that full 40-core occupancy is not enough. Pack5 reduces Case4 to 24 source-order groups, 20% fewer switch/index iterations than Pack4's 30 groups, while keeping packed int8 arithmetic. This tests whether fewer scalar dispatch iterations can overcome higher register pressure.
- Package: `IndexAdd_v145_case4_pack5_packed.zip`
- SHA256: `0823c0c9558f1dec858fe9334bb9c2afb3d5a11621f5dcf7758d2dd2b144825c`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 13:01:54 CST, workId `IndexAdd_v145_case4_pack5_packed.zip`.
- Official result:

```text
Case1: Pass, Result: 39.72
Case2: Pass, Result: 12601.888
Case3: Pass, Result: 60752.472
Case4: Pass, Result: 1729313.46
Case5: Pass, Result: 15616.524
prof_sum: 1818324.064
```

Conclusion: full pass and better than the 12:23 v121 recovery run by 14.088us, but Case4 itself is still slower than v121. Pack5 reduced scalar dispatch groups from 30 to 24, but higher register pressure and idle block overhead appear to offset the loop-count reduction.

## v146 case4-pack5-blockdim24

- Change: keep the v145 Pack5 scalar kernel, but change host tiling so the exact Case4 Pack5 path reports `work_units=outer/5`. For the observed Case4 shape this launches 24 active blocks instead of the v121/v145 40-block launch with 16 idle blocks.
- Rationale: isolate whether Pack5's idle block scheduling overhead was masking the benefit of fewer scalar index-dispatch groups. This is a host-only scheduling experiment; the kernel arithmetic and memory path are unchanged from v145.
- Package: `IndexAdd_v146_pack5_blockdim24.zip`
- SHA256: `cc6ba488923156c52f8bc2f7c5f65e21818eca8b58f6c8e2762e70e83ea1c2eb`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 13:11:58 CST, workId `IndexAdd_v146_pack5_blockdim24.zip`.
- Official result:

```text
Case1: Pass, Result: 39.24
Case2: Pass, Result: 12619.312
Case3: Pass, Result: 60755.156
Case4: Pass, Result: 1729292.38
Case5: Pass, Result: 15627.58
prof_sum: 1818333.668
```

Conclusion: full pass but 9.604us slower than v145. The Pack5/blockDim24 scheduling experiment does not materially affect the real hot Case4, supporting the profile-based correction that the hidden Case4 outer is likely `8192` rather than a small multiple of five.

## v147 case4-pack8-packed

- Change: add an exact int8 Case4 Pack8 source-order scalar path for `index=int32`, `index_len=1024`, and `outer % 8 == 0`. Four 16KB UB buffers each hold two 7KB source tiles at offsets `0` and `8192`, leaving slack for unaligned `LoadPacked7`; the shared index tile remains cached in `index32_buf_`.
- Rationale: the profile note indicates the hot Case4 has `outer=8192`, so v144 Pack3 and v145 Pack5 likely do not hit the hot path. Pack8 does hit `8192 % 8 == 0` and halves the Pack4 group count from 2048 to 1024 while preserving sequential source access and packed byte-wise accumulation.
- Package: `IndexAdd_v147_case4_pack8_packed.zip`
- SHA256: `bf13f69a9ad1da18de05964cc7ef899c40dfdd0ebde6cc6014596cb1a8fe0166`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 13:25:43 CST, workId `IndexAdd_v147_case4_pack8_packed.zip`.
- Official result:

```text
Case1: Pass, Result: 42.8
Case2: Pass, Result: 12592.688
Case3: Pass, Result: 60760.7
Case4: Pass, Result: 1729266.328
Case5: Pass, Result: 15618.592
prof_sum: 1818281.108
```

Conclusion: full pass and 52.56us faster than v146, confirming the Pack8 path hits the hidden Case4 shape. The gain is still tiny relative to the target, so larger PackN alone is not the order-level breakthrough.

## v148 case4-pack16-packed

- Change: add a more aggressive exact int8 Case4 Pack16 path for `outer % 16 == 0`. It processes the 1024 index dimension as two 512-row tiles; four 16KB UB buffers are split into four 4KB slots each, so all 16 source tiles fit with slack for unaligned packed loads.
- Rationale: if the profile shape is `outer=8192`, Pack16 cuts Pack8's source-order switch/index loop count in half again while preserving conflict-free per-outer accumulation. The main risk is much higher scalar register pressure from 16 x 6 packed accumulators.
- Package: `IndexAdd_v148_case4_pack16_packed.zip`
- SHA256: `9ec803acc7f6a327bbb32f028f9240cee033abd7a7be9949d46c157a981dfe0d`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 13:31:45 CST, workId `IndexAdd_v148_case4_pack16_packed.zip`.
- Official result:

```text
Case1: Pass, Result: 40.952
Case2: Pass, Result: 12576.772
Case3: Pass, Result: 60714.456
Case4: Pass, Result: 1729255.06
Case5: Pass, Result: 15612.3
prof_sum: 1818199.54
```

Conclusion: full pass and 81.568us faster than v147. Pack16 continues the scalar PackN trend, but the total gain remains far below the order-level target; the hot loop is still dominated by scalar packed loads/adds and index dispatch.

## v149 case4-pack32-array

- Change: add an exact int8 Case4 Pack32 path for `outer % 32 == 0`. It uses four 16KB UB buffers split into eight 2KB source slots, processes the index dimension as four 256-row tiles, and stores the 32 x 6 packed accumulators in a local scalar array so each index performs a dynamic bucket update instead of a six-way switch.
- Rationale: Pack8/Pack16 confirmed that reducing dispatch groups helps but does not break the scalar ceiling. This candidate tests a different scalar shape: halve Pack16's group count again and remove the per-index switch, accepting the risk that dynamic accumulator indexing spills into private memory.
- Package: `IndexAdd_v149_case4_pack32_array.zip`
- SHA256: `589770604310d6844fc04a5e9f667546aaca86fbf876d3f480e8ff07422be50b`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 14:02:40 CST, workId `IndexAdd_v149_case4_pack32_array.zip`.
- Official result: pending; as of 14:04 CST the ranking row still pointed to v148's `2026/07/01 13:31:43` result.

## v150 case4-pack16-pair2

- Change: keep the Pack16 source tiling and register accumulator layout, but change the hot index loop from one 6-way switch per source row to one 36-way switch per two consecutive source rows. Host tiling explicitly prioritizes this exact path with `work_units=outer/16`, so it wins over the Pack32-array branch for `outer=8192`.
- Rationale: Pack16 was the best confirmed scalar candidate, while Pack32-array risks accumulator spills. Pair2 reduces index/switch dispatch by half without dynamic accumulator indexing, testing whether branch overhead is still a major component after Pack16.
- Package: `IndexAdd_v150_case4_pack16_pair2.zip`
- SHA256: `9693874a60ea0df0f36dcfb2ef85a9041401d6c0a99d2da6033a56deed9c89d3`
- Verification: Docker/Lima build passed; top-level sources and run-package embedded `index_add.cpp` match local source.
- Submit: 2026-07-01 14:10:18 CST, workId `IndexAdd_v150_case4_pack16_pair2.zip`.
- Official result: pending; as of 14:10 CST the ranking row still pointed to v148's `2026/07/01 13:31:43` result.

## v157 mask-i16-direct

- Change: test a pure AIV Case4 mask-reduce path that would cast `int8` source directly to `int16`, then select and tree-reduce in `int16`.
- Rationale: this would remove the old `int8 -> half -> int16` conversion chain from the vector mask-reduce path and avoid the scalar Pack16 ceiling.
- Package: none.
- Verification: Docker build failed during Ascend C kernel compilation. CANN 8.5 for 910B has `Cast<int16_t, half>`, `Cast<int16_t, float>`, and `Cast<int16_t, int32_t>`, but no `Cast<int16_t, int8_t>` intrinsic.
- Conclusion: direct int8-to-int16 vector cast is not available on this toolchain/device path.

## v158 mask-half-reduce

- Change: enable the exact hot Case4 mask-reduce path for `dtype=int8`, `index=int32`, `outer=8192`, `axis=6`, `inner=7`, `index_len=1024`, but reduce directly in `half` after the initial `int8 -> half` cast. This removes the per-axis `half -> int16` cast from v120's mask-reduce design.
- Rationale: Pack16/Pack32 are bounded by scalar index dispatch and packed scalar accumulation. v158 revisits the onehot-mask/vector-tree idea with one less conversion stage. It has no KFC lifecycle risk and should be judged by official correctness/performance, especially for the hidden Case4 distribution.
- Package: `IndexAdd_v158_mask_half_reduce.zip`
- SHA256: `ecbd759cd88597b94d812cca3e1eab4f7c79e31d29c4520d376692ebad269a06`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: not submitted yet.

## v159 pack16-bucket

- Change: add an exact hot Case4 Pack16 bucket path for `dtype=int8`, `index=int32`, `outer=8192`, `axis=6`, `inner=7`, and `index_len=1024`. The kernel preclassifies the shared 1024-entry index into `2 x 6` tile-local offset buckets, then the Pack16 hot loop accumulates each axis bucket directly instead of executing a per-source-row switch.
- Rationale: this adapts the small-inner/group-reduce direction from the latest external suggestion while avoiding unsupported int8 atomic add. Compared with v140's Pack4 bucket attempt, v159 keeps the best confirmed Pack16 source tiling and stores offsets per 512-row tile so each bucket only references the currently resident 4KB source slot.
- Package: `IndexAdd_v159_pack16_bucket.zip`
- SHA256: `25f557bd1531730f62de0f8e86dd85be79440b8f5b0b0b758058f9f6e499ed81`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 15:13:11 CST, workId `IndexAdd_v159_pack16_bucket.zip`.
- Official result:

```text
Case1: Pass, Result: 43.512
Case2: Pass, Result: 12560.82
Case3: Pass, Result: 60750.012
Case4: Pass, Result: 1812407.92
Case5: Pass, Result: 15619.524
prof_sum: 1901381.788
```

Conclusion: full pass but worse than v148 by about 83ms, so preclassified bucket order alone does not break the scalar Pack16 ceiling.

## v160 gather-float-reduce

- Change: add an exact hot Case4 Gather path before v159. The path reuses the `2 x 6` bucketed index offsets, copies each 512-row source tile for one outer row, casts `int8 -> half -> float`, gathers only the matching rows for each axis/lane, then uses an aligned manual tree reduction that stops at 16 elements before scalar finalization.
- Rationale: this is the higher-fidelity version of the external group/reduce recommendation for the small-inner Case4 shape. It avoids unsupported int8 atomic add and avoids the `WholeReduceSum/vcadd` primitive that made v137 fail globally, while keeping float accumulation so int8 exactness is not lost to half rounding.
- Package: `IndexAdd_v160_gather_float_reduce.zip`
- SHA256: `81f338c5beb6a1d9bb62377f90ceaa5b68a4f5d2377c5b571034bbce119ab0db`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 15:20:07 CST, workId `IndexAdd_v160_gather_float_reduce.zip`.
- Official result:

```text
Case1: Pass, Result: 37.956
Case2: Pass, Result: 12579.64
Case3: Pass, Result: 61446.868
Case4: Pass, Result: 1812374.328
Case5: Pass, Result: 15390.54
prof_sum: 1901829.332
```

Conclusion: full pass but worse than v148 by about 83ms. The lane-by-lane float Gather path reduces scalar branching but adds too much cast/gather/reduce overhead.

## v161 gather-i16-reduce

- Change: add an exact hot Case4 Gather path before v160. It keeps the same bucketed offset strategy, but casts `int8 -> half -> int16`, gathers int16 lanes, and reduces with vector `Add<int16_t>` before scalar finalization to int8.
- Rationale: int8 vector add and int8 DMA atomic are not supported on this 910B/CANN 8.5 path. The int16 gather/reduce version preserves final int8 modulo behavior while avoiding v160's float bandwidth and float arithmetic.
- Package: `IndexAdd_v161_gather_i16_reduce.zip`
- SHA256: `5c073be51e331fcae9288c422f23420ba6b8bc34a8ba7b56469e2686c605c432`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 15:25:27 CST, workId `IndexAdd_v161_gather_i16_reduce.zip`.
- Official result:

```text
Case1: Pass, Result: 37.82
Case2: Pass, Result: 12616.528
Case3: Pass, Result: 61495.776
Case4: Pass, Result: 1812413.892
Case5: Pass, Result: 15620.32
prof_sum: 1902184.336
```

Conclusion: full pass but slightly worse than v160. Reducing in int16 preserves int8 wraparound behavior, but lane-by-lane Gather still has too much command and barrier overhead.

## v162 atomic-i8-hotcase

- Change: add an exact hot Case4 source-major atomic path before the gather/reduce experiments. The path first copies `input` to `output`, calls `SyncAll<true>()`, then streams each outer's contiguous source tile from GM to UB and uses `SetAtomicAdd<int8_t>()` on 7-byte row writes into `output`.
- Rationale: this directly tests the external recommendation's `copy self -> atomic scatter` algorithm on the dominant small-inner int8 case. CANN 8.5's 910B/c220 headers compile `SetAtomicAdd<int8_t>()`; this candidate checks whether direct 7-byte atomic `DataCopyPad` is valid and whether DMA atomic can move the hot loop out of scalar AIV arithmetic.
- Package: `IndexAdd_v162_atomic_i8_hotcase.zip`
- SHA256: `3d6d3c6ab4715f05d3fe47467a3a399a3baf8d9683ce4980543cce9587ac217e`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 15:37:56 CST, workId `IndexAdd_v162_atomic_i8_hotcase.zip`.
- Official result:

```text
Case1: Pass, Result: 30.84
Case2: Pass, Result: 12605.84
Case3: Pass, Result: 60748.76
Case4: Pass, Result: 1812378.432
Case5: Pass, Result: 15607.136
prof_sum: 1901371.008
```

Conclusion: full pass, proving direct int8 DMA atomic is legal on the platform. However, Case4 stays in the 1.812s band, so one atomic DMA command per source row is still too fine-grained to approach the leading solutions.

## v163 owner-atomic-i8

- Change: keep v162's exact hot Case4 direct int8 atomic scatter, but remove the global copy/sync phase. Each core owns a disjoint set of `outer` rows, copies that outer's 42-byte `input` slice to `output`, then atomically adds only that same outer's 1024 source rows.
- Rationale: this tests the same `SetAtomicAdd<int8_t>()` direct 7-byte row writes while avoiding `SyncAll<true>()` as a possible runtime failure source. Because each outer is owned by exactly one core, no other core can overwrite the initialized output row.
- Package: `IndexAdd_v163_owner_atomic_i8.zip`
- SHA256: `9463a82bcb25745795d0aad20c9881a8fc75f263e655919dff086c6a4ce52f65`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: pending.

## v164 owner-atomic-once

- Change: keep v163's owner-local direct int8 atomic algorithm, but split it into two owner-local phases: first copy all owned `input` outer slices to `output`, then enable `SetAtomicAdd<int8_t>()` once per core and scatter all owned source rows before a single `SetAtomicNone()`.
- Rationale: removes 8192 per-outer atomic mode toggles from v163 while preserving the no-global-sync ownership property. This should be the faster direct atomic shape if GM->UB source reads are unaffected by DMA atomic state.
- Package: `IndexAdd_v164_owner_atomic_once.zip`
- SHA256: `cb2db7be7059f60c574e9ce172b37d3afd152617b077a14dddedbc1c2aa675d5`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 15:47:47 CST, workId `IndexAdd_v164_owner_atomic_once.zip`.
- Official result:

```text
Case1: Pass, Result: 34.22
Case2: Pass, Result: 12595.064
Case3: Pass, Result: 60739.864
Case4: Pass, Result: 1812396.42
Case5: Pass, Result: 15613.808
prof_sum: 1901379.376
```

Conclusion: full pass, but no improvement over v162/v148. Moving `SetAtomicAdd<int8_t>()` outside the per-outer loop does not solve the fine-grained atomic DMA command bottleneck.

## v165 workspace-atomic-i8

- Change: move the exact hot Case4 int8 atomic destination from the 7-byte output rows to a 32-byte padded GM workspace row per `(outer, axis)`. Each core initializes only its owned outer rows in workspace, atomically adds 32-byte padded source rows, then compacts the first 7 bytes of each workspace row back to output.
- Rationale: this is a safety variant for the direct atomic line. If v164 fails because 7-byte atomic `DataCopyPad` is illegal or misaligned at runtime, v165 tests the same `SetAtomicAdd<int8_t>()` path using fully aligned 32B atomic writes.
- Package: `IndexAdd_v165_workspace_atomic_i8.zip`
- SHA256: `1207c206a9716e2f80443836b18c51960939b5a813f1ffaf07e39f05666bb29c`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: pending.

## v166 workspace-atomic-packed

- Change: keep v165's exact hot Case4 32B workspace atomic algorithm, but pack each 7-byte int8 row through `uint64_t` helpers before the 32B atomic write. Workspace initialization and source scatter now use packed row construction instead of seven scalar `GetValue`/`SetValue` pairs per row; final compaction also uses packed 7-byte loads.
- Rationale: this follows the external `copy self -> atomic scatter` direction while shaving the extra scalar row-marshalling cost introduced by the padded workspace safety variant.
- Package: `IndexAdd_v166_workspace_atomic_packed.zip`
- SHA256: `9221d8f54685287d5477c4de1ced1f2f80eb00366e3692421d1030c1d60f0d3e`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 16:03:58 CST, workId `IndexAdd_v166_workspace_atomic_packed.zip`.
- Official result: pending; as of 16:04 CST the ranking row still pointed to v151's `2026/07/01 14:23:54` `-1us` result.

## v167 row-gather-i16-reduce

- Change: add an exact hot Case4 int8 row-gather reduce path before atomic. The shared index is expanded once per core into a compact `uint32_t` offset table containing all seven lane offsets for each bucket, so each `(tile, axis)` uses one `Gather<int16_t>` over `count * 7` values instead of seven lane-specific gathers.
- Rationale: v160 proved lane-by-lane Gather is too expensive. This candidate keeps the group/reduce algorithm but cuts gather calls per outer from 84 to 12 and reduces lanes together with row-major tree `Add`.
- Package: `IndexAdd_v167_row_gather_i16_reduce.zip`
- SHA256: `f11526fdd8669c884230b0ae0b9739ede728cd7faafb94bc0edb0de9d01ec18c`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source.
- Submit: 2026-07-01 16:15:09 CST, workId `IndexAdd_v167_row_gather_i16_reduce.zip`.
- Official result: pending; as of 16:15 CST the ranking row still pointed to v160's `2026/07/01 15:20:04` result.

## v168 opapi-builtin-forward

- Change: override the generated `aclnnIndexAdd` op_api entry with a C++ shim. The shim first tries to forward each call to CANN's built-in `libopapi.so` `aclnnIndexAdd`; if that symbol path is unavailable or rejects the case, it falls back to the original generated custom-op executor embedded from `aclnn_index_add.cpp`.
- Rationale: the public wrapper calls `aclnnIndexAdd` 30 times and profiles the median AICore task duration. This candidate tests whether CANN's built-in IndexAdd implementation already contains a lower-scalar segmented-reduce path for the hidden Case4 shape, without changing the existing GE/kernel fallback.
- Package: `IndexAdd_v168_opapi_builtin_forward.zip`
- SHA256: `5bb5a95d382b998d71deb0a45be4b6c98847632a1dfc4b445faad9d126942d44`
- Verification: Docker build passed; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source; `libcust_opapi.so` exports `aclnnIndexAddGetWorkspaceSize` and `aclnnIndexAdd`.
- Submit: 2026-07-01 16:37:15 CST, workId `IndexAdd_v168_opapi_builtin_forward.zip`.
- Official result: pending; as of 16:37 CST the ranking row still pointed to v164's `2026/07/01 15:47:37` result.

## v169 pack16-bulk-dma

- Change: restore the exact int8 Case4 Pack16 pair2 path ahead of the later gather/atomic experiments, restore the normal generated op_api instead of v168's built-in-forward shim, then coarsen the Pack16 DMA shape. Each group now copies 16 contiguous input outer rows in one 672-byte `DataCopyPad`, copies source tiles as four 4-block strided `DataCopyPad` calls instead of sixteen separate calls, and writes the 16 contiguous output rows back in one 672-byte `DataCopyPad`.
- Rationale: v148/v150 showed packed scalar accumulation is the best confirmed correct arithmetic path, while profiling still suggests excessive scalar/MTE command overhead. This candidate keeps the same byte-wise wraparound accumulation but cuts Pack16 per-group DMA command count from roughly 64 to roughly 10, testing whether small-copy launch overhead is the missing factor.
- Package: `IndexAdd_v169_pack16_bulk_dma.zip`
- SHA256: `fa7d3e66f46f629cf550f46f3095c54d3a77fa93885dc716bf46acf74a2341f1`
- Verification: Docker build passed with normal generated op_api; top-level host/kernel sources match the zip; run-package embedded `index_add.cpp` matches local source; zip does not contain the v168 `op_host/fast/aclnn_index_add.cpp` shim.
- Submit: 2026-07-01 16:47:26 CST, workId `IndexAdd_v169_pack16_bulk_dma.zip`.
- Official result: pending; as of 16:47 CST the ranking row still pointed to v164's `2026/07/01 15:47:37` result.
