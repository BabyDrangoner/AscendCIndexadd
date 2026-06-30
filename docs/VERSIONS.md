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
- Official result:

```text
Case1: Pass, Result: 40.1
Case2: Pass, Result: 12591.652
Case3: Pass, Result: 60715.208
Case4: Pass, Result: 1729338.12
Case5: Pass, Result: 15626.452
prof_sum: 1818311.532
```

Conclusion: full pass and much better than bd40/v93, mainly from Case4. Still slightly slower than the best known 1.818158s result, so continue tuning.

## v96 pack4-workunits

- Change: when host tiling selects the int8 case4 Pack4 shape, set `work_units = outer / 4` so `blockDim` follows Pack4 groups instead of outer rows.
- Rationale: for the likely `(2,3,4,5,6,7)` case4 shape, this maps 120 outer rows to 30 pack4 groups and avoids launching extra idle cores.
- Package: `IndexAdd_v96_pack4_workunits.zip`
- SHA256: `ce257bbe9c82c3c706819690daa7b6c20deaa1742c577ad2d993b1336c694115`
- Official result:

```text
Case1: Pass, Result: 39.592
Case2: Pass, Result: 12627.62
Case3: Pass, Result: 60725.28
Case4: Pass, Result: 1729369.4
Case5: Pass, Result: 15620.372
prof_sum: 1818382.264
```

Conclusion: full pass but slower than v95 by 70.732us. Keep v95-style 40-core host scheduling for Pack4.

## v97 outer1-src32k

- Change: restore the stable `Outer1` case4 int8 dispatch before `Pack2`/`Pack4`, keep host `work_units = outer`, and enlarge only the case4 int8 `src_buf_` from 16KB to 32KB.
- Rationale: Pack2 did not improve official Case4. A larger source tile keeps the proven one-outer register accumulation but roughly halves tile-loop overhead for large `index_len`.
- Package: `IndexAdd_v97_outer1_src32k.zip`
- SHA256: `196a6b4a2e2249d4b28ed9af8089afe8cffa4b28a313d689c4e542758138dec6`
- Status: built but superseded before submit; package inspection showed the real `Outer1` tile cap still used the old 16KB constant.

## v98 outer1-src32k-tilecap

- Change: fix v97 by applying the 32KB source tile cap to `ProcessCase4Outer1Int8()` itself.
- Rationale: this is the intended safe tile-loop reduction for the stable outer1 path.
- Package: `IndexAdd_v98_outer1_src32k_tilecap.zip`
- SHA256: `cbe1120fdd117d02eafe76fcfde789df38c8a500b500da3ca728b6b1ffd714e3`
- Status: built; submit after v96 official result is captured.

## v99 pack4-src32k

- Change: return to v95-style Pack4 priority and 40-core host scheduling, but allocate 32KB buffers for all four Pack4 source tiles and use a 32KB Pack4 tile cap.
- Rationale: v95 proved Pack4 full-pass and improved Case4; larger tiles should reduce per-tile index/source copy and branch overhead without changing arithmetic.
- Package: `IndexAdd_v99_pack4_src32k.zip`
- SHA256: `a64d78f9c45a9a8b349a3ee32d90eddb01d6b960abd171642e6ddc0382007d16`
- Official result:

```text
Case1: Pass, Result: 41.648
Case2: Pass, Result: 12600.94
Case3: Pass, Result: 60737.952
Case4: Pass, Result: 1812401.736
Case5: Pass, Result: 15620.06
prof_sum: 1901402.336
```

Conclusion: full pass, but worse than v95 by 83090.804us. The 32KB Pack4 tile hurts Case4, so do not continue increasing the Pack4 tile size.

## v100 pack3-src32k

- Change: add a Pack3 int8 case4 path before Pack4. For likely `outer=120`, this creates exactly 40 groups, keeping all 40 cores active while reusing each index scan across three outer rows.
- Rationale: v95 Pack4 improved Case4 but only has 30 groups for `outer=120`; Pack3 trades less index reuse for better core occupancy and lower register pressure.
- Package: `IndexAdd_v100_pack3_src32k.zip`
- SHA256: `24cf49f187ffa65d06b9db37619b454a08b37c224c4920bf7a3f519caaf073d8`
- Status: built; submit after v99 official result is captured.

## v101 pack3-row16

- Change: keep v100 Pack3 source tiles at 32KB, but leave `out_buf_` at the normal 16KB for Pack3 because it only stages a 42-byte output row.
- Rationale: reduce UB pressure without changing Pack3 arithmetic or tile size.
- Package: `IndexAdd_v101_pack3_row16.zip`
- SHA256: `ad448feaed8682d7fa15156abbb93a65922e7f5a3c4d83ca39b42b8596459f3e`
- Status: built; submit after v99 official result is captured.

## v102 pack3-src16k

- Change: keep Pack3 priority and row16, but use the original 16KB source tiles for Pack3.
- Rationale: separates Pack3 scheduling/core-occupancy effects from the larger 32KB tile and UB pressure.
- Package: `IndexAdd_v102_pack3_src16k.zip`
- SHA256: `49f9137d447bc3a7d1d63c17b2ca0721d355b82a3d3e4ca169921e748213f6a3`
- Official result:

```text
Case1: Pass, Result: 37.908
Case2: Pass, Result: 12630.28
Case3: Pass, Result: 60773.312
Case4: Pass, Result: 1812396.92
Case5: Pass, Result: 15635.628
prof_sum: 1901474.048
```

Conclusion: full pass but worse than v95 by 83162.516us. Pack3 scheduling without the 32KB tile still falls into the 1.901s band, so Pack3 is not promising.

## v103 class3-tile1024

- Change: add a Case4 int8 Class3 path before Pack3. Each core owns up to three outer rows, classifies a 1024-entry index tile into six position lists, then accumulates source rows by class to remove the inner source-loop switch.
- Rationale: v99 showed larger source tiles hurt. This instead tests whether reducing branch pressure in the hot int8 Case4 loop can beat Pack3/Pack4 while keeping UB usage inside the original 16KB buffers.
- Package: `IndexAdd_v103_class3_tile1024.zip`
- SHA256: `82575150c5b1091798b8b636e239b4b44318f62fa32476c026e0c894bd4a9795`
- Official result:

```text
Case1: Pass, Result: 33.656
Case2: Pass, Result: 12608.208
Case3: Pass, Result: 61471.232
Case4: Pass, Result: 1901334.44
Case5: Pass, Result: 15620.652
prof_sum: 1991068.188
```

Conclusion: full pass, but worse than v95 by 172756.656us and worse than v102 by 19594.14us. Classifying index positions before reading source adds too much UB traffic; skip v104 unless a future signal contradicts this.

## v104 class3-tile512

- Change: reduce the Class3 position/source tile from 1024 to 512 entries.
- Rationale: parameter sweep for the Class3 path. Smaller tiles increase loop count but reduce active UB position/source footprint and may improve local access behavior.
- Package: `IndexAdd_v104_class3_tile512.zip`
- SHA256: `2ea2b733bbab0c58da2082d6c1ed9b6acff1c6cd89b01a2053d0caeb26ffa24f`
- Status: built; submit after v103 if Class3 shows promise.

## v105 class4-tile1024

- Change: switch the classified Case4 int8 path to four outer rows per group with a 1024-entry tile, while keeping host `blockDim` at the v95-style 40.
- Rationale: compares against Pack4 more directly: same four-row index reuse shape and 40-block launch behavior, but source accumulation runs by preclassified index buckets instead of a per-source switch.
- Package: `IndexAdd_v105_class4_tile1024.zip`
- SHA256: `9b093ad36595996cfdb941ed5ca70b62d53882c192ef4e6af918920fb63ae774`
- Status: built; submit after v103/v104 depending on Class3 signal.

## v106 pack4-src8k

- Change: restore the Case4 int8 dispatch priority to the v95-style Pack4 path, reduce only the Pack4 source tile to 8KB, and restore the Outer1 tile to 16KB.
- Rationale: v99 showed 32KB Pack4 buffers were slower, likely from UB pressure. This candidate sweeps the opposite direction while keeping Pack4's four-outer index reuse.
- Package: `IndexAdd_v106_pack4_src8k.zip`
- SHA256: `d9e440d7c8f1ef93d10fd03bc0c2a1b009543925d0858a1b7da7b0e0564fbd9f`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`89d6512568039f883c2036f65e250ccd1103a27e2ccaf637b4948e21a83330de`).
- Submit: 2026-06-30 16:26:50, workId `0000019F17A3573EB01156FB7EA919F4`.
- Official result:

```text
Case1: Pass, Result: 39.848
Case2: Pass, Result: 12589.44
Case3: Pass, Result: 61480.86
Case4: Pass, Result: 1901336.872
Case5: Pass, Result: 15615.52
prof_sum: 1991062.54
```

Conclusion: full pass but much worse than v95. Reducing Pack4 source tile to 8KB increases tile-loop overhead and falls into the 1.991s band, so v107/v108 should be lower priority than an algorithmic change.

## v107 pack4-src12k

- Change: keep the v106 Pack4-priority dispatch but increase the Pack4 source tile from 8KB to 12KB.
- Rationale: parameter sweep between 8KB and the v95 16KB best-known point to determine whether smaller UB footprint or fewer tile loops dominates Case4.
- Package: `IndexAdd_v107_pack4_src12k.zip`
- SHA256: `f1882e6b434166ab1626a5fbca8a39d541c6eb81b4ece1f5abd96c17afaa7306`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`c54eaeb727580f50cca22fe9c47fe312b362252a36a828d44040d00cf0f753d7`).
- Status: built; submit after v106 result is captured.

## v108 pack4-src20k

- Change: keep the v106 Pack4-priority dispatch but increase the Pack4 source tile to 20KB.
- Rationale: parameter sweep between the v95 16KB best-known point and the slower v99 32KB point; 20KB may reduce tile-loop count without the full UB pressure of 32KB.
- Package: `IndexAdd_v108_pack4_src20k.zip`
- SHA256: `790455b055b61070c93961550e8306ce7ca81a23f3bc0ebb9f187d78605474be`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`cdfa893495c3864e49309116d00ed594a2432afe2cf3fcd307ca10bb8ec68625`).
- Status: built; submit after v106/v107 trend is known.

## v109 pack4-packed16k

- Change: replace the Case4 int8 Pack4 accumulator with packed 7-byte rows in `uint64_t`, using byte-wise modulo addition instead of 168 int32 scalar accumulators.
- Rationale: int8 correctness only depends on the low 8 bits of each accumulated element. Packed byte addition should reduce register pressure and UB byte `GetValue` traffic while preserving modulo int8 semantics.
- Package: `IndexAdd_v109_pack4_packed16k.zip`
- SHA256: `e48cbd0451e7c7b00db005c7a671aa992b316ab731c420d25e3a283cfdc96e07`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`11fe6ec19b9cbf87bb62b9632ac356dad03589487e8ec8d87bde1aa34b5cb684`).
- Submit: 2026-06-30 16:42:01, workId `0000019F17B13F43B0139C2F275482B8`.
- Official result:

```text
Case1: Pass, Result: 40.36
Case2: Pass, Result: 12627.048
Case3: Pass, Result: 60912.82
Case4: Pass, Result: 1812570.764
Case5: Pass, Result: 15616.54
prof_sum: 1901767.532
```

Conclusion: full pass, but still worse than v95 by 83456.0us. Packed byte accumulation alone reduces some scalar register work but does not remove the dominant repeated index handling in Case4.

## v110 pack4-packed20k

- Change: keep the v109 packed accumulator and increase the Pack4 source tile to 20KB.
- Rationale: if packed accumulation reduces register/UB pressure enough, a larger tile may reduce loop count without reproducing the scalar 32KB regression.
- Package: `IndexAdd_v110_pack4_packed20k.zip`
- SHA256: `b6c27299dcede27303f973a322a61ccffcee121c7c10b92bf136d0d37d3ebf22`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`41c458da37f356c55e93a6ef0d600b6d30d332b3853aced7f980492837b7fb17`).
- Status: built; submit only if v109 is correct and promising.

## v111 pack4-pos-packed16k

- Change: add a Case4 int8 Pack4 position-list path before the normal Pack4 path. Each core precomputes six shared `index -> position` lists once, then reuses those lists for all four-row Pack4 groups assigned to that core.
- Rationale: the profile shows Case4 is scalar-pipe bound (`aiv_scalar_ratio ~= 0.985`) and the index vector is shared across all outer rows. This candidate attacks repeated per-outer index decoding while keeping the v109 packed int8 accumulator.
- Package: `IndexAdd_v111_pack4_pos_packed16k.zip`
- SHA256: `241815b55768b656b3fccddfdbf57e67c8b2d94dc616b00b06518ae8bc1b12a0`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`a388d2d137151a2e7782c879900c042a468d339428a5f7bf834922f777878388`).
- Submit: 2026-06-30 16:59:37, workId `0000019F17C15B62B0214B3CA1A2A016`.
- Official result:

```text
Case1: Pass, Result: 33.908
Case2: Pass, Result: 12768.6
Case3: Pass, Result: 60566.236
Case4: Pass, Result: 1812631.892
Case5: Pass, Result: 15617.56
prof_sum: 1901618.196
```

Conclusion: full pass but still in the slower packed/position-list band, worse than v95 by 83306.664us. Preclassifying positions avoids repeated index decoding, but the resulting source gather order and packed byte arithmetic do not beat the v95 sequential int32 accumulator path.

## v112 pack4-block8

- Change: add an exact-shape Case4 int8 Pack4Block8 path for `index_len == 1024`. Each core caches the shared int32 index as uint8 once, copies each four-row source slab once, then decodes eight 7-byte rows from seven uint64 loads instead of calling the generic unaligned packed loader for every row.
- Rationale: v111 removes repeated index classification but turns source access into position-list gathers. This candidate keeps source access sequential while reducing packed-load scalar overhead and per-group index GM copies.
- Package: `IndexAdd_v112_pack4_block8.zip`
- SHA256: `1c67a869d91372b6fd20ffc85249904553ba6518dc290eb078c0a53557401f8e`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`b3c9e429089b899de1aa68b99d40b674e351a8c17b573be7b1a3bf6dbc4dca19`).
- Status: built; submit after v111 result is captured.

## v113 pack4-cachedidx

- Change: add an exact-shape Case4 int8 Pack4CachedIdx path for `index_len == 1024`. It returns to the v95-style int32 accumulator and sequential source scan, but caches the shared int32 index as uint8 once per core instead of copying/reading an int32 index tile for every Pack4 group.
- Rationale: official v109 showed packed byte arithmetic was slower than v95. This candidate isolates the cheaper part of the profile idea: reduce repeated index handling without introducing packed bit operations or position-list source gathers.
- Package: `IndexAdd_v113_pack4_cachedidx.zip`
- SHA256: `a05295cddaba71d3379f7bb9df995071d8d3e52d8b55b2004368d0df6019c4ba`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`c40d038ca8f3634c2c61e3a6ee992235112adf0e006a31119f99bf398c200608`).
- Submit: 2026-06-30 17:11:23, workId `0000019F17CC2207B014CF897630FCB6`.
- Official result:

```text
Case1: Pass, Result: 38.344
Case2: Pass, Result: 12714.004
Case3: Pass, Result: 60582.436
Case4: Pass, Result: 1812610.072
Case5: Pass, Result: 15617.412
prof_sum: 1901562.268
```

Conclusion: full pass but still worse than v95 by 83250.736us. Caching the shared index without changing the scalar accumulation shape is not enough; the remaining bottleneck needs a real vectorized reduction path.

## v114 pack4-cachedidx-u8acc

- Change: keep the v113 exact-shape cached-index path, but switch its per-output accumulators from int32 to uint8 modulo accumulation.
- Rationale: int8 correctness on the platform has already accepted low-byte modulo arithmetic in v109. This tests whether avoiding int32 accumulator widening lowers scalar-register pressure without paying the packed-byte bit-operation cost.
- Package: `IndexAdd_v114_pack4_cachedidx_u8acc.zip`
- SHA256: `62f1ed6be4dc20cecdbca0e86dc483d5a32941c275e7998d9789e8a72ee0240f`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`c3dafdaff68a095cf5f8165c43b9f3bb4f991120cef6000ebc8d2711b90c291f`).
- Status: built; submit after v113 result is captured.

## v115 case4-fp32-maskreduce

- Change: add an exact-shape fp32 Case4 path for `axis == 6`, `inner == 7`, and `index_len == 1024`. The path expands the shared index into 512x7 tiles, precomputes 12 `CompareScalar(idx == axis)` masks once per core, then uses `Select(mask, src, 0)` plus aligned vector tree `Add` reductions from 512 rows down to 8 rows before a small scalar tail.
- Rationale: the profile points to scalar execution as the Case4 bottleneck. This candidate tests the onehot-mask/vector-reduction idea directly while avoiding the known int32 `Select` compile failure and leaving the current full-pass int8 path untouched.
- Package: `IndexAdd_v115_case4_fp32_maskreduce.zip`
- SHA256: `2f986b8b1a444360ad5459c50e6d2cea811c2b1a00fb42c12b3a355473135922`
- Verification: top-level source, run-package embedded source, and local `op_kernel/index_add.cpp` match (`03ae775393ecd5d595f686b8a9162d8c624a7afbf969334413730dcae139273a`).
- Submit: 2026-06-30 17:34:23, workId `0000019F17E1372FB017FCF7AA1E099F`.
- Official result:

```text
Case1: Pass, Result: 39.712
Case2: Pass, Result: 12794.46
Case3: Pass, Result: 60793.256
Case4: Pass, Result: 1812409.78
Case5: Pass, Result: 15630.456
prof_sum: 1901667.664
```

Conclusion: full pass. Case4 improves v113 by 200.292us, so the fp32 mask-reduce path likely hits at least part of the Case4 workload, but total time is 105.396us worse than v113 because the other cases regress/noise upward. The onehot mask/tree-reduce idea is correct enough to test scheduling, but it is not yet an order-of-magnitude Case4 fix.

## v116 case4-fp32-maskreduce-b40

- Change: keep the v115 fp32 mask-reduce kernel unchanged, but raise the exact-shape fp32 mask-reduce host `blockDim` cap from 32 to 40.
- Rationale: the v115 kernel parallelizes by outer row. If hidden Case4 has `outer == 120`, 40 blocks give an even three outers per block and match the scheduling that worked best for the int8 Case4 family.
- Package: `IndexAdd_v116_case4_fp32_maskreduce_b40.zip`
- SHA256: `e6381fc8f1938e3d299abb927e660ee5c1b228fcaf5c2f59bfcf1c0ed0c64c16`
- Verification: top-level host source matches (`52775ae1a3152feabebcb8537931c7b34961dbf755207ece8b2e3d6a99fc286a`); top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match (`03ae775393ecd5d595f686b8a9162d8c624a7afbf969334413730dcae139273a`).
- Submit: 2026-06-30 17:46:19, workId `0000019F17EC232DB014D7A794B39119`.
- Official result:

```text
Case1: Pass, Result: 37.8
Case2: Pass, Result: 12824.328
Case3: Pass, Result: 60790.108
Case4: Pass, Result: 1812386.66
Case5: Pass, Result: 15629.62
prof_sum: 1901668.516
```

Conclusion: full pass, but effectively tied with v115 and still in the 1.901s band. Increasing the fp32 mask-reduce block cap does not move the dominant Case4 bottleneck; stop tuning this branch until the int8 Case4 path is back on the v95 baseline.

## v117 restore-v95-pack4

- Change: restore the int8 Case4 dispatch priority to the v95-style Pack4 path and restore `ProcessCase4Pack4Int8()` to its int32 accumulator implementation with `DataCopyPad` row writeback.
- Rationale: v109-v116 replaced or bypassed the v95 Pack4 core with packed/cached/position-list experiments, all of which landed in the slower 1.901s band. This candidate reestablishes the known 1.818s baseline before attempting another algorithmic change.
- Package: `IndexAdd_v117_restore_v95_pack4.zip`
- SHA256: `f93ef986c48b19042f1fb9950bbf620e91995202cd17c94cd24b0b01db8afee1`
- Verification: top-level host source matches (`52775ae1a3152feabebcb8537931c7b34961dbf755207ece8b2e3d6a99fc286a`); top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match (`e742a81c4f8b609ee9750b0dcfc1f45463969f02132db4d52340ade0586f6b96`).
- Submit: 2026-06-30 18:32:17, workId `0000019F18163AE8B0115CF0C2ACE2C8`.
- Official result:

```text
Case1: Pass, Result: 34.008
Case2: Pass, Result: 12602.78
Case3: Pass, Result: 60737.66
Case4: Pass, Result: 1812384.024
Case5: Pass, Result: 15627.86
prof_sum: 1901386.332
```

Conclusion: full pass and slightly better than v115/v116, but it did not recover the older v95 1.729s Case4 band. A direct v95 package/source comparison is needed before assuming the historical result is still reproducible.

## v118 case4-i8-float-maskreduce

- Change: add an exact-shape int8 Case4 mask-reduce path before the restored v95 Pack4 fallback. It precomputes the shared 512x7 onehot masks for both index tiles once per core, casts each int8 source tile through half to float, applies `Select(mask, src, 0)`, and reduces 512 rows to 8 rows with vector `Add` before a small scalar tail and int8 row writeback.
- Rationale: the profile showed the hot Case4 kernel is scalar-pipe bound. This candidate applies the fp32 mask-reduce algorithm to int8 while preserving exact integer accumulation in float and final low-byte int8 semantics.
- Package: `IndexAdd_v118_case4_i8_float_maskreduce.zip`
- SHA256: `bc23bf6f46abc45d618702eb3c83efbe44d33b7ffdc1538ac1585c4580d12f68`
- Verification: top-level host source matches (`52775ae1a3152feabebcb8537931c7b34961dbf755207ece8b2e3d6a99fc286a`); top-level kernel source, run-package embedded kernel source, and local `op_kernel/index_add.cpp` match (`428cae49bb460988c06a827b1cdc289edd76f8b5f72bc295f1104b66c16a9727`).
- Submit: 2026-06-30 18:42:08, workId `0000019F181F3EDCB015A7BF7B3A7280`.
- Status: submitted; waiting for official result.
