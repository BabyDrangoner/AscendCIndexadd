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
- Status: built; submit after v102 official result is captured.

## v104 class3-tile512

- Change: reduce the Class3 position/source tile from 1024 to 512 entries.
- Rationale: parameter sweep for the Class3 path. Smaller tiles increase loop count but reduce active UB position/source footprint and may improve local access behavior.
- Package: `IndexAdd_v104_class3_tile512.zip`
- SHA256: `2ea2b733bbab0c58da2082d6c1ed9b6acff1c6cd89b01a2053d0caeb26ffa24f`
- Status: built; submit after v103 if Class3 shows promise.
