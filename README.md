# AscendCIndexadd

Huawei AscendC S9 contest workspace for the `IndexAdd` operator.

The current baseline is the stable bd40 version:

- `op_host/index_add.cpp`: `use_case4_i8_outer1 ? 40 : 32`
- Official result: `prof_sum=1912481.06us`
- Topic: `IndexAdd`
- Contest id: `41ffbad2024e4ccfa43520c57ffa7b9e`

Common workflow:

```bash
./scripts/build_docker.sh releases IndexAdd_candidate.zip
./scripts/submit_hiascend.sh --zip releases/IndexAdd_candidate.zip --cookie-file .cache/hiascend_headers.txt --topic IndexAdd
./scripts/monitor_hiascend_results.sh --topic IndexAdd --team '请输入团队' --cookie-file .cache/hiascend_headers.txt --out-dir .cache/hiascend_results --interval 5
```
