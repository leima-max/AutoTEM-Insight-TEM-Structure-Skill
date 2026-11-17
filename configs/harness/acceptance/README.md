# RG0 发行合同复核

`traditional-core-contract-v1.json` 冻结 Windows x64 的 CPython 3.12.10、pip 25.0.1、九个 Harness 工具的参数、操作版本与角色、单位、状态和错误。角色 schema 的 SHA256 取自公开 MCP 工具返回值。完整 schema 保存在采集输出的 `mcp.capabilities.result.operations` 中。NPY、TIF 和 TIFF 的发行范围、其他 reader 源码分支以及 EMI 的元数据用途分别登记。`harness-evidence-v1.schema.json` 使用 JSON Schema Draft 2020-12，适用于协议回复、登记产物、补参记录、CLI 科学用例、宿主摘要、宿主索引与 RG0 证书。

## 采集

使用已激活的目标发行包私有 Python，从 checkout 外的目录运行。下列 PowerShell 变量分别指向验证器文件、发行目录、实际冷装证据目录和新建的本次取证目录。`$Evidence` 下须有同一次安装的 `installation.json`、`science/evidence.json`、`science/operations.json`、`science/fixtures/` 和 `science/harness/`。证据目录可自行命名。

```powershell
& $PrivatePython -I $Verifier collect --bundle $Release --evidence $Evidence --work "$Output/contract-samples" --out "$Output/contract-freeze.json"
```

采集器通过私有解释器启动 MCP 和 JSONL 进程，记录真实响应，并用多帧 NPY 与多帧 RGB TIFF 检验图像选择。结果目录包含 `contract-probes.json`、`contract-samples/` 的 SQLite 与原始/选中数组以及 `contract-freeze.json`。若已有完整的探针输出，可将 `collect` 改成 `summarize` 只生成尚未写入的证书。

## 离线复核

```powershell
& $PrivatePython -I $Verifier recheck --bundle $Release --evidence $Evidence --certificate "$Output/contract-freeze.json"
```

复核重新解析 wheel 元数据和目标依赖闭包、协议与操作合同、运行登记、数组、文件哈希、声称结果所引的 artifact，以及父子运行的标定响应。返回码 0 表示重算的 RG0 证书逐字段一致。上层索引可调用 `tools.release.contract_freeze.recheck(bundle, evidence, certificate, work=None, host_index=None)`，参数为 `pathlib.Path`，成功时返回重算后的字典。证书固定验证器、采集器、合同、证据 schema、原始探针和发行指纹。

默认依据发行包内的宿主矩阵生成三宿主 J01–J15 的未观察基准，此时 `host_matrix_source` 为 `empty_schema_baseline`。需要附带既有宿主索引时，在采集和复核两次调用中都传入 `--host-index $Index`。把该索引引用的脱敏单案摘要放在索引旁，工具按 SHA256 寻找并重建版本、变体、计数和汇总。私有宿主会话由 RG3 独立重放。本证书的 `scientific_accuracy` 为 `not_evaluated`，`release_ready` 为 `false`。
