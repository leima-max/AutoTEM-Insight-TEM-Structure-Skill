# Windows 传统算法发行候选

Windows x64 发行目录使用 CPython 3.12.10、pip 25.0.1，包含一个明确的 profile、
私有 Python、启动器、离线 wheel、精确版本与 SHA256 锁、源码归档和许可清单。

| 发行 profile | 使用入口及范围 |
|---|---|
| `traditional-core-mcp`（默认构建） | 传统分析、CLI、JSONL 和 MCP。[基础包配置](../configs/release/traditional-core-mcp.json) |
| `traditional-cli` | 同一套传统服务，提供项目 CLI 和 JSONL。MCP 不在该包的依赖中。[CLI 包配置](../configs/release/traditional-cli.json) |
| `traditional-simulation` | 私有验证版，在传统分析和 MCP 基础上加入 diffsims 的衍射仿真及候选匹配。该版的真实运算、故障分支与再分发许可分别取证。[仿真包配置](../configs/release/traditional-simulation.json) |

RC8 的既有安装与验收记录属于 `traditional-core-mcp`。不同 profile 的归档、运行锁、
manifest 和项目 runtime ID 分别绑定，跨版升级使用对应的完整发行目录。

## 首次安装

1. 将完整发行目录放在可读取的位置，双击其中的 `AutoTEM-Setup.exe`。
2. 选择用于保存分析的项目文件夹和宿主，再点“安装或修复”。确认窗口列出当前活动版本、项目盘可用空间、最低预留空间、预计新增载荷和可能写入的项目路径。新增载荷按私有 Python 和 wheel 中的文件大小估计，最低预留还考虑安装临时空间。确认后，安装器在项目下创建 `.autotem`，验证科学依赖和本地传输，再生成所选宿主资源。
3. 出现 `host_setup_required` 后，按所选宿主完成首次连接。下表路径均从刚选的项目文件夹算起。

| 宿主 | 首次连接 |
|---|---|
| Codex | 在 Codex 中打开并信任这个项目，重新加载项目配置 `.codex/config.toml`，按宿主提示批准 AutoTEM 工具。 |
| WorkBuddy | 在 WorkBuddy 中导入 `autotem-host/workbuddy/autotem-skill.zip`。打开 `autotem-host/workbuddy/adapter.json`，用 `mcp.mcpServers.autotem` 中的 `command` 和 `args` 配置项目连接器，按宿主提示授权。 |
| OpenClaw | 在网关健康的 OpenClaw 中使用这个项目作为工作区，让宿主加载 `skills/autotem/SKILL.md`。通过该宿主支持的项目连接方式使用 `autotem-host/openclaw/adapter.json` 中的命令与参数，按宿主提示授权。 |

首次连接后，请宿主调用 `list_capabilities`，根据实际返回的操作范围提交分析。

CLI 版安装成功后显示 `local_cli_ready`。项目下 `autotem-host/<宿主>/adapter.json`
的 `cli_command` 是命令和参数数组，可在其后加 `capabilities`、`inspect`、`submit`、
`status` 等。`jsonl_command` 可直接启动标准输入输出服务。以下 PowerShell 示例从项目目录运行。

```powershell
$adapter = Get-Content -Raw 'autotem-host/codex/adapter.json' | ConvertFrom-Json
$cli = @($adapter.cli_command)
$cliArgs = @($cli | Select-Object -Skip 1)
& $cli[0] @cliArgs capabilities
$bridge = @($adapter.jsonl_command)
$bridgeArgs = @($bridge | Select-Object -Skip 1)
'{"request_id":"check","tool":"list_capabilities","arguments":{}}' | & $bridge[0] @bridgeArgs
```

从宿主程序调用 JSONL 时，每条请求用 UTF-8 编码并以换行结束。中文路径可直接写入 JSON 字符串，
也可使用标准 JSON 的 `\u` 转义。手工检查单个文件时，可将完整路径作为一个参数传给
`& $cli[0] @cliArgs inspect 'D:\电镜样本\项目 一\原图.tif'`。将示例路径换成实际文件即可。

命令行安装可使用 `AutoTEM-Setup.exe --project <项目路径> --apply`。启动器读取本目录的
唯一 profile。自动化可附加 `--profile traditional-cli` 等明确要求，目录版本不符时会在
项目写入前返回失败。切换后重新启动 CLI 或宿主连接，即可使用新的运行时。

项目级 MCP 配置的信任和授权由宿主管理。[Codex 官方 MCP 说明](https://learn.chatgpt.com/docs/extend/mcp)

## 补充科学参数

当任务返回 `needs_input`，宿主读取 `input_requests`，向用户展示缺失字段、允许单位和
已登记的证据。用户提供数值及依据后，宿主校验完整新计划，通过 `parent_run_id` 创建
child，再读取 child 报告。重复发送同一回复使用同一幂等键。对原图 ROI 的选择使用
`[y0,y1,x0,x1]`，上界不包含在内。[工具合同和标定答复示例](../skills/autotem/references/tool-contracts.md)

对 SAED/NBED 直衍射图，缺少倒易标定时可读取峰相对中心的探测器像素偏移、半径和角度。
此时 d 间距列表为空。要得到 nm 单位的 d 间距，需补充每个探测器像素对应的倒长度
（`1/nm/pixel`）及其来源。对实空间 HRTEM 图像的 FFT，未给出长度标定时仍可报告以原图像素计的周期。

## 运行环境与恢复

项目中的 `.autotem/active-runtime.json` 标识当前已激活版本。
`.autotem/installations/` 保存事务，`.autotem/host-installations/` 保存宿主收据和备份。
科学数据、SQLite 和报告存放在 `.autotem/runs/`。同一版本再次安装会复核包和依赖。
出现 `WAIT_ACTIVE_RUNS` 时，先等作业结束，或在宿主中取消它并确认终态。
出现 `PROJECT_DATABASE_BUSY` 时，保留项目文件，等占用项目数据库的写入结束，再对同一项目重试原操作。
收到 `HOST_RECONNECT_REQUIRED` 时，重新连接项目工具服务。

如果升级后宿主在连接、发现工具或调用工具时确实报错，可用本次升级发行目录中的安装器提交故障记录。将下面的占位内容替换成现场值，保存为 UTF-8 JSON。复制本次升级安装回执的 `installation_id` 和 `release_id`，填写发生故障的 `host`（`codex`、`openclaw` 或 `workbuddy`）。按实际阶段将 `stage` 设为 `connect`、`discover_tools` 或 `call_tool`，在 `error_code`、`message` 中记录可核对的故障。

```json
{
  "schema_version": 1,
  "installation_id": "<本次升级安装回执的 installation_id>",
  "release_id": "<本次升级安装回执的 release_id>",
  "host": "openclaw",
  "stage": "connect",
  "error_code": "<实际故障代码>",
  "message": "<现场诊断>"
}
```

从该发行目录运行 `.\AutoTEM-Setup.exe --project "D:\电镜项目" --host-reconnect-failure "D:\现场记录\宿主重连.json"`。该独立命令会核对报告对应的项目安装事务，尝试恢复上一个有效入口。查看返回的 JSON 了解原始宿主故障、恢复状态及下一步，再从宿主重新连接。旧版安装器不识别该选项时，使用支持该入口且与本次升级对应的发行包。

例如，连接被拒时可填写现场出现的 `ECONNREFUSED`，调用途中连接断开时可填写 `CONNECTOR_DISCONNECTED`。`message` 只需保留可复核的诊断，省略令牌等凭据。首次信任或授权弹窗应由当前用户处理。已收到 `NEEDS_INPUT` 或 `DEFERRED_THIS_PHASE` 的分析请求，应按工具返回的下一步处理。若回退报告 `HOST_FILE_CONFLICT`，处理列出的文件后，再提交同一份故障记录。看到 `rolled_back` 后从宿主重新发现工具并实际调用，以确认旧入口恢复。

“回滚上次安装”会按收据恢复指针和宿主文件。若文件被用户修改或收据缺失，
冲突提示会列出项目相对路径、当前文件和受管版本的哈希。可以保留现有配置，在另一个空项目安装。要继续原项目的操作，请先单独保存个人修改，再按提示使用经过哈希核验的项目内对照文件并重试。旧收据只有安装前备份时，提示会给出所需版本的哈希；请使用自己留存且哈希匹配的副本。回滚保留版本目录和科学数据。旧式固定路径
`.autotem/runtime` 的项目可以继续使用原开发者入口，迁移时需要排空旧会话并核对配置。

需要移除环境时，先结束该项目中的分析并断开宿主连接，再选择“卸载项目环境”。
安装器逐项检查收据、文件哈希和运行中进程，并在原项目保留原始快照、报告和运行历史。
返回 `uninstall_pending` 时，根据结果中的路径或占用进程处理后，可再次运行卸载。

### 配额、保留和磁盘清理

默认每个原始文件的快照上限为 2 GiB，可通过计划的
`budgets.max_snapshot_bytes` 指定。超限时会返回 `SNAPSHOT_QUOTA`。选择
`input_mode: reference` 后，原始文件需要留在原位置供后续校验。项目当前没有所有
运行合计的输出配额或按日期自动清理的功能。可通过项目文件夹属性查看占用空间。

需要归档时，先等工作进程及作业结算、关闭项目连接，再整体备份
`.autotem/runs/`，包括 SQLite 及其可能存在的 WAL 文件、快照和所有运行子目录。
同时保留引用模式的外部原件、发行指纹和所需安装包。重新打开后按原路径核对
报告与产物哈希。只需腾出解释器空间时，可运行安装器的“卸载项目环境”。
选择彻底清理专用项目时，先在备份上验证所需历史记录和报告，再移除该项目副本。
SQLite 的登记记录与对应科学文件需要一起保留。

## 构建与验收

维护者从精确源码归档生成自有 wheel，通过构建输入锁核对上游文件，在离线环境
解析运行依赖。普通用户安装时使用自带解释器和 `--require-hashes --no-index`。
发布者签名与再分发许可有独立状态。[构建说明](../tools/release/README.md)

离线复跑 Windows CI 构建时，向 `tools/release/build_windows_ci.ps1` 传 `-OfflineInputs`。该目录根部存放策略指定的 `python.3.12.10.nupkg`，`pools/<构建输入锁的 SHA-256>/` 存放对应锁列出的原始 wheel。锁的摘要取自当前 Git 检出中的文件，目录名使用小写十六进制。标准 Windows CI 从 NuGet 和 PyPI 取得相同的锁定输入。两种方式均重新构建启动器、wheel 和三个发行档，逐一校验解释器及依赖哈希。重建旧版用于升级验证时可再次传入同一 `-OfflineInputs`。

Windows CI 在源码外的空项目调用相同启动器，再由安装后的解释器执行磁盘用例。
验收驱动保存运行 ID、科学产物、哈希、运行时指纹和逐操作矩阵。另存的
`first-slice.json` 将安装、运行锁、发行 manifest 和科学证据绑定到同一个构建版本。
安装期间还经项目的稳定入口读取磁盘样例，并在回滚后重新校验该运行的快照、产物和作业绑定。
技术计划要求首批通过真实 Codex 对话。操作被宿主审批策略拦截时，宿主记录保留
`blocked`，需在允许审批的会话里由用户授权该项目的 AutoTEM 工具。
合成样例用于检验数值接线，真实样本精度另行取证。
[Windows CI](../.github/workflows/windows-release.yml) · [磁盘验收](../tools/acceptance/README.md) · [宿主取证](../tools/host_acceptance/README.md)

完整发行判定可用私有解释器重新生成。[RG0–RG5 判定与补证](harness-release-qualification.md)
