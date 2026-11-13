# Windows 发行门禁与补证

维护者在已安装该版本的项目中打开 PowerShell，使用项目的私有 Python
运行验证器。发行目录、既有验收目录和输出文件都使用绝对路径。下列命令中
`$tools` 是维护者工具目录，`$evidence` 是同一发行的冷安装、科学与变体验收目录。

```powershell
$active = Get-Content -Raw .autotem/active-runtime.json | ConvertFrom-Json
$py = (Resolve-Path ".autotem/runtimes/$($active.runtime_id)/Scripts/python.exe").Path
& $py -I "$tools/release/qualify_release.py" --bundle $bundle --evidence $evidence --out $index --require RG1
```

验证器复核发行清单中每个文件、解释器指纹、源归档、wheel 和锁，以及登记的安装与科学证据。
每份补证都要与发行清单 SHA256 一致。使用新的输出文件可保留旧判定。指定门禁满足时退出码为 0。
索引已写出但指定门禁未满足时返回 1。证据、运行环境或输出无效时返回 2。
冷安装的磁盘预检若拒绝安装，保留原生回执中的可用字节数和最低预留量，验证器会从发行内容复算阈值并核对空项目。若预检通过后安装失败，则继续从实际安装事务、运行时标识和原生错误码复核。
可多次使用 `--require-check RGn.检查名` 约束单项。例如，`--require-check RG4.interrupted_install_recovery` 要求真实中断恢复通过。RG4 的总体判定仍取决于该门禁内所有检查。
`--require-ready` 检查六个门禁。Windows CI 的 `qualify_ci.py` 将随发行冻结的验证器和样例解到新目录，再用项目的私有解释器采集合同、数值与进程证据。

| 门禁 | 输入及复核 |
|---|---|
| RG0 合同 | `--contract contract-freeze.json` 复核九工具、注册操作、格式/选择、精确锁、错误和补参合同。证书旁保留探针记录与样例。 |
| RG1 安装 | `--evidence` 中的正式启动器冷安装、私有环境、项目入口与磁盘旅程。 |
| RG2 本地执行 | `--local local-execution.json` 复核直接算法对照、物理产物、故障及兼容记录。另用 `--headless headless.json` 检查每个科学 worker 和子解释器对列出 UI/DL 模块的探测与导入。已声明实现的可选操作须在相容发行环境有实算证据。 |
| RG3 宿主 | 每个 `--host-capture case.json 项目目录` 从原始会话重新核对真实工具、产物、宿主回复和运行时。各宿主独立计算。经验证的原生委派自动纳入，可用 `--native 宿主` 声明该宿主必须覆盖原生委派。 |
| RG4 交付 | `--upgrade upgrade.json 旧包目录 演练项目` 从本地项目重读两代运行、配置与科学文件。`--recovery recovery.json 恢复项目` 复读中断事务、原生重试和恢复前后的分析。宿主端还须有首次授权、九工具发现及 J06–J13 的证据。 |
| RG5 发布 | `--source-tests source-regression.json` 对照同名 `.junit.xml` 逐项复算回归。汇总前面门禁，并从评审原件核对 A 的集成结论、B/C 的科学边界及发行来源与再分发决定。 |

完整源码回归使用 `tools/release/run_source_tests.py`。它从发行源归档准备独立检出，
在测试前后核对源码文件，并启用两项真实 diffsims 测试。新版回执的
`source-regression.collection.json` 保存 pytest 实际收集、选中和排除的用例 ID 及筛选表达式。
门禁将选中列表与 JUnit 中每条测试的 ID 对照；JUnit 保留跳过与失败原因。
历史 1.0 版回执继续核对 JUnit，其独立收集清单标为 `not_evaluated`。

### Windows 中断、升级与完整回归

升级演练使用已冻结的上一代核心版和本次核心版。CI 从仓库的稀疏 RC8 基线、原 PSF 包和有哈希的上游 wheel 恢复上一代发行，逐文件对照旧 manifest，并检查至少一个实际应用模块发生变化。`verify_upgrade.py` 用正常启动器完成旧版 → 新版 → 旧版，重读三次分析的产物、配置冲突与已打开的旧会话。核验使用独立项目。[历史基线](../.github/release-baselines/README.md)

`RG4.post_switch_host_failure_recovery` 单独检查升级后的新连接。验证器在合成项目中暂时将新运行时标记为失效，按已安装适配器的实际 MCP 命令发起握手，保存进程退出码和诊断，再恢复测试标记。原生安装器接收绑定本次升级的故障报告，依次演练配置冲突、解决后回退和同一报告重复提交。恢复后的入口须完成 MCP 九工具发现、一次能力调用和一个独立科学作业。原生宿主里的用户授权与对话在各自宿主单独验收。早期升级回执在这项检查中显示 `not_evaluated`。

中断恢复使用另一个空项目。运行 `tools/setup/verify_recovery.py --bundle 核心版目录 --project 恢复项目 --output 新证据目录`。驱动先安装并完成一次分析，回滚及受保护卸载后，在这个专用项目持有一个真实 SQLite 写事务，使下次安装停在运行时就绪、项目入口尚未启用的时点。释放写事务后由同一原生启动器重试。加 `--verify` 可复读两份分析、事务日志、安装收据和落盘文件。旧包的开发演练另留回执，正式门禁要求驱动与候选源归档一致。

同一驱动也在已有分析的专用项目中让 SQLite 拒绝清理一次闲置调度器租约，然后调用原生回滚。故障当时的活动指针、宿主配置和历史产物均须与回滚前一致。撤去测试数据库中的临时触发器后再次回滚，核验正常恢复路径。

维护者可用以下命令在新的独立环境测试冻结源码。`--support-repo` 指向保有构建时 Git 提交的本地检出。`--test-wheelhouse` 可提供已按补充测试锁下载的本地 wheel。

```powershell
& $py -I "$tools/release/run_source_tests.py" --bundle $core --simulation-bundle $simulation --checkout $checkout --scratch $scratch --out $sourceEvidence --support-repo $repository --test-wheelhouse $testWheels
```

1.2 版源码回执在 JUnit 和收集清单旁封存两份发行 manifest、两份运行锁、补充测试锁及环境清单。复核时对照 PSF 解释器、发行源归档、安装包版本与来源。CI 公开摘要使用 `ci_evidence.py public --delivery-evidence 升级回执 恢复回执 源码回执`，只收录经复核的数量、判定与指纹。旧源码回执可以单独重读，环境项标记为 `not_evaluated`。

### 在独立仿真版本补验两项条件操作

在安装了 `traditional-simulation` 的专用验收项目中，用该项目的私有 Python
运行 `tools/acceptance/qualify_simulation.py --bundle 仿真发行目录 --project 项目 --out 新证据目录`。
验证器提交一项使用显式 Si 原子占位的合成衍射任务，重读实际作业、输入快照、全图和倒易单位，
并将匹配候选与直接 API 的结果逐项核对。`verify_headless.py` 在该版本还会监测真实仿真进程。

核心版汇总时可传 `--simulation simulation.json 仿真发行目录 已安装的仿真项目`，并以
`--simulation-headless headless.json` 接入该版本进程证据。`--simulation-faults simulation-faults/faults.json` 接入缺包、坏导入、后端及外层作业超时、缺原子位点五个实测分支。其故障收据绑定同一个仿真 manifest 和正例 run。RG2 同时检查两种进程边界及这些异常路径。两个发行目录必须有完全相同的
AutoTEM wheel 和源归档。核心版本内缺少后端的两项状态仍单独显示。
该仿真版的真实宿主 J15 通过 `--simulation-host-capture case.json 对应宿主项目` 逐条补录，
每个项目由其已安装且版本匹配的解释器复核。

### 补充真实宿主证据

在宿主中打开已安装的项目，按宿主显示的流程完成信任、连接和首次授权。用磁盘样例
提出自然语言分析请求，保留实际交互、工具调用、项目里的 SQLite 和产物。每个宿主版本
单独取证。没有发生的会话保留 `not_evaluated`，有可归属的阻断原件时可记录为
`blocked`。完整变体和留档字段见[真实宿主取证说明](../tools/host_acceptance/README.md)。

### 已有评审原件的登记

通过 `--decisions` 传入私有 JSON。顶层使用 `schema_version` 值 `1.0.0`、
`release_manifest_sha256` 和 `decisions` 映射。仅登记已有的书面决定。每项填写
`reviewer`、含时区的 `recorded_at`、`verdict`、同目录下的原件相对路径 `record` 和
`record_sha256`。原件须写明实际发行及审阅范围。团队职责映射如下。

| 决定键 | 审阅人和内容 |
|---|---|
| `integration` | A / leima-max，确认已审阅的整套结果，并填入本次索引的 `qualification_basis_sha256`。 |
| `diffraction` | B / wangzhe221，确认输入、标定、倒易空间和仿真在该发行条件下的科学边界。 |
| `atomic_strain` | C / yswgityes，确认原子坐标、强度、应变与参考区的科学边界。 |
| `redistribution`、`publisher_origin` | A 登记适用许可与再分发依据，以及来源、签名或其他实际身份核验方式和结果。 |

验证器从原件计算哈希。评审人的身份由团队取得原件的渠道确认。原始宿主消息、路径、真实
实验文件、SQLite、凭证和签认原件由项目保管。对外审阅使用脱敏索引，另记录真实样品精度
和性能的评测范围。合成数据对照只用于对应工程和数值合同。
