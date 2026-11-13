# 本地发行验收

从源码目录之外的工作目录运行安装后的私有解释器。选一个新目录保存本次记录。

```powershell
& $InstalledPython -I -m autotem.acceptance --require-installed --release-bundle $ReleaseRoot --output $EvidenceDirectory
```

默认通过公开 JSONL CLI 启动独立科学 worker，从磁盘读取三份解析样例。

| 场景 | 输入和请求 | 核验 |
|---|---|---|
| J01 | 非方形 HRTEM NPY 的四个主峰和受限相候选 | 重载完整 FFT，逐轴验证倒易尺度、d 值与候选身份 |
| J02 | HAADF 浮点 TIFF 的坐标、间距、相对强度 | processed 检测与 raw 精修采用直接 API 对照，逐点状态、背景及有效掩膜可追溯 |
| J06 | 未记录物理标定的 HRTEM TIFF | 父 run 请求输入，非法答复记录拒绝，有来源的 20 pm/pixel 答复创建 child 并重测 d，逐文件核实父证据及快照哈希 |

加上 `--extended` 会额外执行 J03 的 GPA 场与参考区、J04 的逐原子应变与插值显示、J05 的 SAED 和 NBED 直接衍射图。每项均使用包内的确定性公式生成合成样例。输入来源、轴、形状、类型、尺度、期望与哈希保存在本次证据目录。

`evidence.json` 包含逐场景判定、运行与作业 ID、工具请求、文件哈希、完整数组重载结果、测量与 Artifact 的对应。相邻的 `operations.json` 来自实际注册表，每一行分别记录参数负例、磁盘执行、直接 API 对照及其验证状态。发行绑定模式还验证发行清单、安装标记和 wheel 内文件与私有环境的一致性，记录源归档、依赖锁、包及 Skill/Profile 的哈希。

无效输入、排队或运行时取消、重复及冲突的幂等请求也通过同一 CLI 执行。输出目录保留运行日志和 SQLite，便于查看失败步骤。新一轮验证应使用新目录；已有证据会被保留。CI 可归档两个 JSON 摘要，本机的 Harness 目录包含绝对路径及原始数据快照。

此入口记录本地工程验证。三宿主 LLM 旅程和真实样本精度分别保留各自的验收记录。缺少发行包时可在源码目录运行 `python -m autotem.acceptance --output <新目录>` 进行开发验证，报告会记录实际解释器和代码来源。

## 其他传统操作

在发行源归档中取出同目录的 `verify_variants.py` 和 `operation_variants.json`，从外部工作目录执行：

```powershell
& $InstalledPython -I $VerifierScript --require-installed --release-bundle $ReleaseRoot --output $NewVariantDirectory
```

脚本检查 16 项独立变体，逐项将结果绑定到发行包哈希、依赖锁、操作版本和完整数据文件。汇总为 `operations.json`，每个场景另有 `variant-evidence.json`。脚本也记录自身及样例清单与发行源归档的哈希对应。原子样例刻意令 42 个柱坐标通过 42 的验收条件、未达到 43 的对照条件，因此应导出 diagnostic 报告。

真实 diffsims 属于可选环境。核心环境会记录实际缺少的依赖及下游匹配状态。具备依赖的环境可加 `--simulation` 运行显式冷启动预算的模拟与匹配。
