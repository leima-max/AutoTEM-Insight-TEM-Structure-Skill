# 准备三宿主的中文现场操作卡

一个材料目录覆盖 J01–J15。每张卡列出所需输入、适用发行档、用户可以发送的请求、后续回复的触发条件和应核对的证据。核心版与仿真版使用各自的清单。输入来自解析样本，J07 的双数组 NPZ 用于检验读入器分支，RC17 核心磁盘格式声明包括 NPY、TIF 和 TIFF。

## 从发行包准备

从仓库根目录运行下列 PowerShell 命令。填写已安装的项目目录、解压后的核心发行目录，以及同版资格索引。生成器会将仓库矩阵与该发行包的冻结版本逐字节核对。

```powershell
$trial = (Resolve-Path -LiteralPath (Read-Host '已安装 AutoTEM 的项目目录')).Path
$bundle = (Resolve-Path -LiteralPath (Read-Host '核心发行目录')).Path
$qualification = (Resolve-Path -LiteralPath (Read-Host '同版资格索引 JSON 文件')).Path
$matrix = (Resolve-Path -LiteralPath 'tools/host_acceptance/matrix.json').Path
$active = Get-Content -Raw -LiteralPath (Join-Path $trial '.autotem/active-runtime.json') | ConvertFrom-Json
$python = Join-Path $trial ".autotem/runtimes/$($active.runtime_id)/Scripts/python.exe"
$kit = Join-Path $trial '现场验收'
& $python -I -B tools/host_acceptance/prepare_fieldkit.py prepare --release $bundle --release-index $qualification --matrix $matrix --out $kit
& $python -I -B tools/host_acceptance/prepare_fieldkit.py verify $kit --release $bundle --release-index $qualification --matrix $matrix
```

打开 `现场验收/README.md` 选卡。目录可以整体移到另一项目根目录，保留目录名即可使用卡中的相对路径。`index.json` 记录各个文件的 SHA-256、数组形状和输入来源。外部发行包、资格索引和冻结矩阵可用于独立核对材料身份。已经存在的同版完整材料会被复用，有差异的材料应使用新的目录。

## 现场执行

1. 在所测宿主中打开已安装对应发行档的项目，由操作者审核项目、Skill 和连接器权限。保存宿主版本与真实的工具发现结果。初次用核心版时从 J01 开始。
2. 每张卡新开会话。将给定文件交给宿主，只发送“首轮”的文字。在原会话满足卡上所列条件时，再发送相应回复。需要受控队列、故障项目或原生委派的卡，先核对现场条件。
3. 保存原始会话、实际工具参数和返回、报告与运行编号。每个宿主、发行、场景变体各自建立一条记录。J14 的原生委派验证需要实际派发与返回，J15 的四个仿真版变体需要连接到对应仿真项目。采集与验证命令见[三宿主真实旅程取证](../README.md)。

本地复核 J07 时，可从材料索引读取 `datasets`、`stack`、`rgb` 三个输入路径，用已安装运行时执行 `field_selection_probe.py --dataset <npz> --frame <npy> --channel <tif> --out <新目录>`。该结果记录选中图面的数值、父子运行和科学步骤的输入来源。现场资格矩阵单独读取真实宿主记录。
