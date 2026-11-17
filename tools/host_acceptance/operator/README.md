# Windows 宿主现场取证

在普通用户账户中解压核心 MCP 或内部仿真安装包，保留解压目录。先在本地磁盘创建项目文件夹，再从 PowerShell 指定这个已存在的目录和实际宿主，预览后应用。宿主名可以是 `codex`、`workbuddy` 或 `openclaw`。

```powershell
$project = Read-Host '已选项目的绝对路径'
$release = Read-Host '已解压的安装包目录'
$agentName = Read-Host '实际使用的宿主名'
& (Join-Path $release 'AutoTEM-Setup.exe') --project $project --host $agentName
& (Join-Path $release 'AutoTEM-Setup.exe') --project $project --host $agentName --apply
```

将本现场 ZIP 解压到**受测项目外的操作员目录**。其中的 `现场验收` 和 `现场取证` 应当相邻。打开操作员目录中的[卡片目录](../现场验收/README.md)，选择与已安装档位相符的任务。操作员保管卡上的核对值、后续回复和原始会话记录。宿主打开已安装的受测项目，每轮只收到用户在该时点给出的请求和输入。三宿主现场卡使用核心 MCP 或仿真安装。

## 创建一次真实会话的记录

J01 适用于核心档。由操作员在宿主会话外打开 PowerShell，为本轮选择一个位于受测项目外的新私有记录目录。脚本从项目的活动安装选择私有 Python，`init` 将本卡首轮所需的输入原样放入项目的 `现场验收/inputs/`。已存在且字节相同的文件可复用，文件冲突会返回具体路径。

```powershell
$operator = (Resolve-Path -LiteralPath (Read-Host '已解压现场 ZIP 的操作员目录')).Path
$entry = Join-Path $operator '现场取证/Collect.ps1'
$record = Read-Host '这次取证目录的绝对路径'
& $entry init -Project $project -Release $release -Record $record -HostName $agentName -Case J01
```

分支卡还需要传入卡名中的变体，例如 `-Case J07 -Variant frame`。J08 的 GPA 补充卡接受 `gpa_answered`、`gpa_unanswered`、`gpa_invalid_reply`。仿真正例使用已安装仿真档的项目和 `-Case J15 -Variant executed`。标记 `controlled` 的卡需要卡上列明的故障或环境条件，`native_only` 需要宿主真实的子代理委派能力。每个宿主和变体分别建立一份记录。J11 数值故障卡若进入可选的均匀图对照，再从操作员的 `现场验收/inputs` 将 `flat.npy` 按同名相对路径复制到受测项目。已在项目中存有核对卡的历史会话应保留原件，新测使用干净的项目。

从已选项目打开宿主，按实际提示完成首次授权。WorkBuddy 使用桌面导入项目提供的 Skill ZIP 并通过连接器导入建议配置。OpenClaw 使用指向本项目的 workspace 和现场已启用的工具路由。保存宿主版本、授权提示、工具枚举和本次会话的 Skill 加载记录。按照卡片发送自然语言问题，各轮保存用户原话、实际工具名、参数与完整返回、运行和作业编号、最终答复及引用的产物编号。若连接或授权失败，记下原始错误和时间。补参时保留每次提问与答复。填写初始化的 `case.json`，具体字段与各宿主导出格式见[原件填写说明](取证格式.md)。

## 核验已保存的原件

```powershell
& $entry verify -Project $project -Release $release -Record $record
$LASTEXITCODE
```

核验从记录读取宿主、场景和变体，回到原安装项目检查实际作业与登记产物。J01–J09 同时绑定卡中的输入文件。J10 的预留操作拒绝及后续故障、多输入卡按冻结验证器的各自规则判断。退出码 0 表示本场景通过，1 表示待验、受阻或未通过，2 表示输入、版本或输出文件错误。详情见记录目录里的 `summary.json`。已有 case 和摘要受保护。补全记录后，复验时加 `-Summary (Join-Path $record 'summary-2.json')`。会话原件和 SQLite 留在原机，具名审阅使用核验后的摘要。
