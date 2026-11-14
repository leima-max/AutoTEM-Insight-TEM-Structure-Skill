# 三宿主真实旅程取证

J08 提供独立的 [GPA 选峰与参考区现场卡](field/J08-GPA.zh-CN.md)。使用支持这些变体的发行包，初始化时可在 `init <目录> --host codex --case J08` 后加 `--variant gpa_answered`、`--variant gpa_unanswered` 或 `--variant gpa_invalid_reply`。现场脚本会使用原安装包的冻结校验器。已登记的原子应变记录保留其原变体；GPA 专项结果按宿主单列，使用 J03 合成 HRTEM 输入。

此工具读取已经发生的宿主调用和 Harness 登记产物。每一份记录对应一个宿主版本、一次会话、一个 J01–J15 场景和一个发行包。原始输入、宿主日志、截图、原话和 SQLite 保留在操作者的本地证据目录。可分发摘要包含哈希、运行标识、角色、形状及逐项判定。科学精度保留为 `not_evaluated`。

## 采集一次旅程

Windows 已安装项目可用 `field_capture.py` 在原机建立记录和核验原件。它从原发行包核对并读取同版 `verify.py` 和矩阵，使用当前活动项目的私有解释器与作业库。填写已安装的项目目录、当初解压安装包的目录和本机私有取证目录。工具参数可以使用含空格或中文的路径。

```powershell
$project = (Resolve-Path -LiteralPath (Read-Host '已安装的项目目录')).Path
$bundle = (Resolve-Path -LiteralPath (Read-Host '解压后的发行目录')).Path
$field = (Resolve-Path -LiteralPath (Read-Host 'field_capture.py 的位置')).Path
$capture = Read-Host '本次私有取证目录的绝对路径'
$active = Get-Content -Raw -LiteralPath (Join-Path $project '.autotem/active-runtime.json') | ConvertFrom-Json
$python = Join-Path $project ".autotem/runtimes/$($active.runtime_id)/Scripts/python.exe"
& $python -I -B -X utf8 $field --project $project --release $bundle init $capture --host codex --case J01
```

初始化产生 `observation=not_evaluated` 的 `case.json`。按下表保存实际用户请求、会话中的 Skill 加载、工具返回和最终回答。J01 的测量结论应引用真实返回的 `artifact_id`。对于 Codex 桌面会话，可用 `observed-host-export` 绑定会话导出或界面原件。CLI 的逐轮 JSONL 使用 `codex-exec-jsonl`。用真实记录补齐模板，保留带时区的开始与结束时间，然后单独运行核验。

每份 Codex CLI 原件记录一个完整回合。工具调用和回答都应位于 `turn.started` 与 `turn.completed` 之间。宿主也可能在回合开始前记录启动诊断。同一工具调用条目 ID 只能完成一次，多次查询同一作业可以有各自的条目 ID。若宿主同时保留了 AutoTEM 的文本 JSON 和结构化返回，两份内容必须一致。截断的尾部需要重新取得完整导出，回合外的工具调用或回答、重复完成或返回矛盾会导致核验失败。

```powershell
$sample = Join-Path $project '现场验收/inputs/J01 合成 HRTEM.npy'
& $python -I -B -X utf8 $field --project $project --release $bundle verify (Join-Path $capture 'case.json') --expected-input $sample --out (Join-Path $capture 'summary.json')
```

`--expected-input` 用于本轮各 run 应读取同一输入的场景。核验通过的摘要会列出实际文件哈希和 `checks.expected_input`，J01 的值还应与现场材料 `index.json` 的 `inputs.j01.sha256` 一致。通用场景及有意变更原始输入的任务可省略此参数。

原宿主以安装项目根目录解析相对路径，取证助手也从该目录核验 `reference` 模式的输入。发行包现场验收应使用独立安装项目。当前冻结验证器要求从源码检出之外运行安装态科学验收。

`summary.json` 可交给复审人。每次复验选择新的输出文件名。需要纳入发布索引时，保留原安装项目和取证目录，在原机向 `qualify_release.py` 加入 `--host-capture <case.json> <项目目录>`。仿真档的 J15 使用 `--simulation-host-capture`。资格工具会重放冻结的通用场景校验，随现场卡移交单一输入的比对摘要。SQLite 的快照和产物路径指向原安装位置。

在授权项目选用本次安装的私有运行时。以下命令在新目录生成 `case.json`。

```powershell
python tools/host_acceptance/verify.py init D:/AutoTEM-evidence/J06-codex --host codex --case J06
```

1. 保存宿主版本输出或从版本界面转录的原始版本行、实际开始及结束的带时区时间。使用有许可的 fixture，记录其 SHA256。用自然语言提出科学问题，保留原话、科学限制和宿主最终回答。
2. Codex CLI 可用 `codex exec --json` 记录 stdout 原始 JSONL。stderr 单独保存。每次用户回复对应同一会话中的新一轮 JSONL 和 UTF-8 用户输入文件。CLI 支持通过实际会话 ID 续接。记录中应可辨认完成的 MCP 调用、AutoTEM envelope 和代理最终回答。[OpenAI Docs 非交互运行](https://learn.chatgpt.com/docs/non-interactive-mode)
3. WorkBuddy 从实际桌面流程导入项目提供的 Skill ZIP，检查连接器、工具列表和首次授权，留存任务会话内的原始调用记录。记录真实桌面版本及 UI 展示的输入、返回和报告引用。[WorkBuddy 技能](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Skills-Market) [WorkBuddy 连接器](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Connector)
4. OpenClaw 在项目工作区验证 Skill 可用性和现有网关健康，使用该版本支持且获准的连接入口。留存真实会话中的工具输入和返回。使用原生委派时另存绑定到父任务的子会话、工具轨迹和返回。[OpenClaw Skills](https://docs.openclaw.ai/tools/skills) [OpenClaw 子代理](https://docs.openclaw.ai/tools/subagents)
5. 补参场景先保留父 run 的 `needs_input`、字段和证据。用户随后提供原话，宿主在同一会话经新 plan 提交 child。保留父、子运行和报告。J06 需要 child 实际产生所需的物理测量。

## 本地记录

所有 capture 文件引用均相对于 `case.json` 的目录。每轮原始提示可保存在 `request.txt`、`reply.txt` 等 UTF-8 文件。阻断时可只记录 `observation=blocked`、简短稳定的 `blocker_code` 与本机实际诊断的 `blocker_file`，例如网关健康日志。完全未执行的场景保留 `not_evaluated`。

`capture_stage` 使用 `host_turn`、`host_preflight` 或 `local_discovery` 标明停止的阶段。已发出用户提问时可把实际提示文件填入 `blocked_prompt_file`。有已安装项目的阻断尝试，用 `activation_file` 引用当时实际项目 `.autotem/active-runtime.json` 的副本，验证命令再传相同的发行目录。仅检查网关健康或本机软件发现的记录保留 `unbound`。

一次已执行的记录需要以下字段。表中的值格式供填写实际观察。

| 字段 | 内容 |
|---|---|
| `observation` | `recorded`、`blocked` 或 `not_evaluated` |
| `host_version/version_file` | 精确版本和包含该版本的取证文件 |
| `started_at/ended_at` | ISO 8601，包含时区 |
| `capture_format/server` | Codex 原始流用 `codex-exec-jsonl`，经操作者核对的宿主导出用 `observed-host-export`。默认服务器名为 `autotem` |
| `turns` | 依序列出 `{"prompt":"request.txt","trace":"turn-1.jsonl"}`，每一项只包含一个宿主回合 |
| `delegation_mode/native_children` | `sequential` 或 `native`。原生模式逐任务填写 `profile/prompt/trace`。父会话须出现该子会话 ID 的派发与返回，子会话须有实际提交。J14 核对三个 Profile |
| `claims` | 可填写宿主最终回答中的原文片段 `quote` 及对应的 `artifact_ids`。若留空，工具从回答逐行提取已返回的真实产物 ID |
| `followup_excerpts` | 若科学答复记录省略了 `raw_reply`，按 request ID 填写 `{"quote":"用户真实回复的片段","answers":{...}}`。answers 与 child 中保存的解析答复逐项相符 |
| `host_observations` | `discovery/authorization/skill_discovery` 可分别指向实际宿主的工具枚举、授权动作和会话 Skill 加载记录。文本导出直接填写相对路径。图形界面原件填写 `{"session_id":"…","original_evidence":"…","observed_text":"实际显示的内容"}` |
| `variant` | J05 用 `saed/nbed`，J07 用 `dataset/frame/channel`，J08 用 `answered/unanswered/invalid_reply`，J09 用 `snapshot/reference`，J11 用 `invalid_file/missing_dependency/broken_dependency/numerical_failure/timeout`，J12 用 `queued_cancel/running_cancel/interrupted`，J14 用 `sequential/native`，J15 用 `executed/missing_dependency/broken_dependency/timeout`。每个变体一次记录 |

WorkBuddy 和 OpenClaw 的手工观察格式用于补充本机可导出的原始证据，记录者应保留每次出现参数和返回的真实截图、日志或宿主导出。单轮 trace JSON 如下，括号内提示需替换成此次真实内容。

```json
{
  "host": "workbuddy",
  "session_id": "<真实任务 ID>",
  "calls": [
    {
      "server": "autotem",
      "tool": "<实际工具>",
      "arguments": {},
      "response": {},
      "original_evidence": "observations/tool-call.png"
    }
  ],
  "answer": "<实际返回的回答>",
  "answer_evidence": "observations/answer.png"
}
```

`response` 填写实际完整 Harness envelope 或 MCP 返回体。原生模式在父 trace 的 `delegation_events` 内逐条转录 `action`（spawn、return）、`session_id` 和 `original_evidence`。合成测试只用于测试校验器。

## 验证与复审

从 checkout 外的当前目录运行，给出本次安装项目的私有 Python 和核对工具的绝对路径。

```powershell
& $InstalledPython -I $Verifier verify $CaseFile --project $InstalledProject --harness-root $RunRoot --release $ReleaseDirectory --out $NewSummaryFile
```

验证器只读打开 Harness SQLite，根据宿主返回找到运行和作业，核对时间、步骤、父子关系、快照或引用文件、登记产物及内容 SHA256。运行时绑定、发行清单、自有 wheel、源码、锁、已部署的 Skill 和 Profile 也参与核对。脱敏摘要中的 `checks` 会显示已达到的检查层。在复审材料齐备后，用新的输出文件名再次执行。

已完成的宿主工具调用直接证明本次看到的工具名和有效权限。`full_tool_inventory` 需要实际枚举的九个名称，`first_authorization` 需要实际发生的授权事件。缺少原始材料的检查保留 `not_evaluated`。Codex 原始回合中成功读取目标 Skill 文件的命令及文件头可证明本次加载；其他宿主可使用带会话依据的真实日志或截图。首次授权和完整枚举也可在该宿主的独立安装旅程统一留证。

J09 先在原会话列出父运行已登记的产物，再接受用户修改。验证时用初次返回中的原文件及产物哈希复查父运行，并核对子运行的新参数与报告。J08 未回复时仅保存首次问答，非法回复保留第二轮实际的校验错误。J11/J12/J15 的内部变体也分别记录，不从单次成功外推其他变体。

终端退出码 0 表示所选场景达到这些门槛。退出码 1 对应场景的 failed、blocked 或 not_evaluated，退出码 2 对应记录或输出文件错误。单场景输出的 `release_ready` 固定为 false。完整发布判定还需要各场景变体、三宿主及 RG0–RG5 的独立证据。[发行技术计划](../../docs/HARNESS_RELEASE_TECHNICAL_DEVELOPMENT_PLAN.md)

## 汇总三宿主矩阵

`index.py` 按实际发行清单建立三宿主各 J01–J15 的索引，接收多次 `--report`。没有报告时也可生成待执行矩阵。

```powershell
python tools/host_acceptance/index.py --release $ReleaseDirectory --report $CodexSummary --report $OpenClawSummary --out $NewIndex
```

每个宿主填写实际版本。可用 `--host-version codex=版本` 等选项固定版本。声明支持原生委派的宿主添加 `--native codex` 等选项。相同宿主、版本、场景、变体只能有一个选定的当前摘要；重测时保留旧文件，在本次索引中选择新文件。索引记录每份摘要和私有原始清单的 SHA256，拒绝混合不同发行包或宿主版本。缺少取证的变体保留 `not_evaluated`。没有给定变体的记录单列，不能代替各变体通过。默认核心 profile 需核实模拟后端缺失的分支；声明 `traditional-simulation` 的发行包还需实际模拟、导入损坏及超时证据。

索引只汇总已选定的本地校验摘要。审计者可用保存的私有原件重跑每份 `verify`。完整矩阵的退出码为 0，待完成或阻断为 1，输入冲突为 2。`release_ready` 始终为 false，供 A 的 RG0–RG5 发布总表引用。

`--discovery workbuddy=文件路径` 记录实际本机入口枚举的哈希；对应宿主的各科学场景仍需独立运行。`--historical-index 旧摘要路径` 只保留旧发行包、宿主版本、结果和摘要哈希，不向当前发行包继承任何状态。
