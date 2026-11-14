# AutoTEM 宿主适配、Skill 与本地工具桥

版本：1.0.0；验证日期：2026-10-05。本阶段交付传统操作注册与宿主调用闭环。前端、交互 revision、DL 权重训练与未开发功能仅保留能力状态，不作为可执行成功结果。

## 1. 真实支持与验证边界

安装了项目 Codex MCP 适配器后，可运行其 `readiness_command`。诊断会在所选工程目录内运行 `codex mcp get autotem --json`，比较启用状态、启动命令、参数和工作目录，并在 `host_project_registration` 中返回结果。若返回 `PROJECT_MCP_SERVER_NOT_DISCOVERED`，请在 Codex 中打开该工程，检查项目授信和项目级 MCP 配置，重新加载后复查。登记匹配以后，还需在真实会话中授权工具并调用 `list_capabilities`。

| 宿主 | 交付资源 | 本机只读检查 | 本文未据此确认的能力 |
|---|---|---|---|
| Codex CLI / Desktop | 项目 `.agents/skills/autotem`；可选项目 MCP 表；native TOML 模板；顺序 fallback | 外部 CLI `0.143.0` 读取 Skill 后执行被策略拒绝；当前 Desktop 协作 API 的三 Profile 已真实完成合成输入闭环 | `.codex/agents` 模板安装/加载未验收；外部 CLI 分析 E2E 仍被阻断；不推断其他 Codex 客户端 |
| OpenClaw | 项目 `skills/autotem`；JSONL/MCP 提议配置；顺序 fallback | `OpenClaw 2026.5.18`；version/skills help/health help 已检查；最终只读 health exit 1，Gateway 未通过 | 该工作区实际 Skill 发现、LLM 对话、native 委派未验收；当前 Gateway 检查失败 |
| 腾讯 WorkBuddy | 项目内 Skill import ZIP、完整展开资源、MCP 提议配置；顺序调用规则 | 当前 PATH 未找到 WorkBuddy CLI；没有执行桌面导入 | GUI 导入/版本、MCP 连接、真实 LLM 调用与 native Agent 均 `not_evaluated` |

能力探测函数仅执行版本/帮助，不读取用户全局配置或凭证、不启动 LLM、不输出完整诊断。`native_subagents_verified=false` 与 `host_llm_e2e_verified=false` 是探测/安装结果的事实边界。Profile 描述职责；复制 Skill 不会安装原生子 Agent。真正的 LLM/工具 E2E 结果须另存具体宿主版本、请求、调用与产物证据；不从一次测试推断所有宿主支持。

当前路径依据：[Codex 官方 Skill 文档](https://learn.chatgpt.com/docs/build-skills)、[Codex MCP 配置字段](https://learn.chatgpt.com/docs/config-file/config-reference)、[OpenClaw 官方 Skill 文档](https://docs.openclaw.ai/tools/skills)。这些页面定义入口和配置，不证明本机已加载。

Codex 三个 native TOML 模板仅包含 `name/description/developer_instructions`，依据[官方 Subagents 文档](https://learn.chatgpt.com/docs/agent-configuration/subagents)。安装器在 `autotem-host/codex/native-agent-templates/` 提供模板，不自动写 `.codex/agents/` 或启用原生功能；不设置模型/权限覆盖。仓库镜像在 [Codex 模板目录](../configs/harness/hosts/codex/)，验证加载/工具继承/实际 spawn 后才可使用。模板语法已解析通过，运行状态仍为 `native_not_verified`。

腾讯平台资源依据其[官方 Skill 格式](https://open.workbuddy.cn/docs/skill)及[Connector 文档](https://open.workbuddy.cn/en/docs/connector)，提供所需双语 metadata 与本地 stdio MCP 提议。适配器不调用第三方同名 “work-buddy” 服务，也不猜测腾讯桌面软件的隐藏配置路径。Import ZIP 把 `SKILL.md` 放在根目录；实际导入须由受支持的 GUI 验证。

## 2. 项目内安装与回滚

普通用户的安装入口应准备运行环境后再部署 Skill，避免逐条排查 Python/网关/权限。
新增标准库引导安装器 `autotem.bootstrap` 与独立 `doctor`，详见
[安装与就绪改进](harness-installation-experience.md)。Codex 优先配置本地 MCP，
CLI/JSONL fallback 使用项目 adapter 记录的绝对解释器及 `-I`；不依赖 Python 别名、
用户 site-packages 或源码 cwd。隔离模式要求依赖完整安装在选定运行环境中。
下面的直接安装命令适用于已具备完整包环境的开发者；Skill 导入、文件复制和宿主真实运行分别验收。

先安装 Python 包。MCP 可选：`pip install "autotem-insight[mcp]"`；不用 MCP 时 Python/CLI/JSONL 可执行传统路线。Windows/Python 3.10 的 TOML 安全合并依赖可选 `tomli`；Python 3.11+ 使用标准库。

```powershell
python -m autotem.harness.cli --root outputs/harness host-probe codex
python -m autotem.harness.cli --root outputs/harness install-host codex --project D:/path/to/project
python -m autotem.harness.cli --root outputs/harness install-host codex --project D:/path/to/project --apply
```

安装默认 dry-run，无写入。`--apply` 仅写指定项目内文件；明确拒绝用户 HOME、全局 `.codex/.openclaw/.workbuddy` 根。安装前解析项目内目标，拒绝越界 symlink/Windows reparse 目标。同名用户 Skill/不同 AutoTEM MCP 表视为冲突，不静默覆盖。

对 Codex，可加 `--with-mcp` 将经过解析校验的 `[mcp_servers.autotem]` 追加到**项目** `.codex/config.toml`，保留无关表和注释。相同表无需修改；不同同名表或无法安全追加的 inline 配置拒绝合并。不会修改用户 `~/.codex/config.toml`，也不调用全局 `codex mcp add`。启动的 Python 路径取安装器当前解释器，项目空间路径通过参数数组传递，无 shell 拼接。

OpenClaw 安装示例将 host 换为 `openclaw`，只复制 workspace Skill 与提议资源，不修改全局 `openclaw.json`。WorkBuddy 换为 `workbuddy`，生成 `autotem-host/workbuddy/autotem-skill.zip` 与 `adapter.json`，下一步明确要求实际 GUI 导入。两者当前不支持 `--with-mcp` 自动配置合并。

应用时写 `.autotem/host-installations/<installation_id>/manifest.json` 与修改前备份，记录前后 hash、动作和安装状态。安装失败只回退本次已写且内容仍相符的文件。重复相同安装不创建新变更。升级资源与既有内容不同会拒绝；先检查旧版本回滚，再安装新版，避免覆盖用户修订。

```powershell
python -m autotem.harness.cli rollback-host --project D:/path/to/project install-RETURNED_ID
python -m autotem.harness.cli rollback-host --project D:/path/to/project install-RETURNED_ID --apply
```

回滚同样默认 dry-run；RETURNED_ID 必须替换为实际完整 ID。变更后文件与安装 hash 不同则拒绝整个回滚；备份 hash 错误也拒绝。只删除确认由该次安装新建的文件或恢复完整原内容，不递归删除用户目录。备份可能含原项目配置，按私有项目文件保护，不上传或打印其内容。

## 3. 九个共享工具与协议

`list_capabilities`、`inspect_input`、`validate_plan`、`submit_analysis`、`get_run_status`、`cancel_run`、`list_artifacts`、`read_artifact`、`export_report` 是公开粗粒度工具。具体参数见 [Skill transport 合同](../skills/autotem/references/tool-contracts.md)和运行时 MCP `tools/list`；内部科学操作由 capabilities 返回，不各自暴露成数百个 MCP 工具。

```powershell
python -m autotem.harness.cli --root outputs/harness stdio
```

JSONL 每行接收 `{"schema_version":"1.0.0","request_id":"q1","tool":"list_capabilities","arguments":{}}`，每行回复包含 `schema_version/request_id/tool/ok/result/error`。非白名单、未知字段/参数、重复 JSON key、NaN/Inf、超大请求与不合法分页被拒绝；库的 stdout 诊断转到 stderr，不能污染 JSON 输出。协议 `1.0.0` 独立于历史 core schema `0.1.0`。

```powershell
python -m autotem.harness.cli --root outputs/harness mcp
```

这里是可选 SDK 的**真正 MCP stdio**，与 JSONL 不同。不能把 MCP 客户端连接到 `stdio` 子命令；也不能把 JSONL 文本当成 MCP JSON-RPC 握手。两者共享白名单、参数校验和 Harness 服务；MCP 不成为分析算法必需依赖。

可选 extra 的最低 SDK 版本采用本轮实测 `mcp 1.27.0`；本机包 metadata 中 mcp 与 tomli 均为 MIT。错误传输保留领域错误码及可修复 fields/roles，同时清除凭证值；Windows CLI/JSONL 使用 ASCII JSON 转义，避免控制台代码页损坏协议。

`ok=true` 表示调用成功返回；仍需读取 run 的执行、科学、复核及 needs-input 状态。CLI 退出 0 不能证明科学通过。输出支持 JSON/HTML/PDF；PDF 为有界证据摘要，完整结构状态仍在 JSON/HTML 中。当前没有统一 frontend/approve/revision 工具，不在 Skill 中虚构调用。

## 4. 用户旅程与人工补充

自然语言需求由已有宿主解释，按三 Profile 的领域规则路由。宿主先发现能力/检查文件，选择合法模态和 dataset/frame/channel，保存计划并 validate，再 submit→status→artifacts→export。三个 Profile 可按有界工作包顺序执行；前包完成后，validate/submit 以 `run_id`（CLI `--run-id`）追加新 step，共享已提交的 artifact roles。只有 completed run 可追加，输入/selection/mode/budget 必须相同，旧 step ID 与参数不可改；科学输入变化仍创建带 `parent_run_id` 的新 run。所有步骤调用同一注册执行层；报告与文件服务是公共能力，不能另建通用报告 Agent。

原生子 Agent 若经实际宿主接口验证可用，才在该宿主中委派；本安装器交付的是顺序 fallback，不添加未经证实的 native TOML/JSON 字段。fake-host/固定计划只验证后端合同；真实宿主验收须让 LLM 从实际自然语言选择/调用工具，记录最终解释所引用的真实产物。

缺标定、可靠 g pair、原图 ROI、候选结构或晶格先验时，返回 needs-input，并由宿主问用户。GPA ROI 使用 `[y0,y1,x0,x1]` 原始像素；图上显示缩放不能直接作为 raw 坐标。用户补充后验证新 plan，以 `parent_run_id` 保存新 run，不改旧输出。未回答不循环，也不生成虚构尺度/GT；未执行科学评测保留 `not_evaluated`。

计划示例：[HRTEM](../configs/harness/hrtem-diffraction.yaml)、[SAED direct pattern](../configs/harness/saed-diffraction.yaml)、[HAADF](../configs/harness/haadf-atomic.yaml)、[GPA](../configs/harness/hrtem-gpa.yaml)。使用 `--input` 替换 `INPUT_FILE`；模板未填写的科学前提必须从用户/可信 metadata 获得。SAED/NBED direct pattern 不再次 FFT；无标定只能像素候选；GPA 模板缺 reference 时是待补输入场景，不是完整应变分析。

## 5. 资源打包与维护

权威可分发资源在 `autotem/harness/resources/skill/`，通过 setuptools package-data 进入 wheel。`skills/autotem`、`configs/harness` 是供仓库浏览的镜像；修改时同步并运行镜像一致性测试。安装器读取包资源，不根据 cwd 查 checkout，因而 wheel 安装后仍可生成宿主资源。

验证应覆盖：dry-run 无写入、同名冲突、无关 TOML 保持、可恢复备份、用户改动阻止回滚、越界 symlink 拒绝、WorkBuddy ZIP metadata、九工具白名单与坏 JSONL 恢复、真实 MCP 握手、wheel 资源及一个真实宿主顺序用户旅程。通过这些工程检查不会自动证明原生多 Agent、三个宿主 LLM E2E 或真实科学精度准入。

## 6. 本轮验收记录与发布边界

以下保留先前算法链交付的验收快照；安装体验增量及新 wheel 的结果另记于
[安装与就绪改进](harness-installation-experience.md)，不把新增源码归入旧 wheel hash 或旧测试数量。

- `python -m pytest tests/test_harness_hosts.py -o addopts=-q --basetemp outputs/harness-implementation-20261005/pytest-full-host-fix`：21 passed；含 Windows 250 字符中文/空格路径、原子替换失败保护和真实 MCP stdio initialize → tools/list → tools/call，SDK 1.27.0。匿名管道逐步握手避免 Windows 沙箱拒绝 SDK 客户端命名管道；服务自身仍为真正 MCP。
- 去掉默认 smoke 过滤、排除 `real_diffsims` 的全量回归：570 passed、2 skipped、1 deselected、3 warnings，483.28 秒；见[全量日志](../outputs/hv-final/pytest.log)。之后的 Windows Numba cache 长路径修复与模拟错误分类另有定向验证；不把它们伪装成该次全量已包含的 case。
- 最终 wheel 从项目缓存目录离线构建（`pip wheel --no-deps --no-build-isolation`），15 个资源及镜像相符，100 个 Python 文件与最新源码、wheel 和非 editable 安装文件逐字节核对，包含短临时文件/短 artifact 名、SQLite close、jobs/PID、Profile 预算、Windows Numba cache 长路径和 dependency-broken 分类。SHA256：`d81a30ea2d1de201cb8168408aecc1847bc62a7a973e03ee9d25c84cc82ff9de`；见[最终源码 manifest](../outputs/hv-final/release-wheel-source-manifest.json)。
- 最终 wheel 在独立 cwd 通过 `pip --target --no-deps` 非 editable 安装；三个宿主的项目 dry-run、实际安装及回滚全部通过。磁盘合成 HRTEM → API data-quality → CLI diffraction/atomic-strain 共享 run `d52be4ec48074ad6bb61af10bc6a4487`，12 steps completed，导出前注册 56 个 artifact，含 5 张图、4 个表，并导出 JSON/HTML/PDF。原输入及全部已注册 artifact 的 hash 已核验。导入 guard 覆盖实际 36 次 Python 启动，UI/DL 模块未加载且没有被拦截的导入尝试。此项复用了当前 Python 3.10.9 的既有依赖，**不证明干净 Windows 新 venv 的完整依赖安装**；`real_host_llm=false`、`scientific_accuracy=not_evaluated`。证据：[最终 wheel 验收](../outputs/hv-final/final-wheel-acceptance.json)、[独立安装运行明细](../outputs/hv-final/installed-wheel-evidence-release.json)。
- 早期 wheel 的 CLI 顺序闭环保留为历史快照：run `af52d03831be4a21a9ed4927b8d57e8a`，12 steps completed、50 个科学 artifact，合成 Gaussian HAADF，JSON/HTML/PDF 已导出，`scientific_accuracy=not_evaluated`、`real_host_llm=false`。证据：[cli-journey-result.json](../outputs/harness-host-validation/wheel-cli-project/cli-journey-result.json)；最终包状态以上一条的新证据为准。
- 真实 Codex 请求先使用现有配置，被该 CLI 拒绝不支持的 `gpt-6.1-sol`。只在隔离验收会话加官方 `--ignore-user-config`，不设置 `--model`、不改全局配置；LLM 成功读 Skill/三 Profile 后，`python -m autotem.harness.cli ... capabilities` 和 `python --version` 被 `blocked by policy` 拒绝。已显式 `--sandbox workspace-write --ephemeral --cd <project>`，未绕过规则或使用全权限。实际分析 E2E 保留 `not_verified`。证据：[真实请求](../outputs/harness-host-validation/codex-request.json)、[宿主进程结果](../outputs/harness-host-validation/codex-process-result.json)与私有本地 `codex-events.jsonl`；输出目录是本轮证据，不随 wheel 分发。
- 当前 Codex Desktop 的协作 API 另行真实委派三个领域 Profile；它们从自然语言请求及已提交证据自主选择/验证/执行工作包，共享 run `e3438274bd774f8cb8defbbf80506cf3` 的 14 steps completed，74 个分析 artifact，另有 JSON/HTML/PDF 三个报告。证据：[data-quality](../outputs/harness-implementation-20261005/native-smoke/data-quality-evidence.json)、[diffraction](../outputs/harness-implementation-20261005/native-smoke/diffraction-evidence.json)、[atomic-strain](../outputs/harness-implementation-20261005/native-smoke/atomic-strain-evidence.json)。这是当前 Desktop 原生委派的有界证据，工具层执行为 Python API/CLI 与 sequential；输入仍为合成 fixture，`scientific_accuracy=not_evaluated`，不证明 `.codex/agents` 模板加载或安装器原生适配。
- OpenClaw 最终只读 health 检查 exit 1、`gateway_health=false`，没有返回已验证 Agent；见[脱敏检查结果](../outputs/harness-host-validation/openclaw-health-final.json)。本轮未启动 Gateway、修改全局配置或发送外部消息；真实 OpenClaw LLM 用户旅程保留未验收。

产品经理复审：当前 Desktop 的有界合成输入旅程、工具层、顺序协作与证据导出可进入本地工程验收。OpenClaw Gateway/LLM、腾讯 WorkBuddy 导入/LLM、TOML 安装后的三 Agent 加载/运行、真实用户 needs-input 回答、干净 Windows 新环境安装和真实标准样本精度分别保留独立验收门槛。预置配方、合成图和手动固定计划不得替代这些门槛；安装器文件成功也不能代表宿主 LLM 执行成功。
