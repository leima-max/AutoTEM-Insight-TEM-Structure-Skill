# AutoTEM-Insight 团队职责与任务认领清单

本文档用于三人协作开发时自行认领模块和任务。每个成员可以按模块认领，也可以按任务认领。认领后请在 OpenClaw / GitHub Issue / 飞书中同步负责人、状态、预计完成时间和相关 PR。

## 认领规则

- 每个任务必须有且只有一个主负责人，可以有多个协作人。
- P0 任务优先认领，形成 MVP 闭环后再扩展 P1/P2/P3。
- 涉及跨模块接口的任务必须提前同步接口契约。
- 涉及科学结论、模型精度、许可证、数据授权的任务必须标记风险。
- 大型实验数据不要直接提交 GitHub；只提交数据说明、示例小数据或外部链接。

## 状态字段

```yaml
owner: ""
collaborators: []
status: "todo | in_progress | blocked | review | done | deferred"
priority: "P0 | P1 | P2 | P3"
milestone: ""
due_date: ""
github_issue: ""
github_pr: ""
notes: ""
```

## 模块总览

| 模块 | 主要职责 | 建议优先级 |
|---|---|---|
| 项目架构与工程基础 | 包结构、数据模型、测试、CI、配置、开发规范 | P0 |
| 数据输入与标定 | 图像读取、元数据、像素标定、质量评估 | P0 |
| 预处理与图像增强 | 去噪、滤波、漂移校正、背景扣除 | P0-P1 |
| FFT / 衍射分析 | FFT、peak detection、g-vector、d-spacing | P0 |
| hkl / 物相 / 带轴 | diffraction indexing、phase matching、zone axis | P1 |
| 模拟与数据库 | CIF、模拟衍射、模拟 STEM、材料数据库 | P1-P2 |
| 原子结构分析 | 原子柱检测、精修、亚晶格、原子间距 | P0-P1 |
| GPA 与应变 | GPA、原子位移、局部应变、剖面 | P1 |
| 晶粒 / 孪晶 / 缺陷 | 晶粒分割、孪晶、位错、层错、缺陷统计 | P2 |
| Z 衬度与成分 | 原子柱强度、Z-contrast、EDS/EELS | P2-P3 |
| 4D-STEM | Bragg disk、orientation、strain、virtual imaging | P2-P3 |
| AI 与主动学习 | 分割/检测模型、训练、标注、模型注册 | P2-P3 |
| 可视化与报告 | overlays、plots、HTML/PDF、导出规范 | P0-P1 |
| CLI / API / Workflow | Python API、CLI、YAML workflow、Agent tools | P0-P1 |
| 验证与 Benchmark | 标准样品、指标、测试集、验收报告 | P0-P2 |
| 文档与协作管理 | README、开发文档、Issue/PR、飞书同步 | P0 |

## P0: MVP 必须完成任务

### 1. 项目架构与工程基础

职责目标：建立可开发、可测试、可扩展的 Python 科学软件骨架。

任务：

- 初始化 Python 包结构：`autotem/`、`tests/`、`configs/`、`examples/`、`notebooks/`。
- 创建 `pyproject.toml`，明确包名、Python 版本、基础依赖、测试命令。
- 建立基础 lint / format / test 约定。
- 建立 GitHub 分支策略：`main`、`dev`、`feature/*`、`fix/*`。
- 建立 GitHub Issue 模板：feature、bug、task、research。
- 建立 PR 模板：范围、验证、影响模块、风险。
- 建立最小 CI：安装依赖、运行单元测试。
- 定义项目版本号策略。
- 定义模块所有权和接口变更流程。

交付物：

- `pyproject.toml`
- Python 包目录骨架
- 基础 CI 配置
- Issue / PR 模板
- 开发规范说明

建议负责人类型：架构/API/工程负责人。

### 2. 核心数据模型

职责目标：统一所有模块的输入输出，避免后期接口混乱。

任务：

- 定义 `TEMDataset`：data、modality、pixel_size、pixel_unit、metadata、provenance。
- 定义 `DiffractionResult`：peaks、center、g_vectors、d_spacings、indexed_spots、phase_candidates、confidence。
- 定义 `AtomLattice`：coordinates、intensities、sublattice_labels、lattice_vectors、zone_axis、confidence。
- 定义 `StrainResult`：displacement_x/y、strain_xx/yy/xy、rotation_xy、uncertainty。
- 定义 `AnalysisReport`：summary、figures、tables、warnings、provenance。
- 定义通用 `Measurement` schema：value、unit、uncertainty、method、parameters、confidence、provenance、warnings。
- 为所有数据模型添加 JSON 序列化和反序列化。
- 添加 schema 稳定性测试。

交付物：

- `autotem/core/models.py`
- `autotem/core/schema.py`
- `tests/test_models.py`

建议负责人类型：架构/API/数据模型负责人。

### 3. 数据输入与元数据解析

职责目标：支持常见 TEM 数据格式的读取和标准化。

任务：

- 实现统一入口 `load_data(path, metadata=None)`。
- 支持 TIFF / PNG / JPG 基础图像读取。
- 支持 DM3/DM4 读取适配调研与基础实现。
- 支持 MRC、HDF5/EMD、SER、Velox/Bruker/Nion 数据的适配方案设计。
- 解析像素尺寸、电压、相机长度、放大倍数、探测器角度等元数据。
- 元数据缺失时输出明确 warning。
- 将原始输入转换为 `TEMDataset`。
- 添加小型示例数据和读取测试。

交付物：

- `autotem/io/readers.py`
- `autotem/io/metadata.py`
- `autotem/io/converters.py`
- `tests/test_io.py`

建议负责人类型：IO/工程负责人。

### 4. 标定与质量评估

职责目标：保证 d-spacing、应变、原子位移等结果有可信标尺。

任务：

- 实现像素尺寸读取和单位转换。
- 实现手动输入 pixel size 的校准路径。
- 实现基于已知 d-spacing 的 FFT 反标定。
- 实现 diffraction calibration 基础接口。
- 设计质量评分字段：SNR、漂移、污染、欠采样、模糊、像散风险。
- 实现基础图像质量评分 baseline。
- 对低质量输入输出 warning。
- 保存校准参数和 provenance。

交付物：

- `autotem/calibration/pixel_calibration.py`
- `autotem/calibration/diffraction_calibration.py`
- `autotem/preprocessing/quality.py`
- 标定结果 JSON 示例

建议负责人类型：IO/FFT/验证负责人。

### 5. 预处理与 FFT 基础管线

职责目标：完成从图像到 FFT 和基础衍射表的闭环。

任务：

- 实现 `preprocess(dataset, recipe="auto")`。
- 实现 Gaussian / median / Wiener / non-local means baseline 去噪。
- 实现局部背景扣除 baseline。
- 实现 FFT 计算并保存 FFT 图。
- 实现 FFT 频谱中心定位。
- 实现 Bragg spot 检测 baseline：LoG/DoG、局部极值。
- 实现 peak 亚像素中心拟合 baseline。
- 输出 peak table。

交付物：

- `autotem/preprocessing/denoise.py`
- `autotem/preprocessing/background.py`
- `autotem/diffraction/fft.py`
- `autotem/diffraction/peak_detection.py`
- FFT 和 peak detection 示例 notebook

建议负责人类型：FFT/图像算法负责人。

### 6. d-spacing / g-vector / 基础 hkl

职责目标：输出第一批可验证结构参数。

任务：

- 根据 pixel size 和 FFT calibration 计算 g-vector。
- 计算 d-spacing 和角度关系。
- 为每个测量值输出 uncertainty 和 confidence。
- 实现基础 hkl indexing 接口。
- 支持用户输入候选晶体参数或 CIF。
- 输出 d-spacing table、g-vector table、基础 indexed spots。
- 添加标准样品验证用例。

交付物：

- `autotem/diffraction/gvector.py`
- `autotem/diffraction/indexing.py`
- `tests/test_diffraction_metrics.py`
- `examples/fft_dspacing_basic.py`

建议负责人类型：衍射/晶体学负责人。

### 7. 原子柱检测与原子间距统计

职责目标：完成原子级结构分析的最小版本。

任务：

- 实现 `detect_atoms(image, mode="auto")` baseline。
- 使用 LoG/DoG + local maxima 检测原子柱。
- 实现 2D Gaussian / center-of-mass 亚像素精修基础版。
- 输出 atom coordinates 和 intensities。
- 实现最近邻/次近邻原子间距统计。
- 输出原子坐标 CSV 和 overlay 图。
- 给出检测置信度和低质量 warning。

交付物：

- `autotem/atomic/atom_detection.py`
- `autotem/atomic/gaussian_refinement.py`
- `autotem/atomic/distance_statistics.py`
- `tests/test_atom_detection.py`

建议负责人类型：原子结构/图像算法负责人。

### 8. 初版应变输出

职责目标：提供 MVP 中的初步 strain map，不追求全自动完美 GPA。

任务：

- 实现 GPA 模块接口 `run_gpa()`。
- 自动推荐两个强且非共线的 g-vectors。
- 支持用户手动传入 g-vectors 和 reference region。
- 实现 Bragg mask 基础生成。
- 实现相位图、位移场、初步 strain tensor 输出。
- 输出 `exx`、`eyy`、`exy`、rotation map。
- 标记参考区不可靠、峰选择不可靠等 warning。

交付物：

- `autotem/strain/gpa.py`
- `autotem/strain/strain_statistics.py`
- `examples/gpa_basic.py`

建议负责人类型：GPA/应变负责人。

### 9. 报告与导出

职责目标：让 MVP 结果可读、可复现、可交付。

任务：

- 实现 JSON 导出。
- 实现 CSV 表格导出。
- 实现 PNG/SVG 图像输出。
- 实现基础 HTML report。
- 报告包含输入文件、参数、软件版本、结果、置信度、warning。
- 报告包含 FFT 图、peak overlay、atom overlay、strain map。
- 保存中间结果目录结构。

交付物：

- `autotem/reporting/export_json.py`
- `autotem/reporting/report_html.py`
- `autotem/visualization/overlays.py`
- `autotem/visualization/plots.py`
- 示例 report

建议负责人类型：报告/可视化/工程负责人。

### 10. Python API / CLI / YAML Workflow

职责目标：让用户能通过代码、命令行和配置运行分析。

任务：

- 实现 `TEMAnalysisSkill` facade。
- 实现 `skill.run(input_path, task, candidate_phases, output_dir)`。
- 设计 YAML workflow schema。
- 实现 `autotem analyze sample.dm4 --workflow ... --output ...`。
- 支持保存完整 workflow config。
- 支持失败时输出清晰错误和日志。

交付物：

- `autotem/__init__.py`
- `autotem/workflows/pipeline.py`
- `autotem/workflows/recipes.py`
- `autotem/cli.py`
- `configs/default.yaml`
- `configs/haadf_atomic.yaml`

建议负责人类型：架构/API/CLI 负责人。

## P1: 第一版核心竞争力任务

### 11. 智能物相识别

任务：

- 支持 CIF 读取。
- 从候选结构生成模拟衍射 peak pattern。
- 实现实验 FFT 与模拟衍射图匹配。
- 计算 phase matching score。
- 输出 top-N phase candidates。
- 输出 mean d-error、mean angle-error、matched spot count。
- 自动推断 zone axis 初版。
- 生成实验/模拟叠加图。

交付物：

- `autotem/diffraction/phase_matching.py`
- `autotem/diffraction/simulated_diffraction.py`
- `autotem/atomic/zone_axis.py`
- phase matching report 示例

### 12. 模拟适配器

任务：

- 调研并封装 `diffsims` 衍射模拟。
- 调研并封装 `pymatgen` diffraction 模块。
- 调研并封装 `abTEM` STEM/TEM 模拟。
- 定义统一 simulation adapter 接口。
- 保存模拟参数和 provenance。

交付物：

- `autotem/simulation/diffsims_adapter.py`
- `autotem/simulation/pymatgen_adapter.py`
- `autotem/simulation/abtem_adapter.py`

### 13. 亚晶格分类与结构建模

任务：

- 根据原子强度、几何关系、局部邻域分类亚晶格。
- 支持候选 CIF 投影辅助分类。
- 输出 A/B/O 等 sublattice labels。
- 拟合理想二维晶格。
- 计算相对理想晶格的原子位移。
- 输出 projected structure model。

交付物：

- `autotem/atomic/sublattice.py`
- `autotem/atomic/displacement.py`
- `autotem/atomic/structure_model.py`

### 14. GPA 完整自动化增强

任务：

- 优化 g-vector 自动选择。
- 自动推荐 reference region。
- phase unwrapping 稳健化。
- 支持 line profile 提取。
- 支持多 reference region 敏感性分析。
- 与 atom-column strain 对比。

交付物：

- `autotem/strain/gpa.py`
- `autotem/strain/atom_strain.py`
- GPA vs atomic strain 对比报告

### 15. 自动 HTML 报告增强

任务：

- 添加报告目录和摘要。
- 添加每个模块的参数、图、表和 warning。
- 支持 phase candidates 可视化。
- 支持 atom overlay 和 strain overlay。
- 支持导出为 PDF 的方案调研或实现。

交付物：

- 增强版 `report_html.py`
- 报告模板
- 示例报告

## P2: 高级科研能力任务

### 16. 晶粒与边界分析

任务：

- 滑窗 FFT 提取局部取向。
- 纹理特征和边缘检测。
- U-Net / SegFormer 分割方案设计。
- watershed / graph-cut 分离接触晶粒。
- 输出 grain mask、boundary curves、grain statistics。

交付物：

- `autotem/microstructure/grain_segmentation.py`
- `autotem/microstructure/boundary_analysis.py`

### 17. 孪晶、位错和层错检测

任务：

- 检测局部晶格镜面对称。
- 分析 FFT spot 成对或镜像关系。
- 匹配候选 twin plane。
- 识别位错相关 GPA 异常和原子位移异常。
- 检测 stacking fault 候选。
- 输出 defect map 和 defect statistics。

交付物：

- `autotem/microstructure/twin_detection.py`
- `autotem/microstructure/defect_detection.py`

### 18. 多相分离

任务：

- 结合图像特征、局部 FFT、原子间距、强度和谱学特征。
- 输出 phase mask。
- 输出每相面积分数、局部 d-spacing、相界面位置。
- 输出每相候选物相和 confidence。

交付物：

- `autotem/microstructure/phase_segmentation.py`
- phase separation 示例报告

### 19. Z 衬度分析

任务：

- 原子柱积分强度。
- 局部背景扣除。
- 强度归一化。
- 亚晶格强度统计。
- 经验 `I ~ Z^n` 拟合。
- 输出相对 Z 衬度。
- 设计多切片仿真校正接口。
- 对厚度、倾斜、探测器角度未知情况输出 warning。

交付物：

- `autotem/atomic/z_contrast.py`
- Z-contrast report 示例

### 20. 4D-STEM 扩展

任务：

- 4D-STEM 数据读取接口。
- Bragg disk 检测。
- orientation mapping。
- 4D-STEM strain mapping 初版。
- virtual BF/DF。
- py4DSTEM / pyxem / LiberTEM 适配方案。

交付物：

- `autotem/fourdstem/bragg_disk.py`
- `autotem/fourdstem/orientation_mapping.py`
- `autotem/fourdstem/virtual_imaging.py`
- `autotem/fourdstem/strain_4dstem.py`

### 21. EDS / EELS 谱学接口

任务：

- EDS 元素面扫读取和 elemental maps。
- EELS 背景扣除。
- EELS core-loss 定量。
- 价态/化学位移 proxy。
- 结构-成分联合分割。
- 原子柱 EELS/EDS 关联方案。

交付物：

- `autotem/spectroscopy/eds.py`
- `autotem/spectroscopy/eels.py`
- `autotem/spectroscopy/chemical_mapping.py`

## P3: 平台化与智能化任务

### 22. AI 模型注册与推理

任务：

- 建立 model registry。
- 定义模型版本、输入输出、训练数据来源、指标。
- 支持 U-Net / Mask R-CNN / YOLO / ViT / GNN 等模型接入。
- 实现统一 inference 接口。
- 对 domain shift 和低置信度输出 warning。

交付物：

- `autotem/ai/model_registry.py`
- `autotem/ai/inference.py`

### 23. 模型训练与主动学习

任务：

- 定义标注数据格式。
- 定义训练集/验证集划分。
- 支持合成数据增强。
- 实现低置信度区域自动标记。
- 建立专家修正 -> 训练集 -> 增量训练闭环。

交付物：

- `autotem/ai/training.py`
- `autotem/ai/active_learning.py`
- 标注规范

### 24. Web / Jupyter UI

任务：

- 上传图像。
- 选择 workflow。
- 查看 FFT、peaks、atoms、strain maps。
- 手动修正 pixel calibration、ROI、g-vectors、reference region。
- 一键生成报告。

交付物：

- Jupyter widget 或轻量 Web UI 原型
- UI 操作说明

### 25. Agent Tools

任务：

- 将核心分析流程封装为 agent-callable tools。
- 支持自然语言选择 workflow。
- 支持解释结果和生成实验建议。
- 支持批量样品统计。

交付物：

- `autotem/workflows/agent_tools.py`
- Agent 使用示例

## 横向职责

### 测试与验证

可独立认领，也可以由各模块负责人共同维护。

任务：

- 单元测试。
- 数值正确性测试。
- schema 稳定性测试。
- 小样本集成测试。
- 标准样品 benchmark。
- 专家标注对比。
- 性能测试：单张 4k 图常规分析 < 30-60 秒。
- CPU/GPU 分开记录。

验收指标：

- FFT peak recall/precision > 90%。
- d-spacing 相对误差 < 1-2%。
- hkl top-3 > 85%，限定候选物相。
- phase top-3 > 80%。
- 原子坐标 RMSE < 5-10 pm，高质量 STEM。
- 原子间距误差 < 1-2%。
- 晶粒分割 Dice/IoU > 0.80。
- 孪晶识别 precision > 80%。

### 文档与示例

任务：

- 更新 `README.md`。
- 维护 `AGENTS.md`。
- 维护开发路线和模块接口文档。
- 编写 quickstart。
- 编写 5 个示例 notebook。
- 编写每个 P0/P1 模块的最小可运行示例。
- 维护数据目录说明。

### 项目管理与飞书同步

任务：

- 每日检查 GitHub PR/Issue 变化。
- 每周生成项目周报。
- 跟踪 blocked 任务。
- 跟踪 PR review 超时。
- 跟踪跨模块接口变更。
- 在飞书同步高信号项目更新。
- 维护 OpenClaw 任务状态。

## 建议三人认领方式

如果团队希望按大块认领，建议如下。

### 方向 A：架构 / Workflow / 报告 / 验证

适合负责：

- 项目架构与工程基础
- 核心数据模型
- Python API / CLI / YAML Workflow
- 报告与导出
- 测试、CI、benchmark
- 项目管理与飞书同步

### 方向 B：FFT / 衍射 / 物相 / 模拟 / 4D-STEM

适合负责：

- 预处理与 FFT 基础管线
- d-spacing / g-vector / 基础 hkl
- 智能物相识别
- 模拟适配器
- 4D-STEM 扩展
- 标准样品衍射验证

### 方向 C：原子 / GPA / 缺陷 / AI / Z 衬度

适合负责：

- 原子柱检测与原子间距统计
- 初版应变输出
- 亚晶格分类与结构建模
- GPA 完整自动化增强
- 晶粒、孪晶、位错、层错、多相分离
- Z 衬度
- AI 模型与主动学习

## 初始认领建议

第一轮建议每个人先认领 2-3 个 P0 任务：

```text
成员 A：
- 项目架构与工程基础
- 核心数据模型
- Python API / CLI / YAML Workflow

成员 B：
- 数据输入与元数据解析
- 标定与质量评估
- 预处理与 FFT 基础管线
- d-spacing / g-vector / 基础 hkl

成员 C：
- 原子柱检测与原子间距统计
- 初版应变输出
- 报告与导出中的 atom/strain 可视化部分
```

第二轮再根据兴趣认领 P1/P2 模块。

## 飞书认领消息模板

```text
【AutoTEM 任务认领】

我认领：
1. 模块/任务：
2. 优先级：
3. 计划开始时间：
4. 预计完成时间：
5. 需要的输入/依赖：
6. 我会创建的 GitHub Issue/PR：
```

## GitHub Issue 模板建议

```markdown
## Task

## Owner

## Priority

## Module

## Scope

## Deliverables

## Acceptance Criteria

## Dependencies

## Risks

## Validation
```

