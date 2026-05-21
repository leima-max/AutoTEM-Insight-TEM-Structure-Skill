# AutoTEM-Insight 三人模块组合分发方案

本文档将项目任务划分为三个适合分发给三位成员的模块组合。划分原则是：

- 同一个组合内任务技术栈相近，适合同一人持续负责。
- 每个组合都有从 P0 到 P1/P2/P3 的顺行路径，能逐步交付。
- 三个组合之间接口清楚，减少频繁互相等待。
- P0 阶段优先形成可运行 MVP 闭环，后续再扩展高级能力。

## 总体分工

| 模块组合 | 推荐负责人 | 核心目标 | 首要交付 |
|---|---|---|---|
| A. 平台架构、数据流与交付闭环 | `leima-max` | 搭建项目骨架、统一数据模型、工作流、CLI/API、报告和验证体系 | 可运行工程骨架 + `TEMAnalysisSkill.run()` + 基础报告 |
| B. 图像输入、标定、FFT、衍射与物相 | `wangzhe221` | 从原始图像到 FFT、d-spacing、g-vector、hkl、物相候选和模拟匹配 | FFT/d-spacing/phase candidate 分析链 |
| C. 原子结构、应变、缺陷与 AI 扩展 | `yswgityes` | 从原子柱检测到原子间距、GPA/局部应变、缺陷、Z 衬度和 AI 模型 | atom coordinates + strain map 分析链 |

## 当前 GitHub 成员映射

| 成员角色 | GitHub 用户名 | 仓库权限 | 默认模块组合 |
|---|---|---|---|
| 成员 A | `leima-max` | owner / admin | A. 平台架构、数据流与交付闭环 |
| 成员 B | `wangzhe221` | collaborator / write | B. 图像输入、标定、FFT、衍射与物相 |
| 成员 C | `yswgityes` | collaborator / write | C. 原子结构、应变、缺陷与 AI 扩展 |

## 当前 Codex 默认视角

当前 Codex 项目会话默认使用成员 A / `leima-max` 的工作视角。

这意味着：

- 用户说“我”时，默认指成员 A / `leima-max`。
- 用户要求“我的任务”“我负责的模块”“成员 A 的交付”时，默认返回 A 组任务。
- 用户要求“团队分工”“项目助理同步”“飞书通知”时，仍按 A/B/C 三组中立汇总。
- 成员 A 负责定义 B/C 需要遵守的数据模型、接口契约、输出目录、报告规范和验证指标。
- 成员 A 不直接接管 B/C 的核心算法任务，除非用户明确要求。

成员 A 的顺行主线：

```text
A0 工程骨架
  -> A1 核心数据模型
  -> A2 Workflow / API / CLI
  -> A3 报告、可视化与导出
  -> A4 验证与 Benchmark
```

## A. 平台架构、数据流与交付闭环

### 适合负责的人

适合工程架构意识强、能维护跨模块接口、愿意做集成、测试、CLI/API、报告和项目节奏的人。

### 职责边界

负责“把所有模块串起来并稳定交付”，不深度负责 FFT 物理算法或原子/GPA 细节算法。

主要 owns：

- 项目目录结构
- Python 包与依赖管理
- 核心数据模型和 schema
- workflow 编排
- Python API / CLI
- 报告导出
- 测试、CI、benchmark 框架
- OpenClaw/GitHub/飞书协作状态同步

### 顺行任务链

#### A0: 工程骨架

优先级：P0

任务：

1. 初始化 Python 包结构：`autotem/`、`tests/`、`examples/`、`notebooks/`。
2. 创建 `pyproject.toml`。
3. 建立 lint / format / test 约定。
4. 建立 GitHub Issue / PR 模板。
5. 建立基础 CI。
6. 建立 `main` / `dev` / `feature/*` 分支策略说明。

交付：

- `pyproject.toml`
- `.github/workflows/tests.yml`
- `.github/ISSUE_TEMPLATE/*`
- `.github/pull_request_template.md`
- 基础包目录

依赖：无。

#### A1: 核心数据模型

优先级：P0

任务：

1. 定义 `TEMDataset`。
2. 定义 `DiffractionResult`。
3. 定义 `AtomLattice`。
4. 定义 `StrainResult`。
5. 定义 `AnalysisReport`。
6. 定义通用 `Measurement` schema。
7. 实现 JSON 序列化/反序列化。
8. 添加 schema 稳定性测试。

交付：

- `autotem/core/models.py`
- `autotem/core/schema.py`
- `tests/test_models.py`

依赖：A0。

对 B/C 的输出：

- B 的 FFT/衍射结果必须写入 `DiffractionResult`。
- C 的原子/GPA 结果必须写入 `AtomLattice` 和 `StrainResult`。

#### A2: Workflow / API / CLI

优先级：P0-P1

任务：

1. 实现 `TEMAnalysisSkill` facade。
2. 实现 `load -> calibrate -> preprocess -> analyze -> report` pipeline。
3. 设计 YAML workflow schema。
4. 实现 `autotem analyze <input> --workflow <yaml> --output <dir>`。
5. 支持 workflow config 保存和复现。
6. 支持错误日志和 warning 汇总。

交付：

- `autotem/__init__.py`
- `autotem/workflows/pipeline.py`
- `autotem/workflows/recipes.py`
- `autotem/cli.py`
- `configs/default.yaml`
- `configs/hrtem_fft_phase.yaml`
- `configs/haadf_atomic.yaml`

依赖：A1；B/C 提供模块函数雏形后可集成。

#### A3: 报告、可视化与导出

优先级：P0-P1

任务：

1. 实现 JSON 导出。
2. 实现 CSV 表格导出。
3. 实现基础 plot/overlay 接口。
4. 实现 HTML report baseline。
5. 汇总 provenance、confidence、warnings。
6. 集成 B 的 FFT/peak/d-spacing 图表。
7. 集成 C 的 atom overlay/strain map 图表。

交付：

- `autotem/reporting/export_json.py`
- `autotem/reporting/report_html.py`
- `autotem/visualization/overlays.py`
- `autotem/visualization/plots.py`
- 示例报告目录

依赖：A1；B5 和 C3 的输出。

#### A4: 验证与 Benchmark

优先级：P0-P2

任务：

1. 建立测试数据目录规范。
2. 定义 benchmark 配置格式。
3. 添加小样本集成测试。
4. 汇总 B 的 d-spacing / hkl / phase 指标。
5. 汇总 C 的 atom coordinate / strain 指标。
6. 生成 benchmark report。

交付：

- `autotem/validation/benchmarks.py`
- `autotem/validation/metrics.py`
- `autotem/validation/uncertainty.py`
- benchmark 示例

依赖：B/C 的指标输出。

### A 组合完成定义

P0 完成时应能运行：

```bash
autotem analyze examples/sample.tif --workflow configs/default.yaml --output outputs/sample
```

并生成：

- 标准化结果 JSON
- CSV 表
- FFT/atom/strain 基础图
- HTML 报告
- provenance 和 warning 汇总

## B. 图像输入、标定、FFT、衍射与物相

### 适合负责的人

适合熟悉图像处理、FFT、电子衍射、晶体学、CIF、物相匹配和模拟的人。

### 职责边界

负责“从图像/衍射图到倒空间结构参数和物相候选”，不负责整体 CLI/report，也不负责原子柱/GPA 的主算法。

主要 owns：

- 数据读取适配中和显微格式强相关的部分
- 元数据和标定
- 图像质量评分中 FFT/衍射相关指标
- FFT 计算
- Bragg spot 检测
- g-vector / d-spacing
- hkl indexing
- phase matching
- zone axis 初步推断
- simulated diffraction
- 4D-STEM 衍射基础扩展

### 顺行任务链

#### B0: 数据输入与元数据解析

优先级：P0

任务：

1. 实现基础图像读取：TIFF / PNG / JPG。
2. 调研 DM3/DM4、MRC、HDF5/EMD、SER 的读取方案。
3. 设计多格式 reader registry。
4. 解析 pixel size、电压、相机长度、放大倍数、探测器角度。
5. 元数据缺失时输出 warning。
6. 将结果转换为 A 定义的 `TEMDataset`。

交付：

- `autotem/io/readers.py`
- `autotem/io/metadata.py`
- `autotem/io/converters.py`
- `tests/test_io.py`

依赖：A1 的 `TEMDataset`。

#### B1: 标定与质量评估

优先级：P0

任务：

1. 实现 pixel size 单位转换。
2. 支持手动 pixel calibration。
3. 支持基于已知 d-spacing 的 FFT 反标定。
4. 实现 diffraction calibration 基础接口。
5. 设计 SNR、漂移、污染、欠采样、模糊、像散风险字段。
6. 输出 quality score 和 warning。

交付：

- `autotem/calibration/pixel_calibration.py`
- `autotem/calibration/diffraction_calibration.py`
- `autotem/preprocessing/quality.py`

依赖：B0；A1 的 `Measurement` schema。

#### B2: 预处理与 FFT 基础

优先级：P0

任务：

1. 实现基础 denoise：Gaussian、median、Wiener、non-local means。
2. 实现背景扣除 baseline。
3. 实现 FFT 计算。
4. 保存 FFT 图和 FFT array。
5. 自动定位 FFT center / direct beam。

交付：

- `autotem/preprocessing/denoise.py`
- `autotem/preprocessing/background.py`
- `autotem/diffraction/fft.py`

依赖：B0/B1。

#### B3: Peak Detection / g-vector / d-spacing

优先级：P0

任务：

1. 实现 Bragg spot 检测 baseline：LoG/DoG、局部极值。
2. 实现 peak 亚像素拟合。
3. 计算 spot 到 center 的 reciprocal vector。
4. 输出 g-vector table。
5. 输出 d-spacing table。
6. 为每个测量输出 uncertainty、method、confidence、warning。

交付：

- `autotem/diffraction/peak_detection.py`
- `autotem/diffraction/gvector.py`
- `tests/test_diffraction_metrics.py`

依赖：B1/B2；A1 的 `DiffractionResult`。

对 A 的输出：

- `DiffractionResult.peaks`
- `DiffractionResult.center`
- `DiffractionResult.g_vectors`
- `DiffractionResult.d_spacings`

#### B4: 基础 hkl Indexing

优先级：P0-P1

任务：

1. 支持用户输入晶格常数或候选 CIF。
2. 根据 d-spacing 和夹角做基础 hkl 匹配。
3. 输出 indexed spots。
4. 输出 top candidates 和 confidence。
5. 添加标准样品验证。

交付：

- `autotem/diffraction/indexing.py`
- `examples/fft_dspacing_basic.py`

依赖：B3。

#### B5: Phase Matching / Zone Axis / Simulation

优先级：P1

任务：

1. 支持 CIF 读取。
2. 封装 `diffsims` 或 `pymatgen` 生成模拟衍射。
3. 实现实验 FFT 与模拟 pattern 匹配。
4. 计算 mean d-error、mean angle-error、matched spot count。
5. 输出 top-N phase candidates。
6. 初步推断 zone axis。
7. 生成实验/模拟叠加图。

交付：

- `autotem/diffraction/phase_matching.py`
- `autotem/diffraction/simulated_diffraction.py`
- `autotem/simulation/diffsims_adapter.py`
- `autotem/simulation/pymatgen_adapter.py`

依赖：B4。

#### B6: 4D-STEM 衍射扩展

优先级：P2-P3

任务：

1. 调研 py4DSTEM / pyxem / LiberTEM 适配。
2. 实现 4D-STEM 数据读取接口。
3. 实现 Bragg disk 检测。
4. 实现 orientation mapping baseline。
5. 实现 virtual BF/DF baseline。

交付：

- `autotem/fourdstem/bragg_disk.py`
- `autotem/fourdstem/orientation_mapping.py`
- `autotem/fourdstem/virtual_imaging.py`
- `autotem/fourdstem/strain_4dstem.py`

依赖：B3/B5。

### B 组合完成定义

P0 完成时应能从单张 HRTEM 图像输出：

- FFT 图
- center/direct beam
- Bragg spot table
- g-vector table
- d-spacing table
- 基础 hkl 候选
- 结构化 `DiffractionResult`

P1 完成时应能输出：

- top-N phase candidates
- zone axis candidate
- experimental/simulated diffraction overlay
- phase matching confidence

## C. 原子结构、应变、缺陷与 AI 扩展

### 适合负责的人

适合熟悉原子分辨 STEM、图像检测、Gaussian fitting、GPA、局部应变、缺陷识别、深度学习模型的人。

### 职责边界

负责“从高分辨图像到原子坐标、原子间距、局部位移、应变、缺陷和强度分析”，不负责项目整体工作流，也不主导衍射物相匹配。

主要 owns：

- atom detection
- subpixel refinement
- sublattice classification
- atom distance statistics
- atom displacement
- GPA
- atom-based strain
- grain/twin/defect analysis
- Z-contrast
- AI model registry / inference / training
- EDS/EELS 的原子结构关联部分

### 顺行任务链

#### C0: 原子柱检测 Baseline

优先级：P0

任务：

1. 实现 `detect_atoms(image, mode="auto")`。
2. 使用 LoG/DoG + local maxima 检测原子柱。
3. 输出初始 coordinates。
4. 输出 peak intensity。
5. 输出 atom overlay 图。
6. 对低 SNR、模糊、过密峰输出 warning。

交付：

- `autotem/atomic/atom_detection.py`
- `tests/test_atom_detection.py`

依赖：A1 的 `AtomLattice`；B0/B1 提供 calibrated image。

#### C1: 亚像素精修与原子间距

优先级：P0

任务：

1. 实现 2D Gaussian refinement。
2. 实现 center-of-mass refinement。
3. 输出 refined coordinates。
4. 计算最近邻/次近邻原子间距。
5. 输出 atom coordinates CSV。
6. 输出 distance statistics。

交付：

- `autotem/atomic/gaussian_refinement.py`
- `autotem/atomic/distance_statistics.py`

依赖：C0。

对 A 的输出：

- `AtomLattice.coordinates`
- `AtomLattice.intensities`
- atom overlay
- atom distance table

#### C2: 初版 GPA / Strain Map

优先级：P0-P1

任务：

1. 实现 `run_gpa()` 接口。
2. 从 B 的 FFT peaks 中推荐两个非共线 g-vectors。
3. 支持用户手动传入 g-vectors。
4. 支持 reference region 输入。
5. 实现 Bragg mask 基础生成。
6. 输出 phase map、displacement map、strain tensor、rotation map。
7. 输出 reference-region / g-vector warning。

交付：

- `autotem/strain/gpa.py`
- `autotem/strain/strain_statistics.py`
- `examples/gpa_basic.py`

依赖：B3 的 g-vectors；A1 的 `StrainResult`。

#### C3: 原子位移与原子级应变

优先级：P1

任务：

1. 拟合理想二维晶格。
2. 计算原子相对理想晶格的 displacement vectors。
3. 基于邻域仿射变换或 Delaunay triangulation 计算 atomic strain。
4. 与 GPA strain 做趋势对比。
5. 输出 displacement overlay 和 local strain map。

交付：

- `autotem/atomic/displacement.py`
- `autotem/strain/atom_strain.py`

依赖：C1/C2。

#### C4: 亚晶格分类与结构建模

优先级：P1-P2

任务：

1. 根据强度、几何关系、局部环境分类 sublattice。
2. 支持 B 的候选 phase/CIF 作为先验。
3. 输出 A/B/O 等 sublattice labels。
4. 推断初步 projected structure model。
5. 输出 structure overlay。

交付：

- `autotem/atomic/sublattice.py`
- `autotem/atomic/structure_model.py`
- `autotem/atomic/zone_axis.py`

依赖：C1；B5 的 phase candidates/CIF。

#### C5: 晶粒、孪晶、缺陷与多相分离

优先级：P2

任务：

1. 基于局部 FFT/纹理/分割识别 grain boundaries。
2. 检测 twin mirror symmetry 和 twin plane candidate。
3. 检测位错、层错、局部异常原子位移。
4. 结合图像、局部 FFT、原子间距、强度做 phase segmentation。
5. 输出 masks、boundary curves、defect map、statistics。

交付：

- `autotem/microstructure/grain_segmentation.py`
- `autotem/microstructure/twin_detection.py`
- `autotem/microstructure/defect_detection.py`
- `autotem/microstructure/phase_segmentation.py`

依赖：B3/B5；C3/C4。

#### C6: Z 衬度与谱学关联

优先级：P2-P3

任务：

1. 原子柱积分强度。
2. 局部背景扣除。
3. 强度归一化。
4. 输出相对 Z-contrast。
5. 设计多切片模拟校正接口。
6. 对厚度、倾斜、探测器角度未知情况输出 warning。
7. 设计 EDS/EELS 与原子柱/相分割关联接口。

交付：

- `autotem/atomic/z_contrast.py`
- `autotem/spectroscopy/eds.py`
- `autotem/spectroscopy/eels.py`
- `autotem/spectroscopy/chemical_mapping.py`

依赖：C1/C4；B5 的 structure prior 可选。

#### C7: AI 模型与主动学习

优先级：P3

任务：

1. 建立 model registry。
2. 定义模型输入输出和版本字段。
3. 实现统一 inference 接口。
4. 支持 U-Net / Mask R-CNN / YOLO / ViT / GNN 的接入规划。
5. 定义标注数据格式。
6. 实现低置信度区域标记。
7. 建立专家修正 -> 训练集 -> 增量训练闭环。

交付：

- `autotem/ai/model_registry.py`
- `autotem/ai/inference.py`
- `autotem/ai/training.py`
- `autotem/ai/active_learning.py`

依赖：C0-C6 的 baseline 结果和标注需求。

### C 组合完成定义

P0 完成时应能从 HAADF-STEM 图像输出：

- atom coordinates
- refined atom coordinates
- atom intensity table
- atom-distance statistics
- atom overlay
- preliminary strain map
- 结构化 `AtomLattice` 和 `StrainResult`

P1 完成时应能输出：

- sublattice labels
- displacement vectors
- atomic strain map
- GPA vs atom strain comparison

## 三个组合之间的接口

### A -> B/C

A 必须提供：

- 数据模型
- workflow 调用规范
- 输出目录规范
- report 接口
- validation metric 接口

### B -> A

B 必须输出：

- `TEMDataset` reader result
- `DiffractionResult`
- FFT image
- peak table
- g-vector table
- d-spacing table
- phase candidates
- diffraction overlays

### C -> A

C 必须输出：

- `AtomLattice`
- `StrainResult`
- atom coordinate table
- atom overlay
- strain maps
- defect/Z-contrast/sublattice tables when available

### B -> C

B 必须提供给 C：

- calibrated pixel size
- FFT peaks
- recommended g-vectors
- phase candidates / CIF priors
- zone axis candidates

### C -> B

C 可反向提供给 B：

- measured atom spacings
- local lattice vectors
- sublattice information
- atom-based phase/structure hints

这些可用于校验 B 的 d-spacing、hkl 和 phase candidate。

## 推荐开发顺序

### 第 1 轮：共同打通 P0 骨架

成员 A：`leima-max`

1. A0 工程骨架
2. A1 核心数据模型
3. A2 workflow/API/CLI baseline

成员 B：`wangzhe221`

1. B0 数据输入
2. B1 标定与质量评估
3. B2 FFT 基础

成员 C：`yswgityes`

1. C0 原子柱检测
2. C1 亚像素精修与原子间距
3. C2 初版 GPA / strain map

目标：产生第一个可运行但很朴素的端到端 demo。

### 第 2 轮：形成 MVP 闭环

成员 A：`leima-max`

1. A3 报告、可视化与导出
2. A4 验证与 benchmark baseline

成员 B：`wangzhe221`

1. B3 peak/g-vector/d-spacing
2. B4 基础 hkl indexing

成员 C：`yswgityes`

1. C3 原子位移与原子级应变
2. C4 亚晶格分类初版

目标：单张 HRTEM/HAADF-STEM 图像能输出 FFT、d-spacing、atom coordinates、strain map 和 HTML report。

### 第 3 轮：增强核心竞争力

成员 A：`leima-max`

1. 强化 CLI/YAML workflow
2. 增强报告模板
3. 完善 benchmark report

成员 B：`wangzhe221`

1. B5 phase matching / simulation
2. zone axis inference
3. experimental/simulated overlay

成员 C：`yswgityes`

1. 完整 GPA 增强
2. atom strain 对比
3. sublattice / structure model 增强

目标：具备 phase candidate + atom structure + strain comparison 的核心科研价值。

### 第 4 轮：高级扩展

成员 A：`leima-max`

1. Web/Jupyter UI 或 Agent tools
2. 批量处理与项目报告

成员 B：`wangzhe221`

1. B6 4D-STEM 扩展
2. 更完整的 simulation/database integration

成员 C：`yswgityes`

1. C5 晶粒/孪晶/缺陷/多相
2. C6 Z 衬度/谱学
3. C7 AI/主动学习

目标：从 MVP 工具扩展到完整 TEM 分析生态。

## 飞书分发消息模板

```text
【AutoTEM 三人模块分发建议】

建议按三个组合认领，减少交叉等待：

1. A 组：平台架构、数据流与交付闭环
   默认负责人：@leima-max
   负责：工程骨架、数据模型、API/CLI、YAML workflow、报告、测试、benchmark、项目同步

2. B 组：图像输入、标定、FFT、衍射与物相
   默认负责人：@wangzhe221
   负责：reader、metadata、calibration、FFT、peak detection、g-vector、d-spacing、hkl、phase matching、simulation、4D-STEM

3. C 组：原子结构、应变、缺陷与 AI 扩展
   默认负责人：@yswgityes
   负责：atom detection、Gaussian refinement、atom distance、GPA、atomic strain、sublattice、defects、Z-contrast、AI

建议第一轮：
- A：工程骨架 + 数据模型 + workflow baseline
- B：数据读取 + 标定 + FFT baseline
- C：原子检测 + 原子间距 + 初版 GPA

请三位按 A/B/C 组合认领，或提出想交换的模块。
```
