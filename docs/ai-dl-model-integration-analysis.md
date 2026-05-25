# AutoTEM 深度学习模型嵌入任务与算法选型分析报告

报告日期：2026-05-26  
适用项目：AutoTEM-Insight / TEM-Structure Skill  
主要依据：`configs/three-person-module-distribution.md`、`D:/edgexiazai/AI_TEM_literature_report.md`、`D:/edgexiazai/全流程TEM资料分析工具开发研究报告.docx`、当前仓库 A2/A3/A4 运行契约与近年公开文献/工具文档。

## 1. 总体判断

AutoTEM 不应设计成一个端到端黑盒大模型，而应设计成“物理/几何基线 + 深度学习增强 + 晶体学/模拟约束 + 不确定性输出”的模块化系统。

原因有三点：

1. TEM/HRTEM/STEM/SAED/NBED/4D-STEM/EDS/EELS 的信号物理不同，单一模型难以同时保证精度、泛化和可解释性。
2. 当前项目已经建立 `TEMDataset`、`DiffractionResult`、`AtomLattice`、`StrainResult`、`AIInferenceResult`、`AnalysisReport` 等统一数据对象，适合把模型输出嵌入为可追溯的中间证据，而不是替代结构化测量。
3. 文献与工具生态都指向同一方向：HyperSpy/pyxem/py4DSTEM/Atomap/abTEM/diffsims 负责数据、衍射、原子几何与模拟主干；深度学习最适合补强低信噪比、复杂分割、候选检测、谱图解混、缺陷识别和高通量推理。

因此，本报告建议把深度学习嵌入分为四类：

| 类型 | 作用 | AutoTEM 中的定位 |
|---|---|---|
| 保守增强模型 | 去噪、低剂量恢复、伪影抑制 | 只作为 `PreprocessingResult` 的可选分支，必须保留 raw/denoised 双轨 |
| 候选检测模型 | FFT 峰、Bragg disk、原子柱、缺陷候选 | 输出候选 mask/heatmap，再由物理/几何算法精修 |
| 语义/实例分割模型 | 相区、晶界、污染、颗粒、缺陷区域 | 输出 `MicrostructureResult`、mask、boundary、confidence |
| 结构/化学推断模型 | 亚晶格、柱高度、Z 衬度、谱学状态、取向快速预测 | 必须与 CIF/模拟/标定/不确定性共同报告 |

## 2. 与 A/B/C 分工的落地边界

| 组别 | 需要深度学习嵌入的职责 | 不应深度接管的部分 |
|---|---|---|
| A 组：平台架构、数据流与交付闭环 | `autotem/ai` 模型注册、统一推理接口、模型版本/许可证/训练数据记录、置信度与 benchmark、报告中 AI 证据展示、主动学习闭环 | 不直接训练 B/C 的科学模型，不替代 B/C 对结果物理意义的解释 |
| B 组：输入、标定、FFT、衍射、物相、4D-STEM | 去噪模型、FFT/SAED 峰点分割模型、Bragg disk 检测模型、4D-STEM 取向快速预测模型、相区识别模型 | d-spacing/g-vector/hkl 的几何计算和 CIF/模拟匹配不应由纯 DL 替代 |
| C 组：原子结构、应变、缺陷、Z 衬度、谱学、AI 扩展 | 原子柱检测、亚晶格分类、原子/缺陷图模型、晶界/孪晶/缺陷分割、Z 衬度/柱成分预测、EDS/EELS AE/VAE 解混、主动学习 | GPA 方程、Gaussian refinement、物理标定和 Z 衬度定量假设不应交给黑盒模型单独决定 |

## 3. 全流程深度学习嵌入点总表

优先级说明：P0 表示当前 MVP 或接口设计必须预留；P1 表示短中期最值得做；P2 表示中期增强；P3 表示远期扩展。

| 流程步骤 | 是否需要 DL | 推荐算法 | 最适配原因 | 输出对象 | Owner | 优先级 |
|---|---:|---|---|---|---|---|
| 模态识别与输入质检 | 可选 | 轻量 CNN/ViT-small 分类器 + 规则特征 | 自动区分 HRTEM/STEM/SAED/4D-STEM，对批量数据路由有价值 | `TEMDataset.metadata`, `QualityAssessment` | A/B | P2 |
| 低剂量去噪/恢复 | 是 | SHINE / Noise2Void 类 blind-spot 自监督；原子分辨场景用 Noise2Atom/AtomSegNet 类去噪头 | 无需干净 GT，适合低剂量和束敏材料；比监督去噪更容易落地 | `PreprocessingResult`, `AIInferenceResult` | B/C | P1 |
| 视觉增强/超分辨 | 可选 | CGRDN/cGAN、DFTNet、Diffusion/LoRA | 对展示和人工判断有帮助，但定量风险高 | `PreprocessingResult.artifacts` | B/C | P2 |
| 污染/遮挡/坏区 mask | 是 | U-Net / nnU-Net | 小样本分割稳定，可作为后续 FFT/atom/GPA 的 exclusion mask | `PreprocessingResult.mask`, `QualityAssessment` | B/C | P1 |
| FFT center/direct beam | 可选 | 传统中心估计为主，heatmap U-Net 辅助 | 中心定位对 d-spacing 误差影响大；DL 只用于复杂背景 | `DiffractionResult.center` | B | P1 |
| FFT/SAED 峰点检测 | 是 | 小型 U-Net/LCA-Unet spot segmentation + Gaussian/subpixel refinement | FFT spot 是高对比小目标，分割模型比纯阈值对污染、条纹、弱峰更鲁棒 | `DiffractionResult.peaks` | B | P1 |
| g-vector/d-spacing 计算 | 否，依赖上游 DL | 物理几何计算 + uncertainty propagation | 这是确定性测量，不应由 DL 回归替代 | `DiffractionResult.g_vectors`, `d_spacings` | B | P0 |
| hkl indexing | 不建议纯 DL | CIF/晶格约束匹配 + 可选学习排序器 | hkl 需要晶体学约束；DL 可做候选排序，不能单独给最终索引 | `DiffractionResult.indexed_spots` | B | P1 |
| HRTEM 相区/局部相识别 | 是 | sliding-window FFT + PCA/NMF + U-Net/ResNet/Swin 分类器 | 先把局部周期性转到频域，再学习相区，比直接 raw image 分类更稳 | `MicrostructureResult.phase_masks`, `PhaseCandidate[]` | B/C | P1 |
| Phase matching / zone axis | 可选 | diffraction-library matching 为主，Siamese/metric learning 或 GNN ranker 辅助 | 相候选必须由 CIF/模拟约束；DL 可提升 top-k 排序速度 | `PhaseCandidate[]`, `SimulationResult` | B | P2 |
| 4D-STEM 去噪 | 是 | 无监督/自监督 4D-STEM denoising | 4D datacube 噪声大，DL 去噪可提升 Bragg disk 检测召回 | `FourDSTEMResult.datacube` | B | P2 |
| 4D-STEM Bragg disk 检测 | 是 | NDPD / YOLO-style detector / RT-DETR-small + subpixel fit | Bragg disk 是天然目标检测问题，高通量下检测器比逐图规则更快 | `FourDSTEMResult.bragg_disk_table` | B | P2 |
| 4D-STEM 取向图 | 是，但不替代模板匹配 | Swin/ResNet/DenseNet Euler angle regression + py4DSTEM/pyxem template matching 校核 | DL 适合快速预测，模板匹配适合作权威验证 | `FourDSTEMResult.orientation_map` | B | P2 |
| 原子柱初检 | 是 | 双 U-Net heatmap + segmentation，或 AtomSegNet 类模型 | 对低 SNR、厚度变化、复杂背景比 LoG/DoG 更稳 | `AtomLattice.atoms` 初始坐标 | C | P1 |
| 原子柱亚像素精修 | 否，DL 只给初值 | Atomap/2D Gaussian + center-of-mass refinement | 亚像素坐标应由可解释拟合给出残差和误差条 | `AtomSite`, coordinate CSV | C | P0 |
| 原子间距/局部晶格 | 否 | Delaunay/邻域图/晶格拟合 | 几何统计不需要 DL，需输出 uncertainty | `AtomLattice.lattice_vectors`, tables | C | P0 |
| 亚晶格分类 | 是 | Graph Neural Network + MLP/RandomForest 基线 + CIF prior | 原子天然构成图结构，强度、邻域、相对位置可联合分类 | `AtomSite.sublattice_label` | C | P2 |
| atom-based strain | 可选 | 几何仿射/Delaunay 为主，GNN/AE 用于异常点过滤 | 应变是物理量，DL 适合剔除离群或补全缺失，不适合作唯一计算器 | `StrainResult`, displacement table | C | P1 |
| GPA strain | 不建议核心 DL | 经典 GPA + 可选 U-Net reference/mask 推荐 | GPA 的核心是相位/反射物理，DL 可帮助 ROI/reference/mask 选择 | `StrainResult` | C | P0-P1 |
| 晶粒/晶界/孪晶/相边界分割 | 是 | U-Net++ / nnU-Net + orientation/FFT 物理约束 | 语义边界分割是 DL 强项，但需用取向/FFT/原子拓扑校验 | `MicrostructureResult.features` | C | P2 |
| 缺陷检测与分类 | 是 | 两阶段：AE/VAE/OCSVM novelty detection + U-Net/Mask R-CNN/YOLO/Graph classifier | 已知缺陷可监督，未知缺陷必须保留 novelty/needs-review 路径 | `DefectCandidate[]` | C | P2 |
| Z 衬度/柱高度/成分推断 | 是，但必须模拟约束 | FCN/encoder-decoder CNN + Bayesian/ensemble uncertainty + multislice calibration | HAADF 强度受厚度、倾斜、探测角影响，DL 只能在校准域内推断 | `ZContrastResult` | C | P2-P3 |
| EDS/EELS 谱图解混 | 是 | NMF/PCA baseline + Autoencoder/VAE；精细结构用 1D CNN/Transformer | 高光谱数据适合潜空间建模，AE/VAE 可提端元和丰度图 | `SpectroscopyResult` | C | P2-P3 |
| 4D-STEM -> EDS 预测 | 远期 | CNN regression / U-Net regression | 可减少束敏样品 EDS 采集压力，但需要严格同位配准训练集 | `SpectroscopyResult.element_maps` | C/B | P3 |
| 原位/视频颗粒或缺陷追踪 | 是，远期 | YOLO/Mask R-CNN/SAM2 预分割 + TrackMate/ByteTrack/TrackNet + ConvLSTM/TCN | 动态事件需要检测、分割、追踪和时间模型组合 | video result extension | C/A | P3 |
| 自动报告文字生成 | 可选 | 规则模板优先，LLM 只基于结构化结果生成摘要 | 报告不能让 LLM 编造结论；只可解释已有 metrics/warnings | `AnalysisReport.summary` | A | P2 |
| benchmark 与不确定性 | 是 | MC Dropout / Deep Ensembles / calibration curves | 所有 DL 输出必须能显示不确定性、低置信区域、OOD 风险 | `AIInferenceResult.uncertainty`, `BenchmarkResult` | A | P0-P1 |

### 3.1 DL 嵌入步骤到具体细分任务的映射

上表中的 Owner 只表示责任组。实际开发时应落到 A/B/C 的细分任务链中，避免“AI 扩展”变成一个笼统包袱。下面表格给出每个可嵌入 DL 的步骤应该挂靠的具体任务。

| DL 嵌入步骤 | 对应细分任务 | 训练/推理目标 | 推荐模型 | 进入 AutoTEM 的输出 |
|---|---|---|---|---|
| 模态识别与输入质检 | B0 数据输入、B1 标定与质量评估、A2 workflow 路由 | 自动判断 HRTEM/STEM/SAED/NBED/4D-STEM/EDS/EELS，标记低质数据 | MobileNetV3 / EfficientNet-B0 / ViT-small + 规则特征 | `TEMDataset.modality`, `QualityAssessment`, routing warning |
| 低剂量去噪/保守恢复 | B2 预处理与 FFT 基础、C0 原子柱检测前处理、C7 AI 模型接口 | 从 noisy raw 生成保守 denoised image，提升峰/原子候选召回 | SHINE / Noise2Void / Noise2Atom 类 blind-spot 自监督 | `PreprocessingResult.image`, `AIInferenceResult`, raw-denoised consistency |
| 展示增强/超分辨 | B2 预处理、C0/C1 原子图展示、A3 报告图 | 生成仅供视觉复核的 enhanced image | CGRDN/cGAN、DFTNet、Diffusion + LoRA | `PreprocessingResult.artifacts`, report figure；禁止覆盖定量 raw |
| 污染/坏区/支撑体 mask | B1 质量评估、B2 预处理、C5 微结构/缺陷 | 分割污染、孔洞、支撑体、坏像素、不可分析区域 | U-Net / nnU-Net / U-Net++ | `PreprocessingResult.mask`, QA flags, exclusion mask |
| FFT center/direct beam 辅助定位 | B2 FFT 基础、B3 peak/g-vector/d-spacing | 在复杂背景下预测中心热图或 direct beam 区域 | heatmap U-Net / small HRNet | `DiffractionResult.center`, center confidence |
| FFT/SAED/NBED spot segmentation | B3 Peak Detection / g-vector / d-spacing | 预测衍射峰概率图，再做连通域与亚像素拟合 | 小型 U-Net / LCA-Unet / StarDist-style spot head | `DiffractionResult.peaks`, peak mask artifact |
| hkl 候选排序 | B4 基础 hkl indexing、B5 phase matching | 对规则法产生的 hkl/phase 候选重新排序 | Siamese metric learner / shallow MLP ranker / GNN ranker | `IndexedSpot.confidence`, `PhaseCandidate.score` |
| HRTEM 局部相区识别 | B5 phase matching、C5 phase segmentation | 基于局部 FFT/频谱特征分割相区和周期性区域 | sliding-window FFT + PCA/NMF + U-Net/ResNet/Swin | `MicrostructureResult.phase_masks`, `PhaseCandidate[]` |
| diffraction-library 快速相匹配 | B5 simulated diffraction、B5 zone axis inference | 从模拟库中快速检索 top-k pattern/zone axis | CNN embedding / Siamese network / metric learning | `SimulationResult.comparison_metrics`, top-k candidates |
| 4D-STEM datacube 去噪 | B6 4D-STEM 衍射扩展 | 对每个 probe diffraction pattern 做无监督去噪 | self-supervised 4D denoising / Noise2Void-4D | `FourDSTEMResult.datacube`, denoised cube artifact |
| 4D-STEM Bragg disk 检测 | B6 Bragg disk detection | 高通量检测 Bragg disk bbox/center/radius | NDPD / YOLOv8-style / RT-DETR-small + subpixel fit | `FourDSTEMResult.bragg_disk_table` |
| 4D-STEM 取向快速预测 | B6 orientation mapping | 从 diffraction pattern 预测 Euler angle/top-k orientation | ResNet/DenseNet/Swin regression/classifier | `FourDSTEMResult.orientation_map`, warm-start for pyxem/py4DSTEM |
| 原子柱候选检测 | C0 原子柱检测、C1 精修与距离 | 预测 atom column heatmap/segmentation，再交给 Gaussian refinement | 双 U-Net / AtomSegNet 类 heatmap model | `AtomLattice.atoms` 初始坐标、atom candidate mask |
| 原子柱亚像素异常过滤 | C1 Gaussian refinement、C3 atom displacement | 对拟合失败、重叠峰、低置信原子做 reject/rescore | small MLP / GNN / autoencoder residual model | `AtomSite.confidence`, refinement warnings |
| 亚晶格分类 | C4 sublattice / structure model、B5 CIF prior | 根据强度、邻域图、局部晶格和 CIF 先验分类 A/B/O 等子晶格 | GNN / GraphSAGE / GAT + MLP baseline | `AtomSite.sublattice_label`, `sublattice_table.csv` |
| GPA ROI/reference/mask 推荐 | C2 初版 GPA、C3 strain cross-validation | 推荐 reference region、Bragg mask、低可信应变区域 | U-Net mask head / classical+ML ranker | `StrainResult.reference_region`, GPA warnings |
| atom-based strain 离群点过滤 | C3 原子位移与原子级应变 | 发现不可信原子位移、缺失点、局部异常邻域 | graph autoencoder / robust GNN / isolation model | `StrainResult.uncertainty`, atomic displacement QA |
| 晶粒/晶界/孪晶/相边界分割 | C5 grain/twin/phase segmentation、B5 phase prior | 分割 grain mask、boundary line、twin plane、phase boundary | nnU-Net / U-Net++ / DeepLabV3+ + orientation features | `MicrostructureResult.features`, boundary artifacts |
| 缺陷检测与分类 | C5 defect detection、C7 active learning | 检测位错、空位、层错、界面缺陷和未知异常 | AE/VAE/OCSVM novelty + YOLO/Mask R-CNN/U-Net/GNN classifier | `DefectCandidate[]`, novelty score, needs-review queue |
| Z 衬度/柱高度/成分预测 | C6 Z-contrast、C4 structure prior、B5 simulation prior | 预测柱高、占位、相对 Z 或候选元素组成 | FCN / encoder-decoder CNN + ensemble uncertainty | `ZContrastResult`, sensitivity/warning |
| EDS/EELS 谱图解混 | C6 spectroscopy、C7 AI model registry | 端元谱、丰度图、化学态潜空间、谱峰异常 | PCA/NMF baseline + Autoencoder/VAE/1D CNN/Transformer | `SpectroscopyResult.element_maps`, latent maps |
| 4D-STEM -> EDS 预测 | B6 4D-STEM、C6 spectroscopy | 用低剂量 4D-STEM 特征预测元素图，减少 EDS 采集负担 | U-Net regression / CNN regression + calibration head | `SpectroscopyResult.element_maps`, uncertainty map |
| 原位 TEM 视频追踪 | C5 defect/particle tracking、C7 AI、A2 workflow | 对颗粒、界面、缺陷做 frame-wise detection + tracking | YOLO/Mask R-CNN/SAM2 pre-mask + ByteTrack/TrackNet/ConvLSTM/TCN | video extension artifact, event table |
| 报告摘要生成 | A3 reporting/export、A2 workflow | 基于结构化结果生成摘要文字和复核提示 | 规则模板优先；可选小模型/LLM summary，不训练科学判断模型 | `AnalysisReport.summary`, review notes |
| 不确定性与置信度校准 | A4 validation/benchmark、C7 AI active learning | 校准 confidence、生成 OOD/low-confidence 区域 | MC Dropout / Deep Ensembles / temperature scaling / conformal prediction | `AIInferenceResult.uncertainty`, `BenchmarkResult` |
| 合成数据与主动学习闭环 | A4 benchmark、B5 simulation、C7 training/active learning | 生成 HRTEM/STEM/FFT/defect 合成训练集，收集专家修正 | abTEM/diffsims/Construction Zone + domain randomization + active learning | dataset manifest, model card, training provenance |

### 3.2 训练硬件配置、效率和时间估算

下面估算用于指导硬件采购和排期，不是最终 SLA。实际时间会随数据规模、标注质量、增强策略、epoch 数、I/O 缓存和超参搜索大幅变化。

统一假设：

- 框架：PyTorch + CUDA + mixed precision；数据在 NVMe SSD 本地缓存。
- 2D 图像任务：以 256-512 px patch 训练，5k-50k patch，50-150 epoch。
- 衍射/FFT 任务：以 128-512 px pattern/FFT patch 训练，10k-100k pattern。
- 4D-STEM 任务：以裁剪后的 diffraction pattern 或 datacube chunk 训练，50k-500k pattern。
- 表中时间表示“第一个可用于项目评估的模型”训练时间；若做正式论文级模型、5 折验证和系统超参搜索，通常需要乘以 2-5。
- 最低配置表示能跑通并可做小规模微调；推荐配置表示适合团队持续迭代。

硬件分层建议：

| 配置层 | 建议硬件 | 适用范围 |
|---|---|---|
| 2D 最低配置 | 1x CUDA GPU 12GB 显存，32-64GB RAM，8-12 核 CPU，1TB NVMe | 小型 U-Net、分类器、AE/VAE、少量 patch 微调 |
| 2D 推荐配置 | 1x RTX 4090 24GB 或 RTX 5090 32GB，64-128GB RAM，16-24 核 CPU，2TB NVMe | 大多数 2D TEM/STEM/FFT 模型训练和快速迭代 |
| 大图/4D 最低配置 | 1x 24GB GPU，128GB RAM，16 核 CPU，2TB NVMe | 4D-STEM 小块训练、YOLO/ResNet 取向模型、较小 diffusion/denoise |
| 大图/4D 推荐配置 | 1x RTX 6000 Ada 48GB / RTX A6000 48GB / A100/H100 80GB，或 2x RTX 4090/5090，256GB RAM，4TB NVMe | 4D-STEM、较大 batch、ensemble、长时间生产训练 |

硬件规格参考：NVIDIA 官方页面显示 RTX 5090 为 32GB GDDR7，RTX 4090 为 24GB GDDR6X，RTX 6000 Ada 为 48GB GDDR6 ECC，H100 提供 80GB/94GB HBM 规格。训练效率估算按显存容量、Tensor Core 世代、batch size 和常见 PyTorch AMP 经验折算。

| DL 嵌入步骤 | 具体细分任务 | 最低配置效率与时间 | 推荐配置效率与时间 |
|---|---|---|---|
| 模态识别与输入质检 | B0/B1/A2 | 12GB GPU；batch 32-64，约 200-600 thumbnail/s；训练 1-4 小时 | RTX 4090/5090；batch 128-256，约 800-2000 thumbnail/s；训练 15-60 分钟 |
| 低剂量去噪/保守恢复 | B2/C0/C7 | 12GB GPU；256 patch batch 4-16，效率 1x；训练 12-36 小时 | RTX 4090/5090；batch 16-64，效率约 3-6x；训练 4-12 小时 |
| 展示增强/超分辨 | B2/C0/A3 | 16-24GB GPU；batch 1-8，效率 1x；cGAN/DFTNet 1-3 天，diffusion 3-7 天 | 32-48GB GPU；batch 8-32 或梯度累积，效率约 3-5x；cGAN/DFTNet 8-24 小时，diffusion 1-3 天 |
| 污染/坏区/支撑体 mask | B1/B2/C5 | 8-12GB GPU；batch 4-16，效率 1x；训练 4-16 小时 | RTX 4090/5090；batch 16-64，效率约 3-7x；训练 1-4 小时 |
| FFT center/direct beam 辅助定位 | B2/B3 | 8-12GB GPU；batch 16-64，效率 1x；训练 1-3 小时 | RTX 4090/5090；batch 64-256，效率约 3-6x；训练 20-60 分钟 |
| FFT/SAED/NBED spot segmentation | B3 | 8-12GB GPU；batch 8-32，效率 1x；训练 3-10 小时 | RTX 4090/5090；batch 32-128，效率约 3-7x；训练 1-3 小时 |
| hkl 候选排序 | B4/B5 | 8-12GB GPU 或 CPU 可跑；小型 ranker 训练 1-6 小时 | RTX 4090/5090；大模拟库 embedding 训练/微调 0.5-2 小时 |
| HRTEM 局部相区识别 | B5/C5 | 12-16GB GPU；512 patch batch 2-8，效率 1x；训练 12-36 小时 | RTX 4090/5090；batch 8-32，效率约 3-6x；训练 4-12 小时 |
| diffraction-library 快速相匹配 | B5 | 12GB GPU；10k-100k simulated pattern，训练 4-12 小时 | RTX 4090/5090；embedding 预计算快 3-6x，训练 1-4 小时 |
| 4D-STEM datacube 去噪 | B6 | 24GB GPU + 128GB RAM；小 chunk batch 1-4，效率 1x；训练 2-5 天 | 48-80GB GPU 或 2x 4090/5090 + 256GB RAM；效率约 3-8x；训练 12-36 小时 |
| 4D-STEM Bragg disk 检测 | B6 | 12-16GB GPU；YOLO small batch 8-32，效率 1x；训练 12-30 小时 | RTX 4090/5090/RTX 6000 Ada；batch 32-128，效率约 3-6x；训练 4-12 小时 |
| 4D-STEM 取向快速预测 | B6 | 24GB GPU；ResNet/Swin-small，batch 8-32，效率 1x；训练 2-5 天 | 48-80GB GPU 或 2x 4090/5090；batch 32-128，效率约 3-8x；训练 12-48 小时 |
| 原子柱候选检测 | C0/C1 | 12-16GB GPU；batch 4-16，效率 1x；训练 8-24 小时 | RTX 4090/5090；batch 16-64，效率约 3-6x；训练 2-8 小时 |
| 原子柱亚像素异常过滤 | C1/C3 | 8GB GPU 或 CPU；MLP/AE 小模型，训练 0.5-4 小时 | RTX 4090/5090；效率提升有限，训练 10-60 分钟 |
| 亚晶格分类 | C4 | 8-12GB GPU；小型 GNN，训练 2-8 小时 | 16-24GB GPU；batch 多图并行，效率约 2-5x；训练 0.5-2 小时 |
| GPA ROI/reference/mask 推荐 | C2 | 8-12GB GPU；U-Net-small，训练 3-12 小时 | RTX 4090/5090；效率约 3-6x；训练 1-3 小时 |
| atom-based strain 离群点过滤 | C3 | 8GB GPU 或 CPU；GNN/AE，训练 1-6 小时 | 16-24GB GPU；效率约 2-4x；训练 0.5-2 小时 |
| 晶粒/晶界/孪晶/相边界分割 | C5 | 12-16GB GPU；512 patch batch 2-8，效率 1x；训练 12-48 小时 | 24-48GB GPU；batch 8-32，效率约 3-7x；训练 4-16 小时 |
| 缺陷检测与分类 | C5/C7 | 12-16GB GPU；novelty + detector/classifier，训练 12-48 小时 | 24-48GB GPU；多模型并行/大 batch，效率约 3-7x；训练 4-16 小时 |
| Z 衬度/柱高度/成分预测 | C6/C4/B5 | 16-24GB GPU；模拟+实验混合，训练 1-3 天 | 48GB GPU 或 A100/H100；大 batch/ensemble，效率约 3-8x；训练 8-24 小时 |
| EDS/EELS 谱图解混 | C6 | 8-12GB GPU；AE/VAE/1D CNN，训练 2-12 小时 | 16-24GB GPU；效率约 2-5x；训练 1-4 小时 |
| 4D-STEM -> EDS 预测 | B6/C6 | 24GB GPU；严格配准数据，训练 3-7 天 | 48-80GB GPU 或 2x 4090/5090；效率约 3-8x；训练 1-3 天 |
| 原位 TEM 视频追踪 | C5/C7/A2 | 16GB GPU；detector + tracker，训练 1-3 天 | 24-48GB GPU；效率约 3-6x；训练 8-24 小时 |
| 报告摘要生成 | A3 | 通常不建议训练；规则模板无 GPU；若做 7B LoRA，24GB GPU 训练 6-24 小时 | 48-80GB GPU；LoRA 训练 2-8 小时；但科学结论仍必须来自结构化结果 |
| 不确定性与置信度校准 | A4/C7 | MC Dropout 通常不需重训，推理慢 1.5-3x；Deep Ensembles 训练时间约为单模型 3-5x | 推荐用 2x GPU 或 48GB GPU 并行 ensemble；总耗时可压到单模型 1.5-3x |
| 合成数据与主动学习闭环 | A4/B5/C7 | 数据生成主要吃 CPU/GPU 模拟；单机 24GB GPU 生成+训练一轮通常 1-7 天 | 48-80GB GPU + 32 核以上 CPU；模拟/训练流水线化，一轮 0.5-3 天 |

## 4. 各任务最推荐算法路线

### 4.1 图像去噪与低剂量恢复

推荐主线：SHINE / Noise2Void 类 blind-spot 自监督模型。

落地方式：

- MVP 保留现有 Gaussian/median/Wiener/NLM 等传统去噪。
- P1 增加 `preprocess.denoise_mode: self_supervised_blindspot`。
- 输出 raw image、denoised image、差分图、FFT 一致性指标和 warning。
- 不允许用去噪图覆盖原始图；所有 d-spacing、atom intensity、Z-contrast 定量必须标明是否来自 raw 或 denoised 分支。

选择理由：

- TEM 真实干净 GT 很难获得，自监督 blind-spot 对项目最现实。
- SHINE 文献显示其面向高通量 EM，且与 Noise2Void/BM3D 等方法比较过。
- Diffusion、cGAN、超分辨模型可作为展示增强或人工复核辅助，但不应默认进入定量链路。

### 4.2 FFT/SAED 峰点检测与衍射分析

推荐主线：传统峰检测 + 小型 U-Net/LCA-Unet spot segmentation + subpixel fitting。

落地方式：

- 当前 B3 已有 FFT peak/g-vector/d-spacing baseline，应保留为权威基线。
- P1 新增 `diffraction.peak_detector: classical | unet_spot | ensemble`。
- U-Net 只输出 spot probability mask；最终峰心仍由 connected components + Gaussian/centroid/subpixel fit 决定。
- `DiffractionPeak.confidence` 来自 mask probability、峰形残差、局部 SNR、与 Friedel/晶格一致性。

不建议：

- 不建议直接用 CNN 回归 d-spacing 或 hkl。
- 不建议只输出单个 phase label；应输出 top-k phase candidates、matched spot count、mean d-error、angle error 和模拟叠图。

### 4.3 HRTEM 相区识别与物相分割

推荐主线：sliding-window FFT + PCA/NMF + U-Net/ResNet/Swin 分类器。

落地方式：

- 每个局部窗口生成 FFT amplitude spectrum。
- PCA/NMF 先估计相区数、频谱基底和空间分布。
- 深度模型只对局部频谱或局部 FFT 特征做分类/分割，不直接把 raw HRTEM 作为唯一证据。
- 与 B5 的 CIF/diffsims/abTEM 模拟结果融合后，写入 `phase_candidates` 和 `phase_masks`。

选择理由：

- 文献中的 HRTEM phase division 方法明确使用滑动窗口 FFT、PCA/NMF 与深度学习组合。
- 这种路线保留晶体周期信息，比通用图像分割更符合 TEM 物理。

### 4.4 4D-STEM Bragg disk 与取向映射

推荐主线：

- Bragg disk：NDPD / YOLO-style detector + subpixel refinement。
- 取向图：py4DSTEM/pyxem template matching 为权威输出，Swin/ResNet/DenseNet 做快速候选预测。

落地方式：

- P2 先接入 py4DSTEM/pyxem 适配器，形成非 DL 基线。
- 再加入 detector 处理大规模 datacube，提高高通量 Bragg disk 提取速度。
- 对取向/Euler angle，DL 输出应作为 top-k candidate 或 warm start，再由模板匹配校正。

### 4.5 原子柱检测、坐标和亚晶格

推荐主线：Atomap/2D Gaussian 几何基线 + 双 U-Net/AtomSegNet 候选增强 + GNN 亚晶格分类。

落地方式：

- P0/P1：LoG/DoG/local maxima + 2D Gaussian refinement，输出 `AtomLattice`。
- P1/P2：U-Net heatmap 生成候选原子柱，segmentor mask 去除支撑体/背景/污染区域。
- 最终坐标仍由 Gaussian/center-of-mass refinement 给出，记录拟合残差、peak intensity、confidence。
- 亚晶格分类使用原子图：节点为 atom column，边为近邻关系，节点特征包括强度、局部 spacing、候选元素、phase prior。

选择理由：

- 双 U-Net 已在 supported nanoparticle 原子柱定位中验证。
- AtomSegNet 类模型适合通过模拟数据迁移到实验 STEM，但坐标精修仍需要可解释拟合。

### 4.6 GPA、原子级应变与缺陷

推荐主线：

- GPA：经典 GPA 为主，DL 只辅助 g-vector/reference region/mask 推荐。
- atom strain：几何仿射/Delaunay 为主，DL 用于异常点过滤。
- defect：AE/VAE/OCSVM novelty detection + 监督 U-Net/Mask R-CNN/graph classifier。

落地方式：

- 对位错、层错、孪晶、晶界等，不要只做语义 mask；必须同时输出几何证据，如 boundary line、orientation jump、Burgers vector candidate、局部位移异常。
- 缺陷 taxonomy 初期不宜过细，应先支持 `unknown/needs_review`。
- 每个 defect candidate 写入 `evidence`，包含原图 patch、mask、局部 FFT、原子位移图。

### 4.7 Z 衬度、柱高度与谱学

推荐主线：

- Z 衬度：FCN/encoder-decoder CNN + multislice/abTEM 校准 + Bayesian/ensemble uncertainty。
- EELS/EDS：PCA/NMF baseline + Autoencoder/VAE；精细结构可用 1D CNN/Transformer。

落地方式：

- 先做 relative Z-contrast，不直接承诺绝对成分。
- 如果缺少厚度、倾斜、探测器角、束流、电压或模拟校准，报告必须 warning。
- 谱学结果应关联 atom column 或 phase mask，而不是只输出孤立 element map。

### 4.8 合成数据、主动学习与模型治理

推荐主线：Construction Zone/abTEM/diffsims 合成数据 + 小量真实标注微调 + active learning。

落地方式：

- A 组定义 model registry 与数据版本字段。
- B/C 提供训练数据生成脚本和标注规范。
- 每次模型推理必须记录 `model.name/version/uri/training_data/metrics/license`。
- 低置信区域自动导出为 review queue，专家修正后进入下一轮训练集。

## 5. 建议的模型注册与推理接口

当前 `autotem.core.models` 已有 `AIModelInfo` 和 `AIInferenceResult`，建议 A 组优先将其实体化为：

```text
autotem/ai/
  model_registry.py      # 模型元数据、版本、许可证、输入输出 schema
  inference.py           # 统一 predict() / batch_predict()
  postprocess.py         # mask -> peaks/atoms/defects/regions
  uncertainty.py         # MC dropout / ensemble / calibration
  active_learning.py     # low-confidence review queue
  training.py            # 训练入口和数据集 manifest
```

统一推理结果必须包含：

| 字段 | 要求 |
|---|---|
| model | `AIModelInfo`，包含模型名、版本、任务、路径、训练数据、指标、许可证 |
| predictions/masks | 保存为 artifact，不只在内存中传递 |
| confidence | 0-1，总体置信度 |
| uncertainty | per-pixel/per-object/per-spectrum uncertainty |
| low_confidence_regions | 需要人工复核的 ROI |
| provenance | 输入 hash、workflow 参数、依赖版本、推理设备、随机种子 |
| warnings | 域外样品、metadata 缺失、低 SNR、校准缺失等 |

## 6. 分阶段执行建议

### 第 0 阶段：当前 A 组应立即做的事

1. 保持现有 A2/A3/A4 契约稳定，新增 AI 模型 registry 文档和空实现。
2. 在 YAML workflow 中预留 `ai_models`、`denoise_mode`、`peak_detector`、`atom_detector`、`uncertainty` 参数。
3. 在报告里新增 “AI Evidence / Model Provenance / Low Confidence Regions” 区块。
4. 在 benchmark 中预留 DL 指标：Dice、mAP、atom RMSE、peak precision/recall、calibration error、ECE/uncertainty coverage。

### 第 1 阶段：最值得优先嵌入的三个模型

1. 自监督去噪：SHINE/Noise2Void 类模型。
2. FFT/SAED spot segmentation：小型 U-Net/LCA-Unet。
3. 原子柱候选检测：双 U-Net/AtomSegNet 类 heatmap。

这三个模型覆盖 B/C 的高频痛点，且不会破坏当前平台契约，因为它们都可以作为候选增强接入现有确定性后处理。

### 第 2 阶段：形成科研价值增强

1. HRTEM 相区识别：sliding-window FFT + PCA/NMF + deep classifier。
2. 晶界/孪晶/缺陷分割：U-Net++/nnU-Net + orientation/atom topology 校验。
3. 4D-STEM Bragg disk：NDPD/YOLO-style detector。
4. 亚晶格分类：GNN + CIF prior。

### 第 3 阶段：高级扩展

1. Z 衬度柱高度/成分：FCN + multislice calibration + uncertainty。
2. EELS/EDS：Autoencoder/VAE/Transformer。
3. 原位视频：detector + tracker + temporal model。
4. 报告摘要：LLM 只基于结构化数据生成文字，不参与科学结论判定。

## 7. 关键风险与控制规则

| 风险 | 控制规则 |
|---|---|
| 去噪生成虚假晶格或改变峰强 | raw/denoised 双轨；定量输出必须标注来源；FFT/atom consistency check |
| 单一材料/仪器过拟合 | 按仪器、样品、日期、剂量、厚度分层切分训练/验证/测试 |
| 相识别黑盒误判 | 必须输出 top-k + CIF/模拟证据 + d/angle residual，不输出单一 hard label |
| 原子柱坐标虚假高精度 | 坐标必须经过 Gaussian/COM refinement，附残差和 uncertainty |
| Z 衬度误当成成分 | 缺厚度/探测角/模拟校准时只输出 relative Z contrast |
| 未知缺陷被强行分类 | 支持 `unknown`、novelty score、needs-review |
| 模型许可证影响发布 | `AIModelInfo.license` 必填；YOLO/部分 GPL/AGPL 组件需隔离为 optional adapter |
| AI 输出无法审计 | 所有模型输出必须进入 `AIInferenceResult` 和 report provenance |

## 8. 推荐最终技术路线

AutoTEM 的模型嵌入路线应遵守以下原则：

1. 先有非 DL 基线，再加 DL 增强。
2. DL 输出先作为候选、mask、ranker 或 uncertainty，不直接替代物理量计算。
3. 所有模型结果必须写回统一数据模型，并可被 JSON/CSV/HTML/PDF report 审计。
4. 模型训练必须依赖合成数据 + 少量真实标注 + 主动学习闭环。
5. P0/P1 阶段优先做“最能改善鲁棒性但不破坏可解释性”的模型：去噪、FFT spot、原子柱候选。

如果按当前三人分工推进，建议 A 组先完成 AI 契约和模型治理，B 组先做去噪/FFT/4D-STEM 检测模型，C 组先做原子柱/缺陷/谱学模型。这样可以让 AutoTEM 在保持科学可追溯性的同时，逐步获得深度学习带来的高通量、抗噪声和复杂场景识别能力。

## 9. 参考资料

项目内资料：

- `configs/three-person-module-distribution.md`
- `D:/edgexiazai/AI_TEM_literature_report.md`
- `D:/edgexiazai/全流程TEM资料分析工具开发研究报告.docx`
- `docs/a2-reserved-step-contracts.md`
- `autotem/core/models.py`

公开文献与工具：

- SHINE self-supervised EM denoising: https://pmc.ncbi.nlm.nih.gov/articles/PMC11963987/
- Noise2Void: https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf
- Single-shot EM restoration with DCNN/cGAN: https://www.nature.com/articles/s41524-023-01188-0
- Atomic column localization with U-Net: https://www.nature.com/articles/s41524-024-01360-0
- AtomSegNet / TEMImageNet: https://arxiv.org/abs/2012.09093
- HRTEM phase division via FFT/PCA/NMF/deep learning: https://www.sciencedirect.com/science/article/pii/S0968432824000829
- Neural Object Detection for 4D-STEM: https://arxiv.org/abs/2506.04477
- py4DSTEM software paper: https://academic.oup.com/mam/article/27/4/712/6888063
- pyxem orientation/template matching: https://arxiv.org/abs/2111.07347
- Atomap documentation: https://atomap.org/finding_atom_lattices.html
- abTEM documentation: https://abtem.readthedocs.io/en/latest/getting_started/overview.html
- diffsims documentation: https://diffsims.readthedocs.io/
- Construction Zone synthetic HRTEM data: https://www.nature.com/articles/s41524-024-01336-0
- MC Dropout uncertainty: https://arxiv.org/abs/1506.02142
- NVIDIA GeForce RTX 5090 official specs: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA GeForce RTX 4090 official specs: https://www.nvidia.com/en-ph/geforce/graphics-cards/40-series/rtx-4090/
- NVIDIA RTX 6000 Ada official specs: https://www.nvidia.com/en-us/design-visualization/rtx-6000/
- NVIDIA H100 official specs: https://www.nvidia.com/en-gb/data-center/h100/
