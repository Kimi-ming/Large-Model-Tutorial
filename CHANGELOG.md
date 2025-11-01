# 开发日志 (Changelog)

本文档记录视觉大模型教程项目的所有开发历程、版本更新和重要变更。

---

## [v0.5.0] - 2025-11-01

### 🎉 重要里程碑

完成 **P0阶段（MVP）** 的核心开发工作，实现教程的最小可用版本。

---

## 第一阶段：项目初始化与基础设施搭建

### [2025-11-01] 项目启动

#### ✨ 新增功能

**1. 项目结构初始化**
- 创建完整的目录结构（docs/、code/、scripts/、tests/等）
- 编写 `scripts/init_project.sh` 项目初始化脚本
- 创建 Python 包初始化文件（`__init__.py`）

**2. 开发环境配置**
- 创建 `requirements.txt` - 生产环境依赖
- 创建 `requirements-dev.txt` - 开发环境依赖
- 创建 `environment.yml` - Conda 环境配置
- 创建 `setup.py` 和 `pyproject.toml` - Python 包配置
- 创建 `.gitignore` - Git 忽略规则
- 创建 `.pre-commit-config.yaml` - 代码质量检查

**3. 环境安装脚本**
- 创建 `scripts/setup.sh` - 自动化环境安装脚本
  - Python 版本检查（>=3.8）
  - GPU 环境检测（CUDA/cuDNN）
  - 依赖包安装（支持镜像源）
  - 开发工具配置
  - 环境验证

**4. 模型下载脚本**
- 创建 `scripts/download_models.sh` - HuggingFace 模型下载工具
  - 支持 CLIP、SAM、LLaVA、BLIP-2、Qwen-VL 等模型
  - 支持镜像源加速（国内网络优化）
  - 支持断点续传
  - 网络环境自动检测
  - 下载进度显示

**5. GitHub 仓库配置**
- 创建 `.github/labels.yml` - Issue 标签配置（P0-P3优先级）
- 创建 `.github/milestones.md` - 里程碑规划文档
- 创建 `.github/ISSUE_TEMPLATE/` - Issue 模板（任务/Bug/问题）
- 创建 `.github/PULL_REQUEST_TEMPLATE.md` - PR 模板

**6. 核心工具模块**
- 创建 `code/utils/model_loader.py` - 模型加载工具类
  - 支持 CLIP、SAM、BLIP-2、LLaVA、Qwen-VL
  - 自动设备检测（CUDA/CPU）
  - 支持量化加载（8bit/4bit）
  - 依赖检查和提示
  - 本地缓存管理

**7. 使用说明文档**
- 创建 `docs/05-使用说明/01-环境安装指南.md`
  - 系统要求说明
  - 详细安装步骤
  - 环境验证方法
  - 常见问题解答
- 创建 `docs/05-使用说明/02-快速开始.md`
  - 10分钟快速入门
  - 第一个示例（CLIP 推理）
  - 环境验证脚本
  - 学习路径指引

**8. 示例代码**
- 创建 `code/01-model-evaluation/examples/clip_inference.py`
  - 完整的 CLIP 图文匹配示例
  - 支持 URL、本地文件、生成图像
  - 错误处理和进度显示

#### 📝 文档更新

- 更新 `README.md` - 项目概览和快速开始指南
- 创建 `设计文档.md` v3.3 - 完整的教程设计文档
  - 从企业项目视角转为教程视角
  - 移除所有时间相关概念
  - 优化验收标准（参考区间而非硬阈值）
  - 添加学习者提示和导航
  - 区分教程必需和维护者可选内容

---

## 第二阶段：模型调研与选型章节开发

### [2025-11-01] 文档开发（P1阶段）

#### ✨ 新增功能

**1. 主流视觉大模型概述**
- 创建 `docs/01-模型调研与选型/01-主流视觉大模型概述.md`
  - 10+个开源和商业模型介绍
    - 开源：CLIP、SAM、BLIP/BLIP-2、LLaVA、Qwen-VL、InternVL、CogVLM、Yi-VL
    - 商业：GPT-4V、Gemini Vision
  - 每个模型的特点、应用场景、性能数据
  - 代码示例和参考输出
  - 官方链接和依赖说明
  - 按任务类型和资源约束的选型建议

**2. 模型对比与评测**
- 创建 `docs/01-模型调研与选型/02-模型对比与评测.md`
  - 评测维度说明（性能指标、效率指标）
  - 术语说明（Top-1/mAP/mIoU/CIDEr/BLEU/R@1）
  - 测试条件说明（硬件/分辨率/batch size/精度/数据来源）
  - 综合性能对比表（图文多模态、VQA）
  - 选型决策树
  - CSV 模板和参考资源

**3. 选型策略**
- 创建 `docs/01-模型调研与选型/03-选型策略.md`
  - 5步系统化选型流程
    1. 需求分析（任务类型识别、性能需求定义）
    2. 资源评估（硬件清单、显存估算、预算时间）
    3. 候选筛选（交互式决策树、对比表）
    4. 小规模验证（环境/性能/易用性验证）
    5. 最终决策（决策矩阵、风险评估）
  - 任务需求分析表格和清单
  - 资源约束评估（6个常用模型的FP32/FP16/INT8显存需求）
  - 3个实际选型案例（电商搜索、医疗分割、智能客服）
  - 实践任务和参考资源

**4. 基准测试实践**
- 创建 `docs/01-模型调研与选型/04-基准测试实践.md`
  - 完整的测试环境搭建指南
  - 推理速度测试方法和代码
  - 显存占用测试方法和代码
  - 准确率测试方法和代码
  - 结果可视化工具说明
  - 一键测试脚本使用指南
  - 自动化报告生成器
  - 学习成果验收清单
  - 常见问题解答

#### 💻 代码开发

**5. Benchmark 测试工具包**
- 创建 `code/01-model-evaluation/benchmark/speed_test.py` (184行)
  - `SpeedBenchmark` 类实现
  - GPU 预热机制（避免冷启动影响）
  - `torch.cuda.synchronize()` 精确计时
  - 多 batch size 对比测试
  - JSON 格式结果持久化
  - 命令行参数解析

- 创建 `code/01-model-evaluation/benchmark/memory_test.py` (90行)
  - `MemoryBenchmark` 类实现
  - 显存清空和重置
  - 模型加载前后对比
  - 峰值显存监控
  - 支持 CPU 降级

- 创建 `code/01-model-evaluation/benchmark/accuracy_test.py` (215行)
  - `CLIPAccuracyBenchmark` 类实现
  - CLIP 图文检索评测
  - 双向检索（Image-to-Text + Text-to-Image）
  - Recall@1/5 指标计算
  - 相似度矩阵计算
  - 内置单元测试（`test_recall_calculation()`）
  - 支持 `--test` 参数运行测试

- 创建 `code/01-model-evaluation/benchmark/visualize_results.py` (123行)
  - matplotlib/seaborn 可视化
  - 速度对比图（吞吐量+延迟）
  - 显存对比图（模型大小+峰值）
  - 高质量 PNG 输出（300dpi）

- 创建 `code/01-model-evaluation/benchmark/generate_report.py` (99行)
  - 自动化 Markdown 报告生成
  - JSON 结果解析
  - 表格格式化
  - 排名和推荐建议

- 创建 `code/01-model-evaluation/benchmark/__init__.py`
  - 模块初始化
  - 版本信息
  - 自动创建结果目录

- 创建 `scripts/run_benchmarks.sh` (108行)
  - 一键测试脚本
  - 环境检查
  - 批量模型测试
  - 彩色终端输出
  - 支持演示模式

- 创建 `code/01-model-evaluation/benchmark/README.md` (156行)
  - 完整使用文档
  - 快速开始指南
  - 命令行示例
  - 输出格式说明
  - 依赖要求和使用建议

#### 📝 文档质量

- 学习目标和先修要求明确
- 难度和预计时间标注
- 实践任务和验收标准
- 参考资源和下一步指引
- 代码示例完整可运行

---

## 第三阶段：Bug 修复与质量优化

### [2025-11-01] 第一轮修复 - 文档质量问题

#### 🐛 Bug 修复

**问题1：代码示例缺少导入**
- 修复 SAM 代码示例缺少 `from PIL import Image` 导入
- 影响：用户运行代码会报错

**问题2：缺少官方链接**
- 为所有模型添加官方链接
  - CLIP、SAM、BLIP-2、LLaVA：GitHub + HuggingFace
  - Qwen-VL、InternVL、CogVLM：GitHub + HuggingFace
  - GPT-4V、Gemini Vision：官方文档

**问题3：依赖说明不清晰**
- 为 SAM 添加依赖说明标注：`pip install git+https://github.com/facebookresearch/segment-anything.git`
- 为 Qwen-VL 添加依赖说明标注：`pip install transformers>=4.32.0 transformers_stream_generator`

**问题4：术语缺少解释**
- 添加术语说明区块：
  - Top-1/Top-5准确率
  - mAP (mean Average Precision)
  - mIoU (mean Intersection over Union)
  - CIDEr/BLEU
  - R@1 (Recall at 1)

**问题5：测试条件不明确**
- 添加测试条件说明：
  - 硬件环境：NVIDIA V100 32GB / A100 40GB
  - 输入分辨率：224×224（CLIP）、384×384（SAM）
  - Batch Size：1（单图推理）
  - 精度：FP16（半精度）
  - 数据来源：各模型官方论文及 HuggingFace Model Card（2023-2024）
  - 免责声明：以下数据仅供参考

**问题6：单位不统一**
- 统一单位标注：
  - 推理速度：images/sec 或 ms/image
  - 显存占用：GB
  - 准确率：%

**问题7：对比表缺少推荐场景**
- 在对比表添加"推荐场景"列
- 添加图标和解读说明

**问题8：Emoji 编码问题**
- 修复文档中的 emoji 显示问题（💡📊📈🌳📝等）

**问题9：MobileVLM 未介绍但被引用**
- 替换 MobileVLM 为 MiniCPM-V（实际存在的轻量级模型）
- 添加说明和官方仓库链接

**问题10：缺少实践工具**
- 添加 CSV 模板供用户下载
- 添加参考资源链接（COCO、VQAv2、ImageNet）

#### 📝 改进内容

- 所有代码示例包含完整导入
- 所有模型都有官方链接
- 依赖要求明确标注
- 术语有详细解释
- 测试条件完整说明
- 单位统一规范
- 提供实用工具和模板

---

### [2025-11-01] 第二轮修复 - 脚本引用和路径错误

#### 🐛 Bug 修复

**问题1：文档指向不存在的脚本（High）**
- 移除 `python scripts/download_test_data.py` 引用
- 移除 `python scripts/prepare_test_data.py` 引用
- 提供3种实际可行的测试数据准备方案：
  1. 手动准备（推荐）：直接将 JPG 图像放入 `data/test_dataset/`
  2. wget 下载：提供具体的下载命令示例
  3. 自行实现：明确标注为"需要额外实现"

**问题2：下载命令与脚本实现不符（High）**
- 修正错误命令：`./scripts/download_models.sh --models clip,sam,blip2`
- 提供5种正确的使用方式：
  ```bash
  # 单个模型
  ./scripts/download_models.sh clip
  
  # 多个模型（空格分隔）
  ./scripts/download_models.sh clip sam blip2
  
  # 所有P0模型
  ./scripts/download_models.sh --all-p0
  
  # 使用镜像源
  ./scripts/download_models.sh --mirror clip sam
  
  # 手动下载（Python代码）
  python -c "..."
  ```

**问题3："下一步"链接路径错误（Medium）**
- 修正 `03-选型策略.md` 链接：
  - `../02-模型微调与训练/` → `../02-模型微调技术/`
- 修正 `04-基准测试实践.md` 链接：
  - `../02-模型微调与训练/` → `../02-模型微调技术/`
  - `../03-数据集准备与处理/` → `../03-数据集准备/`

**问题4：同步修复相关文件**
- 更新 `scripts/run_benchmarks.sh` 数据准备提示
- 更新 `code/01-model-evaluation/benchmark/README.md` 测试数据说明

#### 📝 改进内容

- 用户按文档操作不会遇到"文件不存在"错误
- 所有命令示例可直接复制执行
- 所有"下一步"链接可正常跳转
- 提供了多种灵活的数据准备方案

---

### [2025-11-01] 第三轮修复 - Recall@K 计算逻辑错误

#### 🐛 严重 Bug 修复（High）

**问题：Recall@K 计算逻辑完全错误**

原代码存在的问题：
1. 始终检查索引 0 是否在 Top-K 内，而非当前样本索引 i
2. Recall@5 只统计前 5 个样本，样本数 > 5 时结果不完整
3. Image→Text 和 Text→Image 代码重复，易出错

错误代码示例：
```python
# 错误1：始终检查索引0
i2t_recall_at_1 = np.mean([1 if 0 in i2t_ranks[i, :1] else 0 
                            for i in range(num_samples)])

# 错误2：只统计前5个样本
i2t_recall_at_5 = np.mean([1 if i in i2t_ranks[i, :5] else 0 
                            for i in range(min(num_samples, 5))])
```

实际影响：
- 除了第 0 个样本外，其他样本都无法正确统计
- 如果样本数 > 5，Recall@5 只反映前 5 个样本
- 结果完全不可信，可能虚高或虚低

#### ✨ 修复方案

**1. 新增辅助函数 `_compute_recall_at_k()`**
```python
def _compute_recall_at_k(self, ranks: np.ndarray, k: int) -> float:
    """计算Recall@K"""
    num_samples = ranks.shape[0]
    # 正确逻辑：对第i个样本，检查ground truth索引i是否在Top-K内
    hits = [1 if i in ranks[i, :k] else 0 for i in range(num_samples)]
    return np.mean(hits)
```

关键改进：
- 使用 `i in ranks[i, :k]` 而非 `0 in ranks[i, :k]`
- 循环覆盖所有样本 `range(num_samples)`
- 代码复用，减少重复

**2. 重构 `evaluate_retrieval()` 方法**
```python
# Image-to-Text检索
i2t_ranks = np.argsort(-similarity, axis=1)
i2t_recall_at_1 = self._compute_recall_at_k(i2t_ranks, k=1)
i2t_recall_at_5 = self._compute_recall_at_k(i2t_ranks, k=5)

# Text-to-Image检索
t2i_ranks = np.argsort(-similarity.T, axis=1)
t2i_recall_at_1 = self._compute_recall_at_k(t2i_ranks, k=1)
t2i_recall_at_5 = self._compute_recall_at_k(t2i_ranks, k=5)
```

**3. 新增单元测试 `test_recall_calculation()`**

测试用例设计：
```python
# 3×3相似度矩阵（包含错配）
similarity = np.array([
    [0.5, 0.9, 0.3],  # 图0: 文本1最高（错），文本0次之（对）
    [0.3, 0.8, 0.2],  # 图1: 文本1最高（对）
    [0.6, 0.4, 0.5],  # 图2: 文本0最高（错），文本2次之（对）
])
```

期望结果：
- Recall@1 = 33.33%（只有图1的Top-1正确）
- Recall@5 = 100%（所有图像的正确文本都在Top-3内）

测试通过验证：
```
✅ 测试通过！Recall@K计算逻辑正确。
  Image-to-Text Recall@1: 33.33%
  Image-to-Text Recall@5: 100.00%
```

**4. 支持测试模式**
```bash
# 运行单元测试
python code/01-model-evaluation/benchmark/accuracy_test.py --test
```

**5. 更新文档**
- README.md 添加单元测试使用说明

#### 📝 改进内容

修复前后对比：

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 逻辑正确性 | ❌ 只有第0个样本能统计 | ✅ 所有样本正确统计 |
| 样本覆盖 | ❌ Recall@5只统计前5个 | ✅ 统计所有样本 |
| 代码质量 | ❌ 重复代码，易出错 | ✅ 函数复用，可维护 |
| 可测试性 | ❌ 无测试 | ✅ 内置单元测试 |
| 结果可信度 | ❌ 完全不可信 | ✅ 准确可靠 |

---

## 第四阶段：环境脚本优化

### [2025-11-01] 脚本功能增强（P0阶段完成前）

#### ✨ 功能增强

**1. setup.sh 优化**
- 添加 `set -euo pipefail` 增强错误处理
- 实现 `--yes` 或 `-y` 非交互模式
- 检查 `CI`、`DEBIAN_FRONTEND`、`FORCE_NON_INTERACTIVE` 环境变量
- 集成网络检测（`detect_network_region`）自动选择镜像源
- 国内网络自动使用清华镜像
- 镜像源失败时尝试备用镜像（清华/阿里/中科大）
- 非交互模式跳过开发依赖安装
- 更新帮助信息

**2. download_models.sh 优化**
- 添加 `--yes` 或 `-y` 非交互模式
- 检查 `CI`、`DEBIAN_FRONTEND`、`FORCE_NON_INTERACTIVE` 环境变量
- 更新 `MODEL_INFO` 包含 SAM 和 Qwen-VL
- 添加依赖说明（SAM 需要 segment-anything，Qwen-VL 需要 transformers_stream_generator）
- 扩展 `allow_patterns` 包含更多模型文件：
  - `tokenizer.*`、`merges.txt`、`vocab.*`、`*.tiktoken`
  - `preprocessor_config.json`、`generation_config.json`
- 非交互模式自动启用镜像源（如果网络检测建议）
- 非交互模式自动确认下载

**3. model_loader.py 优化**
- 添加 SAM 和 Qwen-VL 到 `SUPPORTED_MODELS`
- 包含默认仓库、模型/处理器类、依赖、安装说明
- 实现 `_check_dependencies()` 方法检查依赖
- 集成依赖检查到 `load_model()` 方法
- 缺少依赖时提供警告和安装建议

#### 📝 改进内容

- 支持 CI/CD 自动化部署
- 国内网络自动优化
- 模型支持矩阵对齐
- 依赖透明度提升

---

## 统计数据

### 代码统计

| 类别 | 数量 | 行数 |
|------|------|------|
| **Markdown 文档** | 12个 | ~4,500行 |
| **Python 代码** | 10个 | ~1,500行 |
| **Shell 脚本** | 4个 | ~1,200行 |
| **配置文件** | 8个 | ~300行 |
| **总计** | 34个文件 | ~7,500行 |

### 提交统计

| 阶段 | 提交次数 | 分支 |
|------|---------|------|
| 项目初始化 | 3次 | fix → main |
| P0开发 | 3次 | fix → main |
| P1开发 | 3次 | fix → main |
| Bug修复 | 3次 | fix → main |
| **总计** | **12次** | - |

### 功能覆盖

- ✅ 环境配置和安装（100%）
- ✅ 模型下载和管理（100%）
- ✅ 模型调研与选型文档（100%）
- ✅ Benchmark 测试工具（100%）
- ✅ 使用说明文档（100%）
- ⏳ 模型微调技术（0%）
- ⏳ 数据集准备（0%）
- ⏳ 多平台部署（0%）

---

## 质量保证

### 代码质量

- ✅ 无 linter 错误（除导入警告）
- ✅ 完整的 docstring 和类型注解
- ✅ 错误处理和边界情况考虑
- ✅ 单元测试覆盖关键逻辑
- ✅ 命令行参数解析完善

### 文档质量

- ✅ 结构清晰，从概念到实践
- ✅ 包含实际案例和决策模板
- ✅ 术语解释和参考链接完整
- ✅ 所有命令可直接执行
- ✅ 学习目标和验收标准明确

### 工程实践

- ✅ Git Flow 工作流（fix → PR → main）
- ✅ 详细的 commit message
- ✅ 代码审查流程
- ✅ 自动化脚本和文档
- ✅ 问题追踪和修复记录

---

## 下一步计划

### P1 阶段（v1.0）

1. **模型微调技术**
   - 微调方法介绍（LoRA、QLoRA、Adapter等）
   - 微调实践指南
   - 微调代码示例
   - 性能优化技巧

2. **数据集准备**
   - 常用数据集介绍
   - 数据预处理方法
   - 数据增强技术
   - 自定义数据集制作

3. **多平台部署**
   - NVIDIA GPU 部署
   - 华为昇腾部署
   - AMD GPU 部署
   - CPU 和边缘设备部署

4. **更多使用说明**
   - API 使用指南
   - 性能优化建议
   - 故障排查指南

### P2 阶段（v1.5）

1. **实际应用场景**
   - 行业应用案例
   - 最佳实践分享
   - 性能基准数据

2. **高级主题**
   - 模型压缩和量化
   - 分布式训练
   - 模型融合和集成

3. **维护和优化**
   - CI/CD 自动化
   - 性能监控
   - 版本管理

---

## 贡献者

- **主要开发者**：Claude (Anthropic AI Assistant)
- **项目发起人**：[项目维护者]
- **审查和测试**：[项目维护者]

---

## 许可证

本项目采用 [MIT License](LICENSE)

---

## 联系方式

- **GitHub Issues**：[项目 Issues 页面]
- **讨论区**：[项目 Discussions 页面]

---

**最后更新**：2025-11-01  
**当前版本**：v0.5.0 (P0-MVP)  
**下一版本**：v1.0.0 (P1-Release)

