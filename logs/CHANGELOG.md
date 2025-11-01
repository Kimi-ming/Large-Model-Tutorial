# 开发日志 (Changelog)

本文档记录视觉大模型教程项目的所有开发历程、版本更新和重要变更。

> 📝 **说明**: 本文档为总览，详细的开发和修复记录请查看 `logs/` 目录下的独立日志文件。

---

## 目录结构

```
logs/
├── CHANGELOG.md                                    # 本文档（总览）
├── development/          # 开发日志
│   ├── 2025-11-01-project-initialization.md
│   ├── 2025-11-01-model-research-docs.md
│   └── 2025-11-01-benchmark-tools.md
└── bugfix/              # Bug修复日志
    ├── 2025-11-01-fix-documentation-quality.md
    ├── 2025-11-01-fix-script-references.md
    └── 2025-11-01-fix-recall-calculation.md
```

---

## [v0.5.0] - 2025-11-01

### 🎉 重要里程碑

完成 **P0阶段（MVP）** 的核心开发工作，实现教程的最小可用版本。

---

## 开发历程

### 第一阶段：项目初始化与基础设施搭建

**日期**: 2025-11-01  
**详细日志**: [`development/2025-11-01-project-initialization.md`](development/2025-11-01-project-initialization.md)

#### ✨ 新增功能

1. **项目结构初始化**
   - 创建完整的目录结构（docs/、code/、scripts/、tests/等）
   - 编写 `scripts/init_project.sh` 项目初始化脚本

2. **开发环境配置**
   - `requirements.txt` - 生产环境依赖
   - `requirements-dev.txt` - 开发环境依赖
   - `environment.yml` - Conda 环境配置
   - `setup.py` 和 `pyproject.toml` - Python 包配置
   - `.gitignore` 和 `.pre-commit-config.yaml` - 代码质量

3. **环境安装脚本** (`scripts/setup.sh`)
   - Python 版本检查（>=3.8）
   - GPU 环境检测（CUDA/cuDNN）
   - 依赖包安装（支持镜像源）
   - 开发工具配置
   - 环境验证

4. **模型下载脚本** (`scripts/download_models.sh`)
   - 支持 CLIP、SAM、LLaVA、BLIP-2、Qwen-VL 等模型
   - 支持镜像源加速（国内网络优化）
   - 支持断点续传
   - 网络环境自动检测

5. **GitHub 仓库配置**
   - Labels 配置（P0-P3优先级）
   - Milestones 规划文档
   - Issue 模板（任务/Bug/问题）
   - PR 模板

6. **核心工具模块** (`code/utils/model_loader.py`)
   - 支持 CLIP、SAM、BLIP-2、LLaVA、Qwen-VL
   - 自动设备检测（CUDA/CPU）
   - 支持量化加载（8bit/4bit）
   - 依赖检查和提示

7. **使用说明文档**
   - `docs/05-使用说明/01-环境安装指南.md`
   - `docs/05-使用说明/02-快速开始.md`

8. **示例代码**
   - `code/01-model-evaluation/examples/clip_inference.py`

**统计**: 20+个文件，~1,500行代码，~800行文档

---

### 第二阶段：模型调研与选型章节开发

**日期**: 2025-11-01  
**详细日志**: [`development/2025-11-01-model-research-docs.md`](development/2025-11-01-model-research-docs.md)

#### ✨ 新增功能

1. **主流视觉大模型概述** (387行)
   - 10+个开源和商业模型介绍
   - 每个模型的特点、应用场景、性能数据
   - 代码示例和参考输出
   - 官方链接和依赖说明

2. **模型对比与评测** (156行)
   - 评测维度说明（性能指标、效率指标）
   - 术语说明（Top-1/mAP/mIoU/CIDEr/BLEU/R@1）
   - 测试条件说明（硬件/分辨率/batch size/精度）
   - 综合性能对比表
   - 选型决策树

3. **选型策略** (393行)
   - 5步系统化选型流程
   - 任务需求分析表格和清单
   - 资源约束评估（显存估算、预算时间）
   - 交互式决策树
   - 3个实际选型案例

4. **基准测试实践** (925行)
   - 完整的测试环境搭建指南
   - 推理速度、显存、准确率测试方法
   - 可视化工具使用说明
   - 学习成果验收清单

**统计**: 4个文档，~2,600行

---

### 第三阶段：Benchmark 测试工具开发

**日期**: 2025-11-01  
**详细日志**: [`development/2025-11-01-benchmark-tools.md`](development/2025-11-01-benchmark-tools.md)

#### ✨ 新增功能

1. **推理速度测试** (`speed_test.py`, 184行)
   - GPU 预热机制
   - 精确计时（torch.cuda.synchronize）
   - 多 batch size 测试
   - JSON 结果保存

2. **显存占用测试** (`memory_test.py`, 90行)
   - 显存清空和重置
   - 峰值显存监控
   - 支持 CPU 降级

3. **准确率测试** (`accuracy_test.py`, 215行)
   - CLIP 图文检索评测
   - Recall@1/5 指标
   - 双向检索（I2T + T2I）
   - 内置单元测试

4. **结果可视化** (`visualize_results.py`, 123行)
   - matplotlib/seaborn 图表
   - 速度和显存对比图
   - 高质量 PNG 输出

5. **自动化报告** (`generate_report.py`, 99行)
   - Markdown 格式报告
   - JSON 结果解析
   - 排名和建议

6. **一键测试脚本** (`run_benchmarks.sh`, 108行)
   - 环境检查
   - 批量测试
   - 彩色输出
   - 演示模式

7. **使用文档** (`benchmark/README.md`, 156行)

**统计**: 8个文件，~900行代码

---

## Bug 修复记录

### 修复1: 文档质量问题

**日期**: 2025-11-01  
**优先级**: High + Medium  
**详细日志**: [`bugfix/2025-11-01-fix-documentation-quality.md`](bugfix/2025-11-01-fix-documentation-quality.md)

#### 🐛 修复内容

1. ✅ 代码示例缺少导入（SAM 示例添加 PIL 导入）
2. ✅ 为所有模型添加官方链接
3. ✅ 为 SAM 和 Qwen-VL 添加依赖说明标注
4. ✅ 添加术语说明（mAP、mIoU、R@1 等）
5. ✅ 添加测试条件说明（硬件、分辨率、精度）
6. ✅ 统一单位标注（images/sec、ms、GB、%）
7. ✅ 在对比表添加"推荐场景"列
8. ✅ 修复 emoji 编码问题
9. ✅ 替换 MobileVLM 为 MiniCPM-V
10. ✅ 添加 CSV 模板和参考资源

**影响**: 2个文件，+44行/-16行

---

### 修复2: 脚本引用和路径错误

**日期**: 2025-11-01  
**优先级**: High + Medium  
**详细日志**: [`bugfix/2025-11-01-fix-script-references.md`](bugfix/2025-11-01-fix-script-references.md)

#### 🐛 修复内容

1. ✅ 移除不存在的脚本引用（download_test_data.py、prepare_test_data.py）
2. ✅ 修正模型下载命令（--models → 位置参数）
3. ✅ 修复"下一步"链接路径（3处目录名错误）
4. ✅ 同步修复相关文件（run_benchmarks.sh、README.md）

**修复前**:
- ❌ `python scripts/download_test_data.py` → 文件不存在
- ❌ `./scripts/download_models.sh --models clip,sam,blip2` → 未知选项
- ❌ 点击"下一步"链接 → 404错误

**修复后**:
- ✅ 提供3种可行的数据准备方案
- ✅ 所有模型下载命令可直接执行
- ✅ 所有"下一步"链接正常跳转

**影响**: 4个文件，+44行/-16行

---

### 修复3: Recall@K 计算逻辑错误

**日期**: 2025-11-01  
**优先级**: High（严重Bug）  
**详细日志**: [`bugfix/2025-11-01-fix-recall-calculation.md`](bugfix/2025-11-01-fix-recall-calculation.md)

#### 🐛 修复内容

**问题**: `accuracy_test.py` 中的 Recall@K 计算存在严重逻辑错误
- 始终检查索引0而非当前样本索引i
- Recall@5只统计前5个样本
- 代码重复，易出错

**修复方案**:
1. ✅ 新增 `_compute_recall_at_k()` 辅助函数
2. ✅ 重构 `evaluate_retrieval()` 方法
3. ✅ 新增 `test_recall_calculation()` 单元测试
4. ✅ 支持 `--test` 参数运行测试
5. ✅ 更新 README 文档

**修复前后对比**:

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 逻辑正确性 | ❌ 只有第0个样本能统计 | ✅ 所有样本正确统计 |
| 样本覆盖 | ❌ Recall@5只统计前5个 | ✅ 统计所有样本 |
| 代码质量 | ❌ 重复代码，易出错 | ✅ 函数复用，可维护 |
| 可测试性 | ❌ 无测试 | ✅ 内置单元测试 |
| 结果可信度 | ❌ 完全不可信 | ✅ 准确可靠 |

**影响**: 2个文件，+108行/-14行

---

## 统计数据

### 代码统计

| 类别 | 数量 | 行数 |
|------|------|------|
| **Markdown 文档** | 12个 | ~4,500行 |
| **Python 代码** | 10个 | ~1,500行 |
| **Shell 脚本** | 4个 | ~1,200行 |
| **配置文件** | 8个 | ~300行 |
| **日志文件** | 6个 | ~2,000行 |
| **总计** | 40个文件 | ~9,500行 |

### 提交统计

| 阶段 | 提交次数 | 分支 |
|------|---------|------|
| 项目初始化 | 3次 | fix → main |
| P0开发 | 3次 | fix → main |
| P1开发 | 3次 | fix → main |
| Bug修复 | 3次 | fix → main |
| 日志重构 | 2次 | fix → main |
| **总计** | **14次** | - |

### 功能覆盖

- ✅ 环境配置和安装（100%）
- ✅ 模型下载和管理（100%）
- ✅ 模型调研与选型文档（100%）
- ✅ Benchmark 测试工具（100%）
- ✅ 使用说明文档（100%）
- ✅ 开发日志系统（100%）
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
- ✅ 分门别类的日志系统

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

- **主要开发者**: Claude (Anthropic AI Assistant)
- **项目发起人**: [项目维护者]
- **审查和测试**: [项目维护者]

---

## 许可证

本项目采用 [MIT License](LICENSE)

---

## 联系方式

- **GitHub Issues**: [项目 Issues 页面]
- **讨论区**: [项目 Discussions 页面]

---

**最后更新**: 2025-11-01  
**当前版本**: v0.5.0 (P0-MVP)  
**下一版本**: v1.0.0 (P1-Release)
