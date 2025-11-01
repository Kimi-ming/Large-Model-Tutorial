# P0-MVP Issues 列表 (v0.5)

> 本文档列出了v0.5 MVP版本需要创建的所有Issues  
> 按阶段和依赖关系排序，建议按顺序执行

## 使用说明

1. 复制每个Issue的内容
2. 在GitHub仓库中创建新Issue
3. 添加对应的Labels和Milestone
4. 按依赖关系执行

---

## 第一阶段：基础框架搭建（必须最先完成）

### Issue #1: 创建项目目录结构
**标签**: P0-MVP, 🔧维护者, 脚本  
**Milestone**: v0.5 MVP  
**依赖**: 无

**描述**:
创建完整的项目目录结构，严格按照设计文档第3.1节的规定。

**交付物**:
- [ ] `docs/` 目录及7个子目录（01-模型调研与选型 至 07-高级主题）
- [ ] `code/` 目录及5个子目录（01-model-evaluation 至 05-applications）
- [ ] `code/utils/` 工具函数目录
- [ ] `notebooks/` 目录
- [ ] `configs/` 目录（models/training/deployment子目录）
- [ ] `scripts/` 目录
- [ ] `tests/` 目录（unit/integration/e2e子目录）
- [ ] `assets/` 目录（images/videos/templates子目录）
- [ ] `docker/` 目录
- [ ] `.github/` 目录（已部分创建）
- [ ] 执行 `scripts/init_project.sh` 脚本

**验收标准**:
- [ ] 目录结构100%符合设计文档第3章
- [ ] 所有目录使用中划线命名（如 `01-model-evaluation/`）
- [ ] 目录结构可以通过脚本自动创建

**优先级**: ⭐⭐⭐⭐⭐ 最高（阻塞其他任务）

---

### Issue #2: 编写基础依赖文件
**标签**: P0-MVP, 📚教程必需, 文档  
**Milestone**: v0.5 MVP  
**依赖**: #1

**描述**:
创建Python依赖管理文件和基础配置文件。

**交付物**:
- [ ] `requirements.txt` - 生产环境依赖（版本锁定）
- [ ] `requirements-dev.txt` - 开发环境依赖
- [ ] `environment.yml` - Conda环境配置
- [ ] `setup.py` - 包安装配置
- [ ] `pyproject.toml` - 项目配置
- [ ] `.gitignore` - Git忽略文件
- [ ] `.pre-commit-config.yaml` - 代码检查钩子

**验收标准**:
- [ ] 依赖文件包含所有必需的包（PyTorch, transformers, numpy等）
- [ ] 版本号明确锁定
- [ ] `.gitignore` 包含常见的Python/ML项目忽略项

---

### Issue #3: 开发环境安装脚本
**标签**: P0-MVP, 📚教程必需, 脚本  
**Milestone**: v0.5 MVP  
**依赖**: #2

**描述**:
开发 `scripts/setup.sh` 环境安装脚本，帮助学习者快速搭建环境。

**交付物**:
- [ ] `scripts/setup.sh` - Linux/Mac环境安装脚本
- [ ] Python环境检查功能
- [ ] GPU驱动检查功能
- [ ] 依赖自动安装
- [ ] 环境验证功能
- [ ] 错误处理和友好提示

**验收标准**:
- [ ] 能在干净的Linux/Mac环境中一键完成环境搭建
- [ ] 脚本包含详细的日志输出
- [ ] 遇到错误时有明确的提示信息

---

### Issue #4: 模型下载脚本
**标签**: P0-MVP, 📚教程必需, 脚本  
**Milestone**: v0.5 MVP  
**依赖**: #2

**描述**:
开发 `scripts/download_models.sh` 模型下载脚本。

**交付物**:
- [ ] `scripts/download_models.sh` - 模型下载脚本
- [ ] 支持从HuggingFace Hub下载
- [ ] 支持断点续传
- [ ] 支持镜像源配置（国内加速）
- [ ] 下载进度显示
- [ ] 支持下载CLIP、SAM、LLaVA三个模型

**验收标准**:
- [ ] 能成功下载至少3个常用模型（CLIP、SAM、LLaVA）
- [ ] 下载失败时能自动重试
- [ ] 有清晰的进度提示

---

### Issue #5: 基础工具函数库
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #1, #2

**描述**:
开发 `code/utils/` 基础工具函数库。

**交付物**:
- [ ] `code/utils/model_loader.py` - 模型加载工具
- [ ] `code/utils/data_processor.py` - 数据处理工具
- [ ] `code/utils/config_parser.py` - 配置解析工具
- [ ] `code/utils/logger.py` - 日志工具
- [ ] `code/utils/__init__.py` - 包初始化
- [ ] 每个模块包含docstring和类型注解
- [ ] 简单的单元测试

**验收标准**:
- [ ] 工具函数能正常导入和使用
- [ ] 代码符合PEP 8规范
- [ ] 基础测试通过

---

### Issue #6: 快速开始文档
**标签**: P0-MVP, 📚教程必需, 文档  
**Milestone**: v0.5 MVP  
**依赖**: #3, #4

**描述**:
编写 `docs/05-使用说明/01-环境安装指南.md` 和 `02-快速开始.md`。

**交付物**:
- [ ] `docs/05-使用说明/01-环境安装指南.md`
  - 系统要求
  - 安装步骤
  - 常见问题
- [ ] `docs/05-使用说明/02-快速开始.md`
  - 第一个示例：运行预训练模型推理
  - 环境验证
  - 后续学习路径指引

**验收标准**:
- [ ] 文档清晰易懂
- [ ] 步骤可复现
- [ ] 包含学习目标和先修要求

---

### Issue #7: 更新README.md
**标签**: P0-MVP, 📚教程必需, 文档  
**Milestone**: v0.5 MVP  
**依赖**: #6

**描述**:
更新项目主页README.md，提供完整的项目介绍。

**交付物**:
- [ ] 教程简介
- [ ] 目标读者
- [ ] 学习路径图
- [ ] 快速开始指南
- [ ] 目录结构说明
- [ ] 贡献指南链接
- [ ] Badges（MIT License等）

**验收标准**:
- [ ] README内容完整、专业
- [ ] 有吸引力，让学习者想要深入学习
- [ ] 链接都有效

---

## 第二阶段：核心教程内容（MVP精简版）

### Issue #8: CLIP模型推理示例
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #5, #4

**描述**:
开发CLIP模型推理示例代码和文档。

**交付物**:
- [ ] `code/01-model-evaluation/examples/clip_inference.py`
  - 图文匹配示例
  - 详细注释
  - 使用说明
- [ ] `configs/models/clip.yaml` - 配置文件
- [ ] 对应文档片段（可集成到主文档）

**验收标准**:
- [ ] 代码能在Colab上运行
- [ ] 包含完整的使用示例
- [ ] 输出结果清晰

---

### Issue #9: SAM模型推理示例
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #5, #4

**描述**:
开发SAM模型推理示例代码。

**交付物**:
- [ ] `code/01-model-evaluation/examples/sam_inference.py`
  - 图像分割示例
  - 详细注释
- [ ] `configs/models/sam.yaml`

**验收标准**:
- [ ] 代码能运行
- [ ] 分割效果可视化

---

### Issue #10: LLaVA模型推理示例
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #5, #4

**描述**:
开发LLaVA多模态对话示例。

**交付物**:
- [ ] `code/01-model-evaluation/examples/llava_inference.py`
  - 多模态对话示例
- [ ] `configs/models/llava.yaml`

**验收标准**:
- [ ] 代码能运行
- [ ] 对话流畅

---

### Issue #11: 快速开始Notebook
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #8

**描述**:
创建 `notebooks/01-quick-start.ipynb` 快速开始教程。

**交付物**:
- [ ] `notebooks/01-quick-start.ipynb`
  - 环境检查
  - 快速运行CLIP示例
  - 可在Colab上运行
  - 清除输出

**验收标准**:
- [ ] Notebook能完整执行
- [ ] 在Colab上测试通过
- [ ] 包含详细说明

---

### Issue #12: LoRA微调示例代码
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #5

**描述**:
开发LoRA微调完整示例（P0核心）。

**交付物**:
- [ ] `code/02-fine-tuning/lora/train.py` - 训练脚本
- [ ] `code/02-fine-tuning/lora/inference.py` - 推理脚本
- [ ] `code/02-fine-tuning/lora/README.md` - 使用说明
- [ ] `configs/training/lora.yaml` - 配置文件
- [ ] 简单的训练日志示例

**验收标准**:
- [ ] 代码能完整运行
- [ ] 提供可复现的训练日志
- [ ] 验证集效果相比基线有实质提升
- [ ] 提供参考结果区间说明

---

### Issue #13: NVIDIA基础部署示例
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #5

**描述**:
开发NVIDIA平台PyTorch基础部署示例。

**交付物**:
- [ ] `code/04-deployment/nvidia/basic/deploy.py`
- [ ] `code/04-deployment/nvidia/basic/inference.py`
- [ ] `code/04-deployment/nvidia/basic/README.md`
- [ ] `configs/deployment/nvidia.yaml`

**验收标准**:
- [ ] NVIDIA基础部署代码能运行
- [ ] 提供性能测试参考数据

---

### Issue #14: NVIDIA ONNX部署示例
**标签**: P0-MVP, 📚教程必需, 代码  
**Milestone**: v0.5 MVP  
**依赖**: #13

**描述**:
开发NVIDIA平台ONNX部署示例。

**交付物**:
- [ ] `code/04-deployment/nvidia/onnx/export_onnx.py`
- [ ] `code/04-deployment/nvidia/onnx/onnx_inference.py`
- [ ] `code/04-deployment/nvidia/onnx/README.md`

**验收标准**:
- [ ] ONNX导出和推理能正常工作
- [ ] 提供性能对比数据

---

## 第一阶段验收检查点

完成以上所有Issue后，进行MVP验收：

### Issue #15: MVP验收测试
**标签**: P0-MVP, 🔧维护者, 测试  
**Milestone**: v0.5 MVP  
**依赖**: #1-#14

**描述**:
对v0.5 MVP进行全面验收测试。

**验收清单**:
- [ ] 目录结构100%符合设计文档第3章
- [ ] 学习者能在干净环境中完成环境搭建
- [ ] `scripts/setup.sh` 能成功执行
- [ ] `scripts/download_models.sh` 能成功下载3个模型
- [ ] 工具函数能正常导入和使用
- [ ] 至少3个模型推理代码能运行
- [ ] `notebooks/01-quick-start.ipynb` 能完整执行
- [ ] LoRA微调代码能完整运行
- [ ] NVIDIA基础部署和ONNX部署能运行
- [ ] 所有文档链接有效
- [ ] README.md 内容完整

**测试方法**:
1. 在干净的Ubuntu 20.04环境中测试
2. 严格按照文档步骤操作
3. 记录所有问题和改进建议
4. 至少1位外部测试用户验证

**交付物**:
- [ ] 测试报告
- [ ] 问题列表（如有）
- [ ] 改进建议

---

## 创建Issues的步骤

### 方法1：手动创建（推荐用于前几个Issue）
1. 进入GitHub仓库
2. 点击 "Issues" → "New issue"
3. 选择 "📋 任务开发" 模板
4. 复制对应Issue的内容
5. 添加Labels: `P0-MVP`, `📚教程必需`或`🔧维护者`, 类型标签
6. 设置Milestone: `v0.5 MVP`
7. 提交

### 方法2：使用GitHub CLI批量创建
```bash
# 示例：创建Issue #1
gh issue create \
  --title "[P0] 创建项目目录结构" \
  --body "$(cat issue-01-content.md)" \
  --label "P0-MVP,🔧维护者,脚本" \
  --milestone "v0.5 MVP"
```

### 方法3：使用脚本批量导入
创建Python脚本读取本文件并批量创建Issues（可选）

---

## 注意事项

1. **严格按依赖关系执行**：Issue #1必须最先完成
2. **优先级顺序**：P0 > P1 > P2 > P3
3. **验收标准**：每个Issue完成后必须满足验收标准
4. **文档同步**：代码和文档同步开发
5. **质量优先**：宁可慢一点，也要保证质量

---

**文档版本**: v1.0  
**创建日期**: 2025-11-01  
**对应**: TODO清单v2.1 - P0任务

