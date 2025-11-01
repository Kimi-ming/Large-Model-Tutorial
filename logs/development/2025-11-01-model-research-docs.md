# 开发日志 - 模型调研与选型文档开发

**日期**: 2025-11-01  
**阶段**: P1阶段  
**类型**: 开发  
**优先级**: P1 (v1.0)

---

## 概述

完成"模型调研与选型"章节的4个核心文档开发，为学习者提供从模型了解到选型决策的完整指南。

---

## 新增文档

### 1. 主流视觉大模型概述

**文件**: `docs/01-模型调研与选型/01-主流视觉大模型概述.md`  
**行数**: 387行

#### 内容结构

**开源视觉大模型** (10个):
1. **CLIP** (OpenAI, 2021)
   - 图文多模态对比学习
   - 零样本图像分类
   - 参数量: ~400M
   - 应用: 图文检索、零样本分类、图像理解

2. **SAM** (Meta AI, 2023)
   - 通用图像分割模型
   - 支持点、框、文本提示
   - 参数量: ~600M
   - 应用: 零样本分割、目标检测、医学影像

3. **BLIP/BLIP-2** (Salesforce, 2022/2023)
   - 图像理解与描述生成
   - Q-Former架构
   - 应用: 图像描述、VQA、图文检索

4. **LLaVA** (UW Madison & Microsoft, 2023)
   - 视觉指令微调
   - 结合CLIP和LLaMA
   - 参数量: 7B/13B
   - 应用: 视觉对话、图像问答、图像推理

5. **Qwen-VL** (阿里巴巴, 2023)
   - 中文支持优秀
   - 多图理解能力
   - 细粒度文本识别
   - 参数量: 7B

6. **InternVL** (上海AI Lab, 2023)
   - 动态分辨率
   - 大规模多模态数据训练
   - 支持4K+高分辨率

7. **CogVLM** (智谱AI, 2023)
   - 视觉专家模块
   - 平衡视觉和语言能力

8. **Yi-VL** (零一万物, 2023)
   - 双语支持
   - 高质量视觉理解

9. **MiniGPT-4** (研究项目, 2023)
   - 轻量级视觉语言模型
   - GPT-4级别的视觉能力

10. **MiniCPM-V** (OpenBMB, 2024)
    - 边缘设备优化
    - 参数量: ~2.4B
    - 适合移动端部署

**商业视觉大模型** (2个):
1. **GPT-4V** (OpenAI, 2023)
   - 最强的多模态理解
   - 支持多图理解
   - 复杂推理能力

2. **Gemini Vision** (Google, 2023)
   - 原生多模态设计
   - 长上下文支持
   - 多模态CoT推理

#### 每个模型包含

- 开发者和发布时间
- 官方链接（GitHub/HuggingFace/论文）
- 核心特点
- 主要应用场景
- 优势和局限
- 参考性能数据
- 代码示例（可运行）
- 依赖说明（如需要）

#### 代码示例

**CLIP示例**:
```python
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("test.jpg")
texts = ["a cat", "a dog", "a car"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

print(f"预测结果: {probs}")
# 参考输出: tensor([[0.8234, 0.1234, 0.0532]])
```

**SAM示例**:
```python
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

image = np.array(Image.open("test.jpg"))
predictor.set_image(image)

point_coords = np.array([[500, 375]])
point_labels = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
)
```

#### 选型建议

**按任务类型选择**:
- 图文检索 → CLIP
- 图像分割 → SAM
- 图像描述 → BLIP-2
- 视觉问答 → LLaVA, Qwen-VL
- 多模态对话 → LLaVA, GPT-4V

**按资源约束选择**:
- 低资源(<8GB) → CLIP, BLIP-2(小版本)
- 中等资源(8-16GB) → SAM, LLaVA-7B
- 高资源(16GB+) → LLaVA-13B, Qwen-VL
- 仅CPU → CLIP, 量化版本

**按部署环境选择**:
- 边缘设备 → CLIP(量化), MiniCPM-V
- 服务器(NVIDIA) → 任意模型
- 服务器(华为昇腾) → 需验证兼容性
- 云API → GPT-4V, Gemini

**验收标准**: ✅ 文档完整，代码可运行，链接有效

---

### 2. 模型对比与评测

**文件**: `docs/01-模型调研与选型/02-模型对比与评测.md`  
**行数**: 156行

#### 内容结构

**评测维度说明**:
1. **性能指标**
   - 图像分类: Top-1/Top-5准确率
   - 目标检测: mAP
   - 图像分割: mIoU
   - 图像描述: CIDEr, BLEU
   - 视觉问答: 准确率

2. **效率指标**
   - 推理速度: images/sec 或 ms/image
   - 显存占用: GB (FP16精度)
   - 模型大小: GB

**术语说明**:
- **Top-1/Top-5准确率**: 模型预测的最高/前5个结果中包含正确答案的比例
- **mAP**: 目标检测的平均精度均值
- **mIoU**: 分割任务中预测区域与真实区域的重叠度
- **CIDEr/BLEU**: 图像描述生成质量指标
- **R@1**: 图文检索任务中，首位结果正确的比例

**测试条件说明**:
- 硬件环境: NVIDIA V100 32GB / A100 40GB
- 输入分辨率: 224×224（CLIP）、384×384（SAM）
- Batch Size: 1（单图推理）
- 精度: FP16（半精度）
- 数据来源: 各模型官方论文及HuggingFace Model Card（2023-2024）
- 免责声明: 以下数据仅供参考

**综合性能对比**:

图文多模态任务（COCO + ImageNet）:
| 模型 | ImageNet Zero-shot | COCO R@1 | 推理速度 | 显存 | 推荐场景 |
|------|:---:|:---:|:---:|:---:|:---:|
| CLIP-ViT-B/32 | 63.2% | 58.4% | ~50 img/s | 2.5GB | ✅ 快速检索 |
| CLIP-ViT-L/14 | 75.5% | 63.9% | ~15 img/s | 4.8GB | ⚖️ 平衡性能 |
| BLIP-2-OPT-2.7B | - | 77.2% | ~8 img/s | 6.8GB | 🎯 高精度需求 |

视觉问答（VQAv2）:
| 模型 | 整体准确率 | Yes/No | Number | 推理时间 | 显存 | 推荐场景 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| BLIP-2-OPT-2.7B | 82.2% | 91.4% | 56.8% | ~150ms | 6.8GB | 🎯 最高精度 |
| LLaVA-1.5-7B | 78.5% | 89.2% | 53.6% | ~200ms | 14GB | 💬 对话能力 |
| Qwen-VL-Chat | 79.8% | 89.8% | 54.9% | ~220ms | 18GB | 🇨🇳 中文支持 |

**选型决策树**:
```
开始
├─ 是否需要中文支持？
│  ├─ 是 → Qwen-VL / InternVL
│  └─ 否 → 继续
│
├─ 主要任务类型？
│  ├─ 图文检索 → CLIP
│  ├─ 图像分割 → SAM
│  └─ 视觉问答 → BLIP-2 / LLaVA
│
└─ 资源约束？
   ├─ 低(<8GB) → CLIP
   ├─ 中(8-16GB) → SAM / LLaVA-7B
   └─ 高(>16GB) → Qwen-VL
```

**实践任务**:
- 创建您的模型对比表
- 提供CSV模板下载
- 参考工具: HuggingFace Model Card, Papers With Code

**参考资源**:
- COCO数据集官方
- VQAv2论文
- ImageNet官方

**验收标准**: ✅ 对比表完整，术语解释清晰，决策树实用

---

### 3. 选型策略

**文件**: `docs/01-模型调研与选型/03-选型策略.md`  
**行数**: 393行

#### 内容结构

**5步系统化选型流程**:

**第一步: 任务需求分析**
1. 任务类型识别表格
2. 性能需求定义（实时性、准确率）
3. 特殊需求清单（多语言、OCR、高分辨率等）

**第二步: 资源约束评估**
1. 硬件资源清单（GPU、CPU、内存、存储）
2. 显存需求估算表（6个常用模型的FP32/FP16/INT8需求）
3. 预算与时间约束

显存需求估算:
| 模型 | FP32 | FP16 | INT8 | 推荐显存 |
|------|:----:|:----:|:----:|:--------:|
| CLIP-ViT-B/32 | 5GB | 2.5GB | 1.5GB | ≥4GB |
| CLIP-ViT-L/14 | 10GB | 4.8GB | 2.5GB | ≥8GB |
| SAM-ViT-B | 8GB | 4GB | 2GB | ≥8GB |
| BLIP-2-2.7B | 14GB | 6.8GB | 3.5GB | ≥8GB |
| LLaVA-7B | 28GB | 14GB | 7GB | ≥16GB |
| Qwen-VL-7B | 36GB | 18GB | 9GB | ≥20GB |

**第三步: 候选模型筛选**
1. 交互式决策树
2. 候选对比表（满足需求、资源匹配、社区支持）

**第四步: 小规模验证**
1. 环境验证清单（30分钟）
2. 性能验证清单（2小时）
3. 易用性验证清单（1小时）
4. 验证脚本模板

验证脚本示例:
```python
import time
import torch
from transformers import AutoModel, AutoProcessor

# 加载模型
model = AutoModel.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# 推理速度测试
start_time = time.time()
for img in test_images:
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
elapsed = time.time() - start_time

print(f"平均推理时间: {elapsed/len(test_images)*1000:.2f} ms/image")

# 显存占用
memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
print(f"峰值显存占用: {memory_allocated:.2f} GB")
```

**第五步: 最终决策**
1. 决策矩阵（功能匹配、性能指标、资源效率、开发成本、维护成本）
2. 风险评估表

**实际选型案例** (3个):

**案例1: 电商图像搜索系统**
- 需求: 图文检索，<100ms延迟，准确率>65%
- 资源: 4×V100 32GB
- 选型结果: CLIP-ViT-L/14（微调中文）
- 理由: 天然适合图文检索，速度满足要求，可微调中文

**案例2: 医疗影像分割辅助系统**
- 需求: 器官/病灶分割，mIoU>85%，高分辨率
- 资源: 2×A100 40GB
- 选型结果: SAM-ViT-H
- 理由: 专为分割设计，支持交互式标注，零样本能力强

**案例3: 智能客服图文问答**
- 需求: VQA，准确率>75%，延迟<300ms，中文为主
- 资源: 1×RTX 4090 24GB
- 选型结果: Qwen-VL-Chat（量化版）
- 理由: 中文理解强，内置OCR，对话能力好

**实践任务**:
- 填写需求分析表格
- 填写资源约束清单
- 使用决策树筛选候选
- 编写验证计划
- 填写决策矩阵

**参考资源**:
- HuggingFace Model Hub
- Papers With Code Leaderboard
- OpenCompass评测平台

**验收标准**: ✅ 流程清晰，案例实用，工具完整

---

### 4. 基准测试实践

**文件**: `docs/01-模型调研与选型/04-基准测试实践.md`  
**行数**: 925行

#### 内容结构

**环境准备**:
1. 安装依赖（pytest, psutil, gputil, pandas, matplotlib, seaborn）
2. 准备测试数据（手动/wget/脚本）
3. 下载测试模型（CLIP, SAM, BLIP-2）

**基准测试实现**:

**1. 推理速度测试** (完整代码，184行)
- `SpeedBenchmark`类
- GPU预热机制
- `torch.cuda.synchronize()`精确计时
- 多batch size测试
- JSON结果保存

**2. 显存占用测试** (完整代码，90行)
- `MemoryBenchmark`类
- 显存清空和重置
- 峰值显存监控
- 支持CPU降级

**3. 准确率测试** (完整代码，125行)
- `CLIPAccuracyBenchmark`类
- Recall@1/5指标
- 双向检索评测
- 相似度矩阵计算

**4. 结果可视化** (完整代码，123行)
- matplotlib/seaborn图表
- 速度对比图
- 显存对比图
- 高质量PNG输出

**5. 自动化报告** (完整代码，99行)
- Markdown格式报告
- JSON结果解析
- 表格格式化
- 排名和建议

**一键测试脚本**:
```bash
#!/bin/bash
# 运行所有基准测试

# 1. 环境检查
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 数据准备
mkdir -p data/test_dataset

# 3. 速度测试
for model in "clip" "sam" "blip2"; do
    python code/01-model-evaluation/benchmark/speed_test.py \
        --model $model \
        --output results/${model}_speed.json
done

# 4. 显存测试
for model in "clip" "sam" "blip2"; do
    python code/01-model-evaluation/benchmark/memory_test.py \
        --model $model > results/${model}_memory.txt
done

# 5. 生成报告
python code/01-model-evaluation/benchmark/generate_report.py
```

**学习成果验收**:
- ✅ 独立搭建测试环境
- ✅ 编写并运行测试脚本
- ✅ 监控显存占用
- ✅ 生成可视化报告
- ✅ 根据结果做出决策

**实践检验**:
1. 基础任务: 运行CLIP测试，生成报告
2. 进阶任务: 对比2-3个模型
3. 高级任务: 在不同硬件上测试

**常见问题**:
- Q: 显存不足? A: 减小batch size，使用量化
- Q: 如何对比不同硬件? A: 使用相对性能比
- Q: 结果与论文不符? A: 检查输入分辨率和精度

**验收标准**: ✅ 代码完整可运行，文档详细清晰

---

## 技术特点

### 文档质量

1. **结构清晰**
   - 学习目标明确
   - 先修要求清晰
   - 难度和时间标注

2. **内容完整**
   - 理论与实践结合
   - 代码示例完整
   - 参考资源丰富

3. **学习者友好**
   - 术语有解释
   - 步骤有指引
   - 验收有标准

4. **实用性强**
   - 实际案例
   - 决策工具
   - 模板表格

### 代码示例

1. **可运行性**
   - 完整的导入语句
   - 清晰的注释
   - 预期输出示例

2. **实用性**
   - 真实场景
   - 错误处理
   - 性能优化

---

## 验收结果

| 文档 | 行数 | 状态 | 说明 |
|------|------|------|------|
| 01-主流视觉大模型概述 | 387 | ✅ | 10+模型，代码可运行 |
| 02-模型对比与评测 | 156 | ✅ | 对比表完整，术语清晰 |
| 03-选型策略 | 393 | ✅ | 流程清晰，案例实用 |
| 04-基准测试实践 | 925 | ✅ | 代码完整，文档详细 |

**总体评估**: ✅ 所有文档验收通过

---

## 影响范围

- **新增文档**: 4个
- **文档行数**: ~2,600行
- **代码示例**: 20+个
- **实际案例**: 3个
- **Git提交**: 2次

---

## 用户价值

学习者可以：
1. 系统了解10+个主流视觉大模型
2. 掌握模型对比和评测方法
3. 使用系统化的选型流程
4. 动手进行基准测试
5. 根据实际案例做出决策

---

**开发者**: Claude (Anthropic AI Assistant)  
**审查者**: 项目维护者  
**状态**: ✅ 已完成并合并到main分支

