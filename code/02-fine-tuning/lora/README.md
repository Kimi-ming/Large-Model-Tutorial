# LoRA微调示例代码

使用LoRA（Low-Rank Adaptation）方法微调CLIP模型进行犬种分类。

## 📁 文件结构

```
code/02-fine-tuning/lora/
├── __init__.py           # 包初始化
├── config.yaml           # 配置文件
├── dataset.py            # 数据集类
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── inference.py          # 推理脚本
└── README.md             # 本文档
```

## 🚀 快速开始

### 1. 准备数据集

```bash
# 创建数据集目录结构
python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10

# 手动添加图像到对应目录
# data/dogs/train/golden_retriever/*.jpg
# data/dogs/train/labrador/*.jpg
# ...
```

**数据集结构**:
```
data/dogs/
├── train/          # 训练集
│   ├── golden_retriever/
│   ├── labrador/
│   └── ...
├── val/            # 验证集
└── test/           # 测试集
```

### 2. 训练模型

```bash
# 使用默认配置训练
python code/02-fine-tuning/lora/train.py

# 使用自定义配置
python code/02-fine-tuning/lora/train.py \
    --config code/02-fine-tuning/lora/config.yaml \
    --data_dir data/dogs \
    --output_dir outputs/my_model
```

**训练监控**:
```bash
# 启动TensorBoard
tensorboard --logdir logs/lora_finetuning
```

### 3. 评估模型

```bash
# 评估测试集
python code/02-fine-tuning/lora/evaluate.py \
    --checkpoint outputs/lora_finetuning/checkpoint-epoch-10 \
    --data_dir data/dogs \
    --split test \
    --output_dir outputs/evaluation
```

**评估输出**:
- `evaluation_report.txt` - 详细文本报告
- `evaluation_results.json` - JSON格式结果
- `confusion_matrix.png` - 混淆矩阵可视化
- `class_performance.png` - 各类别性能图

### 4. 推理预测

**单张图像**:
```bash
python code/02-fine-tuning/lora/inference.py \
    --checkpoint outputs/lora_finetuning/checkpoint-epoch-10 \
    --image path/to/dog.jpg \
    --top_k 5
```

**批量推理**:
```bash
python code/02-fine-tuning/lora/inference.py \
    --checkpoint outputs/lora_finetuning/checkpoint-epoch-10 \
    --image_dir path/to/images/ \
    --output predictions.txt
```

## ⚙️ 配置说明

配置文件 `config.yaml` 包含以下主要部分：

### 模型配置
```yaml
model:
  name: "openai/clip-vit-base-patch32"
  cache_dir: "models/"
```

### LoRA配置
```yaml
lora:
  r: 8                    # LoRA秩（越大参数越多）
  lora_alpha: 32          # 缩放系数
  target_modules:         # 应用LoRA的模块
    - "q_proj"
    - "v_proj"
  lora_dropout: 0.1       # Dropout率
```

**参数调优建议**:
- `r`: 通常 4-16，越大效果越好但参数越多
- `lora_alpha`: 通常设为 `r * 2` 或 `r * 4`
- `target_modules`: 可选 `["q_proj", "v_proj", "k_proj", "out_proj"]`

### 训练配置
```yaml
training:
  num_epochs: 10
  learning_rate: 5.0e-4
  batch_size: 32
  warmup_ratio: 0.1
  early_stopping:
    enabled: true
    patience: 3
```

**超参数建议**:
- 学习率: LoRA通常比全参数微调高 (1e-4 ~ 5e-4)
- Batch size: 根据显存调整 (8GB显存→16-32)
- Warmup: 建议 10% 的训练步数

## 📊 性能基准

### 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | 8GB (RTX 3070) | 16GB+ (RTX 4080) |
| 内存 | 16GB | 32GB |
| 硬盘 | 10GB | 20GB (SSD) |

### 训练时间估算

| 数据集大小 | GPU | 训练时间 (10 epochs) |
|-----------|-----|---------------------|
| 1K 图像 | RTX 3070 | ~15分钟 |
| 5K 图像 | RTX 3070 | ~1小时 |
| 10K 图像 | RTX 4080 | ~1.5小时 |

### 预期效果

| 指标 | 预训练CLIP | LoRA微调 |
|------|-----------|---------|
| Top-1准确率 | ~60% | ~85%+ |
| Top-5准确率 | ~85% | ~95%+ |
| 可训练参数 | 100% | <1% |

## 🔧 常见问题

### 1. CUDA内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 减小batch_size
data:
  batch_size: 16  # 或 8

# 启用梯度累积
training:
  gradient_accumulation_steps: 2
```

### 2. 数据集加载失败

**问题**: `ValueError: 在目录中未找到任何图像文件`

**解决方案**:
```bash
# 验证数据集结构
python scripts/prepare_dog_dataset.py --output_dir data/dogs --validate

# 确保图像格式正确（JPG/PNG）
# 确保目录结构符合要求
```

### 3. 训练不收敛

**问题**: 验证准确率不提升

**解决方案**:
- 检查学习率（尝试 1e-4 ~ 1e-3）
- 增加训练轮数
- 检查数据质量和标注
- 尝试增大 LoRA rank (`r: 16`)

### 4. 推理速度慢

**问题**: 推理时间过长

**解决方案**:
```python
# 使用批量推理
predictor.predict_batch(images, top_k=5)

# 使用混合精度
# 在config.yaml中启用
hardware:
  mixed_precision: true
```

## 📚 相关文档

- [LoRA微调实践教程](../../../docs/02-模型微调技术/02-LoRA微调实践.md)
- [微调理论基础](../../../docs/02-模型微调技术/01-微调理论基础.md)
- [PEFT库文档](https://huggingface.co/docs/peft)

## 🤝 贡献

如果您发现问题或有改进建议，欢迎提交Issue或Pull Request。

## 📄 许可证

本项目遵循MIT许可证。

