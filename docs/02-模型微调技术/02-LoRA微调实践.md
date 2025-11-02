# 02 - LoRA微调实践

> 📚 **学习目标**  
> - 掌握使用LoRA微调CLIP模型的完整流程
> - 学会准备和处理微调数据
> - 理解训练过程中的关键参数
> - 能够评估和使用微调后的模型

> 🎯 **先修要求**  
> - 完成 [01-微调理论基础](01-微调理论基础.md)
> - 熟悉PyTorch基础操作
> - 了解数据加载和预处理

> ⏱️ **预计学习时间**: 1-2小时（含实践）  
> 🏷️ **难度**: ⭐⭐⭐☆☆ 中级

> ✅ **代码可用性**  
> 本教程的所有示例代码已完整实现，可直接运行：
> - 数据准备脚本: `scripts/prepare_dog_dataset.py`
> - 训练/评估/推理: `code/02-fine-tuning/lora/`
> - 详细使用说明: `code/02-fine-tuning/lora/README.md`

---

## 📖 目录

- [实践概述](#实践概述)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [模型配置](#模型配置)
- [训练过程](#训练过程)
- [模型评估](#模型评估)
- [模型使用](#模型使用)
- [常见问题](#常见问题)
- [学习成果验收](#学习成果验收)

---

## 实践概述

### 本章目标

通过一个完整的实例，学习如何使用LoRA微调CLIP模型，使其在特定领域（如宠物品种识别）上表现更好。

### 实践任务

**任务**：微调CLIP模型进行宠物品种识别

**数据集**：Stanford Dogs Dataset（部分）
- 训练集：1,000张图像（10个犬种）
- 验证集：200张图像
- 测试集：200张图像

**预期效果**：
- 基线（预训练CLIP）：Top-1准确率 ~60%
- 微调后：Top-1准确率 ~85%+

### 代码结构

```
code/02-fine-tuning/lora/
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── inference.py          # 推理脚本
├── dataset.py            # 数据集类
├── config.yaml           # 配置文件
└── README.md             # 使用说明
```

---

## 环境准备

### 1. 安装依赖

```bash
# 基础环境（如果还没安装）
pip install torch torchvision transformers

# LoRA相关
pip install peft

# 训练工具
pip install accelerate
pip install tensorboard

# 数据处理
pip install pillow
pip install scikit-learn
```

### 2. 验证安装

```python
import torch
import transformers
import peft

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**预期输出**:
```
PyTorch: 2.0.0+cu118
Transformers: 4.35.0
PEFT: 0.7.0
CUDA available: True
```

### 3. 硬件要求

**最低配置**:
- GPU: 8GB显存（如RTX 3070）
- 内存: 16GB
- 硬盘: 10GB

**推荐配置**:
- GPU: 16GB+显存（如RTX 4080）
- 内存: 32GB
- 硬盘: 20GB（SSD）

---

## 数据准备

### 1. 数据集下载

我们使用Stanford Dogs Dataset的一个子集作为示例。

**方式1：自动下载（推荐）**

```bash
# 运行数据准备脚本（会自动下载约750MB数据）
python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10
```

**脚本功能**：
- ✅ 自动从官方源下载Stanford Dogs数据集（~750MB）
- ✅ 解压并组织数据集
- ✅ 选择指定数量的犬种（1-120个）
- ✅ 按8:2比例分割训练/测试集
- ✅ 验证数据集完整性

**⏱️ 预计时间**：5-10分钟（取决于网络速度）

**💡 如果下载失败**：
```bash
# 手动下载方案
# 1. 访问 http://vision.stanford.edu/aditya86/ImageNetDogs/
# 2. 下载 images.tar 文件
# 3. 放到 data/dogs/downloads/ 目录
# 4. 运行脚本（跳过下载步骤）
python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10 --no-download
```

**方式2：手动准备**

如果您有自己的数据集，按以下结构组织：

```
data/dogs/
├── train/
│   ├── golden_retriever/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── labrador/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### 2. 数据集类实现

创建 `code/02-fine-tuning/lora/dataset.py`:

```python
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class DogBreedDataset(Dataset):
    """
    犬种分类数据集
    
    Args:
        data_dir: 数据目录路径
        split: 'train', 'val', 或 'test'
        processor: CLIP处理器
        transform: 额外的图像变换（可选）
    """
    def __init__(self, data_dir, split='train', processor=None, transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.processor = processor
        self.transform = transform
        
        # 加载类别和图像路径
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {split} set")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用额外变换
        if self.transform:
            image = self.transform(image)
        
        # 使用CLIP processor处理
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            # 移除batch维度
            pixel_values = inputs['pixel_values'].squeeze(0)
        else:
            pixel_values = image
        
        return {
            'pixel_values': pixel_values,
            'labels': label
        }
```

### 3. 数据加载器

```python
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

# 初始化processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 创建数据集
train_dataset = DogBreedDataset(
    data_dir='data/dogs',
    split='train',
    processor=processor
)

val_dataset = DogBreedDataset(
    data_dir='data/dogs',
    split='val',
    processor=processor
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

---

## 模型配置

### 1. 加载预训练模型

```python
from transformers import CLIPModel, CLIPProcessor
import torch

# 加载预训练模型
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. 配置LoRA

```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=8,                          # LoRA秩（rank）
    lora_alpha=32,                # LoRA缩放因子
    target_modules=[              # 目标模块
        "q_proj",                 # Query投影
        "v_proj",                 # Value投影
    ],
    lora_dropout=0.1,             # Dropout概率
    bias="none",                  # 不训练bias
    task_type="FEATURE_EXTRACTION"  # 任务类型
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
```

**预期输出**:
```
trainable params: 294,912 || all params: 149,620,224 || trainable%: 0.20%
```

### 3. 添加分类头

由于CLIP原本不是分类模型，我们需要添加一个分类头：

```python
import torch.nn as nn

class CLIPClassifier(nn.Module):
    """
    CLIP + 分类头
    """
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(
            clip_model.config.projection_dim,  # CLIP输出维度
            num_classes
        )
    
    def forward(self, pixel_values):
        # 获取图像特征
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        
        # 分类
        logits = self.classifier(image_features)
        return logits

# 创建分类器
num_classes = len(train_dataset.classes)
classifier = CLIPClassifier(model, num_classes).to(device)

print(f"Classifier created for {num_classes} classes")
```

---

## 训练过程

### 1. 训练配置

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

# 优化器
optimizer = AdamW(
    classifier.parameters(),
    lr=5e-4,              # LoRA通常使用较大的学习率
    weight_decay=0.01
)

# 学习率调度器
num_epochs = 10
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * len(train_loader)
)

# 损失函数
criterion = nn.CrossEntropyLoss()
```

### 2. 训练循环

```python
from tqdm import tqdm
import numpy as np

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # 数据移动到GPU
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

### 3. 完整训练脚本

```python
# 训练
best_acc = 0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # 训练
    train_loss, train_acc = train_epoch(
        classifier, train_loader, optimizer, scheduler, criterion, device
    )
    
    # 验证
    val_loss, val_acc = validate(
        classifier, val_loader, criterion, device
    )
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pth')
        print(f"✅ Best model saved! (Val Acc: {val_acc:.2f}%)")

print(f"\n🎉 Training completed! Best Val Acc: {best_acc:.2f}%")
```

### 4. 训练监控（使用TensorBoard）

```python
from torch.utils.tensorboard import SummaryWriter

# 创建writer
writer = SummaryWriter('runs/lora_finetuning')

# 在训练循环中记录
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

# 关闭writer
writer.close()
```

**查看TensorBoard**:
```bash
tensorboard --logdir=runs
```

---

## 模型评估

### 1. 加载最佳模型

```python
# 加载checkpoint
checkpoint = torch.load('best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_acc']:.2f}%")
```

### 2. 测试集评估

```python
# 创建测试数据加载器
test_dataset = DogBreedDataset(
    data_dir='data/dogs',
    split='test',
    processor=processor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# 评估
test_loss, test_acc = validate(classifier, test_loader, criterion, device)
print(f"\n📊 Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
```

### 3. 详细评估指标

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def detailed_evaluation(model, test_loader, class_names, device):
    """详细评估"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(pixel_values)
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 分类报告
    print("\n📈 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    ))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Confusion matrix saved to confusion_matrix.png")

# 运行详细评估
detailed_evaluation(classifier, test_loader, test_dataset.classes, device)
```

---

## 模型使用

### 1. 保存模型

```python
# 保存LoRA权重（推荐）
model.save_pretrained("./lora_weights")

# 或保存完整模型
torch.save(classifier.state_dict(), "classifier_full.pth")
```

### 2. 加载和推理

```python
from PIL import Image

def predict_single_image(image_path, model, processor, class_names, device):
    """
    对单张图像进行预测
    
    Args:
        image_path: 图像路径
        model: 模型
        processor: CLIP处理器
        class_names: 类别名称列表
        device: 设备
    
    Returns:
        predicted_class: 预测类别
        confidence: 置信度
    """
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(1)
    
    predicted_class = class_names[predicted.item()]
    confidence = confidence.item()
    
    return predicted_class, confidence

# 示例使用
image_path = "data/dogs/test/golden_retriever/test_001.jpg"
predicted_class, confidence = predict_single_image(
    image_path, classifier, processor, test_dataset.classes, device
)

print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

### 3. 批量推理

```python
def batch_inference(image_dir, model, processor, class_names, device):
    """批量推理"""
    import glob
    
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    results = []
    
    for img_path in tqdm(image_paths, desc="Inference"):
        pred_class, confidence = predict_single_image(
            img_path, model, processor, class_names, device
        )
        results.append({
            'image': os.path.basename(img_path),
            'prediction': pred_class,
            'confidence': confidence
        })
    
    return results

# 使用
results = batch_inference(
    "data/dogs/test/golden_retriever",
    classifier, processor, test_dataset.classes, device
)

# 保存结果
import json
with open('inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**解决方案**:
1. 减小batch size
2. 使用梯度累积
3. 使用QLoRA（4bit量化）
4. 减小LoRA rank (r)

```python
# 梯度累积示例
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = ...
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q2: 训练过拟合怎么办？

**解决方案**:
1. 增加数据增强
2. 增加Dropout
3. 减少训练轮数
4. 使用Early Stopping

```python
# Early Stopping示例
patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存模型
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

### Q3: 如何调整超参数？

**推荐调整顺序**:
1. 学习率（最重要）：1e-5 ~ 1e-3
2. Batch size：8, 16, 32, 64
3. LoRA rank：4, 8, 16, 32
4. LoRA alpha：通常设为rank的2-4倍

### Q4: 训练需要多长时间？

**参考时间**（1000张图像，10个类别）:
- RTX 3090（24GB）: ~30分钟
- RTX 4080（16GB）: ~40分钟
- V100（16GB）: ~45分钟

---

## 学习成果验收

### 📋 实践检查清单

- [ ] 成功安装所有依赖
- [ ] 准备好训练数据（至少100张图像）
- [ ] 成功配置LoRA并打印可训练参数
- [ ] 完成至少3个epoch的训练
- [ ] 验证集准确率达到合理水平（>60%）
- [ ] 成功保存和加载模型
- [ ] 能对新图像进行推理

### 🎯 进阶挑战

- [ ] 尝试不同的LoRA配置（rank, alpha）
- [ ] 添加数据增强并观察效果
- [ ] 使用TensorBoard监控训练
- [ ] 实现Early Stopping
- [ ] 在自己的数据集上微调

### 📊 预期结果

**训练曲线**:
- 训练损失：稳定下降
- 验证损失：先下降后趋于平稳
- 训练准确率：逐步提升至90%+
- 验证准确率：提升至80-90%

**性能提升**:
- 基线（预训练CLIP）：~60%
- 微调后：~85%+
- 提升：+25个百分点

---

## 下一步

恭喜完成LoRA微调实践！接下来您可以：

1. **学习全参数微调** → [03-全参数微调](03-全参数微调.md)
2. **探索QLoRA** → [04-其他PEFT方法](04-其他PEFT方法.md)
3. **准备部署** → [../04-多平台部署/01-NVIDIA平台部署.md](../04-多平台部署/01-NVIDIA平台部署.md)

---

## 参考资源

### 代码

- 完整训练脚本：`code/02-fine-tuning/lora/train.py`
- 评估脚本：`code/02-fine-tuning/lora/evaluate.py`
- 推理脚本：`code/02-fine-tuning/lora/inference.py`

### 文档

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

---

**📝 文档版本**: v1.0  
**✍️ 最后更新**: 2025-11-01  
**👥 贡献者**: Large-Model-Tutorial Team

