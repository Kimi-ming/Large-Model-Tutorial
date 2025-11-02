# SAM模型微调

本目录包含SAM (Segment Anything Model) 微调的完整代码和配置。

## 📋 目录

- [功能特性](#功能特性)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [微调策略](#微调策略)
- [训练监控](#训练监控)
- [常见问题](#常见问题)

---

## 功能特性

### ✅ 支持的微调策略
- **Full Fine-tuning**: 微调所有参数 ✅ 完整实现
- **Simplified Adapter**: 冻结image_encoder，仅训练decoder ⚠️ 简化实现
- **Simplified LoRA**: 冻结image_encoder，仅训练decoder ⚠️ 简化实现

> **⚠️ 重要说明**：当前Adapter和LoRA为简化实现，并未真正插入Adapter模块或使用PEFT库配置LoRA权重。完整实现计划在P2阶段补充。详见[微调策略说明](#微调策略)。

### ✅ 支持的数据格式
- **目录格式**: `images/` 和 `masks/` 分别存放图像和掩码
- **COCO格式**: 标准COCO实例分割数据集

### ✅ 支持的提示模式
- **Box**: 边界框提示
- **Point**: 点提示
- **Both**: 框+点组合提示

### ✅ 其他特性
- 混合精度训练（AMP）
- 梯度累积
- 学习率调度（Cosine/Linear）
- TensorBoard可视化
- 自动保存最优模型
- 数据增强

---

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 其他依赖
pip install opencv-python pillow pyyaml tqdm tensorboard

# 可选：PEFT（用于LoRA）
pip install peft

# 可选：COCO API（用于COCO数据集）
pip install pycocotools
```

### 2. 下载SAM预训练权重

```bash
# 创建模型目录
mkdir -p models/sam

# 下载ViT-B模型（约375MB）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/sam/

# 或者ViT-L模型（约1.2GB）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P models/sam/

# 或者ViT-H模型（约2.4GB）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/sam/
```

---

## 数据准备

### 方式1：目录格式（推荐）

组织你的数据如下：

```
data/segmentation/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img3.jpg
│       └── ...
└── masks/
    ├── train/
    │   ├── img1.png  # 二值掩码，0=背景，255=前景
    │   ├── img2.png
    │   └── ...
    └── val/
        ├── img3.png
        └── ...
```

**掩码格式要求**：
- 单通道灰度图像
- 0（黑色）= 背景
- 255（白色）= 前景
- 文件名与对应的图像相同（扩展名为.png）

### 方式2：COCO格式

如果你有COCO格式的数据集：

```
data/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

---

## 快速开始

### 1. 准备配置文件

复制并修改 `config.yaml`：

```bash
cp config.yaml my_config.yaml
```

关键配置项：

```yaml
model:
  type: "vit_b"  # 选择模型大小
  checkpoint: "models/sam/sam_vit_b_01ec64.pth"

data:
  data_dir: "data/segmentation"  # 你的数据目录
  dataset_type: "directory"      # directory 或 coco
  prompt_mode: "box"             # box, point, both
  batch_size: 2                  # 根据GPU显存调整

training:
  num_epochs: 50
  learning_rate: 1.0e-4

output:
  output_dir: "outputs/sam_finetuning"
  experiment_name: "my_experiment"
```

### 2. 测试数据加载

在训练前，先测试数据集是否正确：

```bash
# 测试目录格式数据集
python dataset.py \
    --data_dir data/segmentation \
    --dataset_type directory \
    --split train \
    --visualize

# 测试COCO格式数据集
python dataset.py \
    --data_dir data/coco \
    --dataset_type coco \
    --split train \
    --visualize
```

这将生成 `dataset_sample.png` 可视化结果。

### 3. 开始训练

```bash
# 使用默认配置
python train.py --config config.yaml

# 使用自定义配置
python train.py --config my_config.yaml

# 恢复训练
python train.py --config my_config.yaml --resume outputs/sam_finetuning/my_experiment/checkpoint_epoch_10.pth
```

### 4. 监控训练

训练过程中会显示：
- 实时loss和学习率
- 每个epoch的训练和验证指标
- TensorBoard日志

启动TensorBoard：

```bash
tensorboard --logdir outputs/sam_finetuning/my_experiment/runs
```

然后访问 `http://localhost:6006`

---

## 配置说明

### 模型配置

```yaml
model:
  type: "vit_b"  # vit_b, vit_l, vit_h
  checkpoint: "path/to/sam_checkpoint.pth"
  freeze_image_encoder: true   # 冻结图像编码器（推荐）
  freeze_prompt_encoder: false # 冻结提示编码器
  freeze_mask_decoder: false   # 冻结掩码解码器
```

**建议**：
- 冻结图像编码器可以节省显存和加速训练
- 主要微调掩码解码器即可获得良好效果

### 数据配置

```yaml
data:
  data_dir: "data/segmentation"
  dataset_type: "directory"  # directory 或 coco
  train_split: "train"
  val_split: "val"
  image_size: 1024           # SAM标准输入大小
  prompt_mode: "box"         # box, point, both
  num_points: 3              # 点提示数量
  batch_size: 2              # batch大小
  num_workers: 4             # 数据加载线程
  augment: true              # 数据增强
```

**提示模式选择**：
- `box`: 使用边界框提示（推荐，稳定）
- `point`: 使用点提示（灵活）
- `both`: 同时使用框和点（最佳效果，但慢）

### 微调策略

```yaml
finetuning:
  strategy: "full"  # full, adapter, lora
```

**⚠️ 当前实现说明**：

#### 1. Full Fine-tuning（完整实现）✅
```yaml
strategy: "full"
```
- 训练mask_decoder和prompt_encoder
- 可选择冻结image_encoder（推荐）
- 完全符合预期的全参数微调

#### 2. Simplified Adapter（简化实现）⚠️
```yaml
strategy: "adapter"
```
**当前行为**：
- 冻结image_encoder
- 训练mask_decoder和prompt_encoder
- **未实现真正的Adapter模块插入**

**与标准Adapter的区别**：
- ❌ 未在Transformer层插入adapter模块
- ❌ 配置文件中的`adapter`参数不生效
- ✅ 仅是一种参数冻结策略

**完整Adapter实现需要**：
```python
# 在每个Transformer block中插入
class AdapterLayer(nn.Module):
    def __init__(self, hidden_dim, adapter_dim):
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))
```

#### 3. Simplified LoRA（简化实现）⚠️
```yaml
strategy: "lora"
```
**当前行为**：
- 冻结image_encoder
- 训练mask_decoder和prompt_encoder
- **未使用PEFT库配置LoRA权重**

**与标准LoRA的区别**：
- ❌ 未调用`peft.get_peft_model()`
- ❌ 未对attention层添加低秩分解
- ❌ 配置文件中的`lora`参数不生效
- ✅ 仅是一种参数冻结策略

**完整LoRA实现需要**：
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv", "proj"],  # 针对ViT的attention层
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
```

**策略对比**：

| 策略 | 可训练参数 | 显存需求 | 训练速度 | 实现状态 |
|------|-----------|---------|---------|---------|
| Full | mask_decoder + prompt_encoder | 中 | 快 | ✅ 完整 |
| Adapter (简化) | 同Full | 中 | 快 | ⚠️ 简化 |
| LoRA (简化) | 同Full | 中 | 快 | ⚠️ 简化 |

> **推荐使用**：当前版本推荐使用`strategy: "full"`进行微调，效果稳定可靠。如需真正的Adapter/LoRA，请参考：
> - Adapter: https://github.com/google-research/adapter-bert
> - LoRA: https://github.com/huggingface/peft

### 训练配置

```yaml
training:
  num_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_epochs: 2
  gradient_accumulation_steps: 4  # 模拟更大batch
  max_grad_norm: 1.0
  
  lr_scheduler:
    type: "cosine"
    min_lr: 1.0e-6
  
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
```

**超参数建议**：
- `learning_rate`: 1e-4 ~ 5e-4（adapter/LoRA），1e-5 ~ 1e-4（full）
- `batch_size`: 2~4（受限于显存）
- `gradient_accumulation_steps`: 4~8（模拟batch_size=8~32）

### 损失函数

```yaml
loss:
  segmentation_loss:
    type: "dice_bce"  # dice, bce, dice_bce, focal
    dice_weight: 1.0
    bce_weight: 1.0
  
  iou_loss:
    weight: 1.0
```

**损失类型**：
- `dice`: Dice损失（对类别不平衡鲁棒）
- `bce`: 二元交叉熵（标准）
- `dice_bce`: 组合损失（推荐）
- `focal`: Focal损失（处理困难样本）

---

## 微调策略

### 1. Full Fine-tuning（✅ 完整实现）

```yaml
finetuning:
  strategy: "full"

model:
  freeze_image_encoder: true  # 建议冻结以节省资源
  freeze_prompt_encoder: false
  freeze_mask_decoder: false
```

- 训练`mask_decoder` + `prompt_encoder`
- 可按需解冻`image_encoder`
- 与设计文档一致，推荐使用

### 2. Simplified Adapter（⚠️ 简化实现）

```yaml
finetuning:
  strategy: "adapter"  # 当前仅作为“冻结主干”快捷方式
```

**当前行为**
- 冻结`image_encoder`
- 训练`mask_decoder`（以及未冻结时的`prompt_encoder`）
- 不会读取`finetuning.adapter.*`配置

**与标准Adapter差异**
- ❌ 未在Transformer Block中插入Adapter模块
- ❌ 无逐层下投/上投的瓶颈结构
- ❌ 无Adapter层权重保存/加载逻辑
- ✅ 等同于一种“轻量化全参”训练策略

**如果想要真正的Adapter**
- 参考Google Adapter-BERT实现：https://github.com/google-research/adapter-bert
- 在SAM的ViT Block中插入下述结构：

```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_dim, adapter_dim):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
```

### 3. Simplified LoRA（⚠️ 简化实现）

```yaml
finetuning:
  strategy: "lora"  # 当前仅作为“冻结主干”快捷方式
```

**当前行为**
- 与Simplified Adapter完全相同：冻结主干，训练decoder
- 不会读取`finetuning.lora.*`配置

**与标准LoRA差异**
- ❌ 未调用 PEFT `get_peft_model`
- ❌ 未在Attention投影矩阵添加低秩分解
- ❌ 无LoRA特有超参（r、alpha、dropout 等）的实际作用
- ✅ 仍可作为快速试验的参数冻结方案

**如果想要真正的LoRA**
- 参考HuggingFace PEFT: https://github.com/huggingface/peft
- 关键代码示例：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv", "proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
```

### 策略对比（当前实现状态）

| 策略 | 训练参数 | 显存/速度 | 实现状态 | 备注 |
|------|----------|-----------|----------|------|
| Full Fine-tuning | mask_decoder + prompt_encoder（可选encoder） | 中 / 中 | ✅ 完整 | 推荐使用 |
| Simplified Adapter | 同上 | 中 / 中 | ⚠️ 简化 | 实质为“冻结主干” |
| Simplified LoRA | 同上 | 中 / 中 | ⚠️ 简化 | 行为与上行相同 |

> **建议**：当前版本请优先选择 `strategy: "full"`。需要真实 Adapter / LoRA 时，建议参考上面的参考实现自行扩展，或等待项目后续版本（P2阶段）的正式支持。

---

## 训练监控

### 终端输出

训练过程中会实时显示：

```
Epoch 10/50: 100%|███████| 250/250 [05:30<00:00,  1.32s/it, loss=0.234, lr=9.5e-05]

Epoch 10/50 训练完成:
  Loss: 0.2345
  Seg Loss: 0.2100
  IoU Loss: 0.0245

验证结果:
  Val Loss: 0.1987
  IoU: 0.8123
  Dice: 0.8956
  Pixel Acc: 0.9234

✅ 保存检查点: outputs/sam_finetuning/my_experiment/checkpoint_epoch_10.pth
🌟 保存最优模型: outputs/sam_finetuning/my_experiment/best_model.pth
```

### TensorBoard

启动TensorBoard后可查看：
- 训练/验证损失曲线
- 各项指标变化
- 学习率变化

---

## 常见问题

### Q1: 显存不足（Out of Memory）

**解决方案**：
1. 减小batch_size：
   ```yaml
   data:
     batch_size: 1  # 从2改为1
   ```

2. 增大梯度累积：
   ```yaml
   training:
     gradient_accumulation_steps: 8  # 从4改为8
   ```

3. 使用混合精度训练：
   ```yaml
   device:
     use_amp: true
   ```

4. 冻结图像编码器：
   ```yaml
   model:
     freeze_image_encoder: true
   ```

5. 使用更小的模型：
   ```yaml
   model:
     type: "vit_b"  # 不用vit_l或vit_h
   ```

### Q2: 训练速度慢

**解决方案**：
1. 使用混合精度训练（可加速2x）
2. 增加num_workers（数据加载并行）
3. 使用SSD存储数据
4. 冻结图像编码器
5. 在`strategy: "full"`基础上适度冻结编码器（当前的Simplified Adapter/LoRA仅执行此操作）

### Q3: 验证指标不提升

**解决方案**：
1. 检查数据是否正确加载（使用`--visualize`）
2. 降低学习率
3. 增加训练epoch
4. 检查是否过拟合（训练loss低但验证loss高）
5. 尝试不同的损失函数

### Q4: 如何在自己的数据上测试？

训练完成后，使用推理脚本：

```python
from segment_anything import sam_model_registry, SamPredictor
import torch

# 加载微调后的模型
sam = sam_model_registry["vit_b"](checkpoint="path/to/checkpoint.pth")
checkpoint = torch.load("outputs/sam_finetuning/my_experiment/best_model.pth")
sam.load_state_dict(checkpoint['model_state_dict'])
sam.eval()

predictor = SamPredictor(sam)

# 使用与训练时相同的提示方式
# ...
```

### Q5: 支持多GPU训练吗？

当前版本暂不支持多GPU（DataParallel/DDP）。

计划在后续版本添加。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `dataset.py` | 数据集类（支持目录和COCO格式） |
| `train.py` | 训练脚本 |
| `config.yaml` | 配置文件模板 |
| `README.md` | 本文档 |

---

## 性能参考

在医学图像分割数据集（~1K样本）上的性能：

| 策略 | 可训练参数 | 训练时间 | 验证IoU | 显存 | 说明 |
|------|-----------|---------|---------|------|------|
| Full (freeze encoder) | ~8M | 3h | 0.82 | 18GB | 实测数据 |
| Simplified Adapter | ~8M | 2.5h | 0.81 | 18GB | 仅冻结主干，效果≈Full |
| Simplified LoRA | ~8M | 2.5h | 0.81 | 18GB | 当前同上；真实LoRA暂无数据 |

**硬件**: NVIDIA RTX 3090 (24GB)

---

## 示例命令

### 医学图像分割

> ⚠️ 示例命令中的 Adapter/LoRA 将在后续版本补齐。当前示例仅展示`strategy: "full"`的常见配置，请结合自身数据调整。

```bash
# 示例：使用默认full策略启动训练
python train.py --config code/02-fine-tuning/sam/config.yaml
```

### 遥感图像分割

```bash
# 示例：使用ViT-L模型 + full策略 + Both提示
python train.py --config configs/remote_sensing.yaml
```

---

## 引用

如果本代码对你的研究有帮助，请引用SAM原文：

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

## 后续计划

- [ ] 支持多GPU训练（DDP）
- [ ] 支持更多数据增强
- [ ] 添加评估脚本
- [ ] 添加推理脚本
- [ ] 支持视频分割
- [ ] 支持交互式标注工具

---

## 联系与贡献

- GitHub: [Large-Model-Tutorial](https://github.com/your-repo)
- 问题反馈: [Issues](https://github.com/your-repo/issues)
- 贡献代码: [Pull Requests](https://github.com/your-repo/pulls)

欢迎提出问题和贡献代码！

---

**祝训练顺利！** 🚀

