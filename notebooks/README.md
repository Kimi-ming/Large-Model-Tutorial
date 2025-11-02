# 📓 Jupyter Notebook 教程

本目录包含交互式的Jupyter Notebook教程，帮助您通过实践学习视觉大模型的微调和部署。

---

## 📚 教程列表

### 1. LoRA微调实战教程
**文件**: `01_lora_finetuning_tutorial.ipynb`

**内容**:
- LoRA微调原理和优势
- 数据准备和预处理
- 模型配置和训练
- 评估和推理
- 模型保存和加载

**适合人群**: 初学者和中级用户

**预计时间**: 30-45分钟

**前置要求**:
- Python 3.8+
- PyTorch 2.0+
- 8GB+ GPU显存（或16GB+ CPU内存）

---

### 2. 全参数微调进阶教程
**文件**: `02_full_finetuning_tutorial.ipynb`

**内容**:
- 全参数微调原理
- 高级训练技巧（分层学习率、渐进解冻）
- 混合精度训练
- 性能对比分析

**适合人群**: 进阶用户

**预计时间**: 45-60分钟

**前置要求**:
- 完成LoRA教程
- 24GB+ GPU显存

---

## 🚀 快速开始

### 方法1：使用Jupyter Notebook

```bash
# 1. 激活环境
conda activate vlm-tutorial

# 2. 启动Jupyter Notebook
jupyter notebook

# 3. 在浏览器中打开教程文件
```

### 方法2：使用JupyterLab

```bash
# 1. 安装JupyterLab（如果未安装）
pip install jupyterlab

# 2. 启动JupyterLab
jupyter lab

# 3. 在JupyterLab中打开教程文件
```

### 方法3：使用VS Code

```bash
# 1. 安装VS Code Python扩展
# 2. 安装Jupyter扩展
# 3. 在VS Code中直接打开.ipynb文件
```

---

## 📋 数据准备

在运行教程前，需要准备训练数据：

### 自动下载（推荐）

```bash
# 下载Stanford Dogs数据集（10个犬种）
# 注意：会自动下载约750MB的数据集，需要5-10分钟
python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10
```

**脚本功能**：
- ✅ 自动从官方源下载数据集（~750MB）
- ✅ 解压并组织成训练/测试集
- ✅ 验证数据完整性

**如果下载失败**：
```bash
# 手动下载方案
# 1. 访问 http://vision.stanford.edu/aditya86/ImageNetDogs/
# 2. 下载 images.tar
# 3. 放到 data/dogs/downloads/
# 4. 运行（跳过下载）
python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10 --no-download
```

### 手动准备

如果自动下载失败，可以手动准备数据：

1. 创建目录结构：
```
data/dogs/
├── train/
│   ├── breed1/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── breed2/
│   └── ...
└── test/
    ├── breed1/
    └── ...
```

2. 每个类别至少准备50张训练图像和10张测试图像

---

## 💡 使用建议

### 1. 按顺序学习
建议按照教程编号顺序学习：
1. 先完成LoRA微调教程
2. 再尝试全参数微调教程

### 2. 实验和修改
- 尝试修改超参数，观察效果
- 使用自己的数据集进行实验
- 记录实验结果和心得

### 3. 保存结果
- 定期保存Notebook
- 保存训练好的模型
- 记录实验配置和结果

### 4. 资源管理
- 训练前检查GPU显存
- 适当调整batch_size
- 使用混合精度训练节省显存

---

## 🔧 常见问题

### Q1: 如何安装Jupyter Notebook？

```bash
pip install jupyter notebook
# 或
conda install jupyter notebook
```

### Q2: 显存不足怎么办？

**解决方案**:
1. 减小batch_size
2. 使用梯度累积
3. 启用混合精度训练（FP16）
4. 使用CPU训练（较慢）

```python
# 在Notebook中修改配置
config.batch_size = 8  # 减小批次大小
config.use_fp16 = True  # 启用FP16
```

### Q3: 训练太慢怎么办？

**解决方案**:
1. 减少训练轮数（num_epochs）
2. 使用更小的数据集
3. 跳过训练，直接加载预训练模型

```python
# 快速演示模式
config.num_epochs = 3  # 减少到3轮
config.num_classes = 5  # 只使用5个类别
```

### Q4: 如何使用自己的数据？

修改数据路径和类别数：

```python
config.data_dir = "/path/to/your/data"
config.num_classes = 20  # 您的类别数
```

确保数据目录结构符合要求。

### Q5: 如何导出训练好的模型？

在Notebook最后一部分有完整的保存和加载示例：

```python
# 保存模型
model.save_pretrained("outputs/my_model")

# 加载模型
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "outputs/my_model")
```

---

## 📊 教程特点

### ✅ 优点
- **交互式学习**：边学边练，即时反馈
- **可视化丰富**：图表、图像展示清晰
- **代码可修改**：随时调整参数实验
- **结果可保存**：保留实验记录

### 📝 学习建议
- **逐步执行**：不要一次运行所有单元格
- **理解代码**：阅读注释，理解每一步
- **做笔记**：在Markdown单元格中记录心得
- **多实验**：尝试不同的配置和数据

---

## 🔗 相关资源

### 文档
- [LoRA微调文档](../docs/02-模型微调技术/02-LoRA微调实践.md)
- [全参数微调文档](../docs/02-模型微调技术/03-全参数微调.md)
- [数据准备文档](../docs/03-数据集准备/)

### 代码示例
- [LoRA训练脚本](../code/02-fine-tuning/lora/)
- [全参数微调脚本](../code/02-fine-tuning/full-finetuning/)

### 论文
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

---

## 🤝 贡献

欢迎提交Issue或Pull Request来改进教程：
- 报告错误
- 建议改进
- 添加新的教程
- 分享您的实验结果

---

## 📄 许可证

本教程遵循MIT许可证，可自由使用和修改。

---

**🎉 开始您的学习之旅吧！**

如有问题，欢迎在GitHub上提Issue或查看[常见问题文档](../FAQ.md)。

