#!/usr/bin/env python3
"""
生成Jupyter Notebook教程

此脚本自动生成LoRA和全参数微调的Notebook教程
"""

import json
import os
from pathlib import Path


def create_lora_notebook():
    """创建LoRA微调Notebook"""
    
    cells = []
    
    # 标题和介绍
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🎯 LoRA微调实战教程\n",
            "\n",
            "**欢迎来到LoRA微调实战教程！**\n",
            "\n",
            "本教程将指导您使用LoRA技术微调CLIP模型，完成犬种识别任务。\n",
            "\n",
            "---\n",
            "\n",
            "## 📚 学习目标\n",
            "\n",
            "- ✅ 理解LoRA微调的原理和优势\n",
            "- ✅ 掌握数据准备和预处理流程\n",
            "- ✅ 学会配置和训练LoRA模型\n",
            "- ✅ 评估和使用微调后的模型\n",
            "\n",
            "---\n",
            "\n",
            "## ⏱️ 预计学习时间\n",
            "\n",
            "- 完整运行：约 30-45 分钟\n",
            "- 快速浏览：约 10-15 分钟\n",
            "\n",
            "---\n",
            "\n",
            "## 🎯 任务说明\n",
            "\n",
            "**任务**：犬种识别\n",
            "\n",
            "- **数据集**：Stanford Dogs（10个犬种）\n",
            "- **基础模型**：CLIP-ViT-B/32\n",
            "- **微调方法**：LoRA（r=8）\n",
            "\n",
            "---\n",
            "\n",
            "## 📌 前置要求\n",
            "\n",
            "```bash\n",
            "# 1. 准备数据\n",
            "python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10\n",
            "\n",
            "# 2. 安装依赖\n",
            "pip install jupyter torch transformers peft pillow matplotlib tqdm scikit-learn seaborn\n",
            "```"
        ]
    })
    
    # 导入库
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第一部分：环境准备\n",
            "\n",
            "## 1.1 导入必要的库"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 基础库\n",
            "import os\n",
            "import sys\n",
            "import random\n",
            "import numpy as np\n",
            "from pathlib import Path\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# 深度学习\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "\n",
            "# HuggingFace\n",
            "from transformers import CLIPModel, CLIPProcessor\n",
            "from peft import LoraConfig, get_peft_model\n",
            "\n",
            "# 可视化\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "from tqdm.auto import tqdm\n",
            "\n",
            "print(\"✅ 所有库导入成功！\")\n",
            "print(f\"PyTorch版本: {torch.__version__}\")\n",
            "print(f\"CUDA可用: {torch.cuda.is_available()}\")"
        ]
    })
    
    # 配置参数
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1.2 配置参数\n",
            "\n",
            "💡 **提示**：您可以根据自己的硬件条件调整这些参数"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 配置类\n",
            "class Config:\n",
            "    # 模型配置\n",
            "    model_name = \"openai/clip-vit-base-patch32\"\n",
            "    num_classes = 10\n",
            "    \n",
            "    # LoRA配置\n",
            "    lora_r = 8\n",
            "    lora_alpha = 32\n",
            "    lora_dropout = 0.1\n",
            "    target_modules = [\"q_proj\", \"v_proj\"]\n",
            "    \n",
            "    # 训练配置\n",
            "    batch_size = 16\n",
            "    num_epochs = 5\n",
            "    learning_rate = 5e-4\n",
            "    \n",
            "    # 数据和设备\n",
            "    data_dir = \"../data/dogs\"\n",
            "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "    output_dir = \"../outputs/lora_notebook\"\n",
            "\n",
            "config = Config()\n",
            "print(f\"📋 设备: {config.device}\")\n",
            "print(f\"📋 LoRA秩: {config.lora_r}\")\n",
            "print(f\"📋 批次大小: {config.batch_size}\")"
        ]
    })
    
    # 数据准备说明
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第二部分：使用训练脚本\n",
            "\n",
            "## 💡 推荐方式：使用现有的训练脚本\n",
            "\n",
            "我们已经提供了完整的训练脚本，您可以直接使用：\n",
            "\n",
            "### 方法1：命令行训练（推荐）\n",
            "\n",
            "```bash\n",
            "# 使用默认配置训练\n",
            "python code/02-fine-tuning/lora/train.py\n",
            "\n",
            "# 使用自定义配置\n",
            "python code/02-fine-tuning/lora/train.py --config code/02-fine-tuning/lora/config.yaml\n",
            "```\n",
            "\n",
            "### 方法2：在Notebook中调用脚本\n",
            "\n",
            "运行下面的单元格来启动训练："
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 在Notebook中运行训练脚本\n",
            "!python ../code/02-fine-tuning/lora/train.py \\\n",
            "    --config ../code/02-fine-tuning/lora/config.yaml \\\n",
            "    --output_dir {config.output_dir}"
        ]
    })
    
    # 评估模型
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第三部分：评估模型\n",
            "\n",
            "训练完成后，让我们评估模型性能："
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 运行评估脚本\n",
            "!python ../code/02-fine-tuning/lora/evaluate.py \\\n",
            "    --model_path {config.output_dir}/best_model \\\n",
            "    --data_dir {config.data_dir} \\\n",
            "    --output_dir {config.output_dir}/evaluation"
        ]
    })
    
    # 推理示例
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第四部分：模型推理\n",
            "\n",
            "## 4.1 单张图像推理"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 选择一张测试图像\n",
            "test_image = \"path/to/your/test/image.jpg\"  # 修改为实际路径\n",
            "\n",
            "# 运行推理\n",
            "!python ../code/02-fine-tuning/lora/inference.py \\\n",
            "    --model_path {config.output_dir}/best_model \\\n",
            "    --image_path {test_image} \\\n",
            "    --top_k 5"
        ]
    })
    
    # 可视化结果
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.2 批量推理和可视化"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 批量推理\n",
            "!python ../code/02-fine-tuning/lora/inference.py \\\n",
            "    --model_path {config.output_dir}/best_model \\\n",
            "    --image_dir {config.data_dir}/test \\\n",
            "    --output_dir {config.output_dir}/predictions \\\n",
            "    --batch_size 32"
        ]
    })
    
    # 总结
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 🎓 总结\n",
            "\n",
            "## ✅ 完成的内容\n",
            "\n",
            "1. ✅ 配置LoRA微调参数\n",
            "2. ✅ 训练CLIP模型\n",
            "3. ✅ 评估模型性能\n",
            "4. ✅ 进行模型推理\n",
            "\n",
            "## 🔑 关键要点\n",
            "\n",
            "- **LoRA优势**：只训练1-2%的参数，大幅降低计算成本\n",
            "- **超参数**：r和alpha是最重要的超参数\n",
            "- **应用场景**：适合资源受限或需要快速迭代的场景\n",
            "\n",
            "## 🚀 进阶方向\n",
            "\n",
            "1. 尝试不同的LoRA配置（r, alpha, target_modules）\n",
            "2. 使用更大的数据集\n",
            "3. 尝试QLoRA（量化LoRA）\n",
            "4. 部署为API服务\n",
            "\n",
            "## 📚 参考资源\n",
            "\n",
            "- [LoRA论文](https://arxiv.org/abs/2106.09685)\n",
            "- [完整文档](../docs/02-模型微调技术/02-LoRA微调实践.md)\n",
            "- [代码示例](../code/02-fine-tuning/lora/)\n",
            "\n",
            "---\n",
            "\n",
            "**🎉 恭喜完成本教程！**\n",
            "\n",
            "如有问题，欢迎在GitHub上提Issue。"
        ]
    })
    
    # 创建notebook结构
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def create_full_finetuning_notebook():
    """创建全参数微调Notebook"""
    
    cells = []
    
    # 标题
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🎯 全参数微调进阶教程\n",
            "\n",
            "**欢迎来到全参数微调进阶教程！**\n",
            "\n",
            "本教程将指导您进行CLIP模型的全参数微调，并学习高级训练技巧。\n",
            "\n",
            "---\n",
            "\n",
            "## 📚 学习目标\n",
            "\n",
            "- ✅ 理解全参数微调的原理\n",
            "- ✅ 掌握分层学习率技术\n",
            "- ✅ 学习渐进解冻策略\n",
            "- ✅ 对比LoRA和全参数微调\n",
            "\n",
            "---\n",
            "\n",
            "## ⚠️ 资源要求\n",
            "\n",
            "- **GPU显存**：至少 24GB（推荐 A100 40GB）\n",
            "- **训练时间**：约 1-2 小时\n",
            "- **前置知识**：完成LoRA微调教程\n",
            "\n",
            "---\n",
            "\n",
            "## 🎯 任务说明\n",
            "\n",
            "**任务**：犬种识别（全参数微调）\n",
            "\n",
            "- **数据集**：Stanford Dogs（10个犬种）\n",
            "- **基础模型**：CLIP-ViT-B/32\n",
            "- **微调方法**：全参数 + 分层学习率 + 渐进解冻"
        ]
    })
    
    # 配置
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第一部分：配置参数\n",
            "\n",
            "全参数微调需要更精细的配置："
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class FullFinetuningConfig:\n",
            "    # 模型配置\n",
            "    model_name = \"openai/clip-vit-base-patch32\"\n",
            "    num_classes = 10\n",
            "    \n",
            "    # 训练配置\n",
            "    batch_size = 16\n",
            "    num_epochs = 20\n",
            "    base_learning_rate = 1e-5\n",
            "    \n",
            "    # 分层学习率\n",
            "    layerwise_lr_decay = 0.95\n",
            "    \n",
            "    # 渐进解冻\n",
            "    unfreeze_epochs = [2, 4, 6]  # 在这些epoch解冻层\n",
            "    \n",
            "    # 数据和设备\n",
            "    data_dir = \"../data/dogs\"\n",
            "    device = \"cuda\"\n",
            "    output_dir = \"../outputs/full_finetuning_notebook\"\n",
            "\n",
            "config = FullFinetuningConfig()\n",
            "print(f\"📋 基础学习率: {config.base_learning_rate}\")\n",
            "print(f\"📋 分层衰减率: {config.layerwise_lr_decay}\")"
        ]
    })
    
    # 使用训练脚本
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第二部分：全参数微调\n",
            "\n",
            "## 使用训练脚本"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 运行全参数微调\n",
            "!python ../code/02-fine-tuning/full-finetuning/train.py \\\n",
            "    --config ../code/02-fine-tuning/full-finetuning/config.yaml \\\n",
            "    --output_dir {config.output_dir}"
        ]
    })
    
    # 性能对比
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 第三部分：性能对比\n",
            "\n",
            "## 3.1 LoRA vs 全参数微调"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# 对比数据（示例）\n",
            "comparison = pd.DataFrame({\n",
            "    '方法': ['LoRA', '全参数微调'],\n",
            "    '准确率(%)': [85.2, 88.5],\n",
            "    '训练时间(分钟)': [15, 45],\n",
            "    '显存占用(GB)': [8, 24],\n",
            "    '可训练参数(%)': [1.2, 100]\n",
            "})\n",
            "\n",
            "print(comparison)\n",
            "\n",
            "# 可视化对比\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
            "\n",
            "metrics = ['准确率(%)', '训练时间(分钟)', '显存占用(GB)', '可训练参数(%)']\n",
            "for idx, metric in enumerate(metrics):\n",
            "    ax = axes[idx // 2, idx % 2]\n",
            "    comparison.plot(x='方法', y=metric, kind='bar', ax=ax, legend=False)\n",
            "    ax.set_title(metric)\n",
            "    ax.set_xlabel('')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # 总结
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "# 🎓 总结\n",
            "\n",
            "## ✅ 关键发现\n",
            "\n",
            "| 方法 | 优势 | 劣势 | 适用场景 |\n",
            "|------|------|------|----------|\n",
            "| **LoRA** | 快速、低成本 | 性能略低 | 资源受限、快速迭代 |\n",
            "| **全参数** | 性能最优 | 成本高、慢 | 追求极致性能 |\n",
            "\n",
            "## 🔑 最佳实践\n",
            "\n",
            "1. **先用LoRA探索**：快速验证想法\n",
            "2. **再用全参数优化**：追求最佳性能\n",
            "3. **根据场景选择**：权衡成本和性能\n",
            "\n",
            "## 📚 参考资源\n",
            "\n",
            "- [全参数微调文档](../docs/02-模型微调技术/03-全参数微调.md)\n",
            "- [代码示例](../code/02-fine-tuning/full-finetuning/)\n",
            "\n",
            "---\n",
            "\n",
            "**🎉 恭喜完成进阶教程！**"
        ]
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """主函数"""
    # 创建输出目录
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    
    print("📝 生成Jupyter Notebook教程...")
    
    # 生成LoRA Notebook
    print("\n1️⃣ 生成LoRA微调教程...")
    lora_notebook = create_lora_notebook()
    lora_path = notebooks_dir / "01_lora_finetuning_tutorial.ipynb"
    with open(lora_path, 'w', encoding='utf-8') as f:
        json.dump(lora_notebook, f, ensure_ascii=False, indent=1)
    print(f"   ✅ 已保存到: {lora_path}")
    
    # 生成全参数微调Notebook
    print("\n2️⃣ 生成全参数微调教程...")
    full_notebook = create_full_finetuning_notebook()
    full_path = notebooks_dir / "02_full_finetuning_tutorial.ipynb"
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(full_notebook, f, ensure_ascii=False, indent=1)
    print(f"   ✅ 已保存到: {full_path}")
    
    print("\n" + "=" * 60)
    print("🎉 所有Notebook教程生成完成！")
    print("=" * 60)
    print("\n📚 使用方法：")
    print("  1. 启动Jupyter: jupyter notebook")
    print("  2. 打开notebooks目录")
    print("  3. 选择教程文件开始学习")
    print("\n💡 提示：")
    print("  - 先完成01_lora教程")
    print("  - 再尝试02_full_finetuning教程")
    print("  - 详细说明见 notebooks/README.md")


if __name__ == "__main__":
    main()

