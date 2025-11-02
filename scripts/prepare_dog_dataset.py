#!/usr/bin/env python3
"""
Stanford Dogs Dataset 准备脚本

用于下载和准备犬种分类数据集的子集，用于LoRA微调示例。
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List
import urllib.request
import tarfile
import random
from PIL import Image

# 为了演示，我们使用一个简化的方法
# 实际的Stanford Dogs数据集需要从官方源下载

def create_demo_dataset(output_dir: str, num_classes: int = 10, samples_per_class: int = 100):
    """
    创建演示数据集（使用占位符图像）
    
    注意：这是一个演示脚本。实际使用时，您需要：
    1. 从官方源下载真实的Stanford Dogs数据集
    2. 或使用您自己的图像数据集
    
    Args:
        output_dir: 输出目录
        num_classes: 类别数量
        samples_per_class: 每个类别的样本数
    """
    print("=" * 60)
    print("Stanford Dogs Dataset 准备工具")
    print("=" * 60)
    
    # 定义犬种类别（前10个常见品种）
    dog_breeds = [
        "golden_retriever",
        "labrador_retriever",
        "german_shepherd",
        "beagle",
        "bulldog",
        "poodle",
        "rottweiler",
        "yorkshire_terrier",
        "boxer",
        "dachshund"
    ][:num_classes]
    
    print(f"\n📦 准备创建数据集:")
    print(f"   - 类别数: {num_classes}")
    print(f"   - 每类样本数: {samples_per_class}")
    print(f"   - 输出目录: {output_dir}")
    
    # 创建目录结构
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for breed in dog_breeds:
            breed_dir = output_path / split / breed
            breed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n✅ 目录结构创建完成")
    
    # 数据集分割比例
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_samples = int(samples_per_class * train_ratio)
    val_samples = int(samples_per_class * val_ratio)
    test_samples = samples_per_class - train_samples - val_samples
    
    print(f"\n📊 数据分割:")
    print(f"   - 训练集: {train_samples * num_classes} 张 ({train_samples}/类)")
    print(f"   - 验证集: {val_samples * num_classes} 张 ({val_samples}/类)")
    print(f"   - 测试集: {test_samples * num_classes} 张 ({test_samples}/类)")
    
    print("\n" + "=" * 60)
    print("⚠️  重要提示")
    print("=" * 60)
    print("此脚本创建了目录结构，但您需要手动添加图像文件。")
    print("\n推荐的数据获取方式：")
    print("\n1. 使用真实的Stanford Dogs数据集:")
    print("   - 下载地址: http://vision.stanford.edu/aditya86/ImageNetDogs/")
    print("   - 解压后按照创建的目录结构组织图像")
    print("\n2. 使用您自己的犬种图像:")
    print("   - 将图像按品种分类放入对应目录")
    print("   - 确保图像格式为 JPG/PNG")
    print("   - 建议图像尺寸: 224x224 或更大")
    print("\n3. 使用在线图像（用于快速测试）:")
    print("   - 从 Unsplash/Pexels 等网站下载免费图像")
    print("   - 搜索对应的犬种名称")
    print("   - 手动下载并放入对应目录")
    
    # 创建一个README文件
    readme_path = output_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 犬种分类数据集\n\n")
        f.write("## 数据集结构\n\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write("├── train/          # 训练集\n")
        f.write("├── val/            # 验证集\n")
        f.write("└── test/           # 测试集\n")
        f.write("```\n\n")
        f.write("## 类别列表\n\n")
        for i, breed in enumerate(dog_breeds, 1):
            f.write(f"{i}. {breed}\n")
        f.write("\n## 数据来源\n\n")
        f.write("请从以下来源获取图像数据：\n\n")
        f.write("1. **Stanford Dogs Dataset** (推荐)\n")
        f.write("   - URL: http://vision.stanford.edu/aditya86/ImageNetDogs/\n")
        f.write("   - 包含120个犬种，共20,580张图像\n\n")
        f.write("2. **自定义数据集**\n")
        f.write("   - 收集您自己的犬种图像\n")
        f.write("   - 确保每个类别有足够的样本（建议>50张）\n\n")
        f.write("## 图像要求\n\n")
        f.write("- 格式: JPG, PNG\n")
        f.write("- 尺寸: 建议 224x224 或更大\n")
        f.write("- 质量: 清晰，光照良好\n")
        f.write("- 内容: 主体为犬只，背景简洁\n")
    
    print(f"\n📝 已创建说明文件: {readme_path}")
    
    # 创建一个示例的类别映射文件
    classes_file = output_path / "classes.txt"
    with open(classes_file, 'w', encoding='utf-8') as f:
        for breed in dog_breeds:
            f.write(f"{breed}\n")
    
    print(f"📝 已创建类别文件: {classes_file}")
    
    print("\n" + "=" * 60)
    print("✅ 数据集准备完成！")
    print("=" * 60)
    print(f"\n下一步: 将图像文件放入 {output_dir} 的对应目录中")
    print("然后运行训练脚本: python code/02-fine-tuning/lora/train.py")


def download_sample_images(output_dir: str, num_samples: int = 5):
    """
    下载一些示例图像用于快速测试（可选功能）
    
    注意：这需要网络连接，且仅用于演示
    """
    print("\n🌐 正在下载示例图像...")
    print("（此功能需要实现具体的下载逻辑）")
    # 实际实现需要从Unsplash API或其他来源下载
    pass


def validate_dataset(data_dir: str) -> bool:
    """
    验证数据集是否正确准备
    
    Args:
        data_dir: 数据集目录
        
    Returns:
        bool: 数据集是否有效
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        return False
    
    print("\n🔍 验证数据集...")
    
    splits = ['train', 'val', 'test']
    total_images = 0
    
    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"❌ 缺少 {split} 目录")
            return False
        
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        if len(classes) == 0:
            print(f"⚠️  {split} 目录为空")
            continue
        
        split_images = 0
        for class_dir in classes:
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            split_images += len(images)
        
        total_images += split_images
        print(f"   {split}: {len(classes)} 类, {split_images} 张图像")
    
    if total_images == 0:
        print("\n⚠️  警告: 数据集目录结构已创建，但尚未添加图像")
        print("   请按照 README.md 的说明添加图像文件")
        return False
    
    print(f"\n✅ 数据集验证通过！共 {total_images} 张图像")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="准备犬种分类数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 创建10个类别的数据集结构
  python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10
  
  # 验证已有数据集
  python scripts/prepare_dog_dataset.py --output_dir data/dogs --validate
  
注意:
  此脚本创建数据集目录结构，您需要手动添加图像文件。
  详见生成的 README.md 文件。
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/dogs',
        help='输出目录路径（默认: data/dogs）'
    )
    
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='类别数量（默认: 10）'
    )
    
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=100,
        help='每个类别的目标样本数（默认: 100）'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='验证现有数据集'
    )
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset(args.output_dir)
    else:
        create_demo_dataset(
            args.output_dir,
            args.num_classes,
            args.samples_per_class
        )
        
        # 验证创建的结构
        validate_dataset(args.output_dir)


if __name__ == '__main__':
    main()

