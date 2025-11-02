#!/usr/bin/env python3
"""
Stanford Dogs Dataset 准备脚本

自动下载并准备犬种分类数据集，用于LoRA微调示例。
"""

import os
import argparse
import shutil
import tarfile
import random
from pathlib import Path
from typing import List
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载URL
        output_path: 输出路径
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_and_prepare_dataset(
    output_dir: str,
    num_classes: int = 10,
    train_ratio: float = 0.8,
    download: bool = True
):
    """
    下载并准备Stanford Dogs数据集
    
    Args:
        output_dir: 输出目录
        num_classes: 使用的类别数量（1-120）
        train_ratio: 训练集比例
        download: 是否下载数据集（如果已存在则跳过）
    """
    print("=" * 70)
    print("Stanford Dogs Dataset 准备工具")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stanford Dogs 数据集URL
    dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
    lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
    
    # 下载路径
    download_dir = output_dir / "downloads"
    download_dir.mkdir(exist_ok=True)
    
    images_tar = download_dir / "images.tar"
    annotations_tar = download_dir / "annotation.tar"
    lists_tar = download_dir / "lists.tar"
    
    # 解压路径
    extract_dir = output_dir / "raw"
    extract_dir.mkdir(exist_ok=True)
    
    # 1. 下载数据集
    if download:
        print("\n📥 步骤 1/4: 下载数据集")
        print("-" * 70)
        
        if not images_tar.exists():
            print(f"正在下载图像数据集 (~750MB)...")
            try:
                download_file(dataset_url, str(images_tar))
                print("✅ 图像数据集下载完成")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                print("\n💡 备选方案:")
                print("   1. 手动下载: http://vision.stanford.edu/aditya86/ImageNetDogs/")
                print(f"   2. 将 images.tar 放到: {download_dir}")
                print("   3. 重新运行此脚本，使用 --no-download 参数")
                return False
        else:
            print("✅ 图像数据集已存在，跳过下载")
        
        if not lists_tar.exists():
            print(f"\n正在下载数据集列表...")
            try:
                download_file(lists_url, str(lists_tar))
                print("✅ 数据集列表下载完成")
            except Exception as e:
                print(f"⚠️  列表下载失败: {e}，将使用默认分割")
        else:
            print("✅ 数据集列表已存在，跳过下载")
    
    # 2. 解压数据集
    print("\n📦 步骤 2/4: 解压数据集")
    print("-" * 70)
    
    images_dir = extract_dir / "Images"
    if not images_dir.exists():
        print("正在解压图像数据集...")
        try:
            with tarfile.open(images_tar, 'r') as tar:
                tar.extractall(extract_dir)
            print("✅ 解压完成")
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False
    else:
        print("✅ 数据集已解压，跳过")
    
    # 3. 组织数据集
    print("\n📂 步骤 3/4: 组织数据集")
    print("-" * 70)
    
    # 获取所有犬种类别
    all_breeds = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    
    if num_classes > len(all_breeds):
        print(f"⚠️  请求的类别数 ({num_classes}) 超过可用类别数 ({len(all_breeds)})")
        num_classes = len(all_breeds)
    
    # 选择指定数量的类别
    selected_breeds = all_breeds[:num_classes]
    
    print(f"📊 数据集信息:")
    print(f"   总类别数: {len(all_breeds)}")
    print(f"   使用类别数: {num_classes}")
    print(f"   训练/测试比例: {train_ratio:.0%} / {1-train_ratio:.0%}")
    
    # 创建训练和测试目录
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # 处理每个类别
    print(f"\n正在处理 {num_classes} 个犬种类别...")
    
    total_train = 0
    total_test = 0
    
    for breed in tqdm(selected_breeds, desc="处理类别"):
        breed_dir = images_dir / breed
        
        # 获取该类别的所有图像
        image_files = list(breed_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠️  {breed} 没有找到图像，跳过")
            continue
        
        # 随机打乱
        random.shuffle(image_files)
        
        # 分割训练和测试
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        # 创建类别目录
        train_breed_dir = train_dir / breed
        test_breed_dir = test_dir / breed
        train_breed_dir.mkdir(exist_ok=True)
        test_breed_dir.mkdir(exist_ok=True)
        
        # 复制训练集图像
        for img_file in train_files:
            shutil.copy2(img_file, train_breed_dir / img_file.name)
        
        # 复制测试集图像
        for img_file in test_files:
            shutil.copy2(img_file, test_breed_dir / img_file.name)
        
        total_train += len(train_files)
        total_test += len(test_files)
    
    # 4. 验证数据集
    print("\n✅ 步骤 4/4: 验证数据集")
    print("-" * 70)
    
    print(f"📊 最终统计:")
    print(f"   训练样本: {total_train}")
    print(f"   测试样本: {total_test}")
    print(f"   总样本数: {total_train + total_test}")
    print(f"\n📁 数据集位置:")
    print(f"   训练集: {train_dir}")
    print(f"   测试集: {test_dir}")
    
    # 显示类别列表
    print(f"\n🐕 犬种类别:")
    for i, breed in enumerate(selected_breeds, 1):
        breed_name = breed.split('-', 1)[-1].replace('_', ' ').title()
        train_count = len(list((train_dir / breed).glob("*.jpg")))
        test_count = len(list((test_dir / breed).glob("*.jpg")))
        print(f"   {i:2d}. {breed_name:30s} (训练: {train_count:3d}, 测试: {test_count:3d})")
    
    print("\n" + "=" * 70)
    print("✅ 数据集准备完成！")
    print("=" * 70)
    print("\n📝 下一步:")
    print("   1. 查看数据集: ls", train_dir)
    print("   2. 开始训练: python code/02-fine-tuning/lora/train.py")
    print("   3. 或使用Notebook: jupyter notebook notebooks/01_lora_finetuning_tutorial.ipynb")
    
    return True


def validate_dataset(data_dir: str):
    """
    验证数据集是否准备正确
    
    Args:
        data_dir: 数据集目录
    """
    data_dir = Path(data_dir)
    
    print("\n🔍 验证数据集...")
    print("-" * 70)
    
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    
    if not train_dir.exists():
        print(f"❌ 训练集目录不存在: {train_dir}")
        return False
    
    if not test_dir.exists():
        print(f"❌ 测试集目录不存在: {test_dir}")
        return False
    
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    
    if not train_classes:
        print(f"❌ 训练集为空")
        return False
    
    if train_classes != test_classes:
        print(f"⚠️  训练集和测试集的类别不一致")
    
    print(f"✅ 验证通过")
    print(f"   类别数: {len(train_classes)}")
    
    total_train = sum(len(list((train_dir / c).glob("*.jpg"))) for c in train_classes)
    total_test = sum(len(list((test_dir / c).glob("*.jpg"))) for c in test_classes)
    
    print(f"   训练样本: {total_train}")
    print(f"   测试样本: {total_test}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stanford Dogs Dataset 准备工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载并准备10个类别的数据集
  python scripts/prepare_dog_dataset.py --output_dir data/dogs --num_classes 10
  
  # 使用已下载的数据集（跳过下载）
  python scripts/prepare_dog_dataset.py --output_dir data/dogs --no-download
  
  # 验证数据集
  python scripts/prepare_dog_dataset.py --output_dir data/dogs --validate
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dogs",
        help="输出目录 (默认: data/dogs)"
    )
    
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="使用的类别数量 (默认: 10, 最大: 120)"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认: 0.8)"
    )
    
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="跳过下载，使用已存在的数据集"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="验证数据集是否准备正确"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 验证模式
    if args.validate:
        validate_dataset(args.output_dir)
        return
    
    # 准备数据集
    success = download_and_prepare_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        train_ratio=args.train_ratio,
        download=not args.no_download
    )
    
    if not success:
        print("\n❌ 数据集准备失败")
        print("\n💡 如果自动下载失败，您可以:")
        print("   1. 手动下载数据集:")
        print("      - 访问: http://vision.stanford.edu/aditya86/ImageNetDogs/")
        print("      - 下载 images.tar")
        print(f"      - 放到: {args.output_dir}/downloads/")
        print("   2. 使用自己的数据集:")
        print("      - 按以下结构组织:")
        print("        data/dogs/")
        print("          ├── train/")
        print("          │   ├── breed1/")
        print("          │   │   ├── img1.jpg")
        print("          │   │   └── ...")
        print("          │   └── breed2/")
        print("          └── test/")
        print("              ├── breed1/")
        print("              └── breed2/")
        exit(1)


if __name__ == "__main__":
    main()
