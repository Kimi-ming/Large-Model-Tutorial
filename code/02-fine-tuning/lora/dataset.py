"""
犬种分类数据集

用于LoRA微调的图像分类数据集实现
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import torch
from torch.utils.data import Dataset
from PIL import Image
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
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train', 
        processor: Optional[CLIPProcessor] = None,
        transform: Optional[Callable] = None
    ):
        self.data_dir = os.path.join(data_dir, split)
        self.processor = processor
        self.transform = transform
        self.split = split
        
        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        # 加载类别和图像路径
        self.classes = sorted([
            d for d in os.listdir(self.data_dir) 
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        
        if len(self.classes) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到任何类别目录")
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # 加载所有样本
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到任何图像文件")
        
        print(f"✅ 加载 {split} 集: {len(self.samples)} 张图像")
        print(f"   类别数: {len(self.classes)}")
        print(f"   类别: {', '.join(self.classes)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  加载图像失败: {img_path}, 错误: {e}")
            # 返回一个空白图像作为后备
            image = Image.new('RGB', (224, 224), color='white')
        
        # 应用额外的变换（如果有）
        if self.transform:
            image = self.transform(image)
        
        # 使用CLIP处理器处理图像
        if self.processor:
            # 处理器返回的是字典，我们需要提取pixel_values
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch维度
            return pixel_values, label
        else:
            # 如果没有处理器，返回PIL图像
            return image, label
    
    def get_class_name(self, idx: int) -> str:
        """根据索引获取类别名称"""
        return self.idx_to_class.get(idx, "unknown")
    
    def get_class_distribution(self) -> dict:
        """获取类别分布统计"""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution


def create_dataloaders(
    data_dir: str,
    processor: CLIPProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据集根目录
        processor: CLIP处理器
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存（GPU训练时推荐）
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = DogBreedDataset(data_dir, split='train', processor=processor)
    val_dataset = DogBreedDataset(data_dir, split='val', processor=processor)
    test_dataset = DogBreedDataset(data_dir, split='test', processor=processor)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print("\n📊 数据加载器创建完成:")
    print(f"   训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"   验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"   测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """测试数据集加载"""
    from transformers import CLIPProcessor
    
    print("=" * 60)
    print("数据集测试")
    print("=" * 60)
    
    # 加载处理器
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 测试数据集
    try:
        dataset = DogBreedDataset(
            data_dir="data/dogs",
            split="train",
            processor=processor
        )
        
        print(f"\n✅ 数据集加载成功")
        print(f"   样本数: {len(dataset)}")
        print(f"   类别数: {len(dataset.classes)}")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            pixel_values, label = dataset[0]
            print(f"\n📦 样本测试:")
            print(f"   图像张量形状: {pixel_values.shape}")
            print(f"   标签: {label} ({dataset.get_class_name(label)})")
        
        # 显示类别分布
        print(f"\n📊 类别分布:")
        distribution = dataset.get_class_distribution()
        for class_name, count in distribution.items():
            print(f"   {class_name}: {count} 张")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已运行 python scripts/prepare_dog_dataset.py")
        print("2. 已将图像文件放入 data/dogs/ 目录")


if __name__ == '__main__':
    test_dataset()

