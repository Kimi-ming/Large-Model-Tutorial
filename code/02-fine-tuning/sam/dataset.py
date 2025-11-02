"""
SAM分割数据集类

支持多种分割数据集格式：
1. COCO格式（实例分割）
2. 自定义目录格式（image + mask）
3. 医学图像格式（NIfTI等）

作者：Large-Model-Tutorial
许可：MIT
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SAMSegmentationDataset(Dataset):
    """
    SAM分割数据集基类
    
    支持的数据格式：
    - 目录格式：images/ 和 masks/ 分别存放图像和掩码
    - COCO格式：标准COCO实例分割格式
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 1024,
        prompt_mode: str = 'box',  # 'box', 'point', 'both'
        num_points: int = 1,
        augment: bool = True,
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            split: 数据集划分 ('train', 'val', 'test')
            image_size: SAM输入图像大小（默认1024）
            prompt_mode: 提示模式 ('box', 'point', 'both')
            num_points: 点提示数量
            augment: 是否使用数据增强
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.prompt_mode = prompt_mode
        self.num_points = num_points
        self.augment = augment and split == 'train'
        
        # 加载数据列表
        self.samples = self._load_samples()
        
        print(f"加载 {split} 数据集: {len(self.samples)} 个样本")
        print(f"提示模式: {prompt_mode}")
        print(f"数据增强: {'开启' if self.augment else '关闭'}")
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本列表"""
        raise NotImplementedError("子类需要实现此方法")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个数据样本
        
        Returns:
            dict containing:
                - image: (3, H, W) 归一化后的图像
                - mask: (H, W) 二值掩码
                - boxes: (N, 4) 边界框（如果prompt_mode包含box）
                - points: (M, 2) 点坐标（如果prompt_mode包含point）
                - labels: (M,) 点标签（1=前景，0=背景）
        """
        sample = self.samples[idx]
        
        # 加载图像和掩码
        image = self._load_image(sample['image_path'])
        mask = self._load_mask(sample['mask_path'])
        
        # 数据增强
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Resize到SAM标准输入大小
        image, mask = self._resize(image, mask)
        
        # 生成提示
        prompts = self._generate_prompts(mask)
        
        # 转换为tensor
        image = self._image_to_tensor(image)
        mask = torch.from_numpy(mask).long()
        
        output = {
            'image': image,
            'mask': mask,
            'original_size': sample.get('original_size', (self.image_size, self.image_size))
        }
        
        # 添加提示信息
        if 'box' in self.prompt_mode and 'boxes' in prompts:
            output['boxes'] = torch.from_numpy(prompts['boxes']).float()
        
        if 'point' in self.prompt_mode and 'points' in prompts:
            output['points'] = torch.from_numpy(prompts['points']).float()
            output['point_labels'] = torch.from_numpy(prompts['point_labels']).long()
        
        return output
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """加载图像（RGB格式）"""
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    def _load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """加载掩码（单通道，0=背景，>0=前景）"""
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        # 二值化：0=背景，1=前景
        mask = (mask > 128).astype(np.uint8)
        return mask
    
    def _resize(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize图像和掩码到SAM标准大小"""
        h, w = image.shape[:2]
        
        # 保持宽高比resize
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Pad到正方形
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        
        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        mask = np.pad(
            mask,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )
        
        return image, mask
    
    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # 随机亮度和对比度调整
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # 对比度
            beta = random.randint(-20, 20)     # 亮度
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        
        # 随机色相调整
        if random.random() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image, mask
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """将图像转换为tensor并归一化"""
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # ImageNet归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
    
    def _generate_prompts(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """根据掩码生成提示（框和/或点）"""
        prompts = {}
        
        # 找到前景区域
        if mask.sum() == 0:
            # 空掩码，返回默认提示
            if 'box' in self.prompt_mode:
                prompts['boxes'] = np.array([[0, 0, 10, 10]], dtype=np.float32)
            if 'point' in self.prompt_mode:
                prompts['points'] = np.array([[5, 5]], dtype=np.float32)
                prompts['point_labels'] = np.array([1], dtype=np.int64)
            return prompts
        
        # 生成边界框
        if 'box' in self.prompt_mode:
            ys, xs = np.where(mask > 0)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            # 添加小的随机扰动（训练时）
            if self.augment:
                w, h = x_max - x_min, y_max - y_min
                noise_x = int(w * 0.05)
                noise_y = int(h * 0.05)
                x_min = max(0, x_min - random.randint(0, noise_x))
                x_max = min(mask.shape[1], x_max + random.randint(0, noise_x))
                y_min = max(0, y_min - random.randint(0, noise_y))
                y_max = min(mask.shape[0], y_max + random.randint(0, noise_y))
            
            prompts['boxes'] = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        
        # 生成点提示
        if 'point' in self.prompt_mode:
            points = []
            labels = []
            
            # 前景点：从掩码内随机采样（固定数量）
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                # 确保每个样本都有相同数量的点
                num_fg_points = min(self.num_points, len(ys))
                indices = np.random.choice(len(ys), size=num_fg_points, replace=False)
                for idx in indices:
                    points.append([xs[idx], ys[idx]])
                    labels.append(1)  # 前景
                
                # 如果前景点不足num_points，用掩码中心填充
                while len(points) < self.num_points:
                    center_y, center_x = ys.mean(), xs.mean()
                    points.append([center_x, center_y])
                    labels.append(1)
            else:
                # 空掩码，使用图像中心作为默认点
                for _ in range(self.num_points):
                    points.append([mask.shape[1] / 2, mask.shape[0] / 2])
                    labels.append(1)
            
            # 确保所有样本的点数量一致
            prompts['points'] = np.array(points, dtype=np.float32)
            prompts['point_labels'] = np.array(labels, dtype=np.int64)
        
        return prompts


class DirectoryDataset(SAMSegmentationDataset):
    """
    目录格式数据集
    
    数据组织：
    data_dir/
        images/
            train/
                img1.jpg
                img2.jpg
            val/
                img3.jpg
        masks/
            train/
                img1.png
                img2.png
            val/
                img3.png
    """
    
    def _load_samples(self) -> List[Dict]:
        image_dir = self.data_dir / 'images' / self.split
        mask_dir = self.data_dir / 'masks' / self.split
        
        if not image_dir.exists():
            raise ValueError(f"图像目录不存在: {image_dir}")
        if not mask_dir.exists():
            raise ValueError(f"掩码目录不存在: {mask_dir}")
        
        samples = []
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_file in sorted(image_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue
            
            # 对应的掩码文件（假设同名，扩展名为.png）
            mask_file = mask_dir / f"{img_file.stem}.png"
            
            if not mask_file.exists():
                print(f"警告: 找不到掩码文件 {mask_file}, 跳过")
                continue
            
            samples.append({
                'image_path': str(img_file),
                'mask_path': str(mask_file),
            })
        
        if len(samples) == 0:
            raise ValueError(f"在 {image_dir} 中未找到有效的图像-掩码对")
        
        return samples


class COCODataset(SAMSegmentationDataset):
    """
    COCO格式数据集
    
    数据组织：
    data_dir/
        images/
            train2017/
            val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json
    """
    
    def _load_samples(self) -> List[Dict]:
        # 确定split名称
        split_name = {
            'train': 'train2017',
            'val': 'val2017',
            'test': 'test2017'
        }.get(self.split, self.split)
        
        image_dir = self.data_dir / 'images' / split_name
        anno_file = self.data_dir / 'annotations' / f'instances_{split_name}.json'
        
        if not anno_file.exists():
            raise ValueError(f"标注文件不存在: {anno_file}")
        
        # 加载COCO标注
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)
        
        # 构建image_id到文件名的映射
        images = {img['id']: img for img in coco_data['images']}
        
        # 按图像组织标注
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        samples = []
        
        for img_id, anns in image_annotations.items():
            if img_id not in images:
                continue
            
            img_info = images[img_id]
            image_path = image_dir / img_info['file_name']
            
            if not image_path.exists():
                continue
            
            # COCO数据集每个标注作为一个样本
            for ann in anns:
                if 'segmentation' not in ann:
                    continue
                
                samples.append({
                    'image_path': str(image_path),
                    'annotation': ann,
                    'image_info': img_info,
                })
        
        return samples
    
    def _load_mask(self, mask_info: Dict) -> np.ndarray:
        """从COCO标注生成掩码"""
        # 注意：这个方法在COCODataset的__getitem__中被调用
        # mask_info实际是完整的sample字典，包含annotation和image_info
        raise NotImplementedError("此方法不应被直接调用，由COCODataset重写")
        
        # 从RLE或polygon生成掩码
        if isinstance(ann['segmentation'], dict):
            # RLE格式
            from pycocotools import mask as mask_utils
            rle = ann['segmentation']
            mask = mask_utils.decode(rle)
        else:
            # Polygon格式
            import cv2
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
        
        return mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """重写以处理COCO特殊格式"""
        sample = self.samples[idx]
        
        # 加载图像
        image = self._load_image(sample['image_path'])
        
        # 从标注生成掩码（使用正确的图像尺寸）
        ann = sample['annotation']
        img_info = sample['image_info']
        h, w = img_info['height'], img_info['width']
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 从RLE或polygon生成掩码
        if isinstance(ann['segmentation'], dict):
            # RLE格式
            try:
                from pycocotools import mask as mask_utils
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
            except ImportError:
                print("警告: pycocotools未安装，无法解析RLE格式")
        else:
            # Polygon格式
            import cv2
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
        
        # 记录原始尺寸
        original_size = (h, w)
        
        # 数据增强
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Resize
        image, mask = self._resize(image, mask)
        
        # 生成提示
        prompts = self._generate_prompts(mask)
        
        # 转换为tensor
        image = self._image_to_tensor(image)
        mask = torch.from_numpy(mask).long()
        
        output = {
            'image': image,
            'mask': mask,
            'original_size': original_size
        }
        
        if 'box' in self.prompt_mode and 'boxes' in prompts:
            output['boxes'] = torch.from_numpy(prompts['boxes']).float()
        
        if 'point' in self.prompt_mode and 'points' in prompts:
            output['points'] = torch.from_numpy(prompts['points']).float()
            output['point_labels'] = torch.from_numpy(prompts['point_labels']).long()
        
        return output


def create_sam_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 2,
    num_workers: int = 4,
    dataset_type: str = 'directory',  # 'directory' or 'coco'
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    创建SAM数据加载器
    
    Args:
        data_dir: 数据目录
        split: 数据集划分
        batch_size: batch大小
        num_workers: 数据加载线程数
        dataset_type: 数据集类型
        **dataset_kwargs: 传递给数据集的其他参数
    
    Returns:
        DataLoader
    """
    # 选择数据集类
    if dataset_type == 'directory':
        dataset_class = DirectoryDataset
    elif dataset_type == 'coco':
        dataset_class = COCODataset
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 创建数据集
    dataset = dataset_class(
        data_dir=data_dir,
        split=split,
        **dataset_kwargs
    )
    
    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="测试SAM数据集")
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--dataset_type', type=str, default='directory', 
                       choices=['directory', 'coco'], help='数据集类型')
    parser.add_argument('--split', type=str, default='train', help='数据集划分')
    parser.add_argument('--visualize', action='store_true', help='可视化样本')
    
    args = parser.parse_args()
    
    # 创建数据集
    print(f"\n测试 {args.dataset_type} 数据集...")
    
    if args.dataset_type == 'directory':
        dataset = DirectoryDataset(
            data_dir=args.data_dir,
            split=args.split,
            prompt_mode='both',
            num_points=3,
            augment=True
        )
    else:
        dataset = COCODataset(
            data_dir=args.data_dir,
            split=args.split,
            prompt_mode='box',
        )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试加载
    print("\n测试加载第一个样本...")
    sample = dataset[0]
    
    print(f"图像形状: {sample['image'].shape}")
    print(f"掩码形状: {sample['mask'].shape}")
    if 'boxes' in sample:
        print(f"边界框: {sample['boxes']}")
    if 'points' in sample:
        print(f"点坐标: {sample['points']}")
        print(f"点标签: {sample['point_labels']}")
    
    # 可视化
    if args.visualize:
        import matplotlib.pyplot as plt
        
        # 反归一化图像
        image = sample['image'].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        mask = sample['mask'].numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # 掩码
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        # 叠加
        axes[2].imshow(image)
        axes[2].imshow(mask, alpha=0.5, cmap='jet')
        
        # 绘制提示
        if 'boxes' in sample:
            box = sample['boxes'][0].numpy()
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor='red', linewidth=2
            )
            axes[2].add_patch(rect)
        
        if 'points' in sample:
            points = sample['points'].numpy()
            labels = sample['point_labels'].numpy()
            for point, label in zip(points, labels):
                color = 'green' if label == 1 else 'red'
                axes[2].plot(point[0], point[1], 'o', color=color, markersize=10)
        
        axes[2].set_title('Image + Mask + Prompts')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
        print("\n✅ 可视化保存到: dataset_sample.png")
    
    print("\n✅ 数据集测试完成！")

