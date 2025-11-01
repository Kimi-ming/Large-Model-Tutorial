"""
数据处理工具模块

提供图像和文本的预处理、增强等功能
"""

import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Any
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器"""
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        interpolation: str = 'bilinear'
    ):
        """
        初始化图像处理器
        
        Args:
            size: 目标图像大小，可以是单个int或(height, width)元组
            mean: 归一化均值，默认ImageNet均值
            std: 归一化标准差，默认ImageNet标准差
            interpolation: 插值方法 ('nearest', 'bilinear', 'bicubic')
        """
        self.size = size if isinstance(size, tuple) else (size, size)
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = std or [0.229, 0.224, 0.225]    # ImageNet std
        
        # 插值方法映射
        interp_map = {
            'nearest': transforms.InterpolationMode.NEAREST,
            'bilinear': transforms.InterpolationMode.BILINEAR,
            'bicubic': transforms.InterpolationMode.BICUBIC,
        }
        self.interpolation = interp_map.get(interpolation, transforms.InterpolationMode.BILINEAR)
        
        # 构建转换pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=self.interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        logger.info(f"ImageProcessor initialized: size={self.size}, mean={self.mean}, std={self.std}")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            PIL Image对象
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 无法打开图像文件
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            logger.debug(f"图像加载成功: {image_path}, size={image.size}")
            return image
        except Exception as e:
            raise ValueError(f"无法打开图像: {image_path}, 错误: {e}")
    
    def process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, Image.Image]:
        """
        处理图像
        
        Args:
            image: 输入图像，可以是:
                - 文件路径 (str/Path)
                - PIL Image
                - numpy数组
            return_tensor: 是否返回tensor，False则返回PIL Image
        
        Returns:
            处理后的图像 (tensor或PIL Image)
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 应用转换
        if return_tensor:
            return self.transform(image)
        else:
            # 只做resize
            return image.resize(self.size, resample=Image.Resampling.BILINEAR)
    
    def process_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        批量处理图像
        
        Args:
            images: 图像列表
            batch_size: 批次大小，None表示一次处理全部
        
        Returns:
            批次tensor [B, C, H, W]
        """
        processed = []
        
        for img in images:
            try:
                tensor = self.process(img, return_tensor=True)
                processed.append(tensor)
            except Exception as e:
                logger.warning(f"处理图像失败，跳过: {img}, 错误: {e}")
                continue
        
        if not processed:
            raise ValueError("没有成功处理的图像")
        
        # 堆叠为batch
        batch_tensor = torch.stack(processed)
        logger.info(f"批量处理完成: {len(processed)}张图像, shape={batch_tensor.shape}")
        
        return batch_tensor
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        反归一化（用于可视化）
        
        Args:
            tensor: 归一化后的tensor [C, H, W] or [B, C, H, W]
        
        Returns:
            反归一化的tensor，值域[0, 1]
        """
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        if tensor.dim() == 4:  # batch
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将tensor转换为PIL Image
        
        Args:
            tensor: 图像tensor [C, H, W]，值域[0, 1]
        
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一张图
        
        # 反归一化
        tensor = self.denormalize(tensor)
        
        # 转为numpy
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)


class TextProcessor:
    """文本处理器"""
    
    def __init__(
        self,
        max_length: int = 77,
        truncation: bool = True,
        padding: str = 'max_length'
    ):
        """
        初始化文本处理器
        
        Args:
            max_length: 最大文本长度
            truncation: 是否截断
            padding: 填充策略 ('max_length', 'longest', 'do_not_pad')
        """
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        
        logger.info(
            f"TextProcessor initialized: max_length={max_length}, "
            f"truncation={truncation}, padding={padding}"
        )
    
    def process(
        self,
        texts: Union[str, List[str]],
        tokenizer: Any,
        return_tensors: str = 'pt'
    ) -> dict:
        """
        处理文本
        
        Args:
            texts: 单个文本或文本列表
            tokenizer: 分词器
            return_tensors: 返回格式 ('pt' for PyTorch, 'np' for NumPy)
        
        Returns:
            编码后的字典 (input_ids, attention_mask等)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = tokenizer(
            texts,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=return_tensors
        )
        
        logger.debug(f"文本处理完成: {len(texts)}条, shape={encoded['input_ids'].shape}")
        return encoded
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        # 移除多余空白
        text = ' '.join(text.split())
        # 去除首尾空格
        text = text.strip()
        return text


class DataAugmentation:
    """数据增强工具"""
    
    @staticmethod
    def get_train_transform(
        size: int = 224,
        with_augmentation: bool = True
    ) -> transforms.Compose:
        """
        获取训练时的数据增强transform
        
        Args:
            size: 目标大小
            with_augmentation: 是否使用增强
        
        Returns:
            transforms组合
        """
        if with_augmentation:
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    @staticmethod
    def get_val_transform(size: int = 224) -> transforms.Compose:
        """
        获取验证/测试时的transform
        
        Args:
            size: 目标大小
        
        Returns:
            transforms组合
        """
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试图像处理器
    processor = ImageProcessor(size=224)
    print("✓ ImageProcessor创建成功")
    
    # 测试文本处理器
    text_processor = TextProcessor(max_length=77)
    print("✓ TextProcessor创建成功")
    
    # 测试数据增强
    train_transform = DataAugmentation.get_train_transform(224, with_augmentation=True)
    val_transform = DataAugmentation.get_val_transform(224)
    print("✓ DataAugmentation transforms创建成功")

