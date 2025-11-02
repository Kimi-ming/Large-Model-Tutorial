"""
LoRA微调模型推理脚本

使用微调后的CLIP模型进行单张图像或批量图像的推理
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Union
import time

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor
import numpy as np

# 添加项目根目录和当前目录到路径
project_root = Path(__file__).parent.parent.parent.parent
current_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 导入当前目录的模块
from train import CLIPClassifier
from evaluate import load_model


class DogBreedPredictor:
    """犬种预测器"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: torch.device = None
    ):
        """
        初始化预测器
        
        Args:
            checkpoint_dir: 模型检查点目录
            device: 计算设备
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 加载处理器
        print("📦 加载处理器...")
        self.processor = CLIPProcessor.from_pretrained(checkpoint_dir)
        
        # 加载类别名称
        classes_file = Path(checkpoint_dir).parent.parent / 'data' / 'dogs' / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f]
        else:
            # 如果没有classes.txt，尝试从数据目录读取
            data_dir = Path('data/dogs/train')
            if data_dir.exists():
                self.class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            else:
                print("⚠️  警告: 无法加载类别名称")
                self.class_names = [f"class_{i}" for i in range(10)]
        
        num_classes = len(self.class_names)
        print(f"   类别数: {num_classes}")
        
        # 加载模型
        print("🤖 加载模型...")
        self.model = load_model(checkpoint_dir, num_classes, self.device)
        self.model.eval()
        
        print("✅ 预测器初始化完成")
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Image.Image],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        预测单张图像
        
        Args:
            image: 图像路径或PIL图像对象
            top_k: 返回前k个预测结果
            
        Returns:
            [(类别名, 置信度), ...] 列表
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image必须是文件路径或PIL.Image对象")
        
        # 预处理
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # 推理
        start_time = time.time()
        logits = self.model(pixel_values)
        inference_time = time.time() - start_time
        
        # 计算概率
        probs = torch.softmax(logits, dim=1)[0]
        
        # 获取top-k结果
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.class_names)))
        
        results = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return results, inference_time
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Image.Image]],
        top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        批量预测
        
        Args:
            images: 图像路径或PIL图像对象列表
            top_k: 返回前k个预测结果
            
        Returns:
            每张图像的预测结果列表
        """
        # 加载图像
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError("image必须是文件路径或PIL.Image对象")
        
        # 预处理
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # 推理
        start_time = time.time()
        logits = self.model(pixel_values)
        inference_time = time.time() - start_time
        
        # 计算概率
        probs = torch.softmax(logits, dim=1)
        
        # 获取top-k结果
        all_results = []
        for prob in probs:
            top_probs, top_indices = torch.topk(prob, min(top_k, len(self.class_names)))
            results = [
                (self.class_names[idx.item()], p.item())
                for idx, p in zip(top_indices, top_probs)
            ]
            all_results.append(results)
        
        avg_time = inference_time / len(images)
        print(f"⏱️  批量推理: {len(images)}张图像, 总耗时{inference_time:.3f}s, 平均{avg_time:.3f}s/张")
        
        return all_results


def print_predictions(
    image_path: str,
    predictions: List[Tuple[str, float]],
    inference_time: float
):
    """
    打印预测结果
    
    Args:
        image_path: 图像路径
        predictions: 预测结果
        inference_time: 推理时间
    """
    print("\n" + "=" * 60)
    print(f"图像: {image_path}")
    print("-" * 60)
    print(f"推理时间: {inference_time*1000:.2f}ms")
    print("\n预测结果:")
    for i, (class_name, confidence) in enumerate(predictions, 1):
        bar_length = int(confidence * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {i}. {class_name:20s} {bar} {confidence*100:5.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LoRA微调模型推理")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点目录'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='单张图像路径'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        help='图像目录（批量推理）'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='返回前k个预测结果'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径（保存预测结果）'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("必须指定 --image 或 --image_dir")
    
    print("=" * 60)
    print("LoRA微调模型推理")
    print("=" * 60)
    
    # 创建预测器
    predictor = DogBreedPredictor(args.checkpoint)
    
    # 单张图像推理
    if args.image:
        print(f"\n🖼️  推理单张图像: {args.image}")
        predictions, inference_time = predictor.predict(args.image, args.top_k)
        print_predictions(args.image, predictions, inference_time)
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Image: {args.image}\n")
                f.write(f"Inference Time: {inference_time*1000:.2f}ms\n\n")
                f.write("Predictions:\n")
                for i, (class_name, confidence) in enumerate(predictions, 1):
                    f.write(f"{i}. {class_name}: {confidence*100:.2f}%\n")
            print(f"\n✅ 结果已保存: {args.output}")
    
    # 批量推理
    elif args.image_dir:
        print(f"\n📁 批量推理目录: {args.image_dir}")
        
        # 收集所有图像
        image_dir = Path(args.image_dir)
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(image_dir.glob(ext))
        
        if len(image_paths) == 0:
            print(f"❌ 在 {args.image_dir} 中未找到图像文件")
            return
        
        print(f"   找到 {len(image_paths)} 张图像")
        
        # 批量推理
        all_predictions = predictor.predict_batch(
            [str(p) for p in image_paths],
            args.top_k
        )
        
        # 打印结果
        for image_path, predictions in zip(image_paths, all_predictions):
            print(f"\n{image_path.name}:")
            for i, (class_name, confidence) in enumerate(predictions[:3], 1):
                print(f"  {i}. {class_name}: {confidence*100:.2f}%")
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for image_path, predictions in zip(image_paths, all_predictions):
                    f.write(f"\nImage: {image_path.name}\n")
                    f.write("Predictions:\n")
                    for i, (class_name, confidence) in enumerate(predictions, 1):
                        f.write(f"  {i}. {class_name}: {confidence*100:.2f}%\n")
            print(f"\n✅ 结果已保存: {args.output}")
    
    print("\n✅ 推理完成！")


if __name__ == '__main__':
    main()

