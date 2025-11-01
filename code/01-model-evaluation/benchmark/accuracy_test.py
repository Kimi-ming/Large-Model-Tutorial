#!/usr/bin/env python3
"""
准确率基准测试（以CLIP图文检索为例）
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Tuple


class CLIPAccuracyBenchmark:
    """CLIP图文检索准确率测试"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            model_name, cache_dir="./models"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            model_name, cache_dir="./models"
        )
        self.model.eval()
        
    def compute_similarity(
        self, 
        image_paths: List[str], 
        texts: List[str]
    ) -> np.ndarray:
        """
        计算图像-文本相似度矩阵
        
        Returns:
            shape (num_images, num_texts) 的相似度矩阵
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        inputs = self.processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            
        return logits_per_image.cpu().numpy()
    
    def evaluate_retrieval(
        self, 
        image_paths: List[str], 
        texts: List[str]
    ) -> dict:
        """
        评估图文检索性能
        假设第i张图对应第i个文本（ground truth）
        """
        similarity = self.compute_similarity(image_paths, texts)
        
        # 计算Recall@1, Recall@5
        num_samples = len(image_paths)
        
        # Image-to-Text检索
        i2t_ranks = np.argsort(-similarity, axis=1)  # 降序排序
        i2t_recall_at_1 = np.mean([1 if 0 in i2t_ranks[i, :1] else 0 
                                    for i in range(num_samples)])
        i2t_recall_at_5 = np.mean([1 if i in i2t_ranks[i, :5] else 0 
                                    for i in range(min(num_samples, 5))])
        
        # Text-to-Image检索
        t2i_ranks = np.argsort(-similarity.T, axis=1)
        t2i_recall_at_1 = np.mean([1 if 0 in t2i_ranks[i, :1] else 0 
                                    for i in range(num_samples)])
        t2i_recall_at_5 = np.mean([1 if i in t2i_ranks[i, :5] else 0 
                                    for i in range(min(num_samples, 5))])
        
        return {
            "i2t_recall@1": round(i2t_recall_at_1 * 100, 2),
            "i2t_recall@5": round(i2t_recall_at_5 * 100, 2),
            "t2i_recall@1": round(t2i_recall_at_1 * 100, 2),
            "t2i_recall@5": round(t2i_recall_at_5 * 100, 2),
        }


def main():
    """运行准确率测试（使用模拟数据）"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP准确率测试")
    parser.add_argument("--model", type=str, 
                       default="openai/clip-vit-base-patch32",
                       help="CLIP模型名称")
    parser.add_argument("--test_data", type=str,
                       help="测试数据文件（JSON格式，包含image_path和text对）")
    
    args = parser.parse_args()
    
    # 示例：图文对（实际使用时应从文件读取）
    test_data = [
        ("data/test_dataset/cat.jpg", "a photo of a cat"),
        ("data/test_dataset/dog.jpg", "a photo of a dog"),
        ("data/test_dataset/car.jpg", "a red car on the street"),
    ]
    
    image_paths = [x[0] for x in test_data]
    texts = [x[1] for x in test_data]
    
    # 运行测试
    benchmark = CLIPAccuracyBenchmark(args.model)
    results = benchmark.evaluate_retrieval(image_paths, texts)
    
    print("\n=== CLIP Retrieval Accuracy ===")
    for k, v in results.items():
        print(f"{k}: {v}%")


if __name__ == "__main__":
    main()

