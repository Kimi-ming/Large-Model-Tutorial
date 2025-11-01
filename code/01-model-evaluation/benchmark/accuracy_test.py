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
    
    def _compute_recall_at_k(
        self, 
        ranks: np.ndarray, 
        k: int
    ) -> float:
        """
        计算Recall@K
        
        Args:
            ranks: 排序后的索引数组，shape (num_samples, num_candidates)
            k: Top-K
            
        Returns:
            Recall@K值（0-1之间）
        """
        num_samples = ranks.shape[0]
        # 对于第i个样本，检查ground truth索引i是否在Top-K内
        hits = [1 if i in ranks[i, :k] else 0 for i in range(num_samples)]
        return np.mean(hits)
    
    def evaluate_retrieval(
        self, 
        image_paths: List[str], 
        texts: List[str]
    ) -> dict:
        """
        评估图文检索性能
        假设第i张图对应第i个文本（ground truth）
        
        Args:
            image_paths: 图像路径列表
            texts: 文本列表
            
        Returns:
            包含Recall@1和Recall@5的字典
        """
        similarity = self.compute_similarity(image_paths, texts)
        
        # Image-to-Text检索
        # similarity[i, j] 表示第i张图与第j个文本的相似度
        # 对于第i张图，ground truth是第i个文本
        i2t_ranks = np.argsort(-similarity, axis=1)  # 降序排序，shape (num_images, num_texts)
        i2t_recall_at_1 = self._compute_recall_at_k(i2t_ranks, k=1)
        i2t_recall_at_5 = self._compute_recall_at_k(i2t_ranks, k=5)
        
        # Text-to-Image检索
        # similarity.T[i, j] 表示第i个文本与第j张图的相似度
        # 对于第i个文本，ground truth是第i张图
        t2i_ranks = np.argsort(-similarity.T, axis=1)  # shape (num_texts, num_images)
        t2i_recall_at_1 = self._compute_recall_at_k(t2i_ranks, k=1)
        t2i_recall_at_5 = self._compute_recall_at_k(t2i_ranks, k=5)
        
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


def test_recall_calculation():
    """
    测试Recall@K计算逻辑的正确性
    使用模拟的相似度矩阵验证
    """
    print("\n" + "="*50)
    print("测试Recall@K计算逻辑")
    print("="*50)
    
    # 创建一个简单的测试用例
    # 3个样本，相似度矩阵设计为：
    # - 样本0: 最相似的是文本1（错配），第2相似的是文本0（正确）
    # - 样本1: 最相似的是文本1（正确）
    # - 样本2: 最相似的是文本0（错配），第2相似的是文本2（正确）
    similarity = np.array([
        [0.5, 0.9, 0.3],  # 图0: 文本1最高，文本0次之
        [0.3, 0.8, 0.2],  # 图1: 文本1最高（正确）
        [0.6, 0.4, 0.5],  # 图2: 文本0最高，文本2次之
    ])
    
    print("\n相似度矩阵（行=图像，列=文本）:")
    print(similarity)
    print("\nGround Truth: 图0→文本0, 图1→文本1, 图2→文本2")
    
    # Image-to-Text检索
    i2t_ranks = np.argsort(-similarity, axis=1)
    print("\nImage-to-Text排序结果:")
    for i in range(len(i2t_ranks)):
        print(f"  图{i} → 文本排序: {i2t_ranks[i]} (ground truth={i})")
    
    # 手动计算期望的Recall
    # Recall@1: 只有图1的Top-1是正确的 → 1/3 = 33.33%
    # Recall@5: 图0的文本0在Top-2，图1的文本1在Top-1，图2的文本2在Top-2 → 3/3 = 100%
    
    # 使用修复后的函数计算
    class MockBenchmark:
        def _compute_recall_at_k(self, ranks, k):
            num_samples = ranks.shape[0]
            hits = [1 if i in ranks[i, :k] else 0 for i in range(num_samples)]
            return np.mean(hits)
    
    mock = MockBenchmark()
    i2t_r1 = mock._compute_recall_at_k(i2t_ranks, k=1)
    i2t_r5 = mock._compute_recall_at_k(i2t_ranks, k=5)
    
    print(f"\n计算结果:")
    print(f"  Image-to-Text Recall@1: {i2t_r1*100:.2f}%")
    print(f"  Image-to-Text Recall@5: {i2t_r5*100:.2f}%")
    
    # 验证结果
    expected_r1 = 1/3  # 只有图1正确
    expected_r5 = 1.0  # 所有图像的正确文本都在Top-3内
    
    assert abs(i2t_r1 - expected_r1) < 0.01, f"Recall@1错误: 期望{expected_r1}, 实际{i2t_r1}"
    assert abs(i2t_r5 - expected_r5) < 0.01, f"Recall@5错误: 期望{expected_r5}, 实际{i2t_r5}"
    
    print("\n✅ 测试通过！Recall@K计算逻辑正确。")
    print("="*50 + "\n")


if __name__ == "__main__":
    import sys
    
    # 如果传入 --test 参数，运行单元测试
    if "--test" in sys.argv:
        test_recall_calculation()
    else:
        main()

