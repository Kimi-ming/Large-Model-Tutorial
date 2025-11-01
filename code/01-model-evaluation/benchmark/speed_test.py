#!/usr/bin/env python3
"""
模型推理速度基准测试
测试不同batch size和输入尺寸下的推理速度
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SpeedBenchmark:
    """推理速度测试器"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
    def load_model(self):
        """加载模型"""
        print(f"Loading model: {self.model_name}")
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            cache_dir="./models"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir="./models"
        )
        self.model.eval()
        print("Model loaded successfully!")
        
    def warmup(self, num_iterations: int = 10):
        """预热GPU"""
        print(f"Warming up GPU with {num_iterations} iterations...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if hasattr(self.model, 'get_image_features'):
                    # CLIP模型
                    self.model.get_image_features(pixel_values=dummy_input)
                else:
                    # 通用模型
                    try:
                        self.model(pixel_values=dummy_input)
                    except:
                        pass
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        print("Warmup completed!")
        
    def benchmark_batch(
        self, 
        image_paths: List[str], 
        batch_size: int = 1,
        num_iterations: int = 50
    ) -> Dict[str, float]:
        """
        批处理速度测试
        
        Args:
            image_paths: 测试图像路径列表
            batch_size: 批处理大小
            num_iterations: 测试迭代次数
            
        Returns:
            测试结果字典
        """
        print(f"\nTesting batch_size={batch_size}, iterations={num_iterations}")
        
        # 准备输入
        images = [Image.open(p).convert("RGB") for p in image_paths[:batch_size]]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 测试推理时间
        times = []
        with torch.no_grad():
            for i in range(num_iterations):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                
                outputs = self.model(**inputs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)
                
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{num_iterations}")
        
        # 统计结果
        times = np.array(times)
        results = {
            "batch_size": batch_size,
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "throughput": batch_size / np.mean(times),  # images/sec
            "latency": np.mean(times) * 1000 / batch_size,  # ms/image
        }
        
        print(f"  Results: {results['throughput']:.2f} images/sec, "
              f"{results['latency']:.2f} ms/image")
        
        return results
    
    def run_full_benchmark(
        self, 
        image_dir: str,
        batch_sizes: List[int] = [1, 2, 4, 8]
    ) -> List[Dict]:
        """运行完整的速度基准测试"""
        image_paths = list(Path(image_dir).glob("*.jpg"))[:50]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(image_paths)} test images")
        
        # 加载模型和预热
        self.load_model()
        self.warmup()
        
        # 测试不同batch size
        all_results = []
        for bs in batch_sizes:
            if bs > len(image_paths):
                print(f"Skipping batch_size={bs} (not enough images)")
                continue
                
            result = self.benchmark_batch(image_paths, batch_size=bs)
            all_results.append(result)
        
        return all_results


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="模型推理速度测试")
    parser.add_argument("--model", type=str, required=True, 
                       help="模型名称或路径")
    parser.add_argument("--image_dir", type=str, default="data/test_dataset",
                       help="测试图像目录")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4],
                       help="测试的batch size列表")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="结果保存路径")
    
    args = parser.parse_args()
    
    # 运行测试
    benchmark = SpeedBenchmark(args.model, args.device)
    results = benchmark.run_full_benchmark(args.image_dir, args.batch_sizes)
    
    # 保存结果
    import json
    with open(args.output, 'w') as f:
        json.dump({
            "model": args.model,
            "device": args.device,
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()

