#!/usr/bin/env python3
"""
GPU显存占用基准测试
"""

import torch
try:
    import GPUtil
except ImportError:
    GPUtil = None
from transformers import AutoModel, AutoProcessor


class MemoryBenchmark:
    """显存测试器"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        
    def measure_memory(self, batch_size: int = 1) -> dict:
        """测量模型加载和推理的显存占用"""
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # 清空显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始显存
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        
        # 加载模型
        print(f"Loading model: {self.model_name}")
        model = AutoModel.from_pretrained(
            self.model_name, cache_dir="./models"
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir="./models"
        )
        
        after_loading = torch.cuda.memory_allocated() / 1024**3
        
        # 创建dummy输入
        dummy_image = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # 推理
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'get_image_features'):
                model.get_image_features(pixel_values=dummy_image)
            else:
                try:
                    model(pixel_values=dummy_image)
                except:
                    pass
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        current_memory = torch.cuda.memory_allocated() / 1024**3
        
        return {
            "model": self.model_name,
            "batch_size": batch_size,
            "initial_memory_gb": round(initial_memory, 2),
            "model_size_gb": round(after_loading - initial_memory, 2),
            "peak_memory_gb": round(peak_memory, 2),
            "current_memory_gb": round(current_memory, 2),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="显存占用测试")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    benchmark = MemoryBenchmark(args.model)
    result = benchmark.measure_memory(args.batch_size)
    
    print("\n=== Memory Benchmark Results ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

