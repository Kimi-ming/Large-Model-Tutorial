"""
华为昇腾性能测试工具

对比NPU、CUDA、CPU的推理性能
"""

import time
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import torch
    import torch_npu
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages:")
    print("pip install torch transformers pillow")
    exit(1)


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, model_path: str = "openai/clip-vit-base-patch32"):
        """
        初始化基准测试
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.results = {}
    
    def benchmark_device(
        self,
        device: str,
        image_path: str,
        texts: List[str],
        num_runs: int = 100,
        warmup_runs: int = 10,
        use_fp16: bool = False
    ) -> Dict:
        """
        在指定设备上进行基准测试
        
        Args:
            device: 设备名称 ('npu', 'cuda', 'cpu')
            image_path: 测试图像路径
            texts: 测试文本列表
            num_runs: 运行次数
            warmup_runs: 预热次数
            use_fp16: 是否使用FP16
            
        Returns:
            性能统计字典
        """
        print(f"\n{'='*60}")
        print(f"测试设备: {device.upper()}")
        print(f"FP16: {use_fp16}")
        print(f"{'='*60}")
        
        # 检查设备可用性
        if device == 'npu':
            if not torch.npu.is_available():
                print(f"⚠️  NPU不可用，跳过测试")
                return None
            device_obj = torch.device('npu:0')
        elif device == 'cuda':
            if not torch.cuda.is_available():
                print(f"⚠️  CUDA不可用，跳过测试")
                return None
            device_obj = torch.device('cuda:0')
        else:
            device_obj = torch.device('cpu')
        
        # 加载模型
        print(f"📥 加载模型...")
        model = CLIPModel.from_pretrained(self.model_path)
        processor = CLIPProcessor.from_pretrained(self.model_path)
        
        model = model.to(device_obj)
        
        if use_fp16 and device in ['npu', 'cuda']:
            model = model.half()
        
        model.eval()
        
        # 准备输入
        image = Image.open(image_path).convert('RGB')
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}
        
        if use_fp16 and device in ['npu', 'cuda']:
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].half()
        
        # 预热
        print(f"🔥 预热中... ({warmup_runs}次)")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(**inputs)
            
            if device == 'npu':
                torch.npu.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
        
        # 基准测试
        print(f"⏱️  运行基准测试... ({num_runs}次)")
        times = []
        
        for i in range(num_runs):
            # 同步
            if device == 'npu':
                torch.npu.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 同步
            if device == 'npu':
                torch.npu.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            times.append(elapsed * 1000)  # 转换为ms
            
            if (i + 1) % 20 == 0:
                print(f"   进度: {i + 1}/{num_runs}")
        
        times = np.array(times)
        
        # 统计
        stats = {
            'device': device,
            'fp16': use_fp16,
            'num_runs': num_runs,
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_per_sec': float(1000.0 / times.mean())
        }
        
        # 显存统计
        if device == 'npu':
            stats['memory_allocated_mb'] = torch.npu.memory_allocated() / 1024 / 1024
            stats['memory_reserved_mb'] = torch.npu.memory_reserved() / 1024 / 1024
        elif device == 'cuda':
            stats['memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        # 打印结果
        print(f"\n📊 性能统计:")
        print(f"   平均延迟: {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f}ms)")
        print(f"   中位数: {stats['p50_ms']:.2f}ms")
        print(f"   P95: {stats['p95_ms']:.2f}ms")
        print(f"   P99: {stats['p99_ms']:.2f}ms")
        print(f"   最小/最大: {stats['min_ms']:.2f}ms / {stats['max_ms']:.2f}ms")
        print(f"   吞吐量: {stats['throughput_per_sec']:.2f} images/sec")
        
        if 'memory_allocated_mb' in stats:
            print(f"   显存占用: {stats['memory_allocated_mb']:.2f}MB")
            print(f"   显存峰值: {stats['memory_reserved_mb']:.2f}MB")
        
        # 清理
        del model
        if device == 'npu':
            torch.npu.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()
        
        return stats
    
    def run_comparison(
        self,
        image_path: str,
        texts: List[str],
        devices: List[str] = None,
        num_runs: int = 100,
        test_fp16: bool = True
    ) -> Dict:
        """
        运行多设备对比测试
        
        Args:
            image_path: 测试图像路径
            texts: 测试文本列表
            devices: 设备列表，None表示测试所有可用设备
            num_runs: 每个设备的运行次数
            test_fp16: 是否测试FP16
            
        Returns:
            所有测试结果
        """
        if devices is None:
            devices = []
            if torch.npu.is_available():
                devices.append('npu')
            if torch.cuda.is_available():
                devices.append('cuda')
            devices.append('cpu')  # CPU总是可用
        
        results = {}
        
        for device in devices:
            # FP32测试
            result_fp32 = self.benchmark_device(
                device=device,
                image_path=image_path,
                texts=texts,
                num_runs=num_runs,
                warmup_runs=10,
                use_fp16=False
            )
            
            if result_fp32:
                results[f"{device}_fp32"] = result_fp32
            
            # FP16测试（仅NPU和CUDA）
            if test_fp16 and device in ['npu', 'cuda']:
                result_fp16 = self.benchmark_device(
                    device=device,
                    image_path=image_path,
                    texts=texts,
                    num_runs=num_runs,
                    warmup_runs=10,
                    use_fp16=True
                )
                
                if result_fp16:
                    results[f"{device}_fp16"] = result_fp16
        
        self.results = results
        return results
    
    def print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            print("没有测试结果")
            return
        
        print(f"\n{'='*80}")
        print("性能对比表")
        print(f"{'='*80}")
        print(f"{'配置':<20} {'平均延迟':<15} {'吞吐量':<15} {'显存占用':<15} {'相对性能'}")
        print(f"{'-'*80}")
        
        # 找出基准（最快的）
        baseline_throughput = max(
            r['throughput_per_sec'] for r in self.results.values()
        )
        
        for config, result in sorted(self.results.items()):
            device = config.split('_')[0].upper()
            precision = config.split('_')[1].upper()
            config_name = f"{device} ({precision})"
            
            mean_ms = result['mean_ms']
            throughput = result['throughput_per_sec']
            memory = result.get('memory_allocated_mb', 0)
            relative_perf = (throughput / baseline_throughput) * 100
            
            print(f"{config_name:<20} {mean_ms:>8.2f}ms {throughput:>10.2f}/s "
                  f"{memory:>10.2f}MB {relative_perf:>10.1f}%")
        
        print(f"{'='*80}\n")
    
    def save_results(self, output_path: str):
        """保存结果到JSON文件"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 结果已保存: {output_path}")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='华为昇腾性能基准测试')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                        help='模型路径')
    parser.add_argument('--image', type=str, required=True,
                        help='测试图像路径')
    parser.add_argument('--texts', type=str, nargs='+',
                        default=['a photo of a cat', 'a photo of a dog'],
                        help='测试文本')
    parser.add_argument('--devices', type=str, nargs='+',
                        choices=['npu', 'cuda', 'cpu'],
                        help='测试设备列表（默认测试所有可用设备）')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='每个设备的运行次数')
    parser.add_argument('--no-fp16', action='store_true',
                        help='跳过FP16测试')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='结果输出路径')
    
    args = parser.parse_args()
    
    # 运行基准测试
    benchmark = PerformanceBenchmark(model_path=args.model)
    
    results = benchmark.run_comparison(
        image_path=args.image,
        texts=args.texts,
        devices=args.devices,
        num_runs=args.num_runs,
        test_fp16=not args.no_fp16
    )
    
    # 打印对比表
    benchmark.print_comparison_table()
    
    # 保存结果
    benchmark.save_results(args.output)


if __name__ == '__main__':
    main()

