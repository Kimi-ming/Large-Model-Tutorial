"""
PyTorch-NPU推理服务

提供基于华为昇腾NPU的CLIP模型推理服务
支持自动设备选择（NPU/CUDA/CPU）
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional
import time
from pathlib import Path
import warnings

# 尝试导入torch_npu
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    warnings.warn("torch_npu not available, NPU will not be used")


class CLIPInferenceService:
    """
    CLIP推理服务（昇腾NPU适配版）
    
    支持图文匹配、图像特征提取、文本特征提取
    自动选择最优设备：NPU > CUDA > CPU
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_fp16: bool = False
    ):
        """
        初始化推理服务
        
        Args:
            model_path: 模型路径或HuggingFace模型名称
            device: 计算设备 ("auto", "npu", "cuda", "cpu")
            use_fp16: 是否使用FP16混合精度
        """
        self.device = self._get_device(device)
        self.use_fp16 = use_fp16 and self.device.type in ['npu', 'cuda']
        
        print(f"🚀 初始化CLIP推理服务（昇腾适配版）...")
        print(f"   设备: {self.device}")
        print(f"   FP16: {self.use_fp16}")
        print(f"   NPU可用: {NPU_AVAILABLE and torch.npu.is_available()}")
        
        # 加载模型和处理器
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 转换为FP16
        if self.use_fp16:
            self.model = self.model.half()
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"✅ 模型加载完成")
    
    def _get_device(self, device: str) -> torch.device:
        """
        智能选择设备
        
        Args:
            device: 设备字符串
            
        Returns:
            torch.device对象
        """
        if device == "auto":
            # 优先级：NPU > CUDA > CPU
            if NPU_AVAILABLE and torch.npu.is_available():
                return torch.device("npu:0")
            elif torch.cuda.is_available():
                return torch.device("cuda:0")
            else:
                return torch.device("cpu")
        elif device.startswith("npu"):
            if not NPU_AVAILABLE:
                raise RuntimeError("torch_npu not installed")
            if not torch.npu.is_available():
                raise RuntimeError("NPU not available")
            return torch.device(device)
        else:
            return torch.device(device)
    
    def predict(
        self,
        image: Union[str, Image.Image],
        texts: List[str]
    ) -> Dict[str, any]:
        """
        图文匹配推理
        
        Args:
            image: 图像路径或PIL Image对象
            texts: 候选文本列表
            
        Returns:
            包含预测结果的字典
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 处理输入
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 转换为FP16
        if self.use_fp16:
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].half()
        
        # 推理
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # 同步设备（对于NPU很重要）
        if self.device.type == 'npu':
            torch.npu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        
        # 结果处理
        probs_cpu = probs.cpu().numpy()
        
        results = {
            'texts': texts,
            'probabilities': probs_cpu[0].tolist(),
            'best_match': texts[probs_cpu[0].argmax()],
            'best_score': float(probs_cpu[0].max()),
            'inference_time_ms': inference_time * 1000,
            'device': str(self.device)
        }
        
        return results
    
    def extract_image_features(
        self,
        images: Union[List[str], List[Image.Image]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            images: 图像列表
            normalize: 是否归一化特征
            
        Returns:
            图像特征张量 [batch_size, feature_dim]
        """
        # 加载图像
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)
        
        # 处理输入
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        
        # 提取特征
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def extract_text_features(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        提取文本特征
        
        Args:
            texts: 文本列表
            normalize: 是否归一化特征
            
        Returns:
            文本特征张量 [batch_size, feature_dim]
        """
        # 处理输入
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def batch_predict(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        batch_size: int = 4
    ) -> List[Dict]:
        """
        批量推理
        
        Args:
            images: 图像列表
            texts: 文本列表（对所有图像通用）
            batch_size: 批大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            for img in batch_images:
                result = self.predict(img, texts)
                results.append(result)
        
        return results
    
    def benchmark(
        self,
        image: Union[str, Image.Image],
        texts: List[str],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            image: 测试图像
            texts: 测试文本
            num_runs: 测试次数
            warmup_runs: 预热次数
            
        Returns:
            性能统计
        """
        print(f"🔥 预热中... ({warmup_runs}次)")
        for _ in range(warmup_runs):
            self.predict(image, texts)
        
        print(f"⏱️  开始基准测试... ({num_runs}次)")
        times = []
        
        for i in range(num_runs):
            result = self.predict(image, texts)
            times.append(result['inference_time_ms'])
            
            if (i + 1) % 20 == 0:
                print(f"   进度: {i + 1}/{num_runs}")
        
        import numpy as np
        times = np.array(times)
        
        stats = {
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_per_sec': 1000.0 / times.mean(),
            'device': str(self.device)
        }
        
        # 显存统计
        if self.device.type == 'npu' and NPU_AVAILABLE:
            stats['memory_allocated_mb'] = torch.npu.memory_allocated() / 1024 / 1024
            stats['memory_reserved_mb'] = torch.npu.memory_reserved() / 1024 / 1024
        elif self.device.type == 'cuda':
            stats['memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return stats
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'device'):
            if self.device.type == 'npu' and NPU_AVAILABLE:
                torch.npu.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLIP推理服务（昇腾NPU适配）')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                        help='模型路径或名称')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--texts', type=str, nargs='+',
                        default=['a photo of a cat', 'a photo of a dog'],
                        help='候选文本')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'npu', 'cuda', 'cpu'],
                        help='设备选择')
    parser.add_argument('--fp16', action='store_true',
                        help='使用FP16')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行基准测试')
    
    args = parser.parse_args()
    
    # 初始化服务
    service = CLIPInferenceService(
        model_path=args.model,
        device=args.device,
        use_fp16=args.fp16
    )
    
    if args.benchmark:
        # 基准测试
        stats = service.benchmark(args.image, args.texts, num_runs=100)
        
        print(f"\n📊 性能统计:")
        print(f"   平均延迟: {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f}ms)")
        print(f"   P50: {stats['p50_ms']:.2f}ms")
        print(f"   P95: {stats['p95_ms']:.2f}ms")
        print(f"   P99: {stats['p99_ms']:.2f}ms)")
        print(f"   吞吐量: {stats['throughput_per_sec']:.2f} images/sec")
        if 'memory_allocated_mb' in stats:
            print(f"   显存占用: {stats['memory_allocated_mb']:.2f}MB")
    else:
        # 单次推理
        result = service.predict(args.image, args.texts)
        
        print(f"\n📝 推理结果:")
        for text, prob in zip(result['texts'], result['probabilities']):
            print(f"   {text}: {prob:.4f}")
        print(f"\n🏆 最佳匹配: {result['best_match']} ({result['best_score']:.4f})")
        print(f"⏱️  推理时间: {result['inference_time_ms']:.2f}ms")
        print(f"💻 设备: {result['device']}")


if __name__ == '__main__':
    main()

