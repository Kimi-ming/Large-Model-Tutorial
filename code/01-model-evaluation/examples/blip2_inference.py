"""
BLIP-2推理示例

本脚本演示BLIP-2模型的各种推理场景：
- 图像描述 (Image Captioning)
- 视觉问答 (Visual Question Answering)
- 图像-文本检索 (Image-Text Retrieval)
- 批量推理
- 性能评估

支持的模型：
- Salesforce/blip2-opt-2.7b
- Salesforce/blip2-opt-6.7b
- Salesforce/blip2-flan-t5-xl
- Salesforce/blip2-flan-t5-xxl

作者: Large-Model-Tutorial
日期: 2025-11-02
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import warnings

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# 导入transformers
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    print("❌ 错误: 需要安装transformers库")
    print("   安装方法: pip install transformers")
    sys.exit(1)

# 可选依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  警告: matplotlib未安装，可视化功能将不可用")

warnings.filterwarnings('ignore')


class BLIP2InferenceService:
    """BLIP-2推理服务类"""
    
    SUPPORTED_MODELS = {
        'opt-2.7b': 'Salesforce/blip2-opt-2.7b',
        'opt-6.7b': 'Salesforce/blip2-opt-6.7b',
        'flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',
        'flan-t5-xxl': 'Salesforce/blip2-flan-t5-xxl',
    }
    
    def __init__(
        self,
        model_name: str = 'opt-2.7b',
        device: Optional[str] = None,
        torch_dtype: str = 'float16',
        cache_dir: Optional[str] = None
    ):
        """
        初始化BLIP-2推理服务
        
        Args:
            model_name: 模型名称或路径
            device: 设备 ('cuda', 'cpu', 或None自动选择)
            torch_dtype: 数据类型 ('float16' 或 'float32')
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if torch_dtype == 'float16' else torch.float32
        self.cache_dir = cache_dir
        
        print(f"🚀 初始化BLIP-2推理服务")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
        print(f"   数据类型: {torch_dtype}")
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        # 获取模型路径
        if self.model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[self.model_name]
        else:
            model_path = self.model_name
        
        print(f"📥 加载模型: {model_path}")
        start_time = time.time()
        
        try:
            # 加载处理器
            self.processor = Blip2Processor.from_pretrained(
                model_path,
                cache_dir=self.cache_dir
            )
            
            # 加载模型
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            print(f"✅ 模型加载完成 ({load_time:.2f}秒)")
            
            # 打印模型信息
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n📊 模型信息:")
        print(f"   总参数: {total_params / 1e9:.2f}B")
        print(f"   可训练参数: {trainable_params / 1e9:.2f}B")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"   显存占用: {memory_allocated:.2f}GB")
    
    def generate_caption(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 50,
        **generate_kwargs
    ) -> str:
        """
        生成图像描述
        
        Args:
            image: 图像路径或PIL Image对象
            prompt: 可选的提示文本
            max_new_tokens: 最大生成token数
            **generate_kwargs: 其他生成参数
        
        Returns:
            生成的描述文本
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 处理输入
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
        
        # 解码
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return generated_text
    
    def visual_question_answering(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 30,
        **generate_kwargs
    ) -> str:
        """
        视觉问答
        
        Args:
            image: 图像路径或PIL Image对象
            question: 问题文本
            max_new_tokens: 最大生成token数
            **generate_kwargs: 其他生成参数
        
        Returns:
            答案文本
        """
        # 构建提示
        prompt = f"Question: {question} Answer:"
        
        # 生成答案
        answer = self.generate_caption(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        
        # 清理答案（移除重复的prompt）
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        
        return answer
    
    def batch_inference(
        self,
        images: List[Union[str, Image.Image]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        max_new_tokens: int = 50,
        **generate_kwargs
    ) -> List[str]:
        """
        批量推理
        
        Args:
            images: 图像列表
            prompts: 提示列表（可选）
            batch_size: 批大小
            max_new_tokens: 最大生成token数
            **generate_kwargs: 其他生成参数
        
        Returns:
            生成的文本列表
        """
        results = []
        
        # 如果没有提供prompts，使用None列表
        if prompts is None:
            prompts = [None] * len(images)
        
        # 分批处理
        for i in tqdm(range(0, len(images), batch_size), desc="批量推理"):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            # 加载图像
            pil_images = []
            for img in batch_images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert('RGB'))
                else:
                    pil_images.append(img)
            
            # 处理输入
            if any(p is not None for p in batch_prompts):
                # 有提示的情况
                inputs = self.processor(
                    images=pil_images,
                    text=batch_prompts,
                    return_tensors="pt",
                    padding=True
                )
            else:
                # 无提示的情况
                inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt"
                )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs
                )
            
            # 解码
            texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            results.extend([t.strip() for t in texts])
        
        return results
    
    def extract_features(
        self,
        image: Union[str, Image.Image],
        text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取图像和文本特征
        
        Args:
            image: 图像路径或PIL Image对象
            text: 可选的文本
        
        Returns:
            特征字典，包含'image_embeds'和可选的'text_embeds'
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 处理输入
        if text:
            inputs = self.processor(images=image, text=text, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 返回特征
        features = {}
        
        # 图像特征（Q-Former输出）
        if hasattr(outputs, 'vision_outputs'):
            features['image_embeds'] = outputs.vision_outputs[0][:, 0, :]  # [CLS] token
        
        # 文本特征（如果提供）
        if text and hasattr(outputs, 'language_model_outputs'):
            features['text_embeds'] = outputs.language_model_outputs.last_hidden_state.mean(dim=1)
        
        return features
    
    def compute_similarity(
        self,
        image: Union[str, Image.Image],
        text: str
    ) -> float:
        """
        计算图像-文本相似度
        
        Args:
            image: 图像路径或PIL Image对象
            text: 文本
        
        Returns:
            相似度分数（0-1）
        """
        features = self.extract_features(image, text)
        
        if 'image_embeds' in features and 'text_embeds' in features:
            image_embed = F.normalize(features['image_embeds'], dim=-1)
            text_embed = F.normalize(features['text_embeds'], dim=-1)
            
            similarity = (image_embed * text_embed).sum().item()
            return (similarity + 1) / 2  # 归一化到0-1
        else:
            print("⚠️  警告: 无法提取特征，返回0")
            return 0.0
    
    def benchmark(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            image: 测试图像
            prompt: 测试提示
            num_runs: 运行次数
        
        Returns:
            性能指标字典
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 准备输入
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预热
        print("🔥 预热中...")
        for _ in range(3):
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=20)
        
        # 测试
        print(f"⏱️  运行基准测试 ({num_runs}次)...")
        times = []
        
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=50)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"   运行 {i+1}/{num_runs}: {elapsed*1000:.2f}ms")
        
        # 统计
        times = np.array(times)
        stats = {
            'mean_ms': float(times.mean() * 1000),
            'std_ms': float(times.std() * 1000),
            'min_ms': float(times.min() * 1000),
            'max_ms': float(times.max() * 1000),
            'throughput_imgs_per_sec': float(1.0 / times.mean()),
        }
        
        # 显存统计
        if torch.cuda.is_available():
            stats['memory_allocated_gb'] = torch.cuda.memory_allocated(self.device) / 1024**3
            stats['memory_reserved_gb'] = torch.cuda.max_memory_reserved(self.device) / 1024**3
        
        return stats


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BLIP-2推理示例')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='opt-2.7b',
                        choices=list(BLIP2InferenceService.SUPPORTED_MODELS.keys()),
                        help='模型名称')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='数据类型')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='模型缓存目录')
    
    # 任务参数
    parser.add_argument('--task', type=str, default='caption',
                        choices=['caption', 'vqa', 'batch', 'similarity', 'benchmark'],
                        help='任务类型')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--prompt', type=str, default=None,
                        help='提示文本')
    parser.add_argument('--question', type=str, default=None,
                        help='VQA问题')
    parser.add_argument('--text', type=str, default=None,
                        help='用于相似度计算的文本')
    
    # 生成参数
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='最大生成token数')
    parser.add_argument('--num-beams', type=int, default=1,
                        help='束搜索大小')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='核采样参数')
    
    # 其他参数
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--benchmark-runs', type=int, default=10,
                        help='基准测试运行次数')
    
    args = parser.parse_args()
    
    # 初始化服务
    service = BLIP2InferenceService(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        cache_dir=args.cache_dir
    )
    
    # 生成参数
    generate_kwargs = {
        'num_beams': args.num_beams,
        'temperature': args.temperature,
        'top_p': args.top_p,
    }
    
    # 执行任务
    print(f"\n🎯 执行任务: {args.task}")
    
    if args.task == 'caption':
        # 图像描述
        caption = service.generate_caption(
            image=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            **generate_kwargs
        )
        print(f"\n📝 生成的描述:")
        print(f"   {caption}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(caption)
            print(f"\n💾 已保存到: {args.output}")
    
    elif args.task == 'vqa':
        # 视觉问答
        if not args.question:
            print("❌ 错误: VQA任务需要提供--question参数")
            return
        
        answer = service.visual_question_answering(
            image=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            **generate_kwargs
        )
        print(f"\n❓ 问题: {args.question}")
        print(f"💡 答案: {answer}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({'question': args.question, 'answer': answer}, f, ensure_ascii=False, indent=2)
            print(f"\n💾 已保存到: {args.output}")
    
    elif args.task == 'similarity':
        # 图像-文本相似度
        if not args.text:
            print("❌ 错误: similarity任务需要提供--text参数")
            return
        
        similarity = service.compute_similarity(
            image=args.image,
            text=args.text
        )
        print(f"\n📊 相似度: {similarity:.4f}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({'image': args.image, 'text': args.text, 'similarity': similarity}, f, ensure_ascii=False, indent=2)
            print(f"\n💾 已保存到: {args.output}")
    
    elif args.task == 'benchmark':
        # 性能基准测试
        stats = service.benchmark(
            image=args.image,
            prompt=args.prompt,
            num_runs=args.benchmark_runs
        )
        
        print(f"\n📊 基准测试结果:")
        print(f"   平均延迟: {stats['mean_ms']:.2f}ms (±{stats['std_ms']:.2f}ms)")
        print(f"   最小/最大: {stats['min_ms']:.2f}ms / {stats['max_ms']:.2f}ms")
        print(f"   吞吐量: {stats['throughput_imgs_per_sec']:.2f} images/sec")
        
        if 'memory_allocated_gb' in stats:
            print(f"   显存占用: {stats['memory_allocated_gb']:.2f}GB")
            print(f"   显存峰值: {stats['memory_reserved_gb']:.2f}GB")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"\n💾 已保存到: {args.output}")
    
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()

