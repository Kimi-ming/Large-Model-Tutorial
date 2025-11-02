"""
PyTorch推理服务

提供基于PyTorch的CLIP模型推理服务
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import List, Dict, Union, Tuple
import time
from pathlib import Path


class CLIPInferenceService:
    """
    CLIP推理服务
    
    支持图文匹配、图像特征提取、文本特征提取
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_fp16: bool = False
    ):
        """
        初始化推理服务
        
        Args:
            model_path: 模型路径或HuggingFace模型名称
            device: 计算设备 ("cuda", "cpu", "mps")
            use_fp16: 是否使用FP16混合精度
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        print(f"🚀 初始化CLIP推理服务...")
        print(f"   设备: {self.device}")
        print(f"   FP16: {self.use_fp16}")
        
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
        
        print(f"✅ 模型加载完成: {model_path}")
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """预热模型"""
        print("🔥 预热模型...")
        dummy_image = Image.new('RGB', (224, 224), color='white')
        dummy_text = ["warmup"]
        
        with torch.no_grad():
            inputs = self.processor(
                text=dummy_text,
                images=dummy_image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}
            
            _ = self.model(**inputs)
        
        print("✅ 预热完成")
    
    @torch.no_grad()
    def predict_image_text(
        self,
        image: Union[str, Image.Image],
        texts: List[str],
        return_probs: bool = True
    ) -> Dict:
        """
        图文匹配推理
        
        Args:
            image: 图像路径或PIL Image对象
            texts: 候选文本列表
            return_probs: 是否返回概率（否则返回logits）
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image必须是文件路径或PIL.Image对象")
        
        # 预处理
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # 推理
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image[0]
        
        # 计算概率或返回logits
        if return_probs:
            scores = logits.softmax(dim=0)
        else:
            scores = logits
        
        # 构建结果
        results = [
            {
                "text": text,
                "score": float(score),
                "rank": idx + 1
            }
            for idx, (text, score) in enumerate(
                sorted(zip(texts, scores.cpu().numpy()), 
                      key=lambda x: x[1], reverse=True)
            )
        ]
        
        inference_time = time.time() - start_time
        
        return {
            "results": results,
            "inference_time_ms": inference_time * 1000,
            "device": str(self.device),
            "fp16": self.use_fp16
        }
    
    @torch.no_grad()
    def get_image_features(
        self,
        images: Union[List[str], List[Image.Image]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            images: 图像路径列表或PIL Image列表
            normalize: 是否归一化特征向量
            
        Returns:
            图像特征张量 (batch_size, feature_dim)
        """
        # 加载图像
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError("图像必须是文件路径或PIL.Image对象")
        
        # 预处理
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # 提取特征
        features = self.model.get_image_features(**inputs)
        
        # 归一化
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    @torch.no_grad()
    def get_text_features(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        提取文本特征
        
        Args:
            texts: 文本列表
            normalize: 是否归一化特征向量
            
        Returns:
            文本特征张量 (batch_size, feature_dim)
        """
        # 预处理
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # 提取特征
        features = self.model.get_text_features(**inputs)
        
        # 归一化
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算图像和文本特征的相似度
        
        Args:
            image_features: 图像特征 (N, D)
            text_features: 文本特征 (M, D)
            
        Returns:
            相似度矩阵 (N, M)
        """
        # 确保特征已归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算余弦相似度
        similarity = image_features @ text_features.T
        
        # 缩放（CLIP的logit_scale）
        similarity = similarity * self.model.logit_scale.exp().item()
        
        return similarity


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP推理服务示例")
    parser.add_argument(
        '--model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='模型路径或名称'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='图像路径'
    )
    parser.add_argument(
        '--texts',
        type=str,
        nargs='+',
        required=True,
        help='候选文本列表'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='计算设备'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用FP16混合精度'
    )
    
    args = parser.parse_args()
    
    # 创建推理服务
    service = CLIPInferenceService(
        model_path=args.model,
        device=args.device,
        use_fp16=args.fp16
    )
    
    # 推理
    print(f"\n🖼️  图像: {args.image}")
    print(f"📝 候选文本: {args.texts}")
    print("\n" + "=" * 60)
    
    results = service.predict_image_text(
        image=args.image,
        texts=args.texts
    )
    
    print(f"\n⏱️  推理时间: {results['inference_time_ms']:.2f}ms")
    print(f"🖥️  设备: {results['device']}")
    print(f"🔢 FP16: {results['fp16']}")
    
    print("\n📊 预测结果:")
    for result in results['results']:
        print(f"  {result['rank']}. {result['text']}")
        print(f"     得分: {result['score']:.4f}")


if __name__ == '__main__':
    main()

