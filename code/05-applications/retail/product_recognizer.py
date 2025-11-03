"""
商品识别器

使用CLIP模型进行零样本商品识别
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from typing import List, Dict, Union
import json
from pathlib import Path


class ProductRecognizer:
    """
    商品识别器
    
    使用CLIP进行零样本商品识别，支持：
    - SKU级别识别
    - 多商品匹配
    - 置信度评估
    """
    
    def __init__(
        self,
        model_path: str = "openai/clip-vit-base-patch32",
        product_database: str = None,
        device: str = "auto",
        confidence_threshold: float = 0.7
    ):
        """
        初始化商品识别器
        
        Args:
            model_path: CLIP模型路径
            product_database: 商品数据库JSON文件路径
            device: 计算设备
            confidence_threshold: 置信度阈值
        """
        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        
        print(f"🚀 初始化商品识别器...")
        print(f"   设备: {self.device}")
        print(f"   置信度阈值: {confidence_threshold}")
        
        # 加载模型
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载商品数据库
        self.products = self._load_products(product_database)
        print(f"   商品数量: {len(self.products)}")
        
        # 预计算商品文本特征
        self._precompute_text_features()
        
        print(f"✅ 初始化完成")
    
    def _get_device(self, device: str) -> torch.device:
        """选择设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_products(self, database_path: str) -> List[Dict]:
        """加载商品数据库"""
        if database_path and Path(database_path).exists():
            with open(database_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认演示数据
            return [
                {
                    "sku": "SKU-001",
                    "name": "可口可乐 330ml",
                    "category": "饮料",
                    "brand": "可口可乐",
                    "price": 3.5,
                    "description": "可口可乐经典罐装饮料 330毫升"
                },
                {
                    "sku": "SKU-002",
                    "name": "雪碧 330ml",
                    "category": "饮料",
                    "brand": "可口可乐",
                    "price": 3.5,
                    "description": "雪碧柠檬味汽水 330毫升"
                },
                {
                    "sku": "SKU-003",
                    "name": "农夫山泉 550ml",
                    "category": "饮料",
                    "brand": "农夫山泉",
                    "price": 2.0,
                    "description": "农夫山泉天然水 550毫升"
                },
                {
                    "sku": "SKU-004",
                    "name": "奥利奥饼干",
                    "category": "零食",
                    "brand": "奥利奥",
                    "price": 9.9,
                    "description": "奥利奥夹心饼干原味"
                },
                {
                    "sku": "SKU-005",
                    "name": "旺旺雪饼",
                    "category": "零食",
                    "brand": "旺旺",
                    "price": 5.5,
                    "description": "旺旺雪饼膨化食品"
                }
            ]
    
    def _precompute_text_features(self):
        """预计算所有商品的文本特征"""
        print(f"🔄 预计算商品文本特征...")
        
        texts = [f"{p['name']} {p['description']}" for p in self.products]
        
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.text_features = text_features.cpu().numpy()
    
    def recognize(
        self,
        image: Union[str, Image.Image],
        top_k: int = 5
    ) -> Dict:
        """
        识别商品
        
        Args:
            image: 图像路径或PIL Image
            top_k: 返回top-k结果
            
        Returns:
            识别结果字典
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 提取图像特征
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        image_features = image_features.cpu().numpy()
        
        # 计算相似度
        similarities = (image_features @ self.text_features.T)[0]
        
        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            product = self.products[idx].copy()
            product['confidence'] = float(similarities[idx])
            product['match'] = similarities[idx] >= self.confidence_threshold
            results.append(product)
        
        # 最佳匹配
        best = results[0]
        
        return {
            'best_match': best,
            'top_k_matches': results,
            'recognized': best['confidence'] >= self.confidence_threshold
        }
    
    def batch_recognize(
        self,
        images: List[Union[str, Image.Image]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        批量识别商品
        
        Args:
            images: 图像列表
            top_k: 返回top-k结果
            
        Returns:
            识别结果列表
        """
        results = []
        for image in images:
            result = self.recognize(image, top_k=top_k)
            results.append(result)
        return results


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='商品识别器')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--database', type=str, help='商品数据库JSON文件')
    parser.add_argument('--top-k', type=int, default=5, help='返回top-k结果')
    parser.add_argument('--threshold', type=float, default=0.7, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 初始化识别器
    recognizer = ProductRecognizer(
        product_database=args.database,
        confidence_threshold=args.threshold
    )
    
    # 识别商品
    result = recognizer.recognize(args.image, top_k=args.top_k)
    
    # 打印结果
    print(f"\n📝 识别结果:")
    print(f"="*60)
    
    best = result['best_match']
    print(f"🏆 最佳匹配:")
    print(f"   商品名称: {best['name']}")
    print(f"   SKU: {best['sku']}")
    print(f"   类别: {best['category']}")
    print(f"   品牌: {best['brand']}")
    print(f"   价格: ¥{best['price']}")
    print(f"   置信度: {best['confidence']:.2%}")
    print(f"   匹配: {'✅ 是' if best['match'] else '❌ 否'}")
    
    print(f"\n📊 Top-{args.top_k} 匹配:")
    for i, match in enumerate(result['top_k_matches'], 1):
        print(f"{i}. {match['name']} (置信度: {match['confidence']:.2%})")


if __name__ == '__main__':
    main()

