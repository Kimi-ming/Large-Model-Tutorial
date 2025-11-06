"""
ONNX推理服务

使用ONNX Runtime进行CLIP模型推理
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from typing import List, Dict, Union
import time
from torchvision import transforms


class ONNXCLIPInferenceService:
    """
    基于ONNX Runtime的CLIP推理服务
    """
    
    def __init__(
        self,
        vision_model_path: str,
        text_model_path: str = None,
        use_gpu: bool = True
    ):
        """
        初始化ONNX推理服务
        
        Args:
            vision_model_path: 视觉编码器ONNX模型路径
            text_model_path: 文本编码器ONNX模型路径（可选）
            use_gpu: 是否使用GPU
        """
        print(f"🚀 初始化ONNX CLIP推理服务...")
        
        # 检查可用的providers
        available_providers = ort.get_available_providers()
        print(f"📋 可用的Execution Providers: {available_providers}")
        
        # 智能选择providers
        if use_gpu:
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"✅ 使用GPU推理 (CUDA)")
            else:
                providers = ['CPUExecutionProvider']
                print(f"⚠️  CUDA不可用，回退到CPU推理")
                print(f"💡 提示: 安装 onnxruntime-gpu 以启用GPU加速")
                print(f"   pip install onnxruntime-gpu")
        else:
            providers = ['CPUExecutionProvider']
            print(f"✅ 使用CPU推理")
        
        # 加载视觉编码器
        print(f"📦 加载视觉编码器: {vision_model_path}")
        self.vision_session = ort.InferenceSession(
            vision_model_path,
            providers=providers
        )
        
        # 加载文本编码器（如果提供）
        if text_model_path:
            print(f"📦 加载文本编码器: {text_model_path}")
            self.text_session = ort.InferenceSession(
                text_model_path,
                providers=providers
            )
        else:
            self.text_session = None
        
        # 获取输入输出名称
        self.vision_input_name = self.vision_session.get_inputs()[0].name
        self.vision_output_name = self.vision_session.get_outputs()[0].name
        
        if self.text_session:
            self.text_input_names = [inp.name for inp in self.text_session.get_inputs()]
            self.text_output_name = self.text_session.get_outputs()[0].name
        
        # 打印provider信息
        print(f"✅ 模型加载完成")
        print(f"   视觉编码器 Provider: {self.vision_session.get_providers()[0]}")
        if self.text_session:
            print(f"   文本编码器 Provider: {self.text_session.get_providers()[0]}")
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """预热模型"""
        print("🔥 预热模型...")
        dummy_image = Image.new('RGB', (224, 224), color='white')
        _ = self.get_image_features([dummy_image])
        print("✅ 预热完成")
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 图像路径或PIL Image对象
            
        Returns:
            预处理后的numpy数组
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image必须是文件路径或PIL.Image对象")
        
        # 应用变换
        image_tensor = self.image_transform(image)
        
        # 转换为numpy并添加batch维度
        image_np = image_tensor.numpy()[np.newaxis, :]
        
        return image_np.astype(np.float32)
    
    def get_image_features(
        self,
        images: List[Union[str, Image.Image]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        提取图像特征
        
        Args:
            images: 图像列表
            normalize: 是否归一化特征
            
        Returns:
            图像特征数组 (batch_size, feature_dim)
        """
        # 预处理所有图像
        image_arrays = [self.preprocess_image(img) for img in images]
        batch_images = np.concatenate(image_arrays, axis=0)
        
        # 推理
        features = self.vision_session.run(
            [self.vision_output_name],
            {self.vision_input_name: batch_images}
        )[0]
        
        # 归一化
        if normalize:
            features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        
        return features
    
    def get_text_features(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        提取文本特征
        
        Args:
            input_ids: token IDs
            attention_mask: 注意力掩码
            normalize: 是否归一化特征
            
        Returns:
            文本特征数组 (batch_size, feature_dim)
        """
        if self.text_session is None:
            raise ValueError("文本编码器未加载")
        
        # 推理
        features = self.text_session.run(
            [self.text_output_name],
            {
                self.text_input_names[0]: input_ids,
                self.text_input_names[1]: attention_mask
            }
        )[0]
        
        # 归一化
        if normalize:
            features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        
        return features
    
    def compute_similarity(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        logit_scale: float = 100.0
    ) -> np.ndarray:
        """
        计算图像和文本特征的相似度
        
        Args:
            image_features: 图像特征 (N, D)
            text_features: 文本特征 (M, D)
            logit_scale: logit缩放因子
            
        Returns:
            相似度矩阵 (N, M)
        """
        # 计算余弦相似度
        similarity = image_features @ text_features.T
        
        # 缩放
        similarity = similarity * logit_scale
        
        return similarity
    
    def predict_image_text(
        self,
        image: Union[str, Image.Image],
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        return_probs: bool = True
    ) -> Dict:
        """
        图文匹配推理
        
        Args:
            image: 图像
            input_ids: 文本token IDs
            attention_mask: 注意力掩码
            return_probs: 是否返回概率
            
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        # 提取特征
        image_features = self.get_image_features([image])
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # 计算相似度
        logits = self.compute_similarity(image_features, text_features)[0]
        
        # 计算概率
        if return_probs:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            scores = probs
        else:
            scores = logits
        
        inference_time = time.time() - start_time
        
        return {
            "scores": scores,
            "inference_time_ms": inference_time * 1000
        }


def main():
    """示例用法"""
    import argparse
    from transformers import CLIPTokenizer
    
    parser = argparse.ArgumentParser(description="ONNX CLIP推理示例")
    parser.add_argument(
        '--vision_model',
        type=str,
        required=True,
        help='视觉编码器ONNX模型路径'
    )
    parser.add_argument(
        '--text_model',
        type=str,
        required=True,
        help='文本编码器ONNX模型路径'
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
        '--cpu',
        action='store_true',
        help='使用CPU推理'
    )
    
    args = parser.parse_args()
    
    # 创建推理服务
    service = ONNXCLIPInferenceService(
        vision_model_path=args.vision_model,
        text_model_path=args.text_model,
        use_gpu=not args.cpu
    )
    
    # 加载tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # 处理文本
    text_inputs = tokenizer(
        args.texts,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='np'
    )
    
    # 推理
    print(f"\n🖼️  图像: {args.image}")
    print(f"📝 候选文本: {args.texts}")
    print("\n" + "=" * 60)
    
    results = service.predict_image_text(
        image=args.image,
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask']
    )
    
    print(f"\n⏱️  推理时间: {results['inference_time_ms']:.2f}ms")
    
    print("\n📊 预测结果:")
    sorted_indices = np.argsort(results['scores'])[::-1]
    for idx, i in enumerate(sorted_indices, 1):
        print(f"  {idx}. {args.texts[i]}")
        print(f"     得分: {results['scores'][i]:.4f}")


if __name__ == '__main__':
    main()

