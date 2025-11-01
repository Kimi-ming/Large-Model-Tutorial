#!/usr/bin/env python3
"""
CLIP 推理示例

演示如何使用CLIP模型进行图文匹配
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import requests
from io import BytesIO

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("❌ 错误: 未安装transformers库")
    print("请运行: pip install transformers")
    sys.exit(1)


def load_clip_model(model_name="openai/clip-vit-base-patch32", device=None):
    """
    加载CLIP模型
    
    Args:
        model_name: 模型名称或路径
        device: 设备 (None=自动检测)
    
    Returns:
        (model, processor)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[1/3] 加载模型: {model_name}")
    print(f"使用设备: {device}")
    
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        print("✓ 模型加载成功")
        return model, processor, device
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n提示:")
        print("  1. 检查网络连接")
        print("  2. 尝试手动下载模型: ./scripts/download_models.sh clip")
        print("  3. 使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)


def load_image_from_url(url):
    """从URL加载图片"""
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"⚠️ 从URL加载图片失败: {e}")
        return None


def load_image_from_file(path):
    """从本地文件加载图片"""
    try:
        image = Image.open(path)
        return image
    except Exception as e:
        print(f"❌ 从文件加载图片失败: {e}")
        return None


def create_dummy_image():
    """创建测试图片"""
    import numpy as np
    from PIL import Image
    
    # 创建一个简单的彩色渐变图
    width, height = 224, 224
    array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 红色渐变
    for i in range(height):
        array[i, :, 0] = int(255 * i / height)
    
    # 绿色渐变
    for j in range(width):
        array[:, j, 1] = int(255 * j / width)
    
    # 蓝色固定
    array[:, :, 2] = 128
    
    return Image.fromarray(array)


def perform_inference(model, processor, image, text_candidates, device):
    """
    执行CLIP推理
    
    Args:
        model: CLIP模型
        processor: CLIP处理器
        image: PIL Image
        text_candidates: 文本候选列表
        device: 设备
    
    Returns:
        概率分布
    """
    print("\n[3/3] 执行推理...")
    
    # 处理输入
    inputs = processor(
        text=text_candidates,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    print("✓ 推理完成")
    return probs


def display_results(text_candidates, probs):
    """显示结果"""
    print("\n" + "=" * 60)
    print("📊 匹配结果")
    print("=" * 60)
    
    # 排序结果
    sorted_indices = probs[0].argsort(descending=True)
    
    for rank, idx in enumerate(sorted_indices, 1):
        text = text_candidates[idx]
        confidence = probs[0][idx].item() * 100
        bar_length = int(confidence / 2)  # 50个字符满格
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"\n{rank}. {text}")
        print(f"   {bar} {confidence:.2f}%")
    
    # 最佳匹配
    best_idx = sorted_indices[0]
    best_text = text_candidates[best_idx]
    best_confidence = probs[0][best_idx].item() * 100
    
    print("\n" + "=" * 60)
    print(f"🎯 最佳匹配: {best_text} ({best_confidence:.2f}%)")
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("CLIP 图文匹配示例")
    print("=" * 60)
    print()
    
    # 1. 加载模型
    model, processor, device = load_clip_model()
    
    # 2. 准备图像
    print("\n[2/3] 准备测试图像")
    
    # 尝试从网络加载示例图片
    image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
    image = load_image_from_url(image_url)
    
    if image is None:
        print("⚠️ 无法从网络加载示例图片")
        print("使用本地生成的测试图片")
        image = create_dummy_image()
        print("✓ 创建测试图片成功")
    else:
        print(f"✓ 从网络加载图片成功: {image.size}")
    
    # 3. 准备文本候选
    text_candidates = [
        "一只猫",
        "一只狗",
        "一群鹦鹉",
        "一辆汽车",
        "一座建筑",
        "a photo of a cat",
        "a photo of a dog",
        "a photo of birds",
    ]
    
    print(f"\n候选文本数量: {len(text_candidates)}")
    for i, text in enumerate(text_candidates, 1):
        print(f"  {i}. {text}")
    
    # 4. 执行推理
    probs = perform_inference(model, processor, image, text_candidates, device)
    
    # 5. 显示结果
    display_results(text_candidates, probs)
    
    # 提示
    print("\n✨ 恭喜！您已成功运行CLIP模型推理！")
    print("\n📚 接下来可以:")
    print("  - 修改 text_candidates 尝试不同的文本")
    print("  - 使用自己的图片: load_image_from_file('your_image.jpg')")
    print("  - 查看更多示例: code/01-model-evaluation/examples/")
    print("  - 阅读教程文档: docs/01-模型调研与选型/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

