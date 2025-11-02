#!/usr/bin/env python3
"""
CLIP快速开始示例

这是一个简单的CLIP模型推理示例，帮助您快速上手。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    import requests
    from io import BytesIO
except ImportError as e:
    print("❌ 缺少依赖库，请先安装:")
    print("   pip install torch transformers pillow requests")
    print(f"\n错误详情: {e}")
    sys.exit(1)


def download_sample_image():
    """下载示例图像"""
    print("📥 下载示例图像...")
    
    # 使用一张公开的示例图像
    image_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400"
    
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        print("✅ 图像下载成功")
        return image
    except Exception as e:
        print(f"⚠️  下载失败: {e}")
        print("💡 将创建一个演示图像...")
        # 创建一个简单的演示图像
        image = Image.new('RGB', (224, 224), color=(73, 109, 137))
        return image


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 CLIP 快速开始示例")
    print("=" * 70)
    
    # 1. 检查设备
    print("\n📊 步骤 1/4: 检查环境")
    print("-" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch版本: {torch.__version__}")
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 2. 加载模型
    print("\n📥 步骤 2/4: 加载CLIP模型")
    print("-" * 70)
    print("正在加载 openai/clip-vit-base-patch32...")
    
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 解决方案:")
        print("   1. 检查网络连接")
        print("   2. 或先下载模型: ./scripts/download_models.sh clip")
        sys.exit(1)
    
    # 3. 准备数据
    print("\n🖼️  步骤 3/4: 准备测试数据")
    print("-" * 70)
    
    # 准备图像
    image = download_sample_image()
    
    # 准备候选文本
    texts = [
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a car",
        "a photo of a bird",
        "a photo of a flower"
    ]
    
    print(f"图像尺寸: {image.size}")
    print(f"候选文本数: {len(texts)}")
    print(f"候选文本: {texts}")
    
    # 4. 推理
    print("\n🔮 步骤 4/4: 执行推理")
    print("-" * 70)
    
    try:
        # 预处理
        inputs = processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 计算相似度
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        print("✅ 推理完成\n")
        
        # 显示结果
        print("📊 图文匹配结果:")
        print("-" * 70)
        
        results = []
        for i, (text, prob) in enumerate(zip(texts, probs[0])):
            results.append((text, prob.item()))
        
        # 按概率排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (text, prob) in enumerate(results, 1):
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{i}. {text:30s} {bar} {prob:6.2%}")
        
        print("\n" + "=" * 70)
        print("🎉 快速开始示例完成！")
        print("=" * 70)
        
        # 下一步提示
        print("\n📚 下一步学习:")
        print("   1. 查看详细文档: docs/01-模型调研与选型/")
        print("   2. 运行基准测试: ./scripts/run_benchmarks.sh")
        print("   3. 尝试模型微调: notebooks/01_lora_finetuning_tutorial.ipynb")
        print("   4. 更多示例代码: code/01-model-evaluation/examples/")
        
        print("\n💡 提示:")
        print("   - 修改 texts 列表来测试不同的文本")
        print("   - 使用自己的图像: image = Image.open('your_image.jpg')")
        print("   - 查看完整教程: docs/05-使用说明/02-快速开始.md")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

