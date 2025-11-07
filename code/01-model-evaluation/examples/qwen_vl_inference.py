#!/usr/bin/env python3
"""
Qwen-VL 推理示例

Qwen-VL是阿里巴巴开发的中文视觉语言模型，在中文场景下表现优异。

功能:
- 图像描述生成（中文）
- 视觉问答（VQA）
- OCR文字识别
- 多图理解

依赖:
    pip install transformers>=4.32.0 transformers_stream_generator
    pip install torch torchvision pillow

作者: Large-Model-Tutorial Team
日期: 2025-11-06
版本: v1.1.0
"""

import argparse
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
except ImportError as e:
    print(f"❌ 缺少必要依赖: {e}")
    print("请安装: pip install torch transformers pillow")
    exit(1)


class QwenVLInference:
    """Qwen-VL 推理类"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",
        device: str = "auto",
        trust_remote_code: bool = True
    ):
        """
        初始化Qwen-VL模型
        
        参数:
            model_name: 模型名称或路径
            device: 设备 ('cuda', 'cpu', 'auto')
            trust_remote_code: 是否信任远程代码（Qwen-VL需要）
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        print(f"🚀 加载Qwen-VL模型: {model_name}")
        print(f"📍 使用设备: {self.device}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device == "auto" else None,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).eval()
            
            if device != "auto" and device != "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ 模型加载成功！")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("\n💡 提示:")
            print("   1. 确保已安装: pip install transformers>=4.32.0 transformers_stream_generator")
            print("   2. 首次运行会下载模型（约10GB），请耐心等待")
            print("   3. 确保网络畅通或配置HuggingFace镜像")
            raise
    
    def _setup_device(self, device: str) -> str:
        """设置运行设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def generate_caption(
        self,
        image_path: str,
        prompt: str = "描述这张图片",
        max_length: int = 256
    ) -> str:
        """
        生成图像描述
        
        参数:
            image_path: 图像路径
            prompt: 提示文本
            max_length: 最大生成长度
            
        返回:
            生成的描述文本
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 构建查询
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])
        
        # 生成响应
        response, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            max_length=max_length
        )
        
        return response
    
    def visual_question_answering(
        self,
        image_path: str,
        question: str,
        max_length: int = 256
    ) -> str:
        """
        视觉问答
        
        参数:
            image_path: 图像路径
            question: 问题文本
            max_length: 最大生成长度
            
        返回:
            答案文本
        """
        return self.generate_caption(image_path, question, max_length)
    
    def multi_image_understanding(
        self,
        image_paths: List[str],
        prompt: str,
        max_length: int = 512
    ) -> str:
        """
        多图理解
        
        参数:
            image_paths: 图像路径列表
            prompt: 提示文本
            max_length: 最大生成长度
            
        返回:
            理解结果
        """
        # 构建多图查询
        query_list = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            query_list.append({'image': img_path})
        
        query_list.append({'text': prompt})
        
        query = self.tokenizer.from_list_format(query_list)
        
        # 生成响应
        response, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            max_length=max_length
        )
        
        return response
    
    def ocr_recognition(
        self,
        image_path: str,
        prompt: str = "识别图片中的所有文字",
        max_length: int = 512
    ) -> str:
        """
        OCR文字识别
        
        参数:
            image_path: 图像路径
            prompt: 提示文本
            max_length: 最大生成长度
            
        返回:
            识别的文字
        """
        return self.generate_caption(image_path, prompt, max_length)
    
    def chat(
        self,
        image_path: str,
        history: Optional[List] = None,
        max_length: int = 256
    ) -> tuple:
        """
        多轮对话
        
        参数:
            image_path: 图像路径
            history: 对话历史
            max_length: 最大生成长度
            
        返回:
            (响应, 新的历史)
        """
        # 构建查询
        query = self.tokenizer.from_list_format([
            {'image': image_path},
        ])
        
        # 进行对话
        response, history = self.model.chat(
            self.tokenizer,
            query=query,
            history=history,
            max_length=max_length
        )
        
        return response, history


def demo_caption_generation(model: QwenVLInference, image_path: str):
    """演示：图像描述生成"""
    print("\n" + "="*60)
    print("📝 演示1: 图像描述生成")
    print("="*60)
    
    prompts = [
        "详细描述这张图片",
        "用一句话概括图片内容",
        "图片中有什么物体？"
    ]
    
    for prompt in prompts:
        print(f"\n❓ 提示: {prompt}")
        try:
            caption = model.generate_caption(image_path, prompt)
            print(f"💬 回答: {caption}")
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_vqa(model: QwenVLInference, image_path: str):
    """演示：视觉问答"""
    print("\n" + "="*60)
    print("❓ 演示2: 视觉问答（VQA）")
    print("="*60)
    
    questions = [
        "图片中有多少人？",
        "这是什么场景？",
        "图片的主要颜色是什么？",
        "图片拍摄的是白天还是晚上？"
    ]
    
    for question in questions:
        print(f"\n❓ 问题: {question}")
        try:
            answer = model.visual_question_answering(image_path, question)
            print(f"💬 回答: {answer}")
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_ocr(model: QwenVLInference, image_path: str):
    """演示：OCR文字识别"""
    print("\n" + "="*60)
    print("🔍 演示3: OCR文字识别")
    print("="*60)
    
    prompts = [
        "识别图片中的所有文字",
        "提取图片中的中文文本",
        "图片中有哪些数字？"
    ]
    
    for prompt in prompts:
        print(f"\n❓ 提示: {prompt}")
        try:
            text = model.ocr_recognition(image_path, prompt)
            print(f"💬 识别结果: {text}")
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_multi_image(model: QwenVLInference, image_paths: List[str]):
    """演示：多图理解"""
    print("\n" + "="*60)
    print("🖼️  演示4: 多图理解")
    print("="*60)
    
    if len(image_paths) < 2:
        print("⚠️  需要至少2张图片进行多图理解演示")
        return
    
    prompts = [
        "比较这些图片的异同",
        "这些图片有什么共同点？",
        "按照时间顺序描述这些图片"
    ]
    
    for prompt in prompts:
        print(f"\n❓ 提示: {prompt}")
        try:
            result = model.multi_image_understanding(image_paths[:2], prompt)
            print(f"💬 回答: {result}")
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_chat(model: QwenVLInference, image_path: str):
    """演示：多轮对话"""
    print("\n" + "="*60)
    print("💭 演示5: 多轮对话")
    print("="*60)
    
    conversations = [
        "这是什么？",
        "它的颜色是什么？",
        "它通常用来做什么？"
    ]
    
    history = None
    for i, question in enumerate(conversations, 1):
        print(f"\n第{i}轮对话:")
        print(f"❓ 用户: {question}")
        try:
            # 构建查询
            if history is None:
                query = model.tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': question},
                ])
            else:
                query = question
            
            response, history = model.model.chat(
                model.tokenizer,
                query=query,
                history=history
            )
            print(f"💬 Qwen-VL: {response}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qwen-VL推理示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基础推理
    python qwen_vl_inference.py --image path/to/image.jpg
    
    # 指定模型
    python qwen_vl_inference.py --image image.jpg --model Qwen/Qwen-VL-Chat
    
    # CPU模式
    python qwen_vl_inference.py --image image.jpg --device cpu
    
    # 多图理解
    python qwen_vl_inference.py --images img1.jpg img2.jpg --demo multi_image
    
    # 仅运行特定演示
    python qwen_vl_inference.py --image image.jpg --demo caption
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="输入图像路径"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="多个图像路径（用于多图理解）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-VL-Chat",
        help="模型名称或路径 (默认: Qwen/Qwen-VL-Chat)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备 (默认: auto)"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "caption", "vqa", "ocr", "multi_image", "chat"],
        default="all",
        help="运行的演示 (默认: all)"
    )
    
    args = parser.parse_args()
    
    # 检查输入
    if not args.image and not args.images:
        parser.error("请提供 --image 或 --images 参数")
    
    image_path = args.image or args.images[0]
    image_paths = args.images or [args.image]
    
    # 检查图像文件
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"❌ 图像文件不存在: {img_path}")
            return
    
    print("🎨 Qwen-VL 推理示例")
    print("="*60)
    print(f"📁 图像: {', '.join(image_paths)}")
    print(f"🤖 模型: {args.model}")
    print(f"💻 设备: {args.device}")
    
    try:
        # 初始化模型
        model = QwenVLInference(
            model_name=args.model,
            device=args.device
        )
        
        # 运行演示
        if args.demo == "all" or args.demo == "caption":
            demo_caption_generation(model, image_path)
        
        if args.demo == "all" or args.demo == "vqa":
            demo_vqa(model, image_path)
        
        if args.demo == "all" or args.demo == "ocr":
            demo_ocr(model, image_path)
        
        if args.demo == "all" or args.demo == "multi_image":
            demo_multi_image(model, image_paths)
        
        if args.demo == "all" or args.demo == "chat":
            demo_chat(model, image_path)
        
        print("\n" + "="*60)
        print("✅ 所有演示完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

