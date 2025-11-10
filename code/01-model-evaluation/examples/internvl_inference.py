#!/usr/bin/env python3
"""
InternVL 推理示例

InternVL是上海AI Lab开发的强大视觉语言模型,接近GPT-4V的性能表现。
InternVL3是最新版本,支持多种视觉任务和多语言对话。

功能:
- 图像描述生成
- 视觉问答(VQA)
- OCR文字识别
- 多图理解
- 多轮对话
- 视频理解

模型版本:
- InternVL2-8B: 平衡性能和速度
- InternVL3-1B: 轻量级版本
- InternVL3-8B: 高性能版本
- InternVL3-78B: 旗舰版本

依赖:
    pip install transformers>=4.37.2 torch torchvision pillow
    pip install accelerate

作者: Large-Model-Tutorial Team
日期: 2025-11-10
版本: v1.1.0
"""

import argparse
import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

try:
    import torch
    from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
except ImportError as e:
    print(f"❌ 缺少必要依赖: {e}")
    print("请安装: pip install torch transformers pillow accelerate")
    exit(1)


class InternVLInference:
    """InternVL 推理类"""

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-8B",
        device: str = "auto",
        torch_dtype: str = "bfloat16"
    ):
        """
        初始化InternVL模型

        参数:
            model_name: 模型名称或路径
                - OpenGVLab/InternVL2-8B (推荐)
                - OpenGVLab/InternVL3-1B (轻量级)
                - OpenGVLab/InternVL3-8B (高性能)
            device: 设备 ('cuda', 'cpu', 'auto')
            torch_dtype: 数据类型 ('bfloat16', 'float16', 'float32')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)

        print(f"🚀 加载InternVL模型: {model_name}")
        print(f"📍 使用设备: {self.device}")

        # 设置数据类型 - 根据设备和用户选择自动调整
        self.dtype = self._setup_dtype(torch_dtype, self.device)

        try:
            # 加载processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # 加载模型
            # 使用AutoModelForImageTextToText (Transformers >= 4.37.2)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                device_map=device if device == "auto" else None,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()

            if device != "auto" and device != "cpu":
                self.model = self.model.to(self.device)

            print("✅ 模型加载成功!")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("\n💡 提示:")
            print("   1. 确保已安装: pip install transformers>=4.37.2 accelerate")
            print("   2. 首次运行会下载模型(约16GB),请耐心等待")
            print("   3. 确保网络畅通或配置HuggingFace镜像")
            print("   4. 推荐使用GPU运行,至少需要16GB显存")
            raise

    def _setup_device(self, device: str) -> str:
        """设置运行设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_dtype(self, torch_dtype: str, device: str) -> torch.dtype:
        """
        设置数据类型，根据设备和兼容性自动调整

        参数:
            torch_dtype: 用户指定的数据类型
            device: 运行设备

        返回:
            torch.dtype对象
        """
        # CPU设备必须使用float32
        if device == "cpu":
            if torch_dtype != "float32":
                print(f"⚠️  CPU设备不支持{torch_dtype}，自动切换到float32")
            print(f"💻 使用精度: Float32 (CPU模式)")
            return torch.float32

        # GPU设备根据用户选择
        if torch_dtype == "bfloat16":
            # 检查是否支持BFloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                print(f"⚡ 使用精度: BFloat16 (推荐)")
                return torch.bfloat16
            else:
                print(f"⚠️  GPU不支持BFloat16，切换到Float16")
                print(f"⚡ 使用精度: Float16")
                return torch.float16
        elif torch_dtype == "float16":
            print(f"⚡ 使用精度: Float16")
            return torch.float16
        else:
            print(f"💻 使用精度: Float32")
            return torch.float32

    def _prepare_messages(
        self,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        准备聊天消息格式

        参数:
            image: 单张图片或图片列表
            text: 文本提示

        返回:
            格式化的消息列表
        """
        # 处理图片
        if isinstance(image, (str, Image.Image)):
            images = [image]
        else:
            images = image

        # 加载图片
        pil_images = []
        for img in images:
            if isinstance(img, str):
                if not os.path.exists(img):
                    raise FileNotFoundError(f"图像文件不存在: {img}")
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)

        # 构建消息
        # InternVL使用特殊的消息格式
        content = []
        for _ in pil_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        return messages, pil_images

    def generate(
        self,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        通用生成方法

        参数:
            image: 单张图片路径/对象或图片列表
            prompt: 文本提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数

        返回:
            生成的文本
        """
        # 准备消息
        messages, pil_images = self._prepare_messages(image, prompt)

        # 应用聊天模板
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = self.processor(
            text=prompt_text,
            images=pil_images,
            return_tensors="pt",
            padding=True
        )

        # 移动到设备
        if self.device != "cpu":
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )

        # 解码
        generated_text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # 提取回答部分(移除prompt)
        if prompt in generated_text:
            answer = generated_text.split(prompt)[-1].strip()
        else:
            # 尝试提取assistant的回答
            if "assistant\n" in generated_text:
                answer = generated_text.split("assistant\n")[-1].strip()
            else:
                answer = generated_text.strip()

        return answer

    def generate_caption(
        self,
        image_path: str,
        prompt: str = "Please describe this image in detail.",
        max_new_tokens: int = 256
    ) -> str:
        """
        生成图像描述

        参数:
            image_path: 图像路径
            prompt: 提示文本
            max_new_tokens: 最大生成长度

        返回:
            生成的描述文本
        """
        return self.generate(image_path, prompt, max_new_tokens)

    def visual_question_answering(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        视觉问答

        参数:
            image_path: 图像路径
            question: 问题文本
            max_new_tokens: 最大生成长度

        返回:
            答案文本
        """
        return self.generate(image_path, question, max_new_tokens)

    def multi_image_understanding(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512
    ) -> str:
        """
        多图理解

        参数:
            image_paths: 图像路径列表
            prompt: 提示文本
            max_new_tokens: 最大生成长度

        返回:
            理解结果
        """
        return self.generate(image_paths, prompt, max_new_tokens)

    def ocr_recognition(
        self,
        image_path: str,
        prompt: str = "Please extract all text from this image.",
        max_new_tokens: int = 512
    ) -> str:
        """
        OCR文字识别

        参数:
            image_path: 图像路径
            prompt: 提示文本
            max_new_tokens: 最大生成长度

        返回:
            识别的文字
        """
        return self.generate(image_path, prompt, max_new_tokens)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        images: List[Union[str, Image.Image]],
        max_new_tokens: int = 512
    ) -> str:
        """
        多轮对话

        参数:
            messages: 对话历史消息列表
            images: 图像列表
            max_new_tokens: 最大生成长度

        返回:
            模型回复
        """
        # 加载图片
        pil_images = []
        for img in images:
            if isinstance(img, str):
                if not os.path.exists(img):
                    raise FileNotFoundError(f"图像文件不存在: {img}")
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)

        # 应用聊天模板
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = self.processor(
            text=prompt_text,
            images=pil_images,
            return_tensors="pt",
            padding=True
        )

        # 移动到设备
        if self.device != "cpu":
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )

        # 解码
        generated_text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # 提取最后的回答
        if "assistant\n" in generated_text:
            answer = generated_text.split("assistant\n")[-1].strip()
        else:
            answer = generated_text.strip()

        return answer


def demo_caption_generation(model: InternVLInference, image_path: str):
    """演示:图像描述生成"""
    print("\n" + "="*60)
    print("📝 演示1: 图像描述生成")
    print("="*60)

    prompts = [
        "Please describe this image in detail.",
        "What is the main subject of this image?",
        "Summarize this image in one sentence."
    ]

    for prompt in prompts:
        print(f"\n❓ Prompt: {prompt}")
        try:
            caption = model.generate_caption(image_path, prompt)
            print(f"💬 Response: {caption}")
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_vqa(model: InternVLInference, image_path: str):
    """演示:视觉问答"""
    print("\n" + "="*60)
    print("❓ 演示2: 视觉问答(VQA)")
    print("="*60)

    questions = [
        "How many people are in this image?",
        "What is the weather like in this image?",
        "What colors are dominant in this image?",
        "Is this image taken during day or night?"
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        try:
            answer = model.visual_question_answering(image_path, question)
            print(f"💬 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_ocr(model: InternVLInference, image_path: str):
    """演示:OCR文字识别"""
    print("\n" + "="*60)
    print("🔍 演示3: OCR文字识别")
    print("="*60)

    prompts = [
        "Please extract all text from this image.",
        "What text can you see in this image?",
        "List all the words visible in this image."
    ]

    for prompt in prompts:
        print(f"\n❓ Prompt: {prompt}")
        try:
            text = model.ocr_recognition(image_path, prompt)
            print(f"💬 Extracted Text: {text}")
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_multi_image(model: InternVLInference, image_paths: List[str]):
    """演示:多图理解"""
    print("\n" + "="*60)
    print("🖼️  演示4: 多图理解")
    print("="*60)

    if len(image_paths) < 2:
        print("⚠️  需要至少2张图片进行多图理解演示")
        return

    prompts = [
        "Compare and contrast these images.",
        "What do these images have in common?",
        "Describe the relationship between these images."
    ]

    for prompt in prompts:
        print(f"\n❓ Prompt: {prompt}")
        try:
            result = model.multi_image_understanding(image_paths[:2], prompt)
            print(f"💬 Response: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_chat(model: InternVLInference, image_path: str):
    """演示:多轮对话"""
    print("\n" + "="*60)
    print("💭 演示5: 多轮对话")
    print("="*60)

    # 加载图片
    image = Image.open(image_path).convert('RGB')

    # 定义对话流程
    conversation_turns = [
        "What do you see in this image?",
        "What color is it?",
        "What is it typically used for?"
    ]

    # 构建对话历史
    messages = []
    images = [image]

    for i, user_msg in enumerate(conversation_turns, 1):
        print(f"\n第{i}轮对话:")
        print(f"❓ User: {user_msg}")

        try:
            # 添加用户消息
            content = [{"type": "image"}] if i == 1 else []
            content.append({"type": "text", "text": user_msg})
            messages.append({
                "role": "user",
                "content": content
            })

            # 获取模型回复
            response = model.chat(messages, images)
            print(f"💬 InternVL: {response}")

            # 添加助手回复到历史
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="InternVL推理示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 基础推理
    python internvl_inference.py --image path/to/image.jpg

    # 指定模型
    python internvl_inference.py --image image.jpg --model OpenGVLab/InternVL2-8B

    # CPU模式(不推荐,很慢)
    python internvl_inference.py --image image.jpg --device cpu

    # 多图理解
    python internvl_inference.py --images img1.jpg img2.jpg --demo multi_image

    # 仅运行特定演示
    python internvl_inference.py --image image.jpg --demo caption
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
        help="多个图像路径(用于多图理解)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL2-8B",
        help="模型名称或路径 (默认: OpenGVLab/InternVL2-8B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备 (默认: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="数据类型 (默认: bfloat16)"
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

    print("🎨 InternVL 推理示例")
    print("="*60)
    print(f"📁 图像: {', '.join(image_paths)}")
    print(f"🤖 模型: {args.model}")
    print(f"💻 设备: {args.device}")

    try:
        # 初始化模型
        model = InternVLInference(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype
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
        print("✅ 所有演示完成!")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
