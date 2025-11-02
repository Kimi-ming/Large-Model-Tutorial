"""
ONNX模型转换脚本

将PyTorch CLIP模型转换为ONNX格式
"""

import torch
import onnx
import onnxruntime as ort
from transformers import CLIPModel, CLIPProcessor
import argparse
from pathlib import Path
import numpy as np


def convert_vision_model(
    model_path: str,
    output_path: str,
    opset_version: int = 14,
    dynamic_batch: bool = True
):
    """
    转换CLIP视觉编码器为ONNX
    
    Args:
        model_path: PyTorch模型路径
        output_path: ONNX模型输出路径
        opset_version: ONNX opset版本
        dynamic_batch: 是否支持动态batch size
    """
    print(f"🔄 转换视觉编码器: {model_path} -> {output_path}")
    
    # 加载模型
    model = CLIPModel.from_pretrained(model_path)
    model.eval()
    
    # 准备示例输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 动态维度配置
    if dynamic_batch:
        dynamic_axes = {
            'pixel_values': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    
    # 导出ONNX
    torch.onnx.export(
        model.vision_model,
        dummy_input,
        output_path,
        input_names=['pixel_values'],
        output_names=['pooler_output', 'last_hidden_state'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✅ 视觉编码器转换完成")
    
    # 验证模型
    verify_onnx_model(output_path, dummy_input.numpy())


def convert_text_model(
    model_path: str,
    output_path: str,
    max_length: int = 77,
    opset_version: int = 14,
    dynamic_batch: bool = True
):
    """
    转换CLIP文本编码器为ONNX
    
    Args:
        model_path: PyTorch模型路径
        output_path: ONNX模型输出路径
        max_length: 最大文本长度
        opset_version: ONNX opset版本
        dynamic_batch: 是否支持动态batch size
    """
    print(f"🔄 转换文本编码器: {model_path} -> {output_path}")
    
    # 加载模型
    model = CLIPModel.from_pretrained(model_path)
    model.eval()
    
    # 准备示例输入
    dummy_input_ids = torch.randint(0, 49408, (1, max_length))
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    
    # 动态维度配置
    if dynamic_batch:
        dynamic_axes = {
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'pooler_output': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    
    # 导出ONNX
    torch.onnx.export(
        model.text_model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['pooler_output', 'last_hidden_state'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✅ 文本编码器转换完成")
    
    # 验证模型
    verify_onnx_model(
        output_path,
        {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        }
    )


def verify_onnx_model(onnx_path: str, dummy_input):
    """
    验证ONNX模型
    
    Args:
        onnx_path: ONNX模型路径
        dummy_input: 示例输入
    """
    print(f"🔍 验证ONNX模型: {onnx_path}")
    
    # 加载并检查模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ 模型结构验证通过")
    
    # 打印模型信息
    print(f"  ✓ IR版本: {onnx_model.ir_version}")
    print(f"  ✓ Opset版本: {onnx_model.opset_import[0].version}")
    
    # 打印输入输出
    print("  ✓ 输入:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                for d in inp.type.tensor_type.shape.dim]
        print(f"      {inp.name}: {shape}")
    
    print("  ✓ 输出:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                for d in out.type.tensor_type.shape.dim]
        print(f"      {out.name}: {shape}")
    
    # 使用ONNX Runtime测试推理
    try:
        session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        if isinstance(dummy_input, dict):
            outputs = session.run(None, dummy_input)
        else:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: dummy_input})
        
        print(f"  ✓ ONNX Runtime推理测试通过")
        print(f"  ✓ 输出形状: {[out.shape for out in outputs]}")
    
    except Exception as e:
        print(f"  ✗ ONNX Runtime推理测试失败: {e}")


def optimize_onnx_model(
    input_path: str,
    output_path: str
):
    """
    优化ONNX模型
    
    Args:
        input_path: 输入ONNX模型路径
        output_path: 优化后的输出路径
    """
    print(f"⚡ 优化ONNX模型: {input_path} -> {output_path}")
    
    try:
        from onnxruntime.transformers import optimizer
        
        # 优化模型
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type='bert',  # CLIP使用Transformer架构
            num_heads=12,
            hidden_size=768,
        )
        
        # 保存优化后的模型
        optimized_model.save_model_to_file(output_path)
        
        print(f"✅ 模型优化完成")
        
        # 比较模型大小
        import os
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        optimized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"  原始大小: {original_size:.2f} MB")
        print(f"  优化后大小: {optimized_size:.2f} MB")
        print(f"  压缩比: {(1 - optimized_size/original_size)*100:.1f}%")
    
    except ImportError:
        print("⚠️  未安装onnxruntime.transformers，跳过优化")
        print("   安装命令: pip install onnxruntime-tools")
    except Exception as e:
        print(f"⚠️  优化失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="CLIP模型ONNX转换")
    parser.add_argument(
        '--model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='PyTorch模型路径或HuggingFace模型名称'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='onnx_models',
        help='ONNX模型输出目录'
    )
    parser.add_argument(
        '--vision_only',
        action='store_true',
        help='只转换视觉编码器'
    )
    parser.add_argument(
        '--text_only',
        action='store_true',
        help='只转换文本编码器'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=14,
        help='ONNX opset版本'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='优化ONNX模型'
    )
    parser.add_argument(
        '--static_batch',
        action='store_true',
        help='使用静态batch size（不支持动态）'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CLIP模型ONNX转换工具")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"输出目录: {output_dir}")
    print(f"Opset版本: {args.opset_version}")
    print(f"动态batch: {not args.static_batch}")
    print("=" * 60)
    
    # 转换视觉编码器
    if not args.text_only:
        vision_output = output_dir / "clip_vision.onnx"
        convert_vision_model(
            model_path=args.model,
            output_path=str(vision_output),
            opset_version=args.opset_version,
            dynamic_batch=not args.static_batch
        )
        
        # 优化
        if args.optimize:
            vision_optimized = output_dir / "clip_vision_optimized.onnx"
            optimize_onnx_model(str(vision_output), str(vision_optimized))
    
    # 转换文本编码器
    if not args.vision_only:
        text_output = output_dir / "clip_text.onnx"
        convert_text_model(
            model_path=args.model,
            output_path=str(text_output),
            opset_version=args.opset_version,
            dynamic_batch=not args.static_batch
        )
        
        # 优化
        if args.optimize:
            text_optimized = output_dir / "clip_text_optimized.onnx"
            optimize_onnx_model(str(text_output), str(text_optimized))
    
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)
    print(f"\n输出文件位于: {output_dir}")
    print("\n使用方式:")
    print(f"  python code/04-deployment/nvidia/onnx/onnx_inference.py \\")
    print(f"    --vision_model {output_dir}/clip_vision.onnx \\")
    print(f"    --text_model {output_dir}/clip_text.onnx \\")
    print(f"    --image your_image.jpg \\")
    print(f"    --texts 'text1' 'text2' 'text3'")


if __name__ == '__main__':
    main()

