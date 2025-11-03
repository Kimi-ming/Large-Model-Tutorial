"""
PyTorch模型转换为昇腾OM格式

使用ATC工具将ONNX模型转换为昇腾优化的OM格式
"""

import os
import subprocess
import argparse
from pathlib import Path
import json


class ModelConverter:
    """模型转换器"""
    
    SUPPORTED_SOC = ['Ascend310', 'Ascend910', 'Ascend310P', 'Ascend910B']
    
    def __init__(self, soc_version: str = 'Ascend910'):
        """
        初始化转换器
        
        Args:
            soc_version: 目标芯片版本
        """
        if soc_version not in self.SUPPORTED_SOC:
            raise ValueError(f"Unsupported SOC: {soc_version}, must be one of {self.SUPPORTED_SOC}")
        
        self.soc_version = soc_version
        
        # 检查atc是否可用
        try:
            subprocess.run(['atc', '--help'], capture_output=True, check=True)
            print(f"✅ ATC工具可用")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ATC tool not found. Please install CANN toolkit.")
    
    def convert_onnx_to_om(
        self,
        onnx_path: str,
        output_path: str,
        input_shape: str = None,
        dynamic_dims: str = None,
        **kwargs
    ) -> bool:
        """
        转换ONNX模型为OM格式
        
        Args:
            onnx_path: ONNX模型路径
            output_path: 输出OM模型路径
            input_shape: 输入shape，格式: "input1:1,3,224,224;input2:1,512"
            dynamic_dims: 动态维度，格式: "1;2;4;8"
            **kwargs: 其他ATC参数
            
        Returns:
            转换是否成功
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # 构建ATC命令
        cmd = [
            'atc',
            f'--model={onnx_path}',
            '--framework=5',  # 5 = ONNX
            f'--output={output_path}',
            f'--soc_version={self.soc_version}',
            '--log=error',
        ]
        
        # 添加输入shape
        if input_shape:
            cmd.append(f'--input_shape={input_shape}')
            cmd.append('--input_format=ND')
        
        # 添加动态维度
        if dynamic_dims:
            cmd.append(f'--dynamic_dims={dynamic_dims}')
        
        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                cmd.append(f'--{key}={value}')
        
        print(f"🔄 开始转换: {onnx_path} -> {output_path}.om")
        print(f"📝 命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ 转换成功!")
            print(f"   输出: {output_path}.om")
            
            # 检查输出文件
            om_file = f"{output_path}.om"
            if os.path.exists(om_file):
                size_mb = os.path.getsize(om_file) / 1024 / 1024
                print(f"   大小: {size_mb:.2f}MB")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败!")
            print(f"错误输出:\n{e.stderr}")
            return False
    
    def convert_pytorch_to_onnx(
        self,
        model,
        dummy_input: dict,
        output_path: str,
        input_names: list,
        output_names: list,
        dynamic_axes: dict = None,
        opset_version: int = 11
    ) -> bool:
        """
        转换PyTorch模型为ONNX
        
        Args:
            model: PyTorch模型
            dummy_input: 示例输入
            output_path: 输出ONNX路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态维度定义
            opset_version: ONNX opset版本
            
        Returns:
            转换是否成功
        """
        import torch
        
        model.eval()
        
        print(f"🔄 导出ONNX模型: {output_path}")
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            print(f"✅ ONNX导出成功")
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX模型验证通过")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
            return False
    
    def convert_clip_model(
        self,
        model_path: str,
        output_dir: str,
        batch_size: int = 1,
        image_size: int = 224,
        text_length: int = 77,
        dynamic_batch: bool = False
    ) -> dict:
        """
        转换CLIP模型为OM格式
        
        Args:
            model_path: HuggingFace模型路径
            output_dir: 输出目录
            batch_size: 批大小
            image_size: 图像大小
            text_length: 文本长度
            dynamic_batch: 是否支持动态batch
            
        Returns:
            转换结果信息
        """
        from transformers import CLIPModel
        import torch
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"📥 加载模型: {model_path}")
        model = CLIPModel.from_pretrained(model_path)
        model.eval()
        
        # 准备示例输入
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, text_length)),
            'pixel_values': torch.randn(batch_size, 3, image_size, image_size),
            'attention_mask': torch.ones(batch_size, text_length, dtype=torch.long)
        }
        
        # 导出ONNX
        onnx_path = str(output_dir / "clip_model.onnx")
        
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input_ids': {0: 'batch_size'},
                'pixel_values': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits_per_image': {0: 'batch_size'},
                'logits_per_text': {0: 'batch_size'}
            }
        
        success = self.convert_pytorch_to_onnx(
            model=model,
            dummy_input=(dummy_input,),
            output_path=onnx_path,
            input_names=['input_ids', 'pixel_values', 'attention_mask'],
            output_names=['logits_per_image', 'logits_per_text'],
            dynamic_axes=dynamic_axes,
            opset_version=11
        )
        
        if not success:
            return {'success': False, 'error': 'ONNX export failed'}
        
        # 转换为OM
        om_path = str(output_dir / "clip_model")
        
        if dynamic_batch:
            input_shape = f"input_ids:-1,{text_length};pixel_values:-1,3,{image_size},{image_size};attention_mask:-1,{text_length}"
            dynamic_dims = "1;2;4;8"
        else:
            input_shape = f"input_ids:{batch_size},{text_length};pixel_values:{batch_size},3,{image_size},{image_size};attention_mask:{batch_size},{text_length}"
            dynamic_dims = None
        
        success = self.convert_onnx_to_om(
            onnx_path=onnx_path,
            output_path=om_path,
            input_shape=input_shape,
            dynamic_dims=dynamic_dims
        )
        
        result = {
            'success': success,
            'onnx_path': onnx_path,
            'om_path': f"{om_path}.om" if success else None,
            'config': {
                'batch_size': batch_size,
                'image_size': image_size,
                'text_length': text_length,
                'dynamic_batch': dynamic_batch,
                'soc_version': self.soc_version
            }
        }
        
        # 保存配置
        if success:
            config_path = output_dir / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(result['config'], f, indent=2)
            print(f"💾 配置已保存: {config_path}")
        
        return result


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='PyTorch模型转换为昇腾OM格式')
    
    subparsers = parser.add_subparsers(dest='command', help='转换命令')
    
    # ONNX to OM
    onnx_parser = subparsers.add_parser('onnx', help='转换ONNX模型为OM')
    onnx_parser.add_argument('--model', type=str, required=True, help='ONNX模型路径')
    onnx_parser.add_argument('--output', type=str, required=True, help='输出OM模型路径')
    onnx_parser.add_argument('--input-shape', type=str, help='输入shape')
    onnx_parser.add_argument('--dynamic-dims', type=str, help='动态维度')
    onnx_parser.add_argument('--soc-version', type=str, default='Ascend910',
                            choices=ModelConverter.SUPPORTED_SOC, help='目标芯片')
    
    # CLIP model
    clip_parser = subparsers.add_parser('clip', help='转换CLIP模型')
    clip_parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                            help='HuggingFace模型路径')
    clip_parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    clip_parser.add_argument('--batch-size', type=int, default=1, help='批大小')
    clip_parser.add_argument('--image-size', type=int, default=224, help='图像大小')
    clip_parser.add_argument('--text-length', type=int, default=77, help='文本长度')
    clip_parser.add_argument('--dynamic-batch', action='store_true', help='动态batch')
    clip_parser.add_argument('--soc-version', type=str, default='Ascend910',
                            choices=ModelConverter.SUPPORTED_SOC, help='目标芯片')
    
    args = parser.parse_args()
    
    if args.command == 'onnx':
        converter = ModelConverter(soc_version=args.soc_version)
        converter.convert_onnx_to_om(
            onnx_path=args.model,
            output_path=args.output,
            input_shape=args.input_shape,
            dynamic_dims=args.dynamic_dims
        )
    
    elif args.command == 'clip':
        converter = ModelConverter(soc_version=args.soc_version)
        result = converter.convert_clip_model(
            model_path=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            text_length=args.text_length,
            dynamic_batch=args.dynamic_batch
        )
        
        if result['success']:
            print(f"\n✅ 转换成功!")
            print(f"   OM模型: {result['om_path']}")
        else:
            print(f"\n❌ 转换失败")
            if 'error' in result:
                print(f"   错误: {result['error']}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

