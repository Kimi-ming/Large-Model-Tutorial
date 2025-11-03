#!/usr/bin/env python3
"""
交通场景分析工具

功能：
1. 场景描述生成
2. 异常检测
3. 事故分析

作者：Large Model Tutorial
日期：2023-11-15
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

# BLIP-2导入
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    print("错误: 未安装 transformers")
    print("请运行: pip install transformers accelerate")
    sys.exit(1)


class TrafficSceneAnalyzer:
    """交通场景分析器"""
    
    # 异常类型关键词
    ANOMALY_KEYWORDS = {
        'accident': ['collision', 'crash', 'accident', 'damaged', 'broken'],
        'congestion': ['traffic jam', 'congestion', 'crowded', 'many cars'],
        'violation': ['running red light', 'illegal', 'violation'],
        'obstacle': ['obstacle', 'blocked', 'debris'],
        'weather': ['rain', 'fog', 'snow', 'storm']
    }
    
    def __init__(
        self,
        model_name: str = 'Salesforce/blip2-opt-2.7b',
        device: str = 'auto'
    ):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称
            device: 设备
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_name}")
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成")
    
    def load_image(self, image_path: str) -> Image.Image:
        """加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        return Image.open(image_path).convert('RGB')
    
    def generate_description(
        self,
        image: Image.Image,
        detail_level: str = 'high'
    ) -> str:
        """
        生成场景描述
        
        Args:
            image: PIL图像
            detail_level: 详细程度
            
        Returns:
            场景描述
        """
        prompt = "Describe this traffic scene in detail:"
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if k == 'input_ids' else v.to(self.device) 
                 for k, v in inputs.items()}
        
        max_length = 300 if detail_level == 'high' else 150
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=8 if detail_level == 'high' else 5
            )
        
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return description
    
    def detect_anomaly(self, image: Image.Image) -> Dict:
        """
        检测异常
        
        Args:
            image: PIL图像
            
        Returns:
            异常检测结果
        """
        # 生成场景描述
        description = self.generate_description(image, detail_level='medium')
        
        # 检测异常类型
        anomaly_type = None
        severity = 'low'
        
        for atype, keywords in self.ANOMALY_KEYWORDS.items():
            if any(kw in description.lower() for kw in keywords):
                anomaly_type = atype
                break
        
        if anomaly_type:
            # 评估严重程度
            if anomaly_type in ['accident', 'weather']:
                severity = 'high'
            elif anomaly_type in ['congestion', 'obstacle']:
                severity = 'medium'
            else:
                severity = 'low'
            
            return {
                'has_anomaly': True,
                'type': anomaly_type,
                'severity': severity,
                'description': description,
                'recommendation': self._get_recommendation(anomaly_type, severity)
            }
        else:
            return {
                'has_anomaly': False,
                'type': None,
                'severity': 'none',
                'description': description,
                'recommendation': '正常通行'
            }
    
    def analyze_accident(self, image: Image.Image) -> Dict:
        """
        分析事故
        
        Args:
            image: PIL图像
            
        Returns:
            事故分析结果
        """
        # 事故相关问题
        questions = {
            'type': "What type of accident is shown?",
            'severity': "How severe is this accident?",
            'vehicles': "How many vehicles are involved?",
            'injuries': "Are there any injuries?"
        }
        
        results = {}
        for key, question in questions.items():
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            inputs = {k: v.to(self.device) if k == 'input_ids' else v.to(self.device) 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
            
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            results[key] = answer
        
        # 生成结构化报告
        return {
            'accident_type': results['type'],
            'severity': 'high' if 'severe' in results['severity'].lower() else 'medium',
            'vehicles_involved': self._extract_number(results['vehicles']),
            'injuries_suspected': 'yes' in results['injuries'].lower() or 'injury' in results['injuries'].lower(),
            'scene_description': self.generate_description(image, 'high'),
            'recommendation': '立即通知救援部门' if 'high' in results['severity'] else '清理现场'
        }
    
    def _get_recommendation(self, anomaly_type: str, severity: str) -> str:
        """获取建议"""
        recommendations = {
            'accident': '立即通知救援部门和交警',
            'congestion': '启动交通疏导预案',
            'violation': '自动记录违章信息',
            'obstacle': '派遣清障车辆',
            'weather': '发布恶劣天气预警'
        }
        return recommendations.get(anomaly_type, '继续监控')
    
    def _extract_number(self, text: str) -> int:
        """从文本提取数字"""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 2  # 默认2辆


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='交通场景分析工具')
    
    parser.add_argument('--image', type=str, required=True, help='输入图像')
    parser.add_argument('--task', type=str, required=True,
                        choices=['caption', 'detect_anomaly', 'analyze_accident'])
    parser.add_argument('--detail', type=str, default='high', choices=['low', 'medium', 'high'])
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("交通场景分析工具")
    print("=" * 60)
    
    analyzer = TrafficSceneAnalyzer(device=args.device)
    image = analyzer.load_image(args.image)
    
    if args.task == 'caption':
        description = analyzer.generate_description(image, args.detail)
        print(f"\n场景描述:\n{description}")
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(description)
    
    elif args.task == 'detect_anomaly':
        result = analyzer.detect_anomaly(image)
        print(f"\n异常检测:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    elif args.task == 'analyze_accident':
        result = analyzer.analyze_accident(image)
        print(f"\n事故分析:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存至: {args.output}")
    print("完成！")


if __name__ == '__main__':
    main()

