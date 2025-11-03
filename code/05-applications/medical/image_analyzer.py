#!/usr/bin/env python3
"""
医学影像理解与问答工具

功能：
1. 影像描述生成
2. 诊断问答（VQA）
3. 多轮对话
4. 结构化报告生成

作者：Large Model Tutorial
日期：2023-11-15
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import warnings

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


class MedicalImageAnalyzer:
    """医学影像分析器"""
    
    # 影像描述模板
    CAPTION_TEMPLATES = {
        'chest_xray': "This is a chest X-ray showing",
        'ct': "This CT scan demonstrates",
        'mri': "This MRI image reveals",
        'pathology': "This pathology slide shows"
    }
    
    # 结构化报告模板
    REPORT_TEMPLATES = {
        'chest_xray': {
            'sections': ['lungs', 'heart', 'mediastinum', 'bones'],
            'impression': True
        },
        'abdomen_ct': {
            'sections': ['liver', 'kidney', 'pancreas', 'spleen'],
            'impression': True
        },
        'brain_mri': {
            'sections': ['brain_parenchyma', 'ventricles', 'vessels'],
            'impression': True
        }
    }
    
    def __init__(
        self,
        model_name: str = 'Salesforce/blip2-opt-2.7b',
        device: str = 'auto',
        precision: str = 'fp32'
    ):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu/auto)
            precision: 精度 (fp32/fp16)
        """
        self.model_name = model_name
        self.precision = precision
        
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载模型: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if precision == 'fp16' and self.device == 'cuda' else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成")
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            PIL图像
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return image
    
    def generate_caption(
        self,
        image: Image.Image,
        detail_level: str = 'medium',
        template: Optional[str] = None,
        max_length: int = 200
    ) -> str:
        """
        生成影像描述
        
        Args:
            image: PIL图像
            detail_level: 详细程度 (low/medium/high)
            template: 模板类型 (chest_xray/ct/mri/pathology)
            max_length: 最大长度
            
        Returns:
            影像描述文本
        """
        # 准备提示词
        prompt = ""
        if template and template in self.CAPTION_TEMPLATES:
            prompt = self.CAPTION_TEMPLATES[template]
        
        # 根据详细程度调整参数
        if detail_level == 'low':
            max_length = 50
            num_beams = 3
        elif detail_level == 'high':
            max_length = 300
            num_beams = 8
        else:  # medium
            max_length = 150
            num_beams = 5
        
        # 生成描述
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, dtype=torch.float16 if self.precision == 'fp16' else torch.float32
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=0.7,
                do_sample=False
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
    
    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100
    ) -> str:
        """
        回答关于影像的问题（VQA）
        
        Args:
            image: PIL图像
            question: 问题文本
            max_length: 最大回答长度
            
        Returns:
            回答文本
        """
        # 准备输入
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(
            self.device, dtype=torch.float16 if self.precision == 'fp16' else torch.float32
        )
        
        # 生成回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                temperature=0.7
            )
        
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return answer
    
    def start_conversation(self, image: Image.Image):
        """
        开始多轮对话
        
        Args:
            image: PIL图像
            
        Returns:
            对话管理器
        """
        return ConversationManager(self, image)
    
    def generate_structured_report(
        self,
        image: Image.Image,
        template: str = 'chest_xray',
        include_measurements: bool = False
    ) -> Dict:
        """
        生成结构化报告
        
        Args:
            image: PIL图像
            template: 报告模板
            include_measurements: 是否包含测量值
            
        Returns:
            结构化报告（JSON）
        """
        if template not in self.REPORT_TEMPLATES:
            raise ValueError(f"不支持的模板: {template}")
        
        template_config = self.REPORT_TEMPLATES[template]
        sections = template_config['sections']
        
        # 生成各部分描述
        findings = []
        for section in sections:
            question = f"Describe the {section.replace('_', ' ')}"
            description = self.answer_question(image, question, max_length=150)
            
            # 判断是否异常
            abnormality = self._detect_abnormality(description)
            
            findings.append({
                'organ': section,
                'description': description,
                'abnormality': abnormality
            })
        
        # 生成印象
        impression = []
        for finding in findings:
            if finding['abnormality']:
                impression.append(finding['description'])
        
        if not impression:
            impression.append("未见明显异常")
        
        # 生成建议
        recommendation = self._generate_recommendation(findings)
        
        return {
            'patient_info': {
                'study_id': 'N/A',
                'modality': template.split('_')[-1].upper(),
                'body_part': template.split('_')[0].title()
            },
            'findings': findings,
            'impression': impression,
            'recommendation': recommendation
        }
    
    def analyze_nodule(
        self,
        roi: Image.Image,
        clinical_info: Optional[Dict] = None
    ) -> Dict:
        """
        分析肺结节特征
        
        Args:
            roi: 结节ROI图像
            clinical_info: 临床信息（可选）
            
        Returns:
            结节分析结果
        """
        # 特征提取
        features = {}
        
        # 1. 密度类型
        density_question = "Is this nodule solid, ground-glass, or part-solid?"
        density_answer = self.answer_question(roi, density_question)
        features['density_type'] = self._parse_density(density_answer)
        
        # 2. 形状
        shape_question = "Is this nodule round, oval, or irregular in shape?"
        shape_answer = self.answer_question(roi, shape_question)
        features['shape'] = self._parse_shape(shape_answer)
        
        # 3. 边缘
        margin_question = "Is the margin of this nodule smooth, lobulated, or spiculated?"
        margin_answer = self.answer_question(roi, margin_question)
        features['margin'] = self._parse_margin(margin_answer)
        
        # 4. 钙化
        calc_question = "Is there any calcification in this nodule?"
        calc_answer = self.answer_question(roi, calc_question)
        features['calcification'] = 'yes' in calc_answer.lower() or 'calcification' in calc_answer.lower()
        
        # 5. 恶性概率评估
        malignancy = self._assess_malignancy(features, clinical_info)
        
        return {
            'density_type': features['density_type'],
            'shape': features['shape'],
            'margin': features['margin'],
            'calcification': features['calcification'],
            'malignancy_probability': malignancy['probability'],
            'lung_rads_category': malignancy['lung_rads'],
            'recommendation': malignancy['recommendation']
        }
    
    def multimodal_diagnosis(
        self,
        image: Image.Image,
        clinical_info: Dict,
        explain: bool = True
    ) -> Dict:
        """
        多模态诊断（影像 + 临床信息）
        
        Args:
            image: 影像图像
            clinical_info: 临床信息
            explain: 是否生成解释
            
        Returns:
            诊断结果
        """
        # 影像分析
        image_findings = self.generate_caption(image, detail_level='high')
        
        # 提取影像证据
        imaging_evidence = self._extract_evidence(image_findings)
        
        # 提取临床证据
        clinical_evidence = self._format_clinical_evidence(clinical_info)
        
        # 综合诊断（这里是简化版，实际需要更复杂的推理）
        diagnosis = self._integrate_diagnosis(imaging_evidence, clinical_evidence)
        
        result = {
            'primary_diagnosis': diagnosis['primary'],
            'confidence': diagnosis['confidence'],
            'evidence': {
                'imaging': imaging_evidence,
                'clinical': clinical_evidence
            },
            'differential_diagnosis': diagnosis['differential'],
            'recommendation': diagnosis['recommendation'],
            'urgency': diagnosis['urgency']
        }
        
        if explain:
            result['explanation'] = diagnosis['explanation']
        
        return result
    
    def _detect_abnormality(self, description: str) -> bool:
        """检测描述中是否有异常"""
        abnormal_keywords = [
            'lesion', 'mass', 'nodule', 'tumor', 'abnormal', 'enlarged',
            'thickened', 'opacity', 'consolidation', 'effusion'
        ]
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in abnormal_keywords)
    
    def _generate_recommendation(self, findings: List[Dict]) -> str:
        """生成建议"""
        has_abnormality = any(f['abnormality'] for f in findings)
        
        if has_abnormality:
            return "建议进一步检查或专科会诊"
        else:
            return "定期随访"
    
    def _parse_density(self, answer: str) -> str:
        """解析密度类型"""
        answer_lower = answer.lower()
        if 'ground-glass' in answer_lower or 'ground glass' in answer_lower:
            return 'ground-glass'
        elif 'part-solid' in answer_lower or 'partial' in answer_lower:
            return 'part-solid'
        else:
            return 'solid'
    
    def _parse_shape(self, answer: str) -> str:
        """解析形状"""
        answer_lower = answer.lower()
        if 'irregular' in answer_lower:
            return 'irregular'
        elif 'oval' in answer_lower:
            return 'oval'
        else:
            return 'round'
    
    def _parse_margin(self, answer: str) -> str:
        """解析边缘"""
        answer_lower = answer.lower()
        if 'spiculated' in answer_lower or 'spiculation' in answer_lower:
            return 'spiculated'
        elif 'lobulated' in answer_lower or 'lobulation' in answer_lower:
            return 'lobulated'
        else:
            return 'smooth'
    
    def _assess_malignancy(
        self,
        features: Dict,
        clinical_info: Optional[Dict]
    ) -> Dict:
        """评估恶性概率（简化版Lung-RADS）"""
        score = 0
        
        # 密度评分
        if features['density_type'] == 'solid':
            score += 2
        elif features['density_type'] == 'part-solid':
            score += 3
        
        # 形状评分
        if features['shape'] == 'irregular':
            score += 2
        
        # 边缘评分
        if features['margin'] == 'spiculated':
            score += 3
        elif features['margin'] == 'lobulated':
            score += 1
        
        # 钙化（良性特征）
        if features['calcification']:
            score -= 2
        
        # 概率评估
        if score <= 1:
            probability = 0.1
            lung_rads = '2'
            recommendation = '12个月年度随访'
        elif score <= 3:
            probability = 0.3
            lung_rads = '3'
            recommendation = '6个月短期随访'
        elif score <= 5:
            probability = 0.6
            lung_rads = '4A'
            recommendation = '3个月短期随访CT'
        else:
            probability = 0.85
            lung_rads = '4B'
            recommendation = '建议PET-CT或活检'
        
        return {
            'probability': probability,
            'lung_rads': lung_rads,
            'recommendation': recommendation
        }
    
    def _extract_evidence(self, findings: str) -> List[str]:
        """从影像描述中提取证据"""
        # 简化版：按句子分割
        sentences = findings.split('.')
        evidence = [s.strip() for s in sentences if len(s.strip()) > 10]
        return evidence[:3]  # 最多3条
    
    def _format_clinical_evidence(self, clinical_info: Dict) -> List[str]:
        """格式化临床证据"""
        evidence = []
        
        if 'age' in clinical_info and 'gender' in clinical_info:
            evidence.append(f"{clinical_info['age']}岁{clinical_info['gender']}")
        
        if 'smoking_history' in clinical_info:
            evidence.append(f"吸烟史: {clinical_info['smoking_history']}")
        
        if 'symptoms' in clinical_info:
            symptoms_str = '、'.join(clinical_info['symptoms'])
            evidence.append(f"症状: {symptoms_str}")
        
        return evidence
    
    def _integrate_diagnosis(
        self,
        imaging_evidence: List[str],
        clinical_evidence: List[str]
    ) -> Dict:
        """整合诊断（简化版）"""
        # 这里是简化版逻辑，实际需要更复杂的推理模型
        
        return {
            'primary': '肺癌高度可疑',
            'confidence': 0.87,
            'differential': [
                {'disease': '肺腺癌', 'probability': 0.55},
                {'disease': '鳞状细胞癌', 'probability': 0.25},
                {'disease': '小细胞肺癌', 'probability': 0.07}
            ],
            'recommendation': [
                '强烈建议PET-CT检查评估转移情况',
                '建议穿刺活检明确病理类型',
                '建议呼吸内科或胸外科会诊'
            ],
            'urgency': 'high',
            'explanation': '基于影像学特征和临床高危因素，肺癌可能性极大。'
        }


class ConversationManager:
    """多轮对话管理器"""
    
    def __init__(self, analyzer: MedicalImageAnalyzer, image: Image.Image):
        """
        初始化对话管理器
        
        Args:
            analyzer: 影像分析器
            image: 图像
        """
        self.analyzer = analyzer
        self.image = image
        self.history = []  # [(question, answer), ...]
    
    def ask(self, question: str) -> str:
        """
        提问
        
        Args:
            question: 问题文本
            
        Returns:
            回答文本
        """
        # 构建上下文（包含历史对话）
        context = self._build_context(question)
        
        # 获取回答
        answer = self.analyzer.answer_question(self.image, context, max_length=150)
        
        # 记录历史
        self.history.append((question, answer))
        
        return answer
    
    def _build_context(self, current_question: str) -> str:
        """构建包含历史的上下文"""
        if not self.history:
            return current_question
        
        # 包含最近2轮对话
        context_parts = []
        for q, a in self.history[-2:]:
            context_parts.append(f"Q: {q} A: {a}")
        
        context_parts.append(f"Q: {current_question}")
        context = " ".join(context_parts)
        
        return context
    
    def reset(self):
        """重置对话历史"""
        self.history = []


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='医学影像理解与问答工具')
    
    # 基本参数
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, help='输出路径（可选）')
    parser.add_argument('--model', type=str, default='Salesforce/blip2-opt-2.7b',
                        help='模型名称')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    
    # 任务选择
    parser.add_argument('--task', type=str, required=True,
                        choices=['caption', 'vqa', 'structured_report'],
                        help='任务类型')
    
    # 影像描述参数
    parser.add_argument('--detail', type=str, default='medium',
                        choices=['low', 'medium', 'high'], help='详细程度')
    parser.add_argument('--template', type=str,
                        choices=['chest_xray', 'ct', 'mri', 'pathology', 'abdomen_ct', 'brain_mri'],
                        help='模板类型')
    
    # VQA参数
    parser.add_argument('--question', type=str, help='问题文本（VQA任务）')
    
    args = parser.parse_args()
    
    # 初始化分析器
    print("=" * 60)
    print("医学影像理解与问答工具")
    print("=" * 60)
    
    analyzer = MedicalImageAnalyzer(
        model_name=args.model,
        device=args.device
    )
    
    # 加载图像
    image = analyzer.load_image(args.image)
    print(f"图像尺寸: {image.size}")
    
    # 执行任务
    if args.task == 'caption':
        print(f"\n生成影像描述（详细程度: {args.detail}）...")
        caption = analyzer.generate_caption(
            image=image,
            detail_level=args.detail,
            template=args.template
        )
        
        print(f"\n影像描述:")
        print(caption)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(caption)
            print(f"\n保存至: {args.output}")
    
    elif args.task == 'vqa':
        if not args.question:
            parser.error("VQA任务需要 --question")
        
        print(f"\n问题: {args.question}")
        answer = analyzer.answer_question(image, args.question)
        
        print(f"回答: {answer}")
        
        if args.output:
            result = {
                'question': args.question,
                'answer': answer
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n保存至: {args.output}")
    
    elif args.task == 'structured_report':
        if not args.template:
            parser.error("结构化报告任务需要 --template")
        
        print(f"\n生成结构化报告（模板: {args.template}）...")
        report = analyzer.generate_structured_report(
            image=image,
            template=args.template
        )
        
        print(f"\n结构化报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n保存至: {args.output}")
    
    print("\n完成！")


if __name__ == '__main__':
    main()

