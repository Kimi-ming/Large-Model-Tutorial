#!/usr/bin/env python3
"""
车辆检测与识别工具

功能：
1. 车型识别（Zero-Shot CLIP）
2. 车辆精确分割（SAM）
3. 车牌定位
4. 批量处理

作者：Large Model Tutorial
日期：2023-11-15
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import torch
from PIL import Image
import cv2

# CLIP导入
try:
    import clip
except ImportError:
    print("错误: 未安装 clip")
    print("请运行: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# SAM导入
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError:
    print("错误: 未安装 segment-anything")
    print("请运行: pip install segment-anything")
    sys.exit(1)


class VehicleDetector:
    """车辆检测器"""
    
    # 车辆类型
    VEHICLE_TYPES = [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle',
        'van', 'suv', 'sedan', 'pickup truck'
    ]
    
    # 车牌类型（中国）
    PLATE_TYPES = {
        'blue': '蓝牌（小型车）',
        'yellow': '黄牌（大型车、货车）',
        'green': '绿牌（新能源车）',
        'black': '黑牌（外籍车）'
    }
    
    def __init__(
        self,
        clip_model: str = 'ViT-B/32',
        sam_model: Optional[str] = None,
        device: str = 'auto',
        precision: str = 'fp32'
    ):
        """
        初始化检测器
        
        Args:
            clip_model: CLIP模型 (RN50/ViT-B-32/ViT-L-14)
            sam_model: SAM模型类型 (vit_h/vit_l/vit_b)，None则不加载
            device: 设备 (cuda/cpu/auto)
            precision: 精度 (fp32/fp16)
        """
        self.precision = precision
        
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载CLIP模型
        print(f"加载CLIP模型: {clip_model}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_model.eval()
        
        # 加载SAM模型（可选）
        self.sam = None
        self.sam_predictor = None
        if sam_model:
            print(f"加载SAM模型: {sam_model}")
            checkpoint = self._get_sam_checkpoint(sam_model)
            self.sam = sam_model_registry[sam_model](checkpoint=checkpoint)
            self.sam.to(device=self.device)
            if precision == 'fp16' and self.device == 'cuda':
                self.sam = self.sam.half()
            self.sam_predictor = SamPredictor(self.sam)
        
        print("模型加载完成")
    
    def _get_sam_checkpoint(self, model_type: str) -> str:
        """获取SAM模型路径"""
        checkpoint_dir = Path(__file__).parent.parent.parent.parent / 'checkpoints'
        checkpoint_map = {
            'vit_h': checkpoint_dir / 'sam_vit_h_4b8939.pth',
            'vit_l': checkpoint_dir / 'sam_vit_l_0b3195.pth',
            'vit_b': checkpoint_dir / 'sam_vit_b_01ec64.pth'
        }
        checkpoint_path = checkpoint_map[model_type]
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM模型文件不存在: {checkpoint_path}\n"
                f"请下载: https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        
        return str(checkpoint_path)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            RGB图像数组 (H, W, 3)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        return image
    
    def detect_vehicles(
        self,
        image: np.ndarray,
        vehicle_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.3
    ) -> Dict:
        """
        检测并分类车辆
        
        Args:
            image: RGB图像 (H, W, 3)
            vehicle_types: 车辆类型列表，None则使用默认
            confidence_threshold: 置信度阈值
            
        Returns:
            检测结果
        """
        if vehicle_types is None:
            vehicle_types = self.VEHICLE_TYPES
        
        # 简化版：检测整个图像的车辆类型
        # 实际应用中应先用目标检测模型定位车辆，再分类
        
        # 准备文本提示
        text_inputs = [f"a photo of a {vehicle_type}" for vehicle_type in vehicle_types]
        text_inputs.append("a photo without any vehicle")  # 背景类
        
        # CLIP分类
        image_pil = Image.fromarray(image)
        image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0]
        
        # 提取结果
        vehicles = []
        for i, vehicle_type in enumerate(vehicle_types):
            confidence = float(similarity[i])
            if confidence > confidence_threshold:
                vehicles.append({
                    'type': vehicle_type,
                    'confidence': confidence,
                    'bbox': [0, 0, image.shape[1], image.shape[0]],  # 全图（简化版）
                    'position': 'center'
                })
        
        # 按置信度排序
        vehicles = sorted(vehicles, key=lambda x: x['confidence'], reverse=True)
        
        return {
            'vehicles': vehicles,
            'total_count': len(vehicles)
        }
    
    def classify_vehicles(
        self,
        image: np.ndarray,
        vehicle_types: Optional[List[str]] = None
    ) -> Dict:
        """
        车型分类（别名，调用detect_vehicles）
        
        Args:
            image: RGB图像
            vehicle_types: 车辆类型列表
            
        Returns:
            分类结果
        """
        return self.detect_vehicles(image, vehicle_types)
    
    def segment_vehicles(
        self,
        image: np.ndarray,
        min_confidence: float = 0.7
    ) -> List[Dict]:
        """
        精确分割车辆
        
        Args:
            image: RGB图像 (H, W, 3)
            min_confidence: 最小置信度
            
        Returns:
            分割结果列表
        """
        if self.sam is None:
            raise RuntimeError("SAM模型未加载，请在初始化时指定sam_model参数")
        
        # 使用SAM自动掩码生成器
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=min_confidence,
            stability_score_thresh=0.92,
            min_mask_region_area=1000  # 过滤小区域
        )
        
        masks = mask_generator.generate(image)
        
        # 后处理：过滤非车辆掩码
        segments = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # [x, y, w, h]
            score = mask_data['stability_score']
            
            # 简单的车辆过滤：根据长宽比和位置
            width = bbox[2]
            height = bbox[3]
            aspect_ratio = width / height if height > 0 else 0
            
            # 车辆长宽比通常在 1.5-3.5 之间
            if 1.2 < aspect_ratio < 4.0:
                segments.append({
                    'mask': mask,
                    'bbox': bbox,
                    'area': int(mask.sum()),
                    'confidence': float(score),
                    'lane': self._estimate_lane(bbox, image.shape)
                })
        
        # 按面积排序（通常车辆是较大的对象）
        segments = sorted(segments, key=lambda x: x['area'], reverse=True)
        
        return segments
    
    def detect_license_plate(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        检测车牌区域
        
        Args:
            image: RGB图像（车辆图像）
            confidence_threshold: 置信度阈值
            
        Returns:
            车牌检测结果
        """
        # 使用CLIP检测车牌区域（简化版）
        # 实际应用中建议使用专门的车牌检测模型（如YOLO + 车牌数据集）
        
        # 车牌通常在图像下半部分
        h, w = image.shape[:2]
        plate_region = image[int(h*0.5):, :]  # 下半部分
        
        # CLIP检测
        text_inputs = [
            "a photo of a license plate",
            "a photo of a car without license plate"
        ]
        
        image_pil = Image.fromarray(plate_region)
        image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            confidence = float(similarity[0, 0])
        
        if confidence > confidence_threshold:
            # 简化：返回下半部分中央区域作为车牌位置
            plate_bbox = [w//4, int(h*0.7), w//2, int(h*0.15)]
            
            # 提取车牌图像
            x, y, pw, ph = plate_bbox
            plate_image = image[y:y+ph, x:x+pw]
            
            return {
                'found': True,
                'bbox': plate_bbox,
                'confidence': confidence,
                'plate_image': plate_image,
                'type': 'blue'  # 简化：默认蓝牌
            }
        else:
            return {
                'found': False,
                'bbox': None,
                'confidence': confidence
            }
    
    def detect_vehicles_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 16,
        **kwargs
    ) -> List[Dict]:
        """
        批量检测车辆
        
        Args:
            images: 图像列表
            batch_size: 批处理大小
            **kwargs: 传递给detect_vehicles的参数
            
        Returns:
            检测结果列表
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            for image in batch:
                result = self.detect_vehicles(image, **kwargs)
                results.append(result)
        return results
    
    def _estimate_lane(self, bbox: List[int], image_shape: Tuple) -> str:
        """估计车辆所在车道"""
        x, y, w, h = bbox
        image_height, image_width = image_shape[:2]
        
        center_x = x + w // 2
        
        # 简单划分：左、中、右
        if center_x < image_width // 3:
            return 'left_lane'
        elif center_x < image_width * 2 // 3:
            return 'middle_lane'
        else:
            return 'right_lane'
    
    def save_result(
        self,
        result: Dict,
        output_path: str,
        image: Optional[np.ndarray] = None
    ):
        """
        保存检测结果
        
        Args:
            result: 检测结果
            output_path: 输出路径
            image: 原始图像（用于可视化）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON
        if output_path.suffix == '.json':
            # 移除不可序列化的字段
            result_copy = {k: v for k, v in result.items() if k != 'mask' and k != 'plate_image'}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, indent=2, ensure_ascii=False)
            print(f"保存结果: {output_path}")
        
        # 保存可视化
        elif image is not None and output_path.suffix in ['.png', '.jpg']:
            vis_image = self._visualize_detections(image, result)
            Image.fromarray(vis_image).save(output_path)
            print(f"保存可视化: {output_path}")
    
    def save_segmentation(
        self,
        segments: List[Dict],
        output_path: str,
        image: np.ndarray
    ):
        """
        保存分割结果
        
        Args:
            segments: 分割结果列表
            output_path: 输出路径
            image: 原始图像
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 可视化所有分割
        vis_image = image.copy()
        for i, seg in enumerate(segments):
            color = self._get_color(i)
            vis_image = self._draw_mask(vis_image, seg['mask'], seg['bbox'], color)
        
        Image.fromarray(vis_image).save(output_path)
        print(f"保存分割结果: {output_path} ({len(segments)} 个车辆)")
    
    def _visualize_detections(
        self,
        image: np.ndarray,
        result: Dict
    ) -> np.ndarray:
        """可视化检测结果"""
        vis_image = image.copy()
        
        if 'vehicles' in result:
            for i, vehicle in enumerate(result['vehicles']):
                bbox = vehicle['bbox']
                vehicle_type = vehicle['type']
                confidence = vehicle['confidence']
                
                # 绘制边界框
                x, y, w, h = bbox
                color = self._get_color(i)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 3)
                
                # 绘制标签
                label = f"{vehicle_type}: {confidence:.2f}"
                cv2.putText(vis_image, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image
    
    def _draw_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: List[int],
        color: Tuple[int, int, int],
        alpha: float = 0.5
    ) -> np.ndarray:
        """绘制分割掩码"""
        vis_image = image.copy()
        
        # 绘制掩码
        overlay = vis_image.copy()
        overlay[mask] = color
        vis_image = cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0)
        
        # 绘制边界框
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        return vis_image
    
    def _get_color(self, index: int) -> Tuple[int, int, int]:
        """获取颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        return colors[index % len(colors)]


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='车辆检测与识别工具')
    
    # 基本参数
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        help='CLIP模型 (RN50/ViT-B/32/ViT-L/14)')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    
    # 任务选择
    parser.add_argument('--task', type=str, required=True,
                        choices=['classify', 'segment', 'detect_plate'],
                        help='任务类型')
    
    # 分割参数
    parser.add_argument('--sam-model', type=str, choices=['vit_h', 'vit_l', 'vit_b'],
                        default='vit_b',  # 默认使用vit_b（速度和精度平衡）
                        help='SAM模型（用于分割任务）')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    
    args = parser.parse_args()
    
    # 初始化检测器
    print("=" * 60)
    print("车辆检测与识别工具")
    print("=" * 60)
    
    detector = VehicleDetector(
        clip_model=args.clip_model,
        sam_model=args.sam_model if args.task == 'segment' else None,
        device=args.device
    )
    
    # 加载图像
    image = detector.load_image(args.image)
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 执行任务
    if args.task == 'classify':
        print("\n车型识别中...")
        result = detector.classify_vehicles(image)
        
        print(f"\n检测到 {result['total_count']} 种车辆类型:")
        for vehicle in result['vehicles']:
            print(f"  {vehicle['type']}: {vehicle['confidence']:.2f}")
        
        detector.save_result(result, args.output, image if args.visualize else None)
    
    elif args.task == 'segment':
        print("\n车辆分割中...")
        segments = detector.segment_vehicles(image)
        
        print(f"\n分割出 {len(segments)} 辆车:")
        for i, seg in enumerate(segments):
            print(f"  车辆 {i+1}: 面积 {seg['area']} 像素, 车道 {seg['lane']}")
        
        if args.visualize:
            detector.save_segmentation(segments, args.output, image)
        else:
            # 保存元数据
            metadata = [{'bbox': s['bbox'], 'area': s['area'], 'lane': s['lane']} 
                       for s in segments]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"保存元数据: {args.output}")
    
    elif args.task == 'detect_plate':
        print("\n车牌检测中...")
        result = detector.detect_license_plate(image)
        
        if result['found']:
            print(f"\n检测到车牌！")
            print(f"  位置: {result['bbox']}")
            print(f"  置信度: {result['confidence']:.2f}")
            print(f"  车牌类型: {result['type']}")
            
            # 保存车牌图像
            if 'plate_image' in result:
                plate_path = Path(args.output).with_suffix('.plate.png')
                Image.fromarray(result['plate_image']).save(plate_path)
                print(f"保存车牌图像: {plate_path}")
        else:
            print("\n未检测到车牌")
        
        # 保存结果
        detector.save_result(result, args.output)
    
    print("\n完成！")


if __name__ == '__main__':
    main()

