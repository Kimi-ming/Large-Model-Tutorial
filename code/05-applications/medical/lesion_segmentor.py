#!/usr/bin/env python3
"""
医学影像病灶分割工具

功能：
1. 交互式分割（点/框提示）
2. 自动批量分割
3. 3D分割与可视化
4. DICOM支持

作者：Large Model Tutorial
日期：2023-11-15
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings

import numpy as np
import torch
from PIL import Image
import cv2

# SAM导入
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError:
    print("错误: 未安装 segment-anything")
    print("请运行: pip install segment-anything")
    sys.exit(1)


class LesionSegmentor:
    """病灶分割器"""
    
    def __init__(
        self,
        model_type: str = 'vit_h',
        checkpoint: Optional[str] = None,
        device: str = 'auto',
        precision: str = 'fp32'
    ):
        """
        初始化分割器
        
        Args:
            model_type: 模型类型 (vit_h/vit_l/vit_b)
            checkpoint: 模型权重路径
            device: 设备 (cuda/cpu/auto)
            precision: 精度 (fp32/fp16/bf16)
        """
        self.model_type = model_type
        self.precision = precision
        
        # 设备选择
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 模型路径
        if checkpoint is None:
            checkpoint = self._get_default_checkpoint(model_type)
        
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"模型文件不存在: {checkpoint}\n"
                f"请从以下地址下载:\n"
                f"vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                f"vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                f"vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )
        
        # 加载模型
        print(f"加载模型: {model_type} from {checkpoint}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        
        # 混合精度
        if precision == 'fp16' and self.device == 'cuda':
            self.sam = self.sam.half()
        
        # 预测器
        self.predictor = SamPredictor(self.sam)
        
        # 自动掩码生成器（用于批量分割）
        self.mask_generator = None  # 按需初始化
        
    def _get_default_checkpoint(self, model_type: str) -> str:
        """获取默认模型路径"""
        checkpoint_dir = Path(__file__).parent.parent.parent.parent / 'checkpoints'
        checkpoint_map = {
            'vit_h': checkpoint_dir / 'sam_vit_h_4b8939.pth',
            'vit_l': checkpoint_dir / 'sam_vit_l_0b3195.pth',
            'vit_b': checkpoint_dir / 'sam_vit_b_01ec64.pth'
        }
        return str(checkpoint_map[model_type])
    
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
    
    def segment_interactive(
        self,
        image: np.ndarray,
        prompt_type: str = 'point',
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[List[int]] = None,
        multimask: bool = False
    ) -> Dict:
        """
        交互式分割
        
        Args:
            image: RGB图像 (H, W, 3)
            prompt_type: 提示类型 ('point'/'box')
            points: 点坐标 [[x1,y1], [x2,y2], ...]
            labels: 点标签 [1=前景, 0=背景]
            box: 边界框 [x, y, w, h]
            multimask: 是否生成多个候选掩码
            
        Returns:
            分割结果字典
        """
        # 设置图像
        self.predictor.set_image(image)
        
        # 准备提示
        point_coords = None
        point_labels = None
        box_coords = None
        
        if prompt_type == 'point':
            if points is None or labels is None:
                raise ValueError("点模式需要提供 points 和 labels")
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
            
        elif prompt_type == 'box':
            if box is None:
                raise ValueError("框模式需要提供 box")
            # box: [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = box
            box_coords = np.array([x, y, x + w, y + h], dtype=np.float32)
            
        else:
            raise ValueError(f"不支持的提示类型: {prompt_type}")
        
        # 预测
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=multimask
        )
        
        # 选择最佳掩码
        if multimask:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
        else:
            mask = masks[0]
            score = scores[0]
        
        # 计算指标
        result = self._compute_metrics(mask, image, score)
        
        return result
    
    def segment_auto(
        self,
        image: np.ndarray,
        lesion_type: str = 'nodule',
        min_size: float = 3.0,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.5
    ) -> List[Dict]:
        """
        自动批量分割
        
        Args:
            image: RGB图像 (H, W, 3)
            lesion_type: 病灶类型 (nodule/tumor/lesion)
            min_size: 最小直径（mm）
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测到的病灶列表
        """
        # 初始化自动掩码生成器
        if self.mask_generator is None:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=confidence_threshold,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
        
        # 生成掩码
        masks = self.mask_generator.generate(image)
        
        # 过滤和后处理
        lesions = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            score = mask_data['stability_score']
            
            # 计算指标
            metrics = self._compute_metrics(mask, image, score)
            
            # 过滤小病灶
            if metrics['diameter'] < min_size:
                continue
            
            # 过滤低置信度
            if score < confidence_threshold:
                continue
            
            # 添加类型信息
            metrics['lesion_type'] = lesion_type
            lesions.append(metrics)
        
        # NMS去重
        lesions = self._non_maximum_suppression(lesions, nms_threshold)
        
        # 按面积排序
        lesions = sorted(lesions, key=lambda x: x['area'], reverse=True)
        
        return lesions
    
    def segment_3d(
        self,
        series: np.ndarray,
        seed_slice: int,
        seed_point: List[int],
        propagation: str = 'bidirectional',
        max_slices: int = 100
    ) -> Dict:
        """
        3D分割（多切片）
        
        Args:
            series: 3D图像序列 (D, H, W, 3)
            seed_slice: 种子切片索引
            seed_point: 种子点坐标 [x, y]
            propagation: 传播方向 (forward/backward/bidirectional)
            max_slices: 最大处理切片数
            
        Returns:
            3D分割结果
        """
        num_slices = series.shape[0]
        mask_3d = np.zeros((num_slices, series.shape[1], series.shape[2]), dtype=bool)
        
        # 种子切片分割
        seed_image = series[seed_slice]
        seed_result = self.segment_interactive(
            image=seed_image,
            prompt_type='point',
            points=[seed_point],
            labels=[1]
        )
        mask_3d[seed_slice] = seed_result['mask']
        
        # 向前传播
        if propagation in ['forward', 'bidirectional']:
            for i in range(seed_slice + 1, min(seed_slice + max_slices, num_slices)):
                prev_mask = mask_3d[i - 1]
                if not prev_mask.any():
                    break
                
                # 使用前一切片的掩码中心作为提示
                centroid = self._get_mask_centroid(prev_mask)
                
                curr_image = series[i]
                result = self.segment_interactive(
                    image=curr_image,
                    prompt_type='point',
                    points=[centroid],
                    labels=[1]
                )
                
                mask_3d[i] = result['mask']
        
        # 向后传播
        if propagation in ['backward', 'bidirectional']:
            for i in range(seed_slice - 1, max(seed_slice - max_slices, -1), -1):
                next_mask = mask_3d[i + 1]
                if not next_mask.any():
                    break
                
                centroid = self._get_mask_centroid(next_mask)
                
                curr_image = series[i]
                result = self.segment_interactive(
                    image=curr_image,
                    prompt_type='point',
                    points=[centroid],
                    labels=[1]
                )
                
                mask_3d[i] = result['mask']
        
        # 计算3D指标
        result_3d = self._compute_3d_metrics(mask_3d)
        result_3d['mask_3d'] = mask_3d
        
        return result_3d
    
    def load_series(self, series_dir: str) -> np.ndarray:
        """
        加载图像序列
        
        Args:
            series_dir: 序列目录（包含多张切片图像）
            
        Returns:
            3D图像数组 (D, H, W, 3)
        """
        series_path = Path(series_dir)
        
        # 查找所有图像文件
        image_files = sorted(series_path.glob('*.png')) + sorted(series_path.glob('*.jpg'))
        
        if len(image_files) == 0:
            raise ValueError(f"目录中未找到图像: {series_dir}")
        
        # 加载所有切片
        slices = []
        for img_file in image_files:
            img = self.load_image(str(img_file))
            slices.append(img)
        
        series = np.stack(slices, axis=0)
        print(f"加载序列: {series.shape[0]} 个切片, 尺寸 {series.shape[1]}x{series.shape[2]}")
        
        return series
    
    def _compute_metrics(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        confidence: float
    ) -> Dict:
        """计算分割指标"""
        # 基本指标
        area = np.sum(mask)
        
        # 边界框
        coords = np.argwhere(mask)
        if len(coords) == 0:
            bbox = [0, 0, 0, 0]
            diameter = 0
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            
            # 等效直径（假设1像素=1mm，实际需根据DICOM元数据调整）
            diameter = np.sqrt(4 * area / np.pi)
        
        return {
            'mask': mask,
            'area': int(area),
            'bbox': bbox,
            'diameter': float(diameter),
            'confidence': float(confidence)
        }
    
    def _compute_3d_metrics(self, mask_3d: np.ndarray) -> Dict:
        """计算3D指标"""
        # 体积（像素数）
        volume_pixels = np.sum(mask_3d)
        
        # 假设切片间距1mm，像素间距1mm（实际需从DICOM读取）
        slice_spacing = 1.0  # mm
        pixel_spacing = 1.0  # mm
        volume_mm3 = volume_pixels * pixel_spacing * pixel_spacing * slice_spacing
        
        # 最大直径
        max_diameter = 0
        for slice_mask in mask_3d:
            if slice_mask.any():
                area = np.sum(slice_mask)
                diameter = np.sqrt(4 * area / np.pi)
                max_diameter = max(max_diameter, diameter)
        
        return {
            'volume_pixels': int(volume_pixels),
            'volume_mm3': float(volume_mm3),
            'max_diameter': float(max_diameter),
            'num_slices': int(np.sum([m.any() for m in mask_3d]))
        }
    
    def _get_mask_centroid(self, mask: np.ndarray) -> List[int]:
        """获取掩码质心"""
        coords = np.argwhere(mask)
        if len(coords) == 0:
            # 返回图像中心
            h, w = mask.shape
            return [w // 2, h // 2]
        
        centroid_y, centroid_x = coords.mean(axis=0)
        return [int(centroid_x), int(centroid_y)]
    
    def _non_maximum_suppression(
        self,
        lesions: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """非极大值抑制，去除重叠检测"""
        if len(lesions) == 0:
            return []
        
        # 按置信度排序
        lesions = sorted(lesions, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, lesion1 in enumerate(lesions):
            bbox1 = lesion1['bbox']
            suppress = False
            
            for lesion2 in keep:
                bbox2 = lesion2['bbox']
                iou = self._compute_iou(bbox1, bbox2)
                
                if iou > threshold:
                    suppress = True
                    break
            
            if not suppress:
                keep.append(lesion1)
        
        return keep
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def save_result(
        self,
        result: Dict,
        output_path: str,
        image: Optional[np.ndarray] = None,
        visualize: bool = True
    ):
        """
        保存分割结果
        
        Args:
            result: 分割结果
            output_path: 输出路径
            image: 原始图像（用于可视化）
            visualize: 是否生成可视化
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if visualize and image is not None:
            # 可视化
            vis_image = self._visualize_mask(image, result['mask'], result.get('bbox'))
            Image.fromarray(vis_image).save(output_path)
            print(f"保存可视化结果: {output_path}")
        else:
            # 仅保存掩码
            mask_image = (result['mask'] * 255).astype(np.uint8)
            Image.fromarray(mask_image).save(output_path)
            print(f"保存掩码: {output_path}")
        
        # 保存元数据
        metadata_path = output_path.with_suffix('.json')
        metadata = {k: v for k, v in result.items() if k != 'mask'}
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"保存元数据: {metadata_path}")
    
    def save_batch_results(
        self,
        results: List[Dict],
        output_path: str,
        image: Optional[np.ndarray] = None
    ):
        """
        保存批量分割结果
        
        Args:
            results: 分割结果列表
            output_path: 输出路径
            image: 原始图像
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if image is not None:
            # 可视化所有掩码
            vis_image = image.copy()
            for i, result in enumerate(results):
                vis_image = self._visualize_mask(
                    vis_image,
                    result['mask'],
                    result['bbox'],
                    color=self._get_color(i)
                )
            
            Image.fromarray(vis_image).save(output_path)
            print(f"保存批量可视化: {output_path}")
        
        # 保存元数据
        metadata_path = output_path.with_suffix('.json')
        metadata = []
        for result in results:
            item = {k: v for k, v in result.items() if k != 'mask'}
            metadata.append(item)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"保存元数据: {metadata_path} ({len(results)} 个病灶)")
    
    def save_nifti(self, mask_3d: np.ndarray, output_path: str):
        """
        保存为NIfTI格式（需要nibabel）
        
        Args:
            mask_3d: 3D掩码 (D, H, W)
            output_path: 输出路径 (.nii.gz)
        """
        try:
            import nibabel as nib
        except ImportError:
            warnings.warn("未安装nibabel，无法保存NIfTI格式。请运行: pip install nibabel")
            return
        
        # 创建NIfTI图像
        nifti_img = nib.Nifti1Image(mask_3d.astype(np.uint8), affine=np.eye(4))
        nib.save(nifti_img, output_path)
        print(f"保存NIfTI: {output_path}")
    
    def _visualize_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Optional[List[int]] = None,
        color: Optional[Tuple[int, int, int]] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """可视化掩码"""
        if color is None:
            color = (255, 0, 0)  # 红色
        
        vis_image = image.copy()
        
        # 绘制掩码
        overlay = vis_image.copy()
        overlay[mask] = color
        vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
        
        # 绘制边界框
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # 绘制掩码轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        return vis_image
    
    def _get_color(self, index: int) -> Tuple[int, int, int]:
        """获取可视化颜色"""
        colors = [
            (255, 0, 0),    # 红
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 蓝
            (255, 255, 0),  # 黄
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 青
        ]
        return colors[index % len(colors)]


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='医学影像病灶分割工具')
    
    # 基本参数
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--image-dir', type=str, help='输入图像序列目录（用于3D分割）')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    parser.add_argument('--model-type', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'], help='模型类型')
    parser.add_argument('--checkpoint', type=str, help='模型权重路径')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    
    # 模式选择
    parser.add_argument('--mode', type=str, required=True,
                        choices=['interactive', 'auto', '3d'], help='分割模式')
    
    # 交互式分割参数
    parser.add_argument('--prompt', type=str, choices=['point', 'box'],
                        help='提示类型（交互式模式）')
    parser.add_argument('--coords', type=str,
                        help='点坐标，JSON格式: "[[x1,y1],[x2,y2]]"')
    parser.add_argument('--labels', type=str,
                        help='点标签，JSON格式: "[1,1]" (1=前景, 0=背景)')
    parser.add_argument('--box', type=str,
                        help='边界框，JSON格式: "[x,y,w,h]"')
    
    # 自动分割参数
    parser.add_argument('--lesion-type', type=str, default='nodule',
                        choices=['nodule', 'tumor', 'lesion'], help='病灶类型')
    parser.add_argument('--min-size', type=float, default=3.0, help='最小直径(mm)')
    
    # 3D分割参数
    parser.add_argument('--slice-index', type=int, help='种子切片索引')
    
    # 可视化
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    
    args = parser.parse_args()
    
    # 初始化分割器
    print("=" * 60)
    print("医学影像病灶分割工具")
    print("=" * 60)
    
    segmentor = LesionSegmentor(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device
    )
    
    # 根据模式执行分割
    if args.mode == 'interactive':
        # 交互式分割
        if not args.image:
            parser.error("交互式模式需要 --image")
        if not args.prompt:
            parser.error("交互式模式需要 --prompt")
        
        image = segmentor.load_image(args.image)
        print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        if args.prompt == 'point':
            if not args.coords or not args.labels:
                parser.error("点模式需要 --coords 和 --labels")
            
            points = json.loads(args.coords)
            labels = json.loads(args.labels)
            
            print(f"点提示: {len(points)} 个点")
            result = segmentor.segment_interactive(
                image=image,
                prompt_type='point',
                points=points,
                labels=labels
            )
        else:  # box
            if not args.box:
                parser.error("框模式需要 --box")
            
            box = json.loads(args.box)
            print(f"框提示: {box}")
            result = segmentor.segment_interactive(
                image=image,
                prompt_type='box',
                box=box
            )
        
        print(f"\n分割结果:")
        print(f"  面积: {result['area']} 像素")
        print(f"  直径: {result['diameter']:.1f} mm")
        print(f"  置信度: {result['confidence']:.2f}")
        
        segmentor.save_result(result, args.output, image, args.visualize)
        
    elif args.mode == 'auto':
        # 自动批量分割
        if not args.image:
            parser.error("自动模式需要 --image")
        
        image = segmentor.load_image(args.image)
        print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
        print(f"病灶类型: {args.lesion_type}")
        print(f"最小尺寸: {args.min_size} mm")
        
        results = segmentor.segment_auto(
            image=image,
            lesion_type=args.lesion_type,
            min_size=args.min_size
        )
        
        print(f"\n检测到 {len(results)} 个病灶:")
        for i, lesion in enumerate(results):
            print(f"  病灶 {i+1}:")
            print(f"    位置: {lesion['bbox']}")
            print(f"    直径: {lesion['diameter']:.1f} mm")
            print(f"    置信度: {lesion['confidence']:.2f}")
        
        segmentor.save_batch_results(results, args.output, image)
        
    elif args.mode == '3d':
        # 3D分割
        if not args.image_dir:
            parser.error("3D模式需要 --image-dir")
        if args.slice_index is None:
            parser.error("3D模式需要 --slice-index")
        if not args.coords:
            parser.error("3D模式需要 --coords（种子点）")
        
        series = segmentor.load_series(args.image_dir)
        points = json.loads(args.coords)
        
        if len(points) != 1:
            parser.error("3D模式仅支持单个种子点")
        
        seed_point = points[0]
        print(f"种子切片: {args.slice_index}")
        print(f"种子点: {seed_point}")
        
        result_3d = segmentor.segment_3d(
            series=series,
            seed_slice=args.slice_index,
            seed_point=seed_point
        )
        
        print(f"\n3D分割结果:")
        print(f"  体积: {result_3d['volume_mm3']:.1f} mm³")
        print(f"  最大直径: {result_3d['max_diameter']:.1f} mm")
        print(f"  切片数: {result_3d['num_slices']}")
        
        # 保存为NIfTI
        segmentor.save_nifti(result_3d['mask_3d'], args.output)
    
    print("\n完成！")


if __name__ == '__main__':
    main()

