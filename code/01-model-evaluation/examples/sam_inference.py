"""
SAM (Segment Anything Model) 推理示例

本脚本展示SAM模型的各种使用方式：
1. 点提示分割
2. 框提示分割
3. 掩码提示分割
4. 自动分割（整图）
5. 与CLIP结合的语义分割

作者：Large-Model-Tutorial
许可：MIT
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_sam_installation():
    """检查SAM是否已安装"""
    try:
        import segment_anything
        print("✅ segment_anything 已安装")
        return True
    except ImportError:
        print("❌ segment_anything 未安装")
        print("\n安装方法：")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("或者:")
        print("  pip install segment-anything")
        return False


def download_checkpoint(model_type: str = "vit_b") -> str:
    """
    下载SAM检查点（如果不存在）
    
    Args:
        model_type: 模型类型，可选 'vit_b', 'vit_l', 'vit_h'
    
    Returns:
        检查点文件路径
    """
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    
    checkpoint_dir = project_root / "models" / "sam"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"sam_{model_type}_01ec64.pth"
    checkpoint_path = checkpoint_dir / filename
    
    if checkpoint_path.exists():
        print(f"✅ 检查点已存在: {checkpoint_path}")
        return str(checkpoint_path)
    
    print(f"📥 下载SAM检查点: {model_type}")
    print(f"   URL: {checkpoint_urls[model_type]}")
    print(f"   保存到: {checkpoint_path}")
    print("\n⏳ 下载中... (约375MB-2.4GB，取决于模型大小)")
    
    try:
        import urllib.request
        urllib.request.urlretrieve(
            checkpoint_urls[model_type],
            checkpoint_path,
        )
        print("✅ 下载完成！")
        return str(checkpoint_path)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n手动下载方法：")
        print(f"1. 访问: {checkpoint_urls[model_type]}")
        print(f"2. 下载文件并保存到: {checkpoint_path}")
        sys.exit(1)


class SAMInference:
    """SAM推理包装类"""
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        初始化SAM模型
        
        Args:
            model_type: 模型类型 ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: 检查点路径
            device: 设备 ('cuda' 或 'cpu')
        """
        from segment_anything import sam_model_registry, SamPredictor
        
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载模型
        if checkpoint_path is None:
            checkpoint_path = download_checkpoint(model_type)
        
        print(f"加载SAM模型: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        self.predictor = SamPredictor(sam)
        self.model_type = model_type
        print("✅ SAM模型加载完成")
    
    def set_image(self, image: np.ndarray):
        """
        设置要分割的图像
        
        Args:
            image: RGB图像数组 (H, W, 3)
        """
        self.predictor.set_image(image)
        self.current_image = image
    
    def predict_with_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用点提示进行分割
        
        Args:
            points: 点坐标数组 (N, 2), 格式 [[x1, y1], [x2, y2], ...]
            labels: 点标签数组 (N,), 1=前景, 0=背景
            multimask_output: 是否输出多个候选掩码
        
        Returns:
            masks: 掩码数组 (N, H, W)
            scores: IoU分数 (N,)
            logits: 低分辨率logits (N, 256, 256)
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )
        return masks, scores, logits
    
    def predict_with_box(
        self,
        box: np.ndarray,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用框提示进行分割
        
        Args:
            box: 边界框 [x_min, y_min, x_max, y_max]
            multimask_output: 是否输出多个候选掩码
        
        Returns:
            masks, scores, logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=multimask_output,
        )
        return masks, scores, logits
    
    def predict_with_box_and_points(
        self,
        box: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用框+点提示进行分割
        
        Args:
            box: 边界框
            points: 点坐标
            labels: 点标签
            multimask_output: 是否输出多个候选掩码
        
        Returns:
            masks, scores, logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box[None, :],
            multimask_output=multimask_output,
        )
        return masks, scores, logits
    
    def automatic_mask_generation(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
    ) -> List[dict]:
        """
        自动分割整张图像
        
        Args:
            image: 输入图像
            points_per_side: 每边采样点数
            pred_iou_thresh: IoU阈值
            stability_score_thresh: 稳定性阈值
            min_mask_region_area: 最小掩码区域（像素）
        
        Returns:
            掩码列表，每个元素是一个字典包含:
                - segmentation: 二值掩码
                - area: 掩码面积
                - bbox: 边界框
                - predicted_iou: 预测的IoU
                - stability_score: 稳定性分数
        """
        from segment_anything import SamAutomaticMaskGenerator
        
        mask_generator = SamAutomaticMaskGenerator(
            model=self.predictor.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
        )
        
        print("🔍 执行自动分割...")
        start_time = time.time()
        masks = mask_generator.generate(image)
        elapsed = time.time() - start_time
        print(f"✅ 分割完成！找到 {len(masks)} 个掩码，耗时 {elapsed:.2f}秒")
        
        return masks


def visualize_point_prompts(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """可视化点提示分割结果"""
    n_masks = masks.shape[0]
    fig, axes = plt.subplots(1, n_masks + 1, figsize=(5 * (n_masks + 1), 5))
    
    if n_masks == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # 显示原图+提示点
    axes[0].imshow(image)
    for i, (point, label) in enumerate(zip(points, labels)):
        color = 'green' if label == 1 else 'red'
        marker = 'o' if label == 1 else 'x'
        axes[0].plot(point[0], point[1], marker, markersize=15, color=color, markeredgewidth=3)
    axes[0].set_title("原图 + 提示点\n绿色=前景, 红色=背景")
    axes[0].axis('off')
    
    # 显示每个掩码
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(mask, alpha=0.5, cmap='jet')
        axes[i + 1].set_title(f"掩码 {i+1}\nIoU分数: {score:.3f}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存可视化结果: {output_path}")
    plt.close()


def visualize_box_prompt(
    image: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    score: float,
    output_path: str
):
    """可视化框提示分割结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原图+框
    axes[0].imshow(image)
    x_min, y_min, x_max, y_max = box
    rect = plt.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor='red', linewidth=3
    )
    axes[0].add_patch(rect)
    axes[0].set_title("原图 + 框提示")
    axes[0].axis('off')
    
    # 显示分割结果
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f"分割结果\nIoU分数: {score:.3f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存可视化结果: {output_path}")
    plt.close()


def visualize_automatic_masks(
    image: np.ndarray,
    masks: List[dict],
    output_path: str,
    max_display: int = 20
):
    """可视化自动分割结果"""
    # 按面积排序，显示最大的几个
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    sorted_masks = sorted_masks[:max_display]
    
    # 创建合成的分割图
    segmentation = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, mask_dict in enumerate(sorted_masks):
        mask = mask_dict['segmentation']
        segmentation[mask] = (i + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title(f"原图")
    axes[0].axis('off')
    
    # 分割结果
    axes[1].imshow(image)
    axes[1].imshow(segmentation, alpha=0.6, cmap='tab20')
    axes[1].set_title(f"自动分割结果\n共 {len(masks)} 个掩码（显示前{min(max_display, len(masks))}个）")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存可视化结果: {output_path}")
    plt.close()


def example_point_prompts(sam: SAMInference, image_path: str, output_dir: str):
    """示例1：点提示分割"""
    print("\n" + "="*60)
    print("示例1：点提示分割")
    print("="*60)
    
    # 加载图像
    image = np.array(Image.open(image_path).convert("RGB"))
    sam.set_image(image)
    
    h, w = image.shape[:2]
    
    # 场景1：单个前景点
    print("\n场景1：单个前景点（点击图像中心）")
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    
    masks, scores, logits = sam.predict_with_points(
        input_point, input_label, multimask_output=True
    )
    
    visualize_point_prompts(
        image, masks, scores, input_point, input_label,
        os.path.join(output_dir, "01_single_point.png")
    )
    
    # 场景2：多个前景点
    print("\n场景2：多个前景点")
    input_points = np.array([
        [w // 3, h // 3],
        [w // 2, h // 2],
        [2 * w // 3, 2 * h // 3],
    ])
    input_labels = np.array([1, 1, 1])
    
    masks, scores, logits = sam.predict_with_points(
        input_points, input_labels, multimask_output=False
    )
    
    visualize_point_prompts(
        image, masks, scores, input_points, input_labels,
        os.path.join(output_dir, "02_multiple_points.png")
    )
    
    # 场景3：前景+背景点
    print("\n场景3：前景点+背景点（精细化分割）")
    input_points = np.array([
        [w // 2, h // 2],      # 前景
        [w // 10, h // 10],    # 背景
        [9 * w // 10, 9 * h // 10],  # 背景
    ])
    input_labels = np.array([1, 0, 0])
    
    masks, scores, logits = sam.predict_with_points(
        input_points, input_labels, multimask_output=False
    )
    
    visualize_point_prompts(
        image, masks, scores, input_points, input_labels,
        os.path.join(output_dir, "03_foreground_background_points.png")
    )


def example_box_prompt(sam: SAMInference, image_path: str, output_dir: str):
    """示例2：框提示分割"""
    print("\n" + "="*60)
    print("示例2：框提示分割")
    print("="*60)
    
    # 加载图像
    image = np.array(Image.open(image_path).convert("RGB"))
    sam.set_image(image)
    
    h, w = image.shape[:2]
    
    # 定义一个框（中心区域）
    margin = 0.2
    input_box = np.array([
        int(w * margin),
        int(h * margin),
        int(w * (1 - margin)),
        int(h * (1 - margin))
    ])
    
    print(f"框坐标: {input_box}")
    
    masks, scores, logits = sam.predict_with_box(input_box, multimask_output=False)
    
    visualize_box_prompt(
        image, masks[0], input_box, scores[0],
        os.path.join(output_dir, "04_box_prompt.png")
    )


def example_automatic_segmentation(sam: SAMInference, image_path: str, output_dir: str):
    """示例3：自动分割整图"""
    print("\n" + "="*60)
    print("示例3：自动分割整图")
    print("="*60)
    
    # 加载图像
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # 自动分割
    masks = sam.automatic_mask_generation(
        image,
        points_per_side=16,  # 减少采样点以加快速度（演示用）
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    
    # 打印前5个掩码的信息
    print("\n前5个掩码信息：")
    for i, mask_dict in enumerate(masks[:5]):
        print(f"  掩码 {i+1}:")
        print(f"    面积: {mask_dict['area']} 像素")
        print(f"    边界框: {mask_dict['bbox']}")
        print(f"    预测IoU: {mask_dict['predicted_iou']:.3f}")
        print(f"    稳定性分数: {mask_dict['stability_score']:.3f}")
    
    visualize_automatic_masks(
        image, masks,
        os.path.join(output_dir, "05_automatic_segmentation.png")
    )


def example_iterative_refinement(sam: SAMInference, image_path: str, output_dir: str):
    """示例4：迭代精细化"""
    print("\n" + "="*60)
    print("示例4：迭代精细化（使用掩码提示）")
    print("="*60)
    
    # 加载图像
    image = np.array(Image.open(image_path).convert("RGB"))
    sam.set_image(image)
    
    h, w = image.shape[:2]
    
    # 第一次分割：粗略的点
    print("第一次分割：使用单点提示")
    point1 = np.array([[w // 2, h // 2]])
    label1 = np.array([1])
    
    masks1, scores1, logits1 = sam.predict_with_points(
        point1, label1, multimask_output=True
    )
    
    # 选择最佳掩码
    best_idx = np.argmax(scores1)
    
    # 第二次分割：使用前一次的logits作为掩码提示，添加新的点
    print("第二次分割：使用掩码提示+新的点提示进行精细化")
    point2 = np.array([[w // 3, h // 3]])  # 新的点
    label2 = np.array([1])
    
    masks2, scores2, logits2 = sam.predictor.predict(
        point_coords=point2,
        point_labels=label2,
        mask_input=logits1[best_idx:best_idx+1, :, :],  # 使用上次的logits
        multimask_output=False,
    )
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].plot(point1[0, 0], point1[0, 1], 'go', markersize=15, markeredgewidth=3)
    axes[0].set_title("原图 + 第1次提示点")
    axes[0].axis('off')
    
    axes[1].imshow(image)
    axes[1].imshow(masks1[best_idx], alpha=0.5, cmap='jet')
    axes[1].set_title(f"第1次分割结果\nIoU: {scores1[best_idx]:.3f}")
    axes[1].axis('off')
    
    axes[2].imshow(image)
    axes[2].imshow(masks2[0], alpha=0.5, cmap='jet')
    axes[2].plot(point2[0, 0], point2[0, 1], 'go', markersize=15, markeredgewidth=3)
    axes[2].set_title(f"第2次精细化结果\nIoU: {scores2[0]:.3f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "06_iterative_refinement.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存可视化结果: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SAM推理示例")
    parser.add_argument(
        "--image", type=str, required=True,
        help="输入图像路径"
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM模型类型 (默认: vit_b)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="SAM检查点路径（可选，如不提供将自动下载）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/sam_inference",
        help="输出目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="设备 (默认: cuda)"
    )
    parser.add_argument(
        "--examples", type=str, nargs='+',
        default=["points", "box", "automatic", "iterative"],
        choices=["points", "box", "automatic", "iterative"],
        help="要运行的示例（默认: 全部）"
    )
    
    args = parser.parse_args()
    
    # 检查SAM是否安装
    if not check_sam_installation():
        sys.exit(1)
    
    # 检查图像是否存在
    if not os.path.exists(args.image):
        print(f"❌ 图像不存在: {args.image}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")
    
    # 初始化SAM
    print("\n" + "="*60)
    print("初始化SAM模型")
    print("="*60)
    
    sam = SAMInference(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 运行示例
    if "points" in args.examples:
        example_point_prompts(sam, args.image, args.output_dir)
    
    if "box" in args.examples:
        example_box_prompt(sam, args.image, args.output_dir)
    
    if "automatic" in args.examples:
        example_automatic_segmentation(sam, args.image, args.output_dir)
    
    if "iterative" in args.examples:
        example_iterative_refinement(sam, args.image, args.output_dir)
    
    print("\n" + "="*60)
    print("✅ 所有示例完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

