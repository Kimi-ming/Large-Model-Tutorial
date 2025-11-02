"""
生成SAM Notebook教程
"""
import json

# 创建notebook结构
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# SAM (Segment Anything Model) 分割教程\n\n> 完整演示SAM模型的各种使用方式\n\n**学习目标**：\n- 掌握SAM的点/框提示分割\n- 学会自动分割整图\n- 了解SAM与CLIP的结合\n\n**预计时间**: 45-60分钟"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 导入必要的库\nimport numpy as np\nimport torch\nimport matplotlib.pyplot as plt\nfrom PIL import Image\nimport urllib.request\nfrom pathlib import Path\n\nfrom segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator\n\nprint(f\"PyTorch版本: {torch.__version__}\")\nprint(f\"CUDA可用: {torch.cuda.is_available()}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 1. 加载SAM模型"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 加载模型\nmodel_type = \"vit_b\"\ncheckpoint_path = \"../models/sam/sam_vit_b_01ec64.pth\"\n\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nprint(f\"使用设备: {device}\")\n\nsam = sam_model_registry[model_type](checkpoint=checkpoint_path)\nsam.to(device=device)\npredictor = SamPredictor(sam)\n\nprint(\"✅ SAM模型加载完成\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 2. 准备示例图像"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 加载示例图像\nimage_path = \"sample_image.jpg\"\nimage = Image.open(image_path).convert(\"RGB\")\nimage = np.array(image)\n\n# 显示图像\nplt.figure(figsize=(10, 10))\nplt.imshow(image)\nplt.title(\"原始图像\")\nplt.axis('off')\nplt.show()\n\n# 设置到预测器\npredictor.set_image(image)\nprint(\"✅ 图像已设置\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 3. 点提示分割"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 定义点提示\ninput_point = np.array([[image.shape[1]//2, image.shape[0]//2]])\ninput_label = np.array([1])  # 1 = 前景\n\n# 预测\nmasks, scores, logits = predictor.predict(\n    point_coords=input_point,\n    point_labels=input_label,\n    multimask_output=True,\n)\n\nprint(f\"生成了 {len(masks)} 个候选掩码\")\nprint(f\"IoU分数: {scores}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 4. 框提示分割"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 定义边界框\nh, w = image.shape[:2]\nmargin = 0.2\ninput_box = np.array([\n    int(w * margin), int(h * margin),\n    int(w * (1 - margin)), int(h * (1 - margin))\n])\n\n# 预测\nmasks, scores, logits = predictor.predict(\n    point_coords=None,\n    point_labels=None,\n    box=input_box[None, :],\n    multimask_output=False,\n)\n\nprint(f\"分割完成, IoU: {scores[0]:.3f}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 5. 自动分割"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# 创建自动掩码生成器\nmask_generator = SamAutomaticMaskGenerator(\n    model=sam,\n    points_per_side=16,\n    pred_iou_thresh=0.86,\n    stability_score_thresh=0.92,\n    min_mask_region_area=100,\n)\n\nprint(\"🔍 正在自动分割...\")\nmasks = mask_generator.generate(image)\nprint(f\"✅ 完成！找到 {len(masks)} 个物体\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 总结\n\n本教程演示了SAM的三种主要使用方式：\n\n1. **点提示分割** - 最简单灵活\n2. **框提示分割** - 稳定准确\n3. **自动分割** - 无需提示\n\n### 练习任务\n\n1. 在自己的图像上测试SAM\n2. 尝试不同的提示组合\n3. 调整自动分割参数\n4. 与CLIP结合实现语义分割\n\n### 参考资源\n\n- [SAM官方仓库](https://github.com/facebookresearch/segment-anything)\n- [SAM论文](https://arxiv.org/abs/2304.02643)\n- [SAM详解文档](../docs/01-模型调研与选型/05-SAM模型详解.md)\n\n🎉 恭喜完成本教程！"
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 保存notebook
output_path = "notebooks/03_sam_segmentation_tutorial.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ SAM Notebook已创建: {output_path}")
print(f"   包含 {len(notebook['cells'])} 个cells")
print(f"   涵盖点/框/自动分割三种方式")

