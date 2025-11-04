"""
SAM模型推理测试
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestSAMInference:
    """SAM推理功能测试"""
    
    def test_point_prompt(self, sample_image, mock_sam_model):
        """测试点提示分割"""
        points = np.array([[112, 112]])
        labels = np.array([1])
        
        masks, scores, logits = mock_sam_model.predict(
            point_coords=points,
            point_labels=labels
        )
        
        assert masks.shape[0] >= 1
        assert scores.shape[0] == masks.shape[0]
        assert masks.dtype == bool
    
    def test_box_prompt(self, sample_image, mock_sam_model):
        """测试框提示分割"""
        box = np.array([50, 50, 150, 150])
        
        masks, scores, logits = mock_sam_model.predict(box=box)
        
        assert masks.shape[0] >= 1
        assert np.all(masks.sum(axis=(1,2)) > 0)  # 掩码不为空
    
    def test_mask_quality(self, mock_sam_model):
        """测试掩码质量"""
        masks, scores, _ = mock_sam_model.predict()
        
        # 分数应该在0-1之间
        assert np.all(scores >= 0) and np.all(scores <= 1)
        
        # 高分掩码应该有合理的面积
        best_mask = masks[np.argmax(scores)]
        mask_area = best_mask.sum()
        total_area = best_mask.size
        assert 0 < mask_area < total_area


@pytest.mark.unit  
class TestSAMUtils:
    """SAM工具函数测试"""
    
    def test_mask_to_bbox(self, sample_mask):
        """测试掩码转边界框"""
        coords = np.argwhere(sample_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        assert all(v >= 0 for v in bbox)
        assert bbox[2] > 0 and bbox[3] > 0  # width, height > 0
    
    def test_mask_iou(self, sample_mask):
        """测试掩码IoU计算"""
        # 相同掩码的IoU应该是1
        intersection = np.logical_and(sample_mask, sample_mask).sum()
        union = np.logical_or(sample_mask, sample_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        assert np.isclose(iou, 1.0)

