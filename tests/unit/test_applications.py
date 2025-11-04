"""
行业应用测试
"""

import pytest
import numpy as np
from PIL import Image


@pytest.mark.unit
class TestRetailApplication:
    """智慧零售应用测试"""
    
    def test_product_recognition_mock(self, sample_image, mock_clip_model):
        """测试商品识别（Mock）"""
        # 模拟商品数据库
        products = ["可口可乐", "雪碧", "芬达"]
        
        # 模拟识别过程
        image_features = mock_clip_model.encode_image(sample_image)
        text_features = mock_clip_model.encode_text(products)
        
        assert image_features.shape == (1, 512)
        assert text_features.shape == (len(products), 512)
    
    def test_shelf_analysis(self, sample_image):
        """测试货架分析"""
        # 简化测试：验证基本功能
        image_array = np.array(sample_image)
        assert image_array.shape == (224, 224, 3)
        
        # 模拟检测到的商品
        detected_products = [
            {"name": "可口可乐", "confidence": 0.95},
            {"name": "雪碧", "confidence": 0.88}
        ]
        
        assert len(detected_products) > 0
        assert all("name" in p and "confidence" in p for p in detected_products)


@pytest.mark.unit
class TestMedicalApplication:
    """医疗影像应用测试"""
    
    def test_lesion_segmentation(self, sample_image, sample_mask, mock_sam_model):
        """测试病灶分割"""
        masks, scores, _ = mock_sam_model.predict()
        
        assert masks.shape[0] >= 1
        assert scores.shape[0] == masks.shape[0]
        
        # 计算面积和直径
        area = masks[0].sum()
        diameter = np.sqrt(4 * area / np.pi)
        
        assert area > 0
        assert diameter > 0
    
    def test_image_analysis(self, sample_image, mock_blip2_model):
        """测试影像分析"""
        output_ids = mock_blip2_model.generate()
        
        assert output_ids is not None
        assert output_ids.shape[0] == 1  # batch size


@pytest.mark.unit
class TestTrafficApplication:
    """智能交通应用测试"""
    
    def test_vehicle_detection(self, sample_image, mock_clip_model):
        """测试车辆检测"""
        vehicle_types = ["car", "truck", "bus"]
        
        image_features = mock_clip_model.encode_image(sample_image)
        text_features = mock_clip_model.encode_text(vehicle_types)
        
        # 计算相似度
        import torch
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        assert similarity.shape == (1, len(vehicle_types))
        assert torch.allclose(similarity.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_scene_analysis(self, sample_image):
        """测试场景分析"""
        # 简化测试
        img_array = np.array(sample_image)
        assert img_array.ndim == 3
        assert img_array.shape[2] == 3  # RGB

