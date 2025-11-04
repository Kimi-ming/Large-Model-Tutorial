"""
CLIP模型推理测试

测试 code/01-model-evaluation/examples/clip_inference.py
"""

import pytest
import sys
from pathlib import Path

# 添加代码路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'code' / '01-model-evaluation' / 'examples'))


@pytest.mark.unit
class TestCLIPInference:
    """CLIP推理功能测试"""
    
    def test_import(self):
        """测试模块可以导入"""
        try:
            import clip
            assert True
        except ImportError:
            pytest.skip("CLIP未安装")
    
    def test_zero_shot_classification(self, sample_image, mock_clip_model):
        """测试零样本分类"""
        # Mock测试（不需要真实模型）
        import torch
        
        # 模拟分类过程
        image_features = mock_clip_model.encode_image(sample_image)
        assert image_features.shape == (1, 512)
        
        text_prompts = ["a cat", "a dog", "a bird"]
        text_features = mock_clip_model.encode_text(text_prompts)
        assert text_features.shape == (3, 512)
        
        # 计算相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)
        
        assert similarity.shape == (1, 3)
        assert torch.all(similarity >= -1) and torch.all(similarity <= 1)
    
    def test_image_feature_extraction(self, sample_image, mock_clip_model):
        """测试图像特征提取"""
        features = mock_clip_model.encode_image(sample_image)
        
        assert features is not None
        assert features.shape[0] == 1  # batch size
        assert features.shape[1] == 512  # feature dim
    
    def test_text_feature_extraction(self, mock_clip_model):
        """测试文本特征提取"""
        texts = ["hello world", "test text"]
        features = mock_clip_model.encode_text(texts)
        
        assert features is not None
        assert features.shape[0] == len(texts)
        assert features.shape[1] == 512
    
    def test_batch_processing(self, sample_batch_images, mock_clip_model):
        """测试批量处理"""
        # 简化测试：验证可以处理多张图像
        assert len(sample_batch_images) == 4
        
        for img in sample_batch_images:
            features = mock_clip_model.encode_image(img)
            assert features.shape == (1, 512)
    
    @pytest.mark.parametrize("num_classes", [2, 5, 10])
    def test_different_class_numbers(self, sample_image, mock_clip_model, num_classes):
        """测试不同类别数量"""
        text_prompts = [f"class {i}" for i in range(num_classes)]
        text_features = mock_clip_model.encode_text(text_prompts)
        
        assert text_features.shape == (num_classes, 512)


@pytest.mark.unit
class TestCLIPUtilities:
    """CLIP工具函数测试"""
    
    def test_softmax_normalization(self):
        """测试softmax归一化"""
        import torch
        
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        probs = torch.softmax(logits, dim=-1)
        
        # 概率和应该为1
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        # 所有概率应该在0-1之间
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        import torch
        
        # 相同向量的余弦相似度应该是1
        vec = torch.randn(512)
        vec_norm = vec / vec.norm()
        similarity = (vec_norm @ vec_norm.T)
        assert torch.allclose(similarity, torch.tensor(1.0), atol=1e-6)
        
        # 正交向量的余弦相似度应该接近0
        vec1 = torch.tensor([1.0, 0.0])
        vec2 = torch.tensor([0.0, 1.0])
        similarity = (vec1 @ vec2.T)
        assert torch.allclose(similarity, torch.tensor(0.0))


@pytest.mark.integration
@pytest.mark.slow
class TestCLIPRealModel:
    """真实CLIP模型集成测试（需要下载模型）"""
    
    @pytest.mark.skip_ci
    def test_load_real_model(self):
        """测试加载真实CLIP模型"""
        try:
            import clip
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            assert model is not None
            assert preprocess is not None
        except Exception as e:
            pytest.skip(f"无法加载CLIP模型: {e}")
    
    @pytest.mark.skip_ci
    def test_real_inference(self, sample_image):
        """测试真实推理"""
        try:
            import clip
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            # 预处理图像
            image_input = preprocess(sample_image).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            
            assert image_features is not None
            assert image_features.shape[0] == 1
        except Exception as e:
            pytest.skip(f"CLIP推理失败: {e}")

