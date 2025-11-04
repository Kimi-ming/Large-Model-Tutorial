"""
pytest配置和共享fixtures

提供测试所需的：
- 测试数据生成
- Mock模型
- 临时目录
- 配置fixture
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """测试数据目录"""
    data_dir = project_root / "tests" / "fixtures"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_dir():
    """临时目录（每个测试后清理）"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """生成测试图像（RGB）"""
    # 创建简单的测试图像
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image)


@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """保存测试图像并返回路径"""
    image_path = temp_dir / "test_image.jpg"
    sample_image.save(image_path)
    return str(image_path)


@pytest.fixture
def sample_batch_images():
    """生成批量测试图像"""
    images = []
    for i in range(4):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(img))
    return images


@pytest.fixture
def sample_mask():
    """生成测试掩码"""
    mask = np.zeros((224, 224), dtype=bool)
    # 添加一个圆形区域
    center = (112, 112)
    radius = 50
    y, x = np.ogrid[:224, :224]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask[mask_circle] = True
    return mask


@pytest.fixture
def mock_clip_model():
    """Mock CLIP模型（用于不需要真实模型的测试）"""
    class MockCLIPModel:
        def __init__(self):
            self.device = 'cpu'
        
        def encode_image(self, image):
            # 返回模拟的图像特征
            import torch
            return torch.randn(1, 512)
        
        def encode_text(self, text):
            # 返回模拟的文本特征
            import torch
            return torch.randn(len(text), 512)
    
    return MockCLIPModel()


@pytest.fixture
def mock_sam_model():
    """Mock SAM模型"""
    class MockSAMModel:
        def __init__(self):
            self.device = 'cpu'
        
        def predict(self, *args, **kwargs):
            # 返回模拟的分割结果
            mask = np.zeros((224, 224), dtype=bool)
            mask[50:150, 50:150] = True
            masks = np.array([mask])
            scores = np.array([0.95])
            logits = np.random.randn(1, 256, 256)
            return masks, scores, logits
    
    return MockSAMModel()


@pytest.fixture
def mock_blip2_model():
    """Mock BLIP-2模型"""
    class MockBLIP2Model:
        def __init__(self):
            self.device = 'cpu'
        
        def generate(self, *args, **kwargs):
            # 返回模拟的生成结果
            import torch
            return torch.tensor([[1, 2, 3, 4, 5]])
    
    return MockBLIP2Model()


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        'model': {
            'type': 'vit_b',
            'checkpoint': 'checkpoints/sam_vit_b_01ec64.pth'
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'epochs': 10
        },
        'data': {
            'image_size': 1024,
            'prompt_mode': ['point', 'box']
        }
    }


@pytest.fixture(scope="session")
def skip_if_no_gpu():
    """如果没有GPU则跳过测试"""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("需要GPU才能运行此测试")


@pytest.fixture(scope="session")
def skip_if_no_model(model_name):
    """如果模型文件不存在则跳过测试"""
    def _skip(model_path):
        if not Path(model_path).exists():
            pytest.skip(f"模型文件不存在: {model_path}")
    return _skip


# 自动使用的fixture
@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个测试前重置随机种子（确保可重现性）"""
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """将工作目录更改为项目根目录"""
    monkeypatch.chdir(PROJECT_ROOT)

