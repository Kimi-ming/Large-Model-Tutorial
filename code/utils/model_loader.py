"""
模型加载工具模块

提供统一的模型加载接口，支持从本地和HuggingFace Hub加载各种视觉大模型
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器基类"""
    
    SUPPORTED_MODELS = {
        'clip': {
            'default_repo': 'openai/clip-vit-base-patch32',
            'model_class': CLIPModel,
            'processor_class': CLIPProcessor,
            'dependencies': ['transformers'],
        },
        'sam': {
            'default_repo': 'facebook/sam-vit-base',
            'model_class': AutoModel,
            'processor_class': AutoProcessor,
            'dependencies': ['transformers', 'segment-anything'],
            'notes': '需要安装: pip install git+https://github.com/facebookresearch/segment-anything.git',
        },
        'blip2': {
            'default_repo': 'Salesforce/blip2-opt-2.7b',
            'model_class': AutoModel,
            'processor_class': AutoProcessor,
            'dependencies': ['transformers'],
        },
        'llava': {
            'default_repo': 'liuhaotian/llava-v1.5-7b',
            'model_class': AutoModel,
            'processor_class': AutoProcessor,
            'dependencies': ['transformers'],
        },
        'qwen-vl': {
            'default_repo': 'Qwen/Qwen-VL-Chat',
            'model_class': AutoModel,
            'processor_class': AutoProcessor,
            'dependencies': ['transformers>=4.32.0', 'transformers_stream_generator'],
            'notes': '需要安装: pip install transformers_stream_generator',
        },
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            cache_dir: 模型缓存目录，默认为 ./models/
        """
        self.cache_dir = cache_dir or "./models"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelLoader initialized with cache_dir: {self.cache_dir}")
    
    def _check_dependencies(self, model_name: str):
        """
        检查模型依赖
        
        Args:
            model_name: 模型名称
        """
        model_info = self.SUPPORTED_MODELS[model_name]
        dependencies = model_info.get('dependencies', [])
        
        missing_deps = []
        for dep in dependencies:
            # 解析依赖名（去除版本号）
            dep_name = dep.split('>=')[0].split('==')[0].replace('-', '_')
            try:
                __import__(dep_name)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            notes = model_info.get('notes', '')
            logger.warning(
                f"模型 {model_name} 缺少依赖: {', '.join(missing_deps)}\n"
                f"建议安装: pip install {' '.join(missing_deps)}\n"
                f"{notes}"
            )
    
    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        加载模型和处理器
        
        Args:
            model_name: 模型名称 (clip, sam, blip2, llava, qwen-vl 等)
            model_path: 模型路径，可以是:
                - 本地路径 (如: ./models/clip)
                - HuggingFace repo ID (如: openai/clip-vit-base-patch32)
                - None (使用默认路径)
            device: 设备 ('cuda', 'cpu', None=自动检测)
            load_in_8bit: 是否以8bit精度加载（节省显存）
            load_in_4bit: 是否以4bit精度加载（最大节省显存）
            **kwargs: 传递给模型加载的额外参数
        
        Returns:
            (model, processor): 模型和处理器对象
        
        Raises:
            ValueError: 如果模型名称不支持
            FileNotFoundError: 如果本地模型路径不存在
        """
        # 检查模型是否支持
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}. "
                f"支持的模型: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        model_info = self.SUPPORTED_MODELS[model_name]
        
        # 检查依赖
        self._check_dependencies(model_name)
        
        # 确定模型路径
        if model_path is None:
            # 优先使用本地缓存
            local_path = Path(self.cache_dir) / model_name
            if local_path.exists():
                model_path = str(local_path)
                logger.info(f"使用本地模型: {model_path}")
            else:
                # 使用默认HuggingFace repo
                model_path = model_info['default_repo']
                logger.info(f"从HuggingFace加载: {model_path}")
        
        # 确定设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        # 准备加载参数
        load_kwargs = {
            'cache_dir': self.cache_dir,
            'trust_remote_code': True,
            **kwargs
        }
        
        # 量化配置
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                )
                logger.info(f"启用量化: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
            except ImportError:
                logger.warning(
                    "未安装bitsandbytes库，忽略量化设置。"
                    "安装方法: pip install bitsandbytes"
                )
        
        try:
            # 加载模型
            model_class = model_info['model_class']
            logger.info(f"正在加载模型...")
            model = model_class.from_pretrained(model_path, **load_kwargs)
            
            # 移动到指定设备（如果没有使用量化）
            if not (load_in_8bit or load_in_4bit):
                model = model.to(device)
            
            model.eval()  # 设置为评估模式
            logger.info(f"模型加载成功")
            
            # 加载处理器
            processor_class = model_info['processor_class']
            logger.info(f"正在加载处理器...")
            processor = processor_class.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            logger.info(f"处理器加载成功")
            
            return model, processor
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def load_tokenizer(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        单独加载分词器（用于某些只需要文本处理的场景）
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            **kwargs: 额外参数
        
        Returns:
            tokenizer: 分词器对象
        """
        if model_path is None:
            if model_name in self.SUPPORTED_MODELS:
                model_path = self.SUPPORTED_MODELS[model_name]['default_repo']
            else:
                raise ValueError(f"未知模型: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                **kwargs
            )
            logger.info(f"分词器加载成功: {model_path}")
            return tokenizer
        except Exception as e:
            logger.error(f"分词器加载失败: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型信息字典
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}")
        
        return self.SUPPORTED_MODELS[model_name].copy()
    
    @classmethod
    def list_supported_models(cls) -> list:
        """
        列出所有支持的模型
        
        Returns:
            模型名称列表
        """
        return list(cls.SUPPORTED_MODELS.keys())


def load_model_simple(
    model_name: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    简化的模型加载函数（快捷方式）
    
    Args:
        model_name: 模型名称 (clip, blip2, llava等)
        device: 设备 ('cuda', 'cpu', None=自动)
        cache_dir: 缓存目录
    
    Returns:
        (model, processor): 模型和处理器
    
    Example:
        >>> model, processor = load_model_simple('clip')
        >>> # 使用模型进行推理...
    """
    loader = ModelLoader(cache_dir=cache_dir)
    return loader.load_model(model_name, device=device)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("支持的模型:")
    for model in ModelLoader.list_supported_models():
        print(f"  - {model}")
    
    # 示例：加载CLIP模型
    # model, processor = load_model_simple('clip', device='cpu')
    # print(f"✓ CLIP模型加载成功")

