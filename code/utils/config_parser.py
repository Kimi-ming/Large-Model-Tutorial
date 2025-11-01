"""
配置解析工具模块

提供统一的配置文件加载和解析功能
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import json

logger = logging.getLogger(__name__)


class ConfigParser:
    """配置解析器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置解析器
        
        Args:
            config_path: 配置文件路径，支持.yaml/.yml/.json格式
        """
        self.config = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置字典
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 根据文件扩展名选择解析器
        suffix = config_path.suffix.lower()
        
        try:
            if suffix in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            elif suffix == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(
                    f"不支持的配置文件格式: {suffix}. "
                    f"支持的格式: .yaml, .yml, .json"
                )
            
            logger.info(f"配置文件加载成功: {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {config_path}, 错误: {e}")
            raise
    
    def save(self, save_path: Union[str, Path], format: str = 'yaml'):
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
            format: 保存格式 ('yaml' or 'json')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'yaml':
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(
                        self.config,
                        f,
                        default_flow_style=False,
                        allow_unicode=True
                    )
            elif format == 'json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"配置文件保存成功: {save_path}")
            
        except Exception as e:
            logger.error(f"配置文件保存失败: {save_path}, 错误: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套键，如 'model.name'）
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
        
        Returns:
            配置值
        
        Example:
            >>> config = ConfigParser('config.yaml')
            >>> model_name = config.get('model.name', default='clip')
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"配置键不存在: {key}, 返回默认值: {default}")
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值（支持嵌套键）
        
        Args:
            key: 配置键
            value: 配置值
        
        Example:
            >>> config = ConfigParser()
            >>> config.set('model.name', 'clip')
            >>> config.set('model.device', 'cuda')
        """
        keys = key.split('.')
        current = self.config
        
        # 导航到最后一层
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # 设置值
        current[keys[-1]] = value
        logger.debug(f"配置已设置: {key} = {value}")
    
    def update(self, new_config: Dict[str, Any]):
        """
        更新配置（合并字典）
        
        Args:
            new_config: 新配置字典
        """
        self._deep_update(self.config, new_config)
        logger.info("配置已更新")
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """
        递归更新嵌套字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        返回配置字典
        
        Returns:
            完整配置字典
        """
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作"""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        return f"ConfigParser({len(self.config)} keys)"


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    快捷函数：加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    
    Example:
        >>> config = load_config('configs/training/lora.yaml')
        >>> learning_rate = config['training']['learning_rate']
    """
    parser = ConfigParser()
    return parser.load(config_path)


def save_config(config: Dict[str, Any], save_path: Union[str, Path], format: str = 'yaml'):
    """
    快捷函数：保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
        format: 格式 ('yaml' or 'json')
    
    Example:
        >>> config = {'model': {'name': 'clip', 'device': 'cuda'}}
        >>> save_config(config, 'my_config.yaml')
    """
    parser = ConfigParser()
    parser.config = config
    parser.save(save_path, format=format)


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'model': {
            'name': 'clip',
            'path': None,
            'device': 'auto',  # 'cuda', 'cpu', or 'auto'
            'load_in_8bit': False,
            'load_in_4bit': False,
        },
        'data': {
            'image_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
        },
        'training': {
            'learning_rate': 1e-4,
            'epochs': 10,
            'warmup_steps': 100,
            'gradient_accumulation_steps': 1,
            'mixed_precision': 'fp16',  # 'no', 'fp16', 'bf16'
        },
        'logging': {
            'log_level': 'INFO',
            'log_interval': 10,
            'save_interval': 100,
        },
        'paths': {
            'models_dir': './models',
            'data_dir': './data',
            'output_dir': './outputs',
            'cache_dir': './cache',
        }
    }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建默认配置
    default_config = create_default_config()
    print("默认配置:")
    print(json.dumps(default_config, indent=2, ensure_ascii=False))
    
    # 测试配置解析器
    parser = ConfigParser()
    parser.config = default_config
    
    # 测试get
    print(f"\n模型名称: {parser.get('model.name')}")
    print(f"学习率: {parser.get('training.learning_rate')}")
    print(f"不存在的键: {parser.get('not.exist', default='默认值')}")
    
    # 测试set
    parser.set('model.device', 'cuda')
    parser.set('new_key.nested', 'value')
    
    # 测试保存/加载
    test_path = Path('/tmp/test_config.yaml')
    parser.save(test_path, format='yaml')
    print(f"\n✓ 配置已保存到: {test_path}")
    
    # 加载测试
    loaded = load_config(test_path)
    print(f"✓ 配置已加载: {len(loaded)} keys")

