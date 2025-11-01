"""
日志工具模块

提供统一的日志配置和输出功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于终端输出）"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m',       # 重置
    }
    
    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: Optional[str] = None,
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    use_color: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置并返回logger
    
    Args:
        name: logger名称，None表示root logger
        log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: 日志文件名（不包含路径）
        log_dir: 日志目录，默认为 ./logs/
        use_color: 终端输出是否使用颜色
        format_string: 自定义日志格式
    
    Returns:
        配置好的logger对象
    
    Example:
        >>> logger = setup_logger('my_app', log_level='DEBUG', log_file='app.log')
        >>> logger.info('应用启动')
    """
    # 获取或创建logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除已存在的handlers（避免重复）
    logger.handlers.clear()
    
    # 默认格式
    if format_string is None:
        format_string = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 1. 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if use_color:
        console_formatter = ColoredFormatter(format_string, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(format_string, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 2. 文件handler（如果指定）
    if log_file or log_dir:
        # 确定日志文件路径
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if log_file is None:
                # 自动生成日志文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = f"{name or 'app'}_{timestamp}.log"
            
            log_path = log_dir / log_file
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建文件handler
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # 文件输出不使用颜色
        file_formatter = logging.Formatter(format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取logger（如果不存在则创建）
    
    Args:
        name: logger名称
    
    Returns:
        logger对象
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('消息')
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    日志上下文管理器（用于临时改变日志级别）
    
    Example:
        >>> logger = get_logger('my_app')
        >>> with LoggerContext(logger, level=logging.DEBUG):
        ...     logger.debug('这条debug消息会显示')
        >>> logger.debug('这条不会显示（如果原级别是INFO）')
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        初始化
        
        Args:
            logger: logger对象
            level: 临时日志级别
        """
        self.logger = logger
        self.new_level = level
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    装饰器：记录函数调用
    
    Args:
        logger: logger对象，None则使用默认logger
    
    Example:
        >>> @log_function_call()
        ... def my_function(x, y):
        ...     return x + y
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"调用函数: {func_name}(args={args}, kwargs={kwargs})")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数 {func_name} 返回: {result}")
                return result
            except Exception as e:
                logger.error(f"函数 {func_name} 出错: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


class TqdmLoggingHandler(logging.Handler):
    """
    兼容tqdm的logging handler
    
    避免进度条和日志输出冲突
    
    Example:
        >>> from tqdm import tqdm
        >>> logger = setup_logger('app')
        >>> logger.addHandler(TqdmLoggingHandler())
        >>> for i in tqdm(range(100)):
        ...     logger.info(f'处理 {i}')
    """
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # 如果没有安装tqdm，使用标准输出
            print(self.format(record))


if __name__ == "__main__":
    # 测试代码
    
    # 1. 基础测试
    logger = setup_logger(
        'test_logger',
        log_level='DEBUG',
        log_dir='./logs',
        use_color=True
    )
    
    logger.debug('这是DEBUG消息')
    logger.info('这是INFO消息')
    logger.warning('这是WARNING消息')
    logger.error('这是ERROR消息')
    
    # 2. 测试上下文管理器
    logger2 = setup_logger('test2', log_level='INFO')
    logger2.debug('这条不会显示（INFO级别）')
    
    with LoggerContext(logger2, level=logging.DEBUG):
        logger2.debug('这条会显示（临时DEBUG级别）')
    
    logger2.debug('这条又不显示了（恢复INFO级别）')
    
    # 3. 测试装饰器
    @log_function_call(logger)
    def test_function(a, b):
        """测试函数"""
        return a + b
    
    result = test_function(1, 2)
    
    print("\n✓ 日志系统测试完成")

