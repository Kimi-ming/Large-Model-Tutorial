"""
视觉大模型基准测试工具包

包含以下模块：
- speed_test: 推理速度测试
- memory_test: 显存占用测试
- accuracy_test: 准确率测试（CLIP）
- visualize_results: 结果可视化
- generate_report: 自动化报告生成
"""

__version__ = "1.0.0"
__author__ = "Large-Model-Tutorial"

from pathlib import Path

# 确保结果目录存在
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

__all__ = [
    "speed_test",
    "memory_test",
    "accuracy_test",
    "visualize_results",
    "generate_report",
]

