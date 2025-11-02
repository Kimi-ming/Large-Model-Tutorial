"""
LoRA微调模型评估脚本

评估微调后的CLIP模型在测试集上的性能
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
import json

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录和当前目录到路径
project_root = Path(__file__).parent.parent.parent.parent
current_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 导入当前目录的模块
from train import CLIPClassifier, load_config
from dataset import DogBreedDataset


def load_model(checkpoint_dir: str, num_classes: int, device: torch.device):
    """
    加载微调后的模型
    
    Args:
        checkpoint_dir: 检查点目录
        num_classes: 类别数
        device: 设备
        
    Returns:
        加载的模型
    """
    print(f"📦 加载模型: {checkpoint_dir}")
    
    # 加载CLIP模型（带LoRA）
    clip_model = CLIPModel.from_pretrained(checkpoint_dir)
    
    # 创建分类器
    model = CLIPClassifier(clip_model, num_classes)
    
    # 加载分类头权重
    classifier_path = os.path.join(checkpoint_dir, 'classifier.pt')
    if os.path.exists(classifier_path):
        model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print("✅ 分类头权重加载成功")
    else:
        print("⚠️  未找到分类头权重文件")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        class_names: 类别名称列表
        
    Returns:
        评估结果字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_correct = 0
    total_samples = 0
    
    print("\n🔍 评估中...")
    for pixel_values, labels in tqdm(dataloader):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        # 前向传播
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1)
        
        # 预测
        _, predicted = torch.max(logits, 1)
        
        # 统计
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # 收集结果
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    accuracy = 100.0 * total_correct / total_samples
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 生成分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算Top-5准确率（如果类别数>=5）
    top5_acc = None
    if len(class_names) >= 5:
        top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
        top5_correct = sum([label in top5_preds[i] for i, label in enumerate(all_labels)])
        top5_acc = 100.0 * top5_correct / total_samples
    
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 10))
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_class_performance(
    report: Dict[str, Any],
    class_names: List[str],
    save_path: str
):
    """
    绘制各类别性能
    
    Args:
        report: 分类报告
        class_names: 类别名称
        save_path: 保存路径
    """
    # 提取各类别的指标
    precisions = [report[name]['precision'] for name in class_names]
    recalls = [report[name]['recall'] for name in class_names]
    f1_scores = [report[name]['f1-score'] for name in class_names]
    
    # 绘制条形图
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 类别性能图已保存: {save_path}")
    plt.close()


def save_results(results: Dict[str, Any], output_dir: str, class_names: List[str]):
    """
    保存评估结果
    
    Args:
        results: 评估结果
        output_dir: 输出目录
        class_names: 类别名称
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文本报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总体准确率: {results['accuracy']:.2f}%\n")
        if results['top5_accuracy']:
            f.write(f"Top-5准确率: {results['top5_accuracy']:.2f}%\n")
        f.write("\n")
        
        f.write("各类别详细指标:\n")
        f.write("-" * 60 + "\n")
        report = results['classification_report']
        for class_name in class_names:
            metrics = report[class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
            f.write(f"  Support:   {metrics['support']}\n")
        
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"\n加权平均:\n")
        f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"✅ 评估报告已保存: {report_path}")
    
    # 保存JSON格式
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    json_results = {
        'accuracy': float(results['accuracy']),
        'top5_accuracy': float(results['top5_accuracy']) if results['top5_accuracy'] else None,
        'classification_report': results['classification_report']
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON结果已保存: {json_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    # 绘制类别性能
    perf_path = os.path.join(output_dir, 'class_performance.png')
    plot_class_performance(results['classification_report'], class_names, perf_path)


def main():
    parser = argparse.ArgumentParser(description="评估LoRA微调模型")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点目录'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/dogs',
        help='数据集目录'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='评估的数据集分割'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='评估结果输出目录'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LoRA微调模型评估")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载处理器
    print("\n📦 加载CLIP处理器...")
    processor = CLIPProcessor.from_pretrained(args.checkpoint)
    
    # 加载数据集
    print(f"\n📊 加载{args.split}数据集...")
    dataset = DogBreedDataset(
        data_dir=args.data_dir,
        split=args.split,
        processor=processor
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    class_names = dataset.classes
    num_classes = len(class_names)
    
    # 加载模型
    model = load_model(args.checkpoint, num_classes, device)
    
    # 评估模型
    results = evaluate_model(model, dataloader, device, class_names)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"准确率: {results['accuracy']:.2f}%")
    if results['top5_accuracy']:
        print(f"Top-5准确率: {results['top5_accuracy']:.2f}%")
    
    # 保存结果
    save_results(results, args.output_dir, class_names)
    
    print("\n✅ 评估完成！")


if __name__ == '__main__':
    main()

