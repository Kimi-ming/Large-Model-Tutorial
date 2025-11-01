#!/usr/bin/env python3
"""
基准测试结果可视化
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_speed_comparison(result_files: list, output_path: str):
    """对比多个模型的推理速度"""
    
    data = []
    for file in result_files:
        if not Path(file).exists():
            print(f"Warning: {file} not found, skipping...")
            continue
            
        with open(file) as f:
            result = json.load(f)
            model_name = Path(file).stem.replace("_speed", "")
            
            for r in result["results"]:
                data.append({
                    "Model": model_name,
                    "Batch Size": r["batch_size"],
                    "Throughput (images/sec)": r["throughput"],
                    "Latency (ms/image)": r["latency"]
                })
    
    if not data:
        print("No data to plot!")
        return
    
    df = pd.DataFrame(data)
    
    # 绘制吞吐量对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df, x="Model", y="Throughput (images/sec)", 
                hue="Batch Size", ax=ax1)
    ax1.set_title("Throughput Comparison")
    ax1.set_ylabel("Images/sec (higher is better)")
    ax1.tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x="Model", y="Latency (ms/image)", 
                hue="Batch Size", ax=ax2)
    ax2.set_title("Latency Comparison")
    ax2.set_ylabel("ms/image (lower is better)")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Speed comparison saved to {output_path}")


def plot_memory_comparison(models_memory: dict, output_path: str):
    """对比多个模型的显存占用"""
    
    df = pd.DataFrame(models_memory).T
    df = df.reset_index().rename(columns={"index": "Model"})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df["model_size_gb"], 
           width, label="Model Size", alpha=0.8)
    ax.bar([i + width/2 for i in x], df["peak_memory_gb"], 
           width, label="Peak Memory", alpha=0.8)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Memory Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Memory comparison saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化基准测试结果")
    parser.add_argument("--speed_files", type=str, nargs="+",
                       help="速度测试结果JSON文件列表")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 示例：可视化速度对比
    if args.speed_files:
        plot_speed_comparison(
            args.speed_files, 
            f"{args.output_dir}/speed_comparison.png"
        )
    else:
        # 默认示例
        result_files = [
            "results/clip_speed.json",
            "results/sam_speed.json",
            "results/blip2_speed.json"
        ]
        plot_speed_comparison(result_files, "results/speed_comparison.png")
    
    # 示例：可视化显存对比
    memory_data = {
        "CLIP-B/32": {"model_size_gb": 0.59, "peak_memory_gb": 2.48},
        "SAM-B": {"model_size_gb": 0.35, "peak_memory_gb": 4.12},
        "BLIP-2": {"model_size_gb": 2.61, "peak_memory_gb": 6.85}
    }
    
    plot_memory_comparison(memory_data, f"{args.output_dir}/memory_comparison.png")

