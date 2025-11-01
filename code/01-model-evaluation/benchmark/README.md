# 视觉大模型基准测试工具

本目录包含视觉大模型的性能基准测试工具，用于评估不同模型的推理速度、显存占用和准确率。

## 📁 文件说明

| 文件 | 功能 | 用法 |
|------|------|------|
| `speed_test.py` | 推理速度测试 | 测试不同batch size下的吞吐量和延迟 |
| `memory_test.py` | 显存占用测试 | 测量模型加载和推理的显存需求 |
| `accuracy_test.py` | 准确率测试 | 评估CLIP模型的图文检索准确率 |
| `visualize_results.py` | 结果可视化 | 生成对比图表 |
| `generate_report.py` | 报告生成 | 自动生成Markdown格式报告 |

## 🚀 快速开始

### 1. 一键运行所有测试

```bash
# 从项目根目录运行
./scripts/run_benchmarks.sh
```

### 2. 单独运行测试

**速度测试**：
```bash
python code/01-model-evaluation/benchmark/speed_test.py \
    --model openai/clip-vit-base-patch32 \
    --batch_sizes 1 2 4 \
    --image_dir data/test_dataset \
    --output results/clip_speed.json
```

**显存测试**：
```bash
python code/01-model-evaluation/benchmark/memory_test.py \
    --model openai/clip-vit-base-patch32 \
    --batch_size 1
```

**准确率测试**：
```bash
# 运行CLIP图文检索测试
python code/01-model-evaluation/benchmark/accuracy_test.py \
    --model openai/clip-vit-base-patch32

# 运行单元测试（验证Recall@K计算逻辑）
python code/01-model-evaluation/benchmark/accuracy_test.py --test
```

### 3. 生成可视化报告

```bash
# 可视化速度对比
python code/01-model-evaluation/benchmark/visualize_results.py \
    --speed_files results/clip_speed.json results/sam_speed.json \
    --output_dir results

# 生成Markdown报告
python code/01-model-evaluation/benchmark/generate_report.py \
    --results_dir results \
    --output results/benchmark_report.md
```

## 📊 输出示例

### 速度测试结果（JSON）

```json
{
  "model": "openai/clip-vit-base-patch32",
  "device": "cuda",
  "results": [
    {
      "batch_size": 1,
      "mean_time": 0.0198,
      "throughput": 50.5,
      "latency": 19.8
    }
  ]
}
```

### 显存测试结果

```
=== Memory Benchmark Results ===
model: openai/clip-vit-base-patch32
batch_size: 1
initial_memory_gb: 0.0
model_size_gb: 0.59
peak_memory_gb: 2.48
current_memory_gb: 2.45
```

### 准确率测试结果

```
=== CLIP Retrieval Accuracy ===
i2t_recall@1: 66.67%
i2t_recall@5: 100.0%
t2i_recall@1: 66.67%
t2i_recall@5: 100.0%
```

## 🔧 依赖要求

```bash
pip install torch transformers pillow numpy matplotlib seaborn pandas
```

可选依赖：
```bash
pip install GPUtil  # 用于更详细的GPU监控
```

## 📝 测试数据准备

测试需要一些图像数据，可以通过以下方式准备：

1. **手动准备**（推荐）：将测试图像（JPG格式）放入 `data/test_dataset/` 目录
   - 至少准备10张图像即可进行测试
   - 图像内容不限，不需要标注

2. **从网络下载示例图像**：
   ```bash
   mkdir -p data/test_dataset
   wget -P data/test_dataset/ https://images.unsplash.com/photo-1574158622682-e40e69881006 -O data/test_dataset/cat.jpg
   wget -P data/test_dataset/ https://images.unsplash.com/photo-1587300003388-59208cc962cb -O data/test_dataset/dog.jpg
   ```

3. **使用公开数据集**（需要额外实现脚本）：
   ```bash
   # 此脚本需要自行实现
   python scripts/prepare_test_data.py --dataset coco --num_samples 100
   ```

## 🎯 使用建议

1. **首次使用**：建议先用小batch size（1-2）测试，确保环境正常
2. **显存不足**：减小batch size或使用量化模型
3. **对比测试**：保持相同的测试条件（硬件、数据、参数）
4. **结果参考**：论文和实际结果可能有差异，关注相对性能比

## 📚 相关文档

- [基准测试实践文档](../../../docs/01-模型调研与选型/04-基准测试实践.md)
- [环境安装指南](../../../docs/05-使用说明/01-环境安装指南.md)

---

**版本**: v1.0  
**更新时间**: 2025-11-01

