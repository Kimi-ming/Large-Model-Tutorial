# 华为昇腾部署代码

本目录包含华为昇腾NPU部署的完整代码和工具。

## 📁 文件结构

```
huawei/
├── pytorch_npu_inference.py  # PyTorch-NPU推理服务
├── convert_to_om.py           # 模型转换工具（ONNX→OM）
├── benchmark.py               # 性能测试工具
├── deploy.sh                  # 自动化部署脚本
└── README.md                  # 本文件
```

## 🚀 快速开始

### 环境要求

- 昇腾AI处理器（Atlas 300/500/800等）
- CANN工具链 ≥ 5.1.RC2
- Python ≥ 3.7
- PyTorch（昇腾适配版）

### 安装依赖

```bash
# 1. 安装CANN（参考官方文档）
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/softwareinstall

# 2. 配置昇腾PyTorch源
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple

# 3. 安装PyTorch和torch-npu
pip install torch==1.11.0
pip install torch-npu==1.11.0 -i https://repo.huaweicloud.com/repository/pypi/simple

# 4. 安装其他依赖
pip install transformers pillow numpy
```

## 📖 使用指南

### 1. PyTorch-NPU推理

直接使用PyTorch模型在NPU上推理：

```bash
# 单次推理
python pytorch_npu_inference.py \
    --model openai/clip-vit-base-patch32 \
    --image test.jpg \
    --texts "a cat" "a dog" "a bird" \
    --device auto \
    --fp16

# 性能测试
python pytorch_npu_inference.py \
    --model openai/clip-vit-base-patch32 \
    --image test.jpg \
    --texts "a cat" "a dog" \
    --device npu \
    --fp16 \
    --benchmark
```

**设备选择**：
- `auto`: 自动选择（NPU > CUDA > CPU）
- `npu`: 强制使用NPU
- `cuda`: 使用CUDA（如果可用）
- `cpu`: 使用CPU

### 2. 模型转换（ONNX→OM）

将模型转换为昇腾优化的OM格式以获得更好性能：

#### 转换CLIP模型

```bash
python convert_to_om.py clip \
    --model openai/clip-vit-base-patch32 \
    --output-dir ./models/clip_om \
    --batch-size 1 \
    --soc-version Ascend910
```

#### 转换自定义ONNX模型

```bash
python convert_to_om.py onnx \
    --model model.onnx \
    --output model_om \
    --input-shape "input1:1,3,224,224;input2:1,512" \
    --soc-version Ascend910
```

**动态batch支持**：

```bash
python convert_to_om.py clip \
    --model openai/clip-vit-base-patch32 \
    --output-dir ./models/clip_om \
    --dynamic-batch \
    --soc-version Ascend910

# 将支持batch size: 1, 2, 4, 8
```

### 3. 性能测试

对比NPU、CUDA、CPU的推理性能：

```bash
python benchmark.py \
    --model openai/clip-vit-base-patch32 \
    --image test.jpg \
    --texts "a cat" "a dog" "a bird" \
    --num-runs 100 \
    --output benchmark_results.json
```

**只测试特定设备**：

```bash
python benchmark.py \
    --image test.jpg \
    --devices npu cuda \
    --num-runs 100
```

### 4. 自动化部署

使用部署脚本一键部署：

```bash
bash deploy.sh \
    --model openai/clip-vit-base-patch32 \
    --output-dir /opt/models/clip \
    --soc-version Ascend910
```

## 💡 代码示例

### Python API使用

```python
from pytorch_npu_inference import CLIPInferenceService

# 初始化服务
service = CLIPInferenceService(
    model_path="openai/clip-vit-base-patch32",
    device="auto",  # 自动选择NPU
    use_fp16=True
)

# 单张图像推理
result = service.predict(
    image="test.jpg",
    texts=["a cat", "a dog", "a bird"]
)

print(f"最佳匹配: {result['best_match']}")
print(f"置信度: {result['best_score']:.4f}")
print(f"设备: {result['device']}")
print(f"延迟: {result['inference_time_ms']:.2f}ms")

# 批量推理
results = service.batch_predict(
    images=["img1.jpg", "img2.jpg", "img3.jpg"],
    texts=["a cat", "a dog"],
    batch_size=4
)

# 性能测试
stats = service.benchmark(
    image="test.jpg",
    texts=["a cat", "a dog"],
    num_runs=100
)

print(f"平均延迟: {stats['mean_ms']:.2f}ms")
print(f"吞吐量: {stats['throughput_per_sec']:.2f} images/sec")
```

## 🔧 常见问题

### Q1: torch_npu导入失败

**错误**：`ModuleNotFoundError: No module named 'torch_npu'`

**解决**：
```bash
# 确保从昇腾源安装
pip uninstall torch-npu -y
pip install torch-npu==1.11.0 -i https://repo.huaweicloud.com/repository/pypi/simple
```

### Q2: NPU不可用

**错误**：`torch.npu.is_available()` 返回 `False`

**排查**：
```bash
# 1. 检查NPU设备
npu-smi info

# 2. 检查CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. 验证torch-npu
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### Q3: ATC转换失败

**错误**：`atc: command not found`

**解决**：
```bash
# 设置CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证ATC
atc --help
```

### Q4: 性能不如预期

**优化建议**：

1. **使用FP16**：
   ```python
   service = CLIPInferenceService(use_fp16=True)
   ```

2. **转换为OM格式**：
   ```bash
   python convert_to_om.py clip --model your_model --output-dir ./om
   ```

3. **使用批量推理**：
   ```python
   results = service.batch_predict(images, texts, batch_size=8)
   ```

4. **检查NPU利用率**：
   ```bash
   npu-smi info -l
   ```

## 📊 性能参考

测试环境：Atlas 800 (Ascend 910)，CANN 6.0.1

| 模型 | 设备 | 精度 | 延迟 | 吞吐量 |
|------|------|------|------|--------|
| CLIP ViT-B/32 | NPU | FP32 | 4.5ms | 222 img/s |
| CLIP ViT-B/32 | NPU | FP16 | 3.0ms | 333 img/s |
| CLIP ViT-B/32 | OM | FP16 | 2.5ms | 400 img/s |

> 实际性能取决于硬件配置和workload

## 🔗 相关资源

- [昇腾部署文档](../../../docs/04-多平台部署/03-华为昇腾部署.md)
- [多平台对比](../../../docs/04-多平台部署/04-多平台对比.md)
- [昇腾社区](https://www.hiascend.com/)
- [CANN文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition)

## 📝 许可

MIT License

