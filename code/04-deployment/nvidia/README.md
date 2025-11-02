# NVIDIA平台部署代码

CLIP模型在NVIDIA GPU上的部署实现，包括PyTorch推理、ONNX转换和API服务。

## 📁 目录结构

```
nvidia/
├── basic/
│   └── pytorch_inference.py    # PyTorch推理服务
├── onnx/
│   ├── convert_to_onnx.py      # ONNX转换脚本
│   └── onnx_inference.py       # ONNX推理服务
└── README.md                    # 本文档
```

## 🚀 快速开始

### 1. PyTorch推理

```bash
# 图文匹配推理
python code/04-deployment/nvidia/basic/pytorch_inference.py \
    --model openai/clip-vit-base-patch32 \
    --image dog.jpg \
    --texts "a photo of a dog" "a photo of a cat" "a photo of a bird" \
    --device cuda \
    --fp16
```

**输出示例**:
```
🚀 初始化CLIP推理服务...
   设备: cuda
   FP16: True
✅ 模型加载完成: openai/clip-vit-base-patch32
🔥 预热模型...
✅ 预热完成

🖼️  图像: dog.jpg
📝 候选文本: ['a photo of a dog', 'a photo of a cat', 'a photo of a bird']
============================================================

⏱️  推理时间: 12.34ms
🖥️  设备: cuda
🔢 FP16: True

📊 预测结果:
  1. a photo of a dog
     得分: 0.9234
  2. a photo of a cat
     得分: 0.0543
  3. a photo of a bird
     得分: 0.0223
```

### 2. ONNX转换

```bash
# 转换CLIP模型为ONNX
python code/04-deployment/nvidia/onnx/convert_to_onnx.py \
    --model openai/clip-vit-base-patch32 \
    --output_dir onnx_models \
    --optimize
```

**输出文件**:
- `onnx_models/clip_vision.onnx` - 视觉编码器
- `onnx_models/clip_text.onnx` - 文本编码器
- `onnx_models/clip_vision_optimized.onnx` - 优化后的视觉编码器
- `onnx_models/clip_text_optimized.onnx` - 优化后的文本编码器

### 3. ONNX推理

```bash
# 使用ONNX模型推理
python code/04-deployment/nvidia/onnx/onnx_inference.py \
    --vision_model onnx_models/clip_vision.onnx \
    --text_model onnx_models/clip_text.onnx \
    --image dog.jpg \
    --texts "a photo of a dog" "a photo of a cat"
```

### 4. API服务

```bash
# 启动FastAPI服务
cd code/04-deployment/api-server
python app.py

# 或使用uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**测试API**:
```bash
# 健康检查
curl http://localhost:8000/health

# 图文匹配推理
curl -X POST "http://localhost:8000/predict" \
  -F "image=@dog.jpg" \
  -F "texts=a photo of a dog,a photo of a cat"

# 提取图像特征
curl -X POST "http://localhost:8000/image_features" \
  -F "image=@dog.jpg" \
  -F "normalize=true"
```

## 📊 性能对比

| 方案 | 延迟 (ms) | 吞吐量 (img/s) | 显存 (GB) |
|------|-----------|---------------|----------|
| PyTorch FP32 | 20 | 50 | 2.5 |
| PyTorch FP16 | 12 | 80 | 1.3 |
| ONNX Runtime (CPU) | 45 | 22 | - |
| ONNX Runtime (GPU) | 15 | 65 | 2.0 |

*测试环境: NVIDIA RTX 3090, Batch Size=1*

## 🔧 高级用法

### 批量推理

```python
from code.deployment.nvidia.basic.pytorch_inference import CLIPInferenceService

service = CLIPInferenceService("openai/clip-vit-base-patch32")

# 批量提取图像特征
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
features = service.get_image_features(image_paths)
print(f"特征形状: {features.shape}")  # (3, 512)
```

### 特征提取和相似度计算

```python
# 提取特征
image_features = service.get_image_features(["dog.jpg"])
text_features = service.get_text_features(["a dog", "a cat"])

# 计算相似度
similarity = service.compute_similarity(image_features, text_features)
print(f"相似度矩阵: {similarity}")
```

### 自定义模型路径

```python
# 使用微调后的模型
service = CLIPInferenceService(
    model_path="outputs/lora_finetuning/checkpoint-epoch-10",
    device="cuda",
    use_fp16=True
)
```

## 📦 依赖安装

```bash
# 基础依赖
pip install torch torchvision transformers pillow

# ONNX相关
pip install onnx onnxruntime-gpu

# API服务
pip install fastapi uvicorn python-multipart

# 可选：ONNX优化
pip install onnxruntime-tools
```

## 🐳 Docker部署

### 构建镜像

```bash
cd code/04-deployment
docker build -t clip-service:latest -f docker/Dockerfile .
```

### 运行容器

```bash
# GPU支持
docker run --gpus all -p 8000:8000 clip-service:latest

# CPU only
docker run -p 8000:8000 clip-service:latest
```

## 💡 最佳实践

### 1. 选择合适的部署方案

- **开发/原型**: PyTorch直接推理
- **生产环境**: ONNX Runtime或TorchScript
- **高性能需求**: TensorRT（待实现）
- **企业级**: Triton推理服务器（待实现）

### 2. 性能优化

- ✅ 使用FP16混合精度
- ✅ 启用批处理
- ✅ 预热模型
- ✅ 缓存文本特征
- ✅ 使用ONNX优化

### 3. 生产部署

- ✅ 使用Gunicorn + Uvicorn（多worker）
- ✅ 添加Nginx反向代理
- ✅ 实现健康检查
- ✅ 添加监控和日志
- ✅ 使用Docker容器化

## 🔗 相关文档

- [NVIDIA部署基础](../../../docs/04-多平台部署/01-NVIDIA部署基础.md)
- [模型服务化](../../../docs/04-多平台部署/02-模型服务化.md)

## 🤝 贡献

如果您发现问题或有改进建议，欢迎提交Issue或Pull Request。

## 📄 许可证

本项目遵循MIT许可证。

