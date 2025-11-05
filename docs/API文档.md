# API文档

本文档提供项目中所有工具函数和命令行工具的详细API说明。

---

## 📋 目录

1. [工具函数API](#工具函数api)
2. [命令行工具](#命令行工具)
3. [配置文件](#配置文件)
4. [REST API](#rest-api)

---

## 工具函数API

### 模型加载器 (utils/model_loader.py)

#### `load_model`

加载预训练的CLIP模型。

**签名**:
```python
def load_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    **kwargs
) -> Tuple[nn.Module, Callable]
```

**参数**:
- `model_name` (str): 模型名称或路径
  - 预定义: `"openai/clip-vit-base-patch32"`, `"openai/clip-vit-large-patch14"`
  - 自定义: 本地路径
- `device` (str): 设备，`"cuda"` 或 `"cpu"`
- `**kwargs`: 传递给模型的额外参数

**返回**:
- `model` (nn.Module): 加载的模型
- `preprocess` (Callable): 预处理函数

**示例**:
```python
from utils.model_loader import load_model

# 加载默认模型
model, preprocess = load_model()

# 加载特定模型
model, preprocess = load_model(
    model_name="openai/clip-vit-large-patch14",
    device="cuda:0"
)

# 加载本地模型
model, preprocess = load_model(
    model_name="models/my_finetuned_model",
    device="cpu"
)
```

**异常**:
- `FileNotFoundError`: 模型文件不存在
- `RuntimeError`: CUDA不可用但指定了cuda设备

---

#### `save_model`

保存模型权重。

**签名**:
```python
def save_model(
    model: nn.Module,
    save_path: str,
    save_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **metadata
) -> None
```

**参数**:
- `model` (nn.Module): 要保存的模型
- `save_path` (str): 保存路径
- `save_optimizer` (bool): 是否保存优化器状态
- `optimizer` (Optional[Optimizer]): 优化器实例
- `**metadata`: 额外的元数据（如epoch, metrics等）

**示例**:
```python
from utils.model_loader import save_model

# 只保存模型
save_model(model, "checkpoints/model.pth")

# 保存模型和优化器
save_model(
    model,
    "checkpoints/checkpoint.pth",
    save_optimizer=True,
    optimizer=optimizer,
    epoch=10,
    val_acc=0.95
)
```

---

### 数据处理器 (utils/data_processor.py)

#### `create_dataloader`

创建PyTorch DataLoader。

**签名**:
```python
def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    **kwargs
) -> DataLoader
```

**参数**:
- `data_dir` (str): 数据目录路径
- `batch_size` (int): 批大小，默认32
- `shuffle` (bool): 是否打乱数据
- `num_workers` (int): 数据加载进程数
- `transform` (Optional[Callable]): 数据变换函数
- `**kwargs`: 传递给DataLoader的额外参数

**返回**:
- `DataLoader`: PyTorch数据加载器

**示例**:
```python
from utils.data_processor import create_dataloader
from torchvision import transforms

# 创建训练数据加载器
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

train_loader = create_dataloader(
    "data/train",
    batch_size=64,
    shuffle=True,
    num_workers=8,
    transform=transform
)

# 创建验证数据加载器
val_loader = create_dataloader(
    "data/val",
    batch_size=32,
    shuffle=False,
    transform=transform
)
```

---

#### `preprocess_image`

预处理单张图片。

**签名**:
```python
def preprocess_image(
    image: Union[str, Path, PIL.Image.Image, np.ndarray],
    size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor
```

**参数**:
- `image`: 输入图片
  - `str/Path`: 图片文件路径
  - `PIL.Image`: PIL图片对象
  - `np.ndarray`: NumPy数组
- `size` (Tuple[int, int]): 目标尺寸
- `normalize` (bool): 是否归一化

**返回**:
- `torch.Tensor`: 预处理后的张量 `[C, H, W]`

**示例**:
```python
from utils.data_processor import preprocess_image

# 从文件路径
tensor = preprocess_image("path/to/image.jpg")

# 从PIL Image
from PIL import Image
img = Image.open("image.jpg")
tensor = preprocess_image(img, size=(256, 256))

# 从NumPy数组
import numpy as np
img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
tensor = preprocess_image(img_array, normalize=False)
```

---

### 配置解析器 (utils/config_parser.py)

#### `load_config`

加载YAML配置文件。

**签名**:
```python
def load_config(
    config_path: str,
    override: Optional[Dict] = None
) -> Dict
```

**参数**:
- `config_path` (str): 配置文件路径
- `override` (Optional[Dict]): 覆盖配置项

**返回**:
- `Dict`: 配置字典

**示例**:
```python
from utils.config_parser import load_config

# 加载配置
config = load_config("configs/base.yaml")

# 覆盖部分配置
config = load_config(
    "configs/base.yaml",
    override={
        "training.batch_size": 64,
        "training.learning_rate": 2e-5
    }
)

# 访问配置
batch_size = config['training']['batch_size']
lr = config['training']['learning_rate']
```

---

#### `save_config`

保存配置到YAML文件。

**签名**:
```python
def save_config(
    config: Dict,
    save_path: str
) -> None
```

**参数**:
- `config` (Dict): 配置字典
- `save_path` (str): 保存路径

**示例**:
```python
from utils.config_parser import save_config

config = {
    'model': {
        'name': 'openai/clip-vit-base-patch32',
        'num_classes': 10
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-5
    }
}

save_config(config, "configs/my_experiment.yaml")
```

---

### 日志记录器 (utils/logger.py)

#### `setup_logger`

设置日志系统。

**签名**:
```python
def setup_logger(
    name: str = __name__,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger
```

**参数**:
- `name` (str): Logger名称
- `log_dir` (str): 日志目录
- `level` (int): 日志级别
- `console` (bool): 是否输出到控制台

**返回**:
- `logging.Logger`: 配置好的Logger实例

**示例**:
```python
from utils.logger import setup_logger

# 设置logger
logger = setup_logger(
    name="my_experiment",
    log_dir="logs/exp1",
    level=logging.DEBUG
)

# 使用logger
logger.info("Training started")
logger.debug("Batch size: 32")
logger.warning("Learning rate is very high")
logger.error("CUDA out of memory")
```

---

## 命令行工具

### 训练脚本

#### `train.py`

训练CLIP模型。

**用法**:
```bash
python code/02-fine-tuning/lora/train.py [OPTIONS]
```

**参数**:
```
--config PATH          配置文件路径 [默认: config.yaml]
--data-dir PATH        数据目录 [默认: data/train]
--output-dir PATH      输出目录 [默认: outputs]
--batch-size INT       批大小 [默认: 32]
--epochs INT           训练轮数 [默认: 10]
--lr FLOAT            学习率 [默认: 1e-5]
--device STR          设备 (cuda/cpu) [默认: cuda]
--resume PATH         从检查点恢复
--seed INT            随机种子 [默认: 42]
--log-every INT       日志频率 [默认: 100]
--save-every INT      保存频率 [默认: 1]
```

**示例**:
```bash
# 使用默认配置
python code/02-fine-tuning/lora/train.py

# 指定配置文件
python code/02-fine-tuning/lora/train.py --config configs/base.yaml

# 或使用自定义配置
python code/02-fine-tuning/lora/train.py --config configs/my_experiment.yaml

# 自定义参数
python code/02-fine-tuning/lora/train.py \
    --data-dir data/my_dataset \
    --batch-size 64 \
    --epochs 20 \
    --lr 2e-5

# 从检查点恢复
python code/02-fine-tuning/lora/train.py \
    --resume checkpoints/checkpoint_epoch_5.pth

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python code/02-fine-tuning/lora/train.py \
    --batch-size 128
```

---

### 评估脚本

#### `evaluate.py`

评估模型性能。

**用法**:
```bash
python code/02-fine-tuning/lora/evaluate.py [OPTIONS]
```

**参数**:
```
--model-path PATH      模型权重路径 [必需]
--data-dir PATH        测试数据目录 [必需]
--batch-size INT       批大小 [默认: 32]
--output-file PATH     结果保存路径
--metrics LIST         评估指标 [默认: accuracy,f1]
--device STR          设备 [默认: cuda]
```

**示例**:
```bash
# 基本评估
python code/02-fine-tuning/lora/evaluate.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/test

# 详细评估
python code/02-fine-tuning/lora/evaluate.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/test \
    --metrics accuracy,precision,recall,f1 \
    --output-file results/eval_results.json

# 批量评估
for model in checkpoints/*.pth; do
    python code/02-fine-tuning/lora/evaluate.py \
        --model-path $model \
        --data-dir data/test \
        --output-file results/$(basename $model .pth).json
done
```

---

### 推理脚本

#### `inference.py`

对图片进行推理。

**用法**:
```bash
python code/02-fine-tuning/lora/inference.py [OPTIONS] IMAGE_PATH
```

**参数**:
```
IMAGE_PATH            输入图片路径 [必需]
--model-path PATH     模型权重路径 [必需]
--labels LIST         类别标签
--top-k INT          返回Top-K结果 [默认: 5]
--device STR         设备 [默认: cuda]
--output-file PATH   保存结果到文件
```

**示例**:
```bash
# 单张图片推理
python code/02-fine-tuning/lora/inference.py \
    test.jpg \
    --model-path checkpoints/best_model.pth \
    --labels "dog,cat,bird,fish"

# 批量推理
for img in images/*.jpg; do
    python code/02-fine-tuning/lora/inference.py \
        $img \
        --model-path checkpoints/best_model.pth \
        --top-k 3 \
        --output-file results/$(basename $img .jpg).json
done

# 目录推理
python code/02-fine-tuning/lora/inference.py \
    images/ \
    --model-path checkpoints/best_model.pth \
    --output-file results/batch_results.json
```

---

### 模型转换脚本

#### `convert_to_onnx.py`

将PyTorch模型转换为ONNX格式。

**用法**:
```bash
python code/04-deployment/nvidia/onnx/convert_to_onnx.py [OPTIONS]
```

**参数**:
```
--model-path PATH      PyTorch模型路径 [必需]
--output-path PATH     ONNX输出路径 [必需]
--input-size LIST      输入尺寸 [默认: 1,3,224,224]
--opset-version INT    ONNX opset版本 [默认: 11]
--dynamic-axes BOOL    是否使用动态轴 [默认: True]
--simplify BOOL        是否简化模型 [默认: True]
```

**示例**:
```bash
# 基本转换
python code/04-deployment/nvidia/onnx/convert_to_onnx.py \
    --model-path checkpoints/best_model.pth \
    --output-path models/model.onnx

# 自定义输入尺寸
python code/04-deployment/nvidia/onnx/convert_to_onnx.py \
    --model-path checkpoints/best_model.pth \
    --output-path models/model_256.onnx \
    --input-size 1,3,256,256

# 固定batch size
python code/04-deployment/nvidia/onnx/convert_to_onnx.py \
    --model-path checkpoints/best_model.pth \
    --output-path models/model_batch8.onnx \
    --input-size 8,3,224,224 \
    --dynamic-axes false
```

---

### 基准测试脚本

#### `run_benchmarks.sh`

运行所有基准测试。

**用法**:
```bash
bash scripts/run_benchmarks.sh [MODEL_PATH] [DATA_DIR]
```

**参数**:
```
MODEL_PATH           模型路径 [默认: checkpoints/best_model.pth]
DATA_DIR            测试数据目录 [默认: data/test]
```

**示例**:
```bash
# 使用默认路径
bash scripts/run_benchmarks.sh

# 指定路径
bash scripts/run_benchmarks.sh \
    checkpoints/my_model.pth \
    data/my_test

# 查看结果
cat benchmark_results.txt
```

**输出**:
```
=== Benchmark Results ===
Accuracy: 92.5%
Speed: 45.2 ms/image
Memory: 1250 MB
Throughput: 22.1 images/sec
```

---

## 配置文件

### 训练配置 (configs/training/*.yaml)

**结构**:
```yaml
# 模型配置
model:
  name: "openai/clip-vit-base-patch32"
  num_classes: 10
  freeze_backbone: false

# 数据配置
data:
  train_dir: "data/train"
  val_dir: "data/val"
  batch_size: 32
  num_workers: 4
  augmentation: true

# 训练配置
training:
  epochs: 50
  learning_rate: 1e-5
  weight_decay: 0.01
  warmup_steps: 500
  gradient_clip: 1.0
  mixed_precision: true
  
# 优化器配置
optimizer:
  type: "adam"
  betas: [0.9, 0.999]
  eps: 1e-8

# 学习率调度器
scheduler:
  type: "cosine"
  T_max: 50
  eta_min: 1e-7

# 保存配置
checkpoint:
  save_dir: "checkpoints"
  save_every: 5
  keep_last_n: 3
  save_best: true

# 日志配置
logging:
  log_dir: "logs"
  log_every: 100
  use_tensorboard: true
```

**使用**:
```python
from utils.config_parser import load_config

# 使用基础配置
config = load_config("configs/base.yaml")
batch_size = config['data']['batch_size']
learning_rate = config['training']['learning_rate']

# 或使用模块特定配置
lora_config = load_config("code/02-fine-tuning/lora/config.yaml")
```

---

### 部署配置 (configs/deployment/*.yaml)

**结构**:
```yaml
# 服务配置
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 60

# 模型配置
model:
  path: "models/production/model.onnx"
  device: "cuda"
  batch_size: 32
  
# 预处理配置
preprocessing:
  image_size: [224, 224]
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# 性能配置
performance:
  use_fp16: true
  use_batch_inference: true
  max_batch_wait_time: 0.1
  
# 缓存配置
cache:
  enabled: true
  max_size: 1000
  ttl: 3600

# 限流配置
rate_limit:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000
```

---

## REST API

### API服务 (code/04-deployment/api-server/app.py)

FastAPI服务，提供模型推理接口。

#### 启动服务

```bash
# 开发模式
python code/04-deployment/api-server/app.py

# 生产模式（使用gunicorn）
gunicorn code.04-deployment.api-server.app:app \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --worker-class uvicorn.workers.UvicornWorker
```

---

#### `POST /predict`

对单张图片进行预测。

**请求**:
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
labels: ["dog", "cat", "bird"]  # 可选
top_k: 5  # 可选
```

**响应**:
```json
{
  "predictions": [
    {
      "label": "dog",
      "confidence": 0.95
    },
    {
      "label": "cat",
      "confidence": 0.03
    }
  ],
  "inference_time_ms": 45.2
}
```

**示例**:
```bash
# cURL
curl -X POST "http://localhost:8000/predict" \
    -F "file=@test.jpg" \
    -F "labels=dog,cat,bird" \
    -F "top_k=3"

# Python
import requests

files = {'file': open('test.jpg', 'rb')}
data = {'labels': 'dog,cat,bird', 'top_k': 3}
response = requests.post('http://localhost:8000/predict', files=files, data=data)
print(response.json())
```

---

#### `POST /batch_predict`

批量预测多张图片。

**请求**:
```http
POST /batch_predict
Content-Type: multipart/form-data

files: [<image_file_1>, <image_file_2>, ...]
labels: ["dog", "cat", "bird"]
```

**响应**:
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "predictions": [...]
    },
    {
      "filename": "image2.jpg",
      "predictions": [...]
    }
  ],
  "total_time_ms": 120.5
}
```

---

#### `GET /health`

健康检查接口。

**请求**:
```http
GET /health
```

**响应**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "memory_usage_mb": 1250.5
}
```

---

#### `GET /metrics`

获取服务指标。

**请求**:
```http
GET /metrics
```

**响应**:
```json
{
  "total_requests": 1523,
  "average_inference_time_ms": 45.2,
  "requests_per_second": 12.5,
  "error_rate": 0.02
}
```

---

## 📝 注意事项

### 版本兼容性

- Python: >= 3.8
- PyTorch: >= 2.0.0
- transformers: >= 4.35.0

### 常见问题

1. **模型加载失败**: 检查模型路径和CUDA是否可用
2. **OOM错误**: 减小batch_size或使用FP16
3. **API超时**: 增加timeout或优化模型推理速度

### 最佳实践

1. 使用配置文件管理参数
2. 记录详细日志便于调试
3. 定期保存检查点
4. 使用混合精度训练加速
5. 部署前进行性能测试

---

## 🔗 相关资源

- [使用说明](docs/05-使用说明/)
- [常见问题FAQ](docs/05-使用说明/03-常见问题FAQ.md)
- [最佳实践](docs/05-使用说明/04-最佳实践.md)

---

**最后更新**: 2025-11-05  
**版本**: v1.0

