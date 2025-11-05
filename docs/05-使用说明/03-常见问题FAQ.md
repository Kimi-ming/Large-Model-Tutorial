# 常见问题FAQ

本文档汇总项目使用过程中的常见问题和解决方案，帮助你快速解决遇到的问题。

---

## 📋 目录

1. [环境安装问题](#环境安装问题)
2. [模型下载问题](#模型下载问题)
3. [训练相关问题](#训练相关问题)
4. [推理部署问题](#推理部署问题)
5. [硬件相关问题](#硬件相关问题)
6. [数据处理问题](#数据处理问题)
7. [性能优化问题](#性能优化问题)

---

## 环境安装问题

### Q1: pip install 时报错 "No module named 'torch'"

**问题描述**: 安装其他依赖时提示找不到PyTorch

**解决方案**:
```bash
# 先安装PyTorch（根据CUDA版本）
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# 然后安装其他依赖
pip install -r requirements.txt
```

### Q2: CUDA版本不匹配

**问题描述**: 
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**:
```bash
# 1. 检查CUDA版本
nvcc --version

# 2. 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"

# 3. 重新安装匹配的PyTorch版本
# 如果系统CUDA是11.8
pip uninstall torch torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Q3: ImportError: libcudnn.so.8: cannot open shared object file

**问题描述**: 找不到cuDNN库

**解决方案**:
```bash
# 方法1: 安装cuDNN
# Ubuntu/Debian
sudo apt-get install libcudnn8 libcudnn8-dev

# 方法2: 添加库路径
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 方法3: 使用conda安装（推荐）
conda install cudnn
```

### Q4: 虚拟环境中pip install很慢

**问题描述**: 使用pip安装依赖速度非常慢

**解决方案**:
```bash
# 使用国内镜像源
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或编辑 ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

### Q5: AttributeError: module 'clip' has no attribute 'load'

**问题描述**: CLIP库安装不正确

**解决方案**:
```bash
# 可能是安装了错误的clip包
pip uninstall clip

# 重新安装正确的CLIP
pip install git+https://github.com/openai/CLIP.git

# 或者
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

---

## 模型下载问题

### Q6: HuggingFace模型下载失败

**问题描述**: 
```
OSError: Can't load weights for 'openai/clip-vit-base-patch32'
```

**解决方案**:
```bash
# 方法1: 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法2: 手动下载并指定路径
# 下载模型到本地
git clone https://hf-mirror.com/openai/clip-vit-base-patch32 models/clip-vit-base-patch32

# 使用本地路径
python code/xxx.py --model_path models/clip-vit-base-patch32

# 方法3: 使用离线模式
export TRANSFORMERS_OFFLINE=1
# 确保模型已缓存在 ~/.cache/huggingface/
```

### Q7: 模型下载中断，如何继续下载？

**问题描述**: 下载大模型时网络中断

**解决方案**:
```python
# HuggingFace会自动断点续传
from transformers import CLIPModel

# 会自动从缓存继续下载
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    resume_download=True  # 支持断点续传
)
```

### Q8: 磁盘空间不足

**问题描述**: `/root/.cache/huggingface/` 占用过多空间

**解决方案**:
```bash
# 查看缓存大小
du -sh ~/.cache/huggingface/

# 清理旧模型
huggingface-cli delete-cache

# 或手动删除不需要的模型
rm -rf ~/.cache/huggingface/hub/models--xxx

# 修改缓存目录
export HF_HOME=/path/to/large/disk/huggingface
export TRANSFORMERS_CACHE=/path/to/large/disk/huggingface

# 永久设置（添加到~/.bashrc）
echo 'export HF_HOME=/path/to/large/disk/huggingface' >> ~/.bashrc
```

---

## 训练相关问题

### Q9: CUDA Out of Memory (OOM)

**问题描述**: 
```
RuntimeError: CUDA out of memory. Tried to allocate XX.XX MiB
```

**解决方案**:
```python
# 1. 减小batch size
batch_size = 8  # 从32减到8

# 2. 使用梯度累积
accumulation_steps = 4
for i, (images, labels) in enumerate(dataloader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. 梯度检查点（牺牲速度换内存）
from torch.utils.checkpoint import checkpoint

output = checkpoint(model.layer, input)

# 5. 使用LoRA而不是全参数微调
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, lora_config)
```

### Q10: 训练loss不下降

**问题描述**: 训练几个epoch后loss没有明显下降

**解决方案**:
```python
# 1. 检查学习率
# 视觉大模型微调建议使用小学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 而不是1e-3

# 2. 使用学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 3. 检查数据标签是否正确
for images, labels in dataloader:
    print(f"Labels: {labels}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    break

# 4. 使用更小的学习率warm-up
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

# 5. 检查是否需要解冻更多层
# 只微调最后几层可能不够
for name, param in model.named_parameters():
    if 'layer.10' in name or 'layer.11' in name or 'classifier' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

### Q11: loss变成NaN

**问题描述**: 训练过程中loss突然变成NaN

**解决方案**:
```python
# 1. 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # 更小的学习率

# 2. 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 检查数据中是否有异常值
# 添加数据验证
def validate_batch(images, labels):
    assert not torch.isnan(images).any(), "Images contain NaN"
    assert not torch.isinf(images).any(), "Images contain Inf"
    assert (labels >= 0).all() and (labels < num_classes).all(), "Invalid labels"

# 4. 使用混合精度时添加梯度缩放
from torch.cuda.amp import GradScaler

scaler = GradScaler()
# ... (如Q9所示)

# 5. 检查损失函数
# 确保使用合适的损失函数
criterion = nn.CrossEntropyLoss(reduction='mean')  # 使用mean而不是sum
```

### Q12: 训练速度很慢

**问题描述**: 训练一个epoch要很长时间

**解决方案**:
```python
# 1. 使用多GPU训练
import torch.nn as nn
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# 2. 增加num_workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # 增加数据加载线程
    pin_memory=True  # 加速数据传输到GPU
)

# 3. 使用混合精度训练（加速2x）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

# 4. 优化数据预处理
# 预先处理数据并缓存
# 或使用更快的图像解码库（如turbojpeg）

# 5. 使用编译优化（PyTorch 2.0+）
model = torch.compile(model)

# 6. 减少日志输出频率
if step % 100 == 0:  # 每100步打印一次，而不是每步
    print(f"Step {step}, Loss: {loss.item()}")
```

---

## 推理部署问题

### Q13: 推理速度慢

**问题描述**: 单张图片推理需要很长时间

**解决方案**:
```python
# 1. 使用eval模式
model.eval()

# 2. 禁用梯度计算
with torch.no_grad():
    outputs = model(images)

# 3. 使用FP16推理
model = model.half()
images = images.half()

# 4. 使用批处理
# 将多个请求合并成batch处理
batch_images = torch.stack([img1, img2, img3, ...])
with torch.no_grad():
    batch_outputs = model(batch_images)

# 5. 转换为ONNX
import torch.onnx
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11
)

# 使用ONNX Runtime推理
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data})

# 6. 使用TensorRT（NVIDIA GPU）
import torch_tensorrt
trt_model = torch_tensorrt.compile(model, ...)
```

### Q14: API服务内存泄漏

**问题描述**: FastAPI服务运行一段时间后内存占用越来越高

**解决方案**:
```python
# 1. 确保使用torch.no_grad()
@app.post("/predict")
async def predict(file: UploadFile):
    image = load_image(file)
    
    with torch.no_grad():  # 重要！防止内存累积
        output = model(image)
    
    return {"result": output.tolist()}

# 2. 及时释放大对象
@app.post("/predict")
async def predict(file: UploadFile):
    image = load_image(file)
    
    with torch.no_grad():
        output = model(image)
    
    result = output.cpu().tolist()
    
    # 显式删除
    del image, output
    torch.cuda.empty_cache()
    
    return {"result": result}

# 3. 使用进程池而不是线程池
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

# 4. 定期重启worker
# 使用gunicorn的--max-requests参数
# gunicorn app:app --max-requests 1000 --max-requests-jitter 100

# 5. 监控内存使用
import psutil
import gc

@app.get("/health")
def health():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 8000:  # 超过8GB
        gc.collect()
        torch.cuda.empty_cache()
    
    return {"memory_mb": memory_mb}
```

### Q15: Docker容器中GPU不可用

**问题描述**: 
```
RuntimeError: Found no NVIDIA driver on your system
```

**解决方案**:
```bash
# 1. 安装nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. 使用--gpus参数运行容器
docker run --gpus all -it your-image

# 3. 使用docker-compose时指定runtime
docker-compose.yml:
services:
  app:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all

# 4. 测试GPU是否可用
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 硬件相关问题

### Q16: 多GPU训练时显存不均衡

**问题描述**: GPU 0显存占用很高，其他GPU很低

**解决方案**:
```python
# 1. 使用DistributedDataParallel而不是DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式训练
dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])

# 2. 使用balanced batch sampler
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# 3. 启动脚本
# torchrun --nproc_per_node=4 train.py

# 4. 或使用torch.multiprocessing
import torch.multiprocessing as mp

def train_worker(rank, world_size):
    setup(rank, world_size)
    # ... training code

mp.spawn(train_worker, args=(world_size,), nprocs=world_size)
```

### Q17: 华为昇腾NPU无法识别

**问题描述**: `npu-smi info` 显示no device

**解决方案**:
```bash
# 1. 检查驱动安装
ls /usr/local/Ascend/driver/

# 2. 检查设备
ls -l /dev/davinci*

# 3. 设置环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export ASCEND_HOME=/usr/local/Ascend/latest
export PATH=$ASCEND_HOME/bin:$PATH

# 4. 重启驱动服务
sudo systemctl restart ascend-device-driver

# 5. 检查用户权限
sudo usermod -a -G HwHiAiUser $(whoami)
# 重新登录使权限生效

# 6. 检查CANN版本
cat /usr/local/Ascend/latest/version.cfg
```

### Q18: CPU推理太慢

**问题描述**: 在没有GPU的环境下推理速度不可接受

**解决方案**:
```python
# 1. 使用INT8量化
import torch
from torch.quantization import quantize_dynamic

model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 2. 使用ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=['CPUExecutionProvider']
)

# 3. 使用OpenVINO（Intel CPU优化）
# 需要安装openvino
from openvino.runtime import Core

ie = Core()
model = ie.read_model("model.xml")
compiled_model = ie.compile_model(model, "CPU")

# 4. 使用更小的模型
# ViT-B/32 -> ViT-B/16 -> 蒸馏小模型

# 5. 批处理
# 累积多个请求一起处理
```

---

## 数据处理问题

### Q19: 自定义数据集加载失败

**问题描述**: 使用自己的数据集时报错

**解决方案**:
```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 加载所有图片
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, class_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, class_name
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个占位图片
            return torch.zeros(3, 224, 224), class_name

# 使用
dataset = CustomDataset("data/images", transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Q20: 图片尺寸不一致导致batch错误

**问题描述**: 
```
RuntimeError: stack expects each tensor to be equal size
```

**解决方案**:
```python
# 1. 在transform中统一尺寸
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),  # 先缩放
    transforms.CenterCrop(224),  # 再裁剪到固定大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 2. 或使用自定义collate_fn
def custom_collate(batch):
    images, labels = zip(*batch)
    
    # 统一尺寸
    images = [transforms.Resize((224, 224))(img) for img in images]
    images = torch.stack(images)
    
    return images, labels

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate
)
```

---

## 性能优化问题

### Q21: 如何提高训练速度？

**综合优化方案**:
```python
# 1. 数据加载优化
dataloader = DataLoader(
    dataset,
    batch_size=64,  # 尽可能大（不OOM的情况下）
    num_workers=8,  # CPU核心数
    pin_memory=True,  # 加速数据传输到GPU
    persistent_workers=True,  # 保持worker进程存活
    prefetch_factor=2  # 预取数据
)

# 2. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 3. 模型编译（PyTorch 2.0+）
model = torch.compile(model, mode="reduce-overhead")

# 4. 梯度累积（模拟大batch size）
accumulation_steps = 4

for i, (images, labels) in enumerate(dataloader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 5. 使用LoRA而不是全参数微调
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)

# 6. 冻结backbone，只微调head
for param in model.visual.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
```

### Q22: 如何减小模型体积？

**模型压缩方案**:
```python
# 1. 模型量化
import torch.quantization as quantization

# 动态量化（最简单）
model_int8 = quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 保存
torch.save(model_int8.state_dict(), "model_int8.pth")

# 2. 模型剪枝
import torch.nn.utils.prune as prune

# 剪枝30%的权重
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')

# 3. 知识蒸馏
# 使用大模型训练小模型
teacher_model = load_large_model()
student_model = load_small_model()

# ... 蒸馏训练代码（参考高级主题文档）

# 4. 只保存必要的权重
# 去除优化器状态、梯度等
torch.save(model.state_dict(), "model_weights_only.pth")

# 5. 使用更小的模型变体
# ViT-L/14 -> ViT-B/32 -> ViT-B/16
```

### Q23: 如何评估模型性能？

**完整评估方案**:
```python
import time
import psutil
from torch.profiler import profile, ProfilerActivity

def comprehensive_evaluation(model, test_loader, device="cuda"):
    """全面评估模型"""
    model.eval()
    
    # 1. 准确率评估
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    
    # 2. 速度评估
    model.eval()
    torch.cuda.synchronize()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(torch.randn(1, 3, 224, 224).to(device))
    
    # 测速
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(torch.randn(1, 3, 224, 224).to(device))
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} images/sec")
    
    # 3. 内存评估
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(torch.randn(32, 3, 224, 224).to(device))
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory: {peak_memory:.2f} MB")
    
    # 4. 模型大小
    torch.save(model.state_dict(), "temp_model.pth")
    model_size = os.path.getsize("temp_model.pth") / 1024 / 1024
    os.remove("temp_model.pth")
    print(f"Model size: {model_size:.2f} MB")
    
    # 5. 详细性能分析（可选）
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            _ = model(torch.randn(1, 3, 224, 224).to(device))
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return {
        "accuracy": accuracy,
        "avg_inference_time_ms": avg_time * 1000,
        "throughput": 1 / avg_time,
        "peak_memory_mb": peak_memory,
        "model_size_mb": model_size
    }

# 使用
results = comprehensive_evaluation(model, test_loader)
```

---

## 📞 获取帮助

如果以上FAQ没有解决你的问题，可以通过以下方式获取帮助：

### 1. 查看文档
- [环境安装指南](./01-环境安装指南.md)
- [快速开始](./02-快速开始.md)
- [最佳实践](./04-最佳实践.md)
- [故障排查指南](./05-故障排查指南.md)

### 2. 提交Issue
- GitHub Issues: https://github.com/YourRepo/Large-Model-Tutorial/issues
- 提供详细的错误信息和复现步骤

### 3. 社区讨论
- GitHub Discussions: 与其他用户交流经验
- Stack Overflow: 使用 `clip` `vision-transformer` 标签

### 4. 查看日志
```bash
# 查看项目日志
cat logs/training.log

# 查看系统日志
dmesg | grep -i cuda
dmesg | grep -i nvidia
```

---

## 🔄 持续更新

本FAQ会持续更新，添加更多常见问题。如果你遇到了新的问题并找到了解决方案，欢迎贡献到项目中！

**最后更新**: 2025-11-05  
**版本**: v1.0

