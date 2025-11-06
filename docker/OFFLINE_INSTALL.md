# Docker离线安装指南

本文档介绍如何在离线环境中构建和使用Docker镜像。

---

## 📋 目录

- [华为昇腾离线安装](#华为昇腾离线安装)
- [NVIDIA GPU离线安装](#nvidia-gpu离线安装)
- [常见问题](#常见问题)

---

## 华为昇腾离线安装

### 1. 准备工作

#### 1.1 下载基础镜像

在有网络的环境中下载基础镜像：

```bash
# 下载昇腾基础镜像
docker pull ascendhub.huawei.com/public-ascendhub/ascend-pytorch:23.0.RC3-ubuntu18.04

# 保存镜像为tar文件
docker save ascendhub.huawei.com/public-ascendhub/ascend-pytorch:23.0.RC3-ubuntu18.04 \
    -o ascend-pytorch-23.0.RC3.tar

# 压缩（可选）
gzip ascend-pytorch-23.0.RC3.tar
```

#### 1.2 准备Python依赖包

创建下载脚本 `download_wheels.sh`：

```bash
#!/bin/bash
# 下载所有Python依赖包

mkdir -p wheels/ascend

# 基础依赖
pip download -d wheels/ascend \
    transformers==4.35.0 \
    pillow==10.1.0 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    pandas==2.1.3 \
    scikit-learn==1.3.2 \
    albumentations==1.3.1 \
    peft==0.6.2 \
    accelerate==0.25.0 \
    onnx==1.15.0 \
    onnxruntime==1.16.3 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    loguru==0.7.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    tqdm==4.66.1 \
    requests==2.31.0 \
    aiohttp==3.9.1 \
    huggingface-hub==0.19.4 \
    datasets==2.15.0 \
    fire==0.5.0

echo "✅ 依赖包下载完成，保存在 wheels/ascend/ 目录"
```

执行下载：

```bash
chmod +x download_wheels.sh
./download_wheels.sh
```

#### 1.3 准备torch-npu（重要）

**注意**：torch-npu **不能**从PyPI安装！

方式一：从华为官方镜像源下载（推荐）

```bash
# 访问华为昇腾社区获取对应CANN版本的torch-npu包
# https://www.hiascend.com/software/cann/community

# 示例：CANN 7.0 对应的 torch-npu
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/PyTorch/torch_npu/... \
    -O wheels/ascend/torch_npu-2.1.0-py3-none-linux_aarch64.whl
```

方式二：从已安装环境复制

```bash
# 如果有已安装torch-npu的环境，直接复制wheel包
cp /path/to/torch_npu-*.whl wheels/ascend/
```

方式三：使用基础镜像自带的

基础镜像 `ascend-pytorch:23.0.RC3` 已包含 torch 和 torch_npu，无需额外安装。

### 2. 修改Dockerfile支持离线安装

创建 `Dockerfile.huawei.offline`：

```dockerfile
FROM ascendhub.huawei.com/public-ascendhub/ascend-pytorch:23.0.RC3-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV ASCEND_HOME=/usr/local/Ascend \
    LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH \
    PATH=/usr/local/Ascend/latest/bin:$PATH \
    PYTHONPATH=/usr/local/Ascend/latest/python/site-packages:$PYTHONPATH

WORKDIR /workspace

# 安装系统依赖（需要缓存deb包）
# 如果完全离线，需要提前下载deb包
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# 复制离线依赖包
COPY wheels/ascend /tmp/wheels

# 从本地wheel目录安装依赖（离线模式）
RUN pip install --no-index --find-links=/tmp/wheels \
    transformers \
    pillow \
    opencv-python \
    numpy \
    pandas \
    scikit-learn \
    albumentations \
    peft \
    accelerate \
    onnx \
    onnxruntime \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    pyyaml \
    python-dotenv \
    loguru \
    matplotlib \
    seaborn \
    tqdm \
    requests \
    aiohttp \
    huggingface-hub \
    datasets \
    fire

# 验证torch-npu（基础镜像已包含）
RUN python -c "import torch; import torch_npu; print(f'✅ torch version: {torch.__version__}'); print(f'✅ torch_npu available')" || \
    echo "⚠️  torch-npu验证失败，请检查基础镜像"

# 复制项目代码
COPY . /workspace/

ENV PYTHONPATH=/workspace:$PYTHONPATH

RUN mkdir -p /workspace/logs \
    /workspace/outputs \
    /workspace/models \
    /workspace/data \
    /workspace/checkpoints

# 清理临时文件
RUN rm -rf /tmp/wheels

EXPOSE 8000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_npu; assert torch.npu.is_available()" || exit 1

CMD ["/bin/bash"]
```

### 3. 离线构建步骤

#### 3.1 传输文件到离线服务器

```bash
# 打包所有需要的文件
tar -czf offline-build.tar.gz \
    ascend-pytorch-23.0.RC3.tar \
    wheels/ \
    Dockerfile.huawei.offline \
    .

# 传输到离线服务器
scp offline-build.tar.gz user@offline-server:/path/to/build/
```

#### 3.2 在离线服务器上构建

```bash
# 解压
cd /path/to/build/
tar -xzf offline-build.tar.gz

# 加载基础镜像
docker load -i ascend-pytorch-23.0.RC3.tar

# 构建镜像
docker build -f Dockerfile.huawei.offline \
    -t large-model-tutorial:ascend-offline \
    .
```

### 4. 验证安装

```bash
# 启动容器
docker run --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -it --rm \
    large-model-tutorial:ascend-offline \
    /bin/bash

# 在容器内验证
python -c "
import torch
import torch_npu

print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ NPU是否可用: {torch.npu.is_available()}')
print(f'✅ NPU设备数: {torch.npu.device_count()}')
"
```

---

## NVIDIA GPU离线安装

### 1. 准备工作

#### 1.1 下载基础镜像

```bash
# 下载NVIDIA CUDA镜像
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 保存镜像
docker save nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
    -o nvidia-cuda-11.8.tar

gzip nvidia-cuda-11.8.tar
```

#### 1.2 下载Python依赖

```bash
#!/bin/bash
mkdir -p wheels/nvidia

# PyTorch（CUDA 11.8版本）
pip download -d wheels/nvidia \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ONNX Runtime GPU版本
pip download -d wheels/nvidia \
    onnxruntime-gpu==1.16.3

# 其他依赖
pip download -d wheels/nvidia \
    -r requirements-gpu.txt

echo "✅ NVIDIA依赖包下载完成"
```

### 2. 修改Dockerfile

创建 `Dockerfile.nvidia.offline`：

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

# 复制离线依赖包
COPY wheels/nvidia /tmp/wheels

# 离线安装PyTorch
RUN pip install --no-index --find-links=/tmp/wheels \
    torch torchvision torchaudio

# 离线安装其他依赖
RUN pip install --no-index --find-links=/tmp/wheels \
    onnxruntime-gpu \
    transformers \
    # ... 其他依赖 ...

COPY . /workspace/

ENV PYTHONPATH=/workspace:$PYTHONPATH

RUN mkdir -p /workspace/logs \
    /workspace/outputs \
    /workspace/models \
    /workspace/data \
    /workspace/checkpoints

RUN rm -rf /tmp/wheels

EXPOSE 8000 8888 6006

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["/bin/bash"]
```

### 3. 离线构建

```bash
# 传输文件
tar -czf offline-build-nvidia.tar.gz \
    nvidia-cuda-11.8.tar \
    wheels/ \
    Dockerfile.nvidia.offline \
    .

# 在离线服务器上
docker load -i nvidia-cuda-11.8.tar
docker build -f Dockerfile.nvidia.offline \
    -t large-model-tutorial:nvidia-offline \
    .
```

---

## 常见问题

### Q1: torch-npu安装失败

**问题**：直接 `pip install torch-npu` 失败或下载到CPU版本

**解决方案**：
1. 使用基础镜像自带的torch-npu（推荐）
2. 从华为官方渠道获取对应CANN版本的wheel包
3. 不要使用PyPI上的通用包

**验证方法**：
```python
import torch
import torch_npu

# 正确的torch-npu会有这些方法
assert hasattr(torch, 'npu')
assert torch.npu.is_available()
print(f"✅ NPU设备: {torch.npu.get_device_name(0)}")
```

### Q2: onnxruntime-gpu在离线环境安装失败

**问题**：onnxruntime-gpu依赖CUDA库，离线环境可能缺失

**解决方案**：
1. 确保基础镜像包含CUDA Runtime
2. 使用完整的CUDA开发镜像（cudnn8-devel）
3. 预先下载所有依赖：
```bash
pip download onnxruntime-gpu==1.16.3 -d wheels/
```

### Q3: 如何验证离线镜像的完整性

**验证脚本**：

```bash
#!/bin/bash
# verify_offline_image.sh

echo "🔍 验证离线镜像..."

# 启动容器
CONTAINER_ID=$(docker run -d large-model-tutorial:ascend-offline sleep 3600)

# 验证Python包
docker exec $CONTAINER_ID python -c "
import torch
import transformers
import onnxruntime
print('✅ 所有依赖包导入成功')
"

# 验证NPU
docker exec $CONTAINER_ID python -c "
import torch
import torch_npu
assert torch.npu.is_available(), 'NPU不可用'
print(f'✅ NPU验证成功: {torch.npu.get_device_name(0)}')
"

# 清理
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo "✅ 离线镜像验证完成"
```

### Q4: 离线环境如何更新模型

**方案一：打包模型到镜像**

```dockerfile
# 在Dockerfile中
COPY models/clip-vit-base-patch32 /workspace/models/clip-vit-base-patch32
```

**方案二：使用数据卷**

```bash
# 在有网络的环境下载模型
python -c "
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.save_pretrained('./models/clip-vit-base-patch32')
processor.save_pretrained('./models/clip-vit-base-patch32')
"

# 传输到离线环境
tar -czf models.tar.gz models/

# 在离线环境使用
docker run -v $(pwd)/models:/workspace/models \
    large-model-tutorial:ascend-offline
```

### Q5: deb包依赖缺失

如果完全离线（无法访问apt源），需要提前下载deb包：

```bash
# 在有网络的环境
apt-get download $(apt-cache depends --recurse --no-recommends \
    build-essential cmake git wget curl \
    libopencv-dev libglib2.0-0 libsm6 libxext6 | \
    grep "^\w" | sort -u)

# 打包deb文件
mkdir debs
mv *.deb debs/
tar -czf debs.tar.gz debs/
```

在Dockerfile中：

```dockerfile
COPY debs /tmp/debs
RUN dpkg -i /tmp/debs/*.deb || apt-get install -f -y
RUN rm -rf /tmp/debs
```

---

## 📝 检查清单

离线构建前的检查清单：

- [ ] 基础Docker镜像已下载并保存
- [ ] Python wheel包已全部下载
- [ ] torch-npu wheel包已准备（昇腾）
- [ ] onnxruntime-gpu已下载（NVIDIA）
- [ ] 系统deb包已准备（如需完全离线）
- [ ] 预训练模型已下载（可选）
- [ ] Dockerfile已修改为离线模式
- [ ] 验证脚本已准备

---

## 🔗 参考资源

**华为昇腾**：
- 昇腾社区：https://www.hiascend.com/
- CANN文档：https://www.hiascend.com/document/
- torch-npu仓库：https://gitee.com/ascend/pytorch

**NVIDIA**：
- CUDA镜像：https://hub.docker.com/r/nvidia/cuda
- PyTorch官方：https://pytorch.org/
- ONNX Runtime：https://onnxruntime.ai/

---

**提示**：离线安装步骤较为复杂，建议先在有网络的测试环境验证流程，再应用到生产环境。

