# Docker部署指南

本目录包含视觉大模型教程项目的Docker配置文件，支持NVIDIA GPU和华为昇腾NPU两种部署方式。

## 📋 目录结构

```
docker/
├── Dockerfile.nvidia       # NVIDIA GPU镜像
├── Dockerfile.huawei       # 华为昇腾NPU镜像
├── docker-compose.yml      # Docker Compose编排配置
├── .dockerignore          # Docker构建忽略文件
└── README.md              # 本文档
```

## 🚀 快速开始

### 方式1：使用Docker Compose（推荐）

#### 启动开发环境
```bash
# 进入项目根目录
cd Large-Model-Tutorial

# 启动NVIDIA GPU开发环境
docker-compose up -d nvidia-dev

# 进入容器
docker-compose exec nvidia-dev bash
```

#### 启动API服务
```bash
# 启动API服务和Redis
docker-compose up -d nvidia-api redis

# 测试API
curl http://localhost:8001/health
```

#### 启动Jupyter Notebook
```bash
# 启动Jupyter
docker-compose up -d jupyter

# 访问 http://localhost:8889
# 默认无需密码（仅供开发使用）
```

#### 启动完整服务栈
```bash
# 启动所有服务
docker-compose up -d nvidia-dev nvidia-api jupyter redis tensorboard

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f nvidia-api
```

### 方式2：使用Docker命令

#### NVIDIA GPU镜像

**1. 构建镜像**
```bash
docker build -f docker/Dockerfile.nvidia -t large-model-tutorial:nvidia .
```

**2. 运行容器（交互式）**
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  large-model-tutorial:nvidia
```

**3. 运行API服务（后台）**
```bash
docker run --gpus all -d \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  --name vlm-tutorial \
  large-model-tutorial:nvidia \
  python code/04-deployment/api-server/app.py
```

**4. 运行Jupyter Notebook**
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  large-model-tutorial:nvidia \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### 华为昇腾NPU镜像

**1. 构建镜像（需在昇腾服务器上）**
```bash
docker build -f docker/Dockerfile.huawei -t large-model-tutorial:ascend .
```

**2. 运行容器**
```bash
docker run \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $(pwd):/workspace \
  -it --rm \
  large-model-tutorial:ascend
```

**3. 使用Docker Compose启动昇腾环境**
```bash
docker-compose --profile ascend up -d ascend-dev
```

## 📦 镜像说明

### NVIDIA GPU镜像特性

- **基础镜像**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Python版本**: 3.10
- **PyTorch版本**: 2.0.1 (CUDA 11.8)
- **主要功能**:
  - 模型训练
  - 模型推理
  - API服务部署
  - Jupyter Notebook
  - TensorBoard可视化

- **暴露端口**:
  - `8000`: FastAPI服务
  - `8888`: Jupyter Notebook
  - `6006`: TensorBoard

### 华为昇腾镜像特性

- **基础镜像**: `ascend-pytorch:23.0.RC3`
- **CANN版本**: 7.0
- **主要功能**:
  - 昇腾NPU训练
  - ACL推理
  - 模型转换（ONNX → OM）

- **暴露端口**:
  - `8000`: API服务
  - `8888`: Jupyter Notebook

## 🔧 环境配置

### 资源限制

可以通过环境变量和Docker参数配置资源使用：

```bash
# 限制使用的GPU
docker run --gpus '"device=0,1"' ...

# 限制CPU和内存
docker run --cpus="4.0" --memory="16g" ...
```

### 环境变量

在`docker-compose.yml`中可配置的环境变量：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 指定GPU设备 | `0` |
| `API_HOST` | API服务监听地址 | `0.0.0.0` |
| `API_PORT` | API服务端口 | `8000` |
| `MODEL_NAME` | 默认模型名称 | `openai/clip-vit-base-patch32` |
| `DEVICE` | 推理设备 | `cuda` |

## 📊 服务说明

### 1. nvidia-dev（开发环境）
- 完整的GPU开发环境
- 包含所有开发工具和依赖
- 适合：模型训练、代码开发、测试

### 2. nvidia-api（API服务）
- 生产级API服务
- 自动重启
- 健康检查
- 适合：模型推理服务部署

### 3. jupyter（交互式开发）
- Jupyter Notebook/Lab
- GPU加速
- 适合：数据探索、模型实验

### 4. redis（缓存服务）
- 用于API结果缓存
- 提升响应速度
- 持久化存储

### 5. tensorboard（可视化）
- 训练过程可视化
- 实时监控指标
- 访问：http://localhost:6007

## 🎯 常用操作

### 查看容器状态
```bash
docker-compose ps
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs -f nvidia-api

# 查看最近100行
docker-compose logs --tail=100 nvidia-api
```

### 进入容器
```bash
# 进入开发环境
docker-compose exec nvidia-dev bash

# 进入API容器
docker-compose exec nvidia-api bash
```

### 重启服务
```bash
# 重启特定服务
docker-compose restart nvidia-api

# 重启所有服务
docker-compose restart
```

### 停止服务
```bash
# 停止所有服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v

# 停止特定服务
docker-compose stop nvidia-api
```

### 重新构建镜像
```bash
# 重新构建所有镜像
docker-compose build

# 重新构建特定镜像（不使用缓存）
docker-compose build --no-cache nvidia-dev
```

## 📝 开发工作流

### 1. 本地开发
```bash
# 启动开发环境
docker-compose up -d nvidia-dev

# 进入容器
docker-compose exec nvidia-dev bash

# 在容器内开发和测试
python code/02-fine-tuning/lora/train.py

# 代码会自动同步（通过volume挂载）
```

### 2. 训练模型
```bash
# 在容器内运行训练
docker-compose exec nvidia-dev python code/02-fine-tuning/lora/train.py \
  --config code/02-fine-tuning/lora/config.yaml

# 或者直接运行
docker-compose run --rm nvidia-dev \
  python code/02-fine-tuning/lora/train.py
```

### 3. 部署API
```bash
# 启动API服务
docker-compose up -d nvidia-api redis

# 测试API
curl -X POST http://localhost:8001/classify \
  -F "file=@test_image.jpg" \
  -F "labels=dog,cat,bird"
```

### 4. 使用Jupyter
```bash
# 启动Jupyter
docker-compose up -d jupyter

# 访问 http://localhost:8889
# 打开notebooks目录下的教程
```

## 🐛 故障排查

### 1. GPU不可用

**问题**：容器内无法使用GPU

**解决**：
```bash
# 检查nvidia-docker安装
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 检查Docker版本（需要 >= 19.03）
docker --version

# 安装nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 容器启动失败

**问题**：服务无法启动

**解决**：
```bash
# 查看详细日志
docker-compose logs nvidia-api

# 检查端口占用
netstat -tuln | grep 8000

# 强制重新创建容器
docker-compose up -d --force-recreate nvidia-api
```

### 3. 模型下载慢

**问题**：HuggingFace模型下载缓慢

**解决**：
```bash
# 设置镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或在docker-compose.yml中添加环境变量
environment:
  - HF_ENDPOINT=https://hf-mirror.com
```

### 4. 内存不足

**问题**：OOM错误

**解决**：
```bash
# 增加Docker内存限制
docker-compose.yml:
  nvidia-api:
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
```

### 5. 昇腾设备映射失败

**问题**：无法访问NPU设备

**解决**：
```bash
# 检查设备是否存在
ls -l /dev/davinci*

# 检查设备权限
sudo chmod 666 /dev/davinci*

# 确认CANN驱动版本
npu-smi info
```

## 📚 参考资料

### Docker相关
- [Docker官方文档](https://docs.docker.com/)
- [Docker Compose文档](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### 项目相关
- [项目README](../README.md)
- [部署文档](../docs/04-多平台部署/)
- [API文档](../docs/04-多平台部署/02-模型服务化.md)

## 💡 最佳实践

### 1. 镜像优化
- 使用多阶段构建减小镜像大小
- 合并RUN命令减少层数
- 使用.dockerignore排除不必要文件
- 固定依赖版本确保可重现性

### 2. 数据管理
- 使用Docker volumes持久化数据
- 模型和数据通过挂载目录共享
- 定期备份重要数据

### 3. 安全性
- 生产环境设置Jupyter密码
- 使用环境变量管理敏感信息
- 定期更新基础镜像
- 限制容器资源使用

### 4. 性能优化
- 使用Redis缓存API结果
- 配置合适的GPU内存分配
- 使用批处理提高吞吐量
- 启用混合精度训练

## 🤝 贡献

如有问题或建议，请提交Issue或Pull Request。

## 📄 许可证

本项目遵循MIT许可证。

