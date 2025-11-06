# Bug修复日志 - 部署依赖问题

**日期**: 2025-11-06  
**类型**: 高优先级Bug修复  
**影响范围**: Docker部署、ONNX GPU推理  
**修复人员**: Large-Model-Tutorial Team

---

## 📋 问题概述

修复了两个高优先级的部署相关Bug：
1. ONNX推理强制使用GPU导致CPU环境失败
2. 华为Docker镜像torch-npu安装失败

---

## 🐛 Bug #1: ONNX推理GPU依赖问题

### 问题描述

**文件**: `code/04-deployment/nvidia/onnx/onnx_inference.py`

**现象**:
- 代码默认 `use_gpu=True` 创建会话
- 强制把 `CUDAExecutionProvider` 放在provider列表
- 如果环境只安装了 CPU 版 `onnxruntime`，会抛出异常：
  ```
  CUDAExecutionProvider is not available
  ```
- 当前 `requirements.txt` 和 `docker/Dockerfile.nvidia` 只安装 `onnxruntime==1.16.3`（CPU版本）
- 按照文档运行GPU ONNX推理必定失败

### 根本原因

1. 代码未检查provider可用性就直接使用
2. 依赖文件未区分CPU和GPU版本
3. Docker镜像未安装GPU版本的onnxruntime

### 解决方案

#### 1. 修改代码智能检测provider

```python
# 修改前
if use_gpu:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']

# 修改后
available_providers = ort.get_available_providers()
print(f"📋 可用的Execution Providers: {available_providers}")

if use_gpu:
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"✅ 使用GPU推理 (CUDA)")
    else:
        providers = ['CPUExecutionProvider']
        print(f"⚠️  CUDA不可用，回退到CPU推理")
        print(f"💡 提示: 安装 onnxruntime-gpu 以启用GPU加速")
        print(f"   pip install onnxruntime-gpu")
else:
    providers = ['CPUExecutionProvider']
    print(f"✅ 使用CPU推理")
```

**改进点**：
- ✅ 运行时检测可用的provider
- ✅ 自动回退到CPU（而非崩溃）
- ✅ 给出友好的提示信息
- ✅ 保持向后兼容

#### 2. 创建GPU专用依赖文件

创建 `requirements-gpu.txt`：
```txt
# GPU版本依赖
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# ONNX Runtime GPU版本
onnxruntime-gpu==1.16.3  # 支持CUDA加速

# ... 其他依赖
```

**特点**：
- 明确区分CPU和GPU依赖
- 包含详细的安装说明
- 注明CUDA版本要求

#### 3. 更新Dockerfile.nvidia

```dockerfile
# 修改前
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# 修改后
COPY requirements.txt /workspace/
COPY requirements-gpu.txt /workspace/
RUN pip install --no-cache-dir -r requirements-gpu.txt
```

**效果**：
- Docker镜像包含GPU版本的onnxruntime
- 支持GPU加速推理
- 文档和实际一致

### 验证测试

```bash
# 测试1: CPU环境
pip install onnxruntime==1.16.3
python code/04-deployment/nvidia/onnx/onnx_inference.py --cpu

# 预期输出：
# 📋 可用的Execution Providers: ['CPUExecutionProvider']
# ✅ 使用CPU推理

# 测试2: GPU环境
pip install onnxruntime-gpu==1.16.3
python code/04-deployment/nvidia/onnx/onnx_inference.py

# 预期输出：
# 📋 可用的Execution Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# ✅ 使用GPU推理 (CUDA)

# 测试3: CPU环境但请求GPU
pip install onnxruntime==1.16.3
python code/04-deployment/nvidia/onnx/onnx_inference.py

# 预期输出：
# 📋 可用的Execution Providers: ['CPUExecutionProvider']
# ⚠️  CUDA不可用，回退到CPU推理
# 💡 提示: 安装 onnxruntime-gpu 以启用GPU加速
```

---

## 🐛 Bug #2: 华为Docker镜像torch-npu安装失败

### 问题描述

**文件**: `docker/Dockerfile.huawei`

**现象**:
- Dockerfile 直接执行 `pip install torch-npu`
- 华为官方明确说明：PyPI上的通用包**无法用于昇腾NPU**
- 要么下载到CPU版本，要么直接安装失败
- 导致Docker构建流程中断
- 与项目文档里"不要使用PyPI版本"的说明相矛盾

### 根本原因

1. Dockerfile使用了不支持的安装方式
2. 基础镜像已包含torch-npu，无需额外安装
3. 文档与实际代码不一致

### 解决方案

#### 1. 修改Dockerfile.huawei

```dockerfile
# 修改前（错误）
RUN pip install --no-cache-dir \
    onnxruntime==1.16.3 \
    torch-npu \    # ❌ 这行会失败
    apex

# 修改后（正确）
# 注意：torch-npu 需要从华为官方源或基础镜像中获取，不能使用PyPI版本
# 基础镜像 ascend-pytorch:23.0.RC3 已包含 torch 和 torch_npu
RUN pip install --no-cache-dir \
    onnxruntime==1.16.3
# apex 需要从源码安装或使用预编译包
# RUN pip install --no-cache-dir apex  # 如需使用请从源码安装
```

**改进点**：
- ✅ 移除错误的torch-npu安装命令
- ✅ 添加详细的注释说明
- ✅ 依赖基础镜像自带的torch-npu
- ✅ 保持与文档一致

#### 2. 创建离线安装文档

创建 `docker/OFFLINE_INSTALL.md`，包含：

1. **torch-npu正确获取方式**：
   - 方式一：使用基础镜像自带（推荐）
   - 方式二：从华为官方源下载wheel
   - 方式三：从已安装环境复制

2. **离线构建完整流程**：
   - 下载基础镜像
   - 准备Python依赖包
   - 修改Dockerfile支持离线
   - 传输和构建步骤

3. **常见问题解答**：
   - torch-npu安装失败处理
   - onnxruntime-gpu离线安装
   - 离线镜像验证方法

### 验证测试

```bash
# 测试1: 构建镜像
docker build -f docker/Dockerfile.huawei \
    -t large-model-tutorial:ascend-test \
    .

# 预期：构建成功，无torch-npu安装错误

# 测试2: 验证torch-npu
docker run --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -it --rm \
    large-model-tutorial:ascend-test \
    python -c "
import torch
import torch_npu
print(f'✅ torch version: {torch.__version__}')
print(f'✅ NPU available: {torch.npu.is_available()}')
print(f'✅ NPU device: {torch.npu.get_device_name(0)}')
"

# 预期输出：
# ✅ torch version: 2.1.0
# ✅ NPU available: True
# ✅ NPU device: Ascend910
```

---

## 📊 影响范围

### 受影响的文件

1. **代码文件**:
   - `code/04-deployment/nvidia/onnx/onnx_inference.py` ✅ 已修复

2. **Docker文件**:
   - `docker/Dockerfile.nvidia` ✅ 已更新
   - `docker/Dockerfile.huawei` ✅ 已修复

3. **依赖文件**:
   - `requirements-gpu.txt` ✅ 新增

4. **文档文件**:
   - `docker/OFFLINE_INSTALL.md` ✅ 新增

### 受影响的用户场景

1. **ONNX GPU推理**: 
   - 之前：CPU环境直接崩溃
   - 现在：自动回退到CPU，给出提示

2. **Docker GPU部署**:
   - 之前：镜像不支持GPU推理
   - 现在：完整的GPU支持

3. **华为昇腾部署**:
   - 之前：Docker构建失败
   - 现在：构建成功，正常使用

4. **离线部署**:
   - 之前：无文档支持
   - 现在：完整的离线安装指南

---

## 🔍 技术细节

### ONNX Runtime Provider机制

ONNX Runtime支持多种Execution Provider：
- `CUDAExecutionProvider`: NVIDIA GPU加速
- `TensorrtExecutionProvider`: TensorRT优化
- `CPUExecutionProvider`: CPU回退
- `CoreMLExecutionProvider`: Apple设备
- `OpenVINOExecutionProvider`: Intel优化

**最佳实践**：
1. 运行时检测可用provider
2. 按优先级排序（GPU > CPU）
3. 自动回退到可用的provider
4. 给出清晰的日志提示

### torch-npu安装机制

**华为昇腾的特殊性**：
- torch-npu与CANN版本强绑定
- PyPI上的包是通用占位包，不包含NPU支持
- 必须从华为官方渠道获取对应版本

**正确安装方式**：
```bash
# 方式1: 使用华为镜像源
pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/...

# 方式2: 从本地wheel安装
pip install /path/to/torch_npu-2.1.0-py3-none-linux_aarch64.whl

# 方式3: 使用基础镜像自带（Docker推荐）
# 基础镜像已包含，无需安装
```

---

## 📝 经验教训

### 1. 依赖管理

**教训**：
- 区分CPU和GPU依赖
- 明确版本兼容性要求
- 提供清晰的安装文档

**改进**：
- 创建独立的requirements-gpu.txt
- 添加版本兼容性说明
- 文档中明确标注依赖关系

### 2. 错误处理

**教训**：
- 不要假设资源一定可用
- 提供友好的错误提示
- 支持自动降级策略

**改进**：
- 运行时检测而非假设
- 清晰的错误信息和解决建议
- 优雅的回退机制

### 3. 平台特殊性

**教训**：
- 不同硬件平台有特殊要求
- 文档要与实际实现保持一致
- 充分测试各种环境

**改进**：
- 详细的平台特定文档
- 代码注释中说明特殊处理
- 提供验证脚本

### 4. 离线部署

**教训**：
- 生产环境常需要离线部署
- 依赖链可能很复杂
- 需要完整的离线方案

**改进**：
- 创建离线安装专门文档
- 提供依赖下载脚本
- 包含验证和故障排查

---

## ✅ 验收标准

### 功能验收

- [x] ONNX推理在CPU环境正常运行
- [x] ONNX推理在GPU环境使用CUDA加速
- [x] 华为Docker镜像成功构建
- [x] torch-npu在容器中可用
- [x] 离线安装文档完整可用

### 代码质量

- [x] 代码有清晰的注释
- [x] 错误处理健壮
- [x] 日志输出友好
- [x] 保持向后兼容

### 文档完整性

- [x] 修改记录在changelog
- [x] 相关文档已更新
- [x] 提供使用示例
- [x] 包含故障排查

---

## 🚀 后续计划

### v1.1改进

1. **依赖管理增强**:
   - [ ] 添加依赖版本检测工具
   - [ ] 自动生成requirements文件
   - [ ] 依赖冲突检查

2. **部署工具完善**:
   - [ ] 环境诊断脚本
   - [ ] 一键离线打包工具
   - [ ] 自动化验证测试

3. **文档改进**:
   - [ ] 添加视频教程
   - [ ] 常见问题FAQ扩充
   - [ ] 多平台部署对比

---

## 📞 相关链接

- Issue讨论: （待创建）
- Pull Request: （待创建）
- 相关文档:
  - [ONNX推理文档](../../docs/04-多平台部署/01-NVIDIA部署基础.md)
  - [华为昇腾部署](../../docs/04-多平台部署/03-华为昇腾部署.md)
  - [Docker使用指南](../README.md)
  - [离线安装指南](../OFFLINE_INSTALL.md)

---

**修复完成日期**: 2025-11-06  
**测试状态**: ✅ 通过  
**发布版本**: v1.0.1（计划）

