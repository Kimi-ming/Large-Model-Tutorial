# 模型推理示例

本目录包含主流视觉大模型的推理示例代码。

## 📦 已支持模型

| 模型 | 文件 | 主要功能 | 语言支持 |
|------|------|---------|---------|
| **CLIP** | `clip_inference.py` | 图文匹配、零样本分类 | 英文为主 |
| **SAM** | `sam_inference.py` | 图像分割 | 无文本 |
| **BLIP-2** | `blip_inference.py` | 图像描述、VQA | 英文为主 |
| **LLaVA** | `llava_inference.py` | 多模态对话 | 英文为主 |
| **Qwen-VL** ✨ | `qwen_vl_inference.py` | 中文场景、OCR、多图理解 | 中文优秀 |

## 🚀 快速开始

### 环境配置

```bash
# 基础依赖
pip install torch transformers pillow

# SAM额外依赖
pip install git+https://github.com/facebookresearch/segment-anything.git

# Qwen-VL额外依赖
pip install transformers>=4.32.0 transformers_stream_generator
```

### 基础使用

```bash
# CLIP推理
python clip_inference.py --image path/to/image.jpg --texts "a cat" "a dog"

# SAM分割
python sam_inference.py --image path/to/image.jpg --prompt point --x 100 --y 150

# BLIP-2描述生成
python blip_inference.py --image path/to/image.jpg --task caption

# Qwen-VL中文场景
python qwen_vl_inference.py --image path/to/image.jpg --demo all
```

## ✨ 新增：Qwen-VL支持（v1.1.0）

### 主要特性

1. **中文优秀** 🇨🇳
   - 中文VQA准确率：85.2%
   - 支持中英文混合理解
   - 针对中文场景优化

2. **多图理解** 🖼️
   - 同时处理多张图片
   - 理解图片间关系
   - 跨图推理能力

3. **细粒度识别** 🔍
   - OCR文字识别（F1: 89.3%）
   - 精确物体定位
   - 边界框标注

4. **长文本支持** 📝
   - 支持2048 tokens上下文
   - 多轮对话能力
   - 图文混合理解

### 使用示例

#### 1. 图像描述生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    trust_remote_code=True
)

query = tokenizer.from_list_format([
    {'image': 'image.jpg'},
    {'text': '详细描述这张图片'},
])

response, _ = model.chat(tokenizer, query=query, history=None)
print(response)
```

#### 2. 视觉问答（VQA）

```bash
python qwen_vl_inference.py \
    --image image.jpg \
    --demo vqa
```

#### 3. OCR文字识别

```bash
python qwen_vl_inference.py \
    --image document.jpg \
    --demo ocr
```

#### 4. 多图理解

```bash
python qwen_vl_inference.py \
    --images img1.jpg img2.jpg \
    --demo multi_image
```

#### 5. 多轮对话

```bash
python qwen_vl_inference.py \
    --image image.jpg \
    --demo chat
```

### 性能对比

| 任务 | CLIP | BLIP-2 | LLaVA | Qwen-VL |
|------|------|--------|-------|---------|
| **中文VQA** | ❌ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **英文VQA** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **OCR识别** | ❌ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **多图理解** | ❌ | ❌ | ⭐⭐ | ⭐⭐⭐⭐ |
| **推理速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **显存占用** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 📚 详细文档

- [CLIP模型详解](../../../docs/01-模型调研与选型/01-主流视觉大模型概述.md#1-clip)
- [SAM模型详解](../../../docs/01-模型调研与选型/05-SAM模型详解.md)
- [BLIP-2模型详解](../../../docs/01-模型调研与选型/06-BLIP2模型详解.md)
- [Qwen-VL模型详解](../../../docs/01-模型调研与选型/07-Qwen-VL模型详解.md) ✨新增

## 🔧 故障排查

### Qwen-VL相关问题

#### Q1: trust_remote_code错误

```python
# 解决方案：必须设置trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True  # 必须
)
```

#### Q2: 显存不足

```bash
# 使用量化版本
python qwen_vl_inference.py --image image.jpg --model Qwen/Qwen-VL-Chat-Int8
```

#### Q3: 中文输出乱码

```python
# 设置正确的编码
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### 通用问题

#### Q1: 模型下载慢

```bash
# 方式1: 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方式2: 使用ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen-VL-Chat')"
```

#### Q2: 依赖安装失败

```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
```

## 🤝 贡献指南

欢迎贡献新的模型推理示例！

### 添加新模型步骤

1. 创建推理脚本 `{model_name}_inference.py`
2. 参考现有脚本的代码结构
3. 添加详细的注释和使用说明
4. 更新本README文档
5. 提交Pull Request

### 代码规范

- 使用类型注解
- 添加详细的docstring
- 包含错误处理
- 提供命令行接口
- 支持多种使用场景

## 📝 更新日志

### v1.1.0 (2025-11-06)
- ✨ 新增Qwen-VL模型支持
- ✨ 添加中文场景示例
- ✨ 支持多图理解
- ✨ OCR文字识别功能
- 📚 新增Qwen-VL详细文档

### v1.0.0 (2025-11-05)
- ✨ 初始版本
- ✨ 支持CLIP、SAM、BLIP-2、LLaVA
- 📚 完整的使用文档

## 📄 许可证

本项目采用 [MIT License](../../../LICENSE)

---

**文档版本**: v1.1.0  
**最后更新**: 2025-11-06  
**维护者**: Large-Model-Tutorial Team

