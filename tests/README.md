# 测试文档

## 测试结构

```
tests/
├── conftest.py              # pytest配置和共享fixtures
├── unit/                    # 单元测试（当前可用）
│   ├── test_clip_inference.py    ✅ CLIP推理测试
│   ├── test_sam_inference.py     ✅ SAM推理测试
│   └── test_applications.py      ✅ 应用功能测试
├── integration/             # 集成测试（规划中 - P2阶段）
└── fixtures/                # 测试数据
```

> **注意**：当前实现为P1阶段基础测试，包含单元测试。
> 
> P2阶段将补充：
> - `integration/test_fine_tuning.py` - 微调流程集成测试
> - `integration/test_deployment.py` - 部署工具集成测试
> - `integration/test_end_to_end.py` - 端到端测试
> - `unit/test_blip2_inference.py` - BLIP-2推理测试

## 运行测试

### 安装依赖

```bash
pip install pytest pytest-cov pytest-mock
```

### 运行所有测试

```bash
pytest
```

### 运行特定测试

```bash
# 仅单元测试
pytest tests/unit/

# 运行特定文件
pytest tests/unit/test_clip_inference.py
pytest tests/unit/test_sam_inference.py
pytest tests/unit/test_applications.py

# 运行特定测试函数
pytest tests/unit/test_clip_inference.py::TestCLIPInference::test_zero_shot_classification

# 集成测试（P2阶段）
# pytest tests/integration/  # 规划中
```

### 使用标记

```bash
# 仅运行单元测试
pytest -m unit

# 跳过慢速测试
pytest -m "not slow"

# 仅运行需要GPU的测试
pytest -m gpu
```

### 查看覆盖率

```bash
# 生成HTML覆盖率报告
pytest --cov=code --cov-report=html

# 查看报告
open htmlcov/index.html
```

## 测试类型

### 单元测试（Unit Tests）
- 测试单个函数/类
- 使用Mock避免依赖外部资源
- 快速执行
- 标记为`@pytest.mark.unit`

### 集成测试（Integration Tests）
- 测试多个组件协作
- 可能需要真实模型
- 执行较慢
- 标记为`@pytest.mark.integration`

### 慢速测试（Slow Tests）
- 需要下载模型
- 需要GPU训练
- 标记为`@pytest.mark.slow`
- CI环境可能跳过

## Fixtures说明

### 图像相关
- `sample_image`: 单张测试图像
- `sample_image_path`: 测试图像文件路径
- `sample_batch_images`: 批量测试图像
- `sample_mask`: 测试掩码

### Mock模型
- `mock_clip_model`: Mock CLIP模型
- `mock_sam_model`: Mock SAM模型
- `mock_blip2_model`: Mock BLIP-2模型

### 工具
- `temp_dir`: 临时目录（自动清理）
- `project_root`: 项目根目录
- `test_data_dir`: 测试数据目录

## CI/CD集成

测试会在CI/CD流程中自动运行：

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest -m "not slow and not skip_ci"
```

## 当前测试覆盖

### ✅ 已实现（P1阶段）

**单元测试**：
- `test_clip_inference.py` - CLIP零样本分类、特征提取、批量处理
- `test_sam_inference.py` - SAM分割（点/框提示）、掩码质量
- `test_applications.py` - 零售/医疗/交通应用基础功能

**测试用例数**：15+个
**Mock模型**：CLIP、SAM、BLIP-2

### 🔜 规划（P2阶段）

- `test_blip2_inference.py` - BLIP-2推理测试
- `integration/test_fine_tuning.py` - SAM微调流程测试
- `integration/test_deployment.py` - 部署工具测试
- `integration/test_end_to_end.py` - 端到端应用测试

## 贡献指南

添加新测试时：

1. **单元测试**放在`tests/unit/`
2. **集成测试**放在`tests/integration/`（P2阶段）
3. **使用合适的标记**：`@pytest.mark.unit`、`@pytest.mark.integration`等
4. **使用fixtures**避免重复代码
5. **使用Mock**避免依赖外部资源（单元测试）
6. **编写清晰的测试名称**和文档字符串

## 常见问题

**Q: 测试失败："No module named 'clip'"**

A: 安装CLIP：`pip install git+https://github.com/openai/CLIP.git`

**Q: 如何跳过需要GPU的测试？**

A: 使用`-m "not gpu"`：`pytest -m "not gpu"`

**Q: 如何Mock外部模型？**

A: 使用`conftest.py`中的mock fixtures，如`mock_clip_model`

---

*测试覆盖率目标：>80%*

