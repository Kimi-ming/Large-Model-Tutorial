# 贡献指南

感谢你对 Large-Model-Tutorial 项目的关注！我们欢迎所有形式的贡献。

---

## 📋 目录

1. [如何贡献](#如何贡献)
2. [开发环境搭建](#开发环境搭建)
3. [代码规范](#代码规范)
4. [提交规范](#提交规范)
5. [Pull Request流程](#pull-request流程)
6. [文档贡献](#文档贡献)
7. [问题报告](#问题报告)

---

## 如何贡献

你可以通过以下方式贡献：

### 1. 代码贡献
- 🐛 修复Bug
- ✨ 添加新功能
- ⚡ 性能优化
- 🎨 代码重构

### 2. 文档贡献
- 📝 完善文档
- 🌍 翻译文档
- 📖 添加教程
- 💡 改进示例

### 3. 测试贡献
- ✅ 添加测试用例
- 🧪 提高测试覆盖率
- 🔍 发现和报告Bug

### 4. 社区贡献
- 💬 回答问题
- 📣 分享经验
- 🎓 组织活动

---

## 开发环境搭建

### 1. Fork 和 Clone

```bash
# Fork 项目到你的GitHub账号
# 然后Clone到本地
git clone https://github.com/YOUR_USERNAME/Large-Model-Tutorial.git
cd Large-Model-Tutorial

# 添加上游仓库
git remote add upstream https://github.com/ORIGINAL_OWNER/Large-Model-Tutorial.git
```

### 2. 创建开发分支

```bash
# 从main分支创建新分支
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

### 3. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装测试依赖
pip install -r requirements-test.txt
```

### 4. 验证环境

```bash
# 运行测试
pytest tests/

# 检查代码风格
flake8 code/
black --check code/
```

---

## 代码规范

### Python 代码风格

我们遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 代码风格。

**使用工具**:
```bash
# 格式化代码
black code/

# 检查代码风格
flake8 code/

# 排序import
isort code/

# 类型检查
mypy code/
```

**代码示例**:
```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    训练模型一个epoch。
    
    Args:
        model: 要训练的模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 训练设备
    
    Returns:
        包含训练指标的字典
    """
    model.train()
    total_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return {
        "loss": total_loss / len(dataloader)
    }
```

### 命名规范

- **变量**: `snake_case`
- **函数**: `snake_case`
- **类**: `PascalCase`
- **常量**: `UPPER_SNAKE_CASE`
- **私有**: `_leading_underscore`

### 文档字符串

使用Google风格的docstring:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    函数简短描述。
    
    详细描述（如果需要）。
    
    Args:
        param1: 参数1的描述
        param2: 参数2的描述
    
    Returns:
        返回值的描述
    
    Raises:
        ValueError: 什么情况下抛出
    
    Examples:
        >>> function_name(1, "test")
        True
    """
    pass
```

---

## 提交规范

### Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type类型**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

**示例**:
```
feat(training): 添加混合精度训练支持

- 使用torch.cuda.amp实现混合精度
- 添加GradScaler支持
- 更新文档

Closes #123
```

### Commit 最佳实践

1. **一次提交只做一件事**
2. **提交信息清晰描述改动**
3. **提交前运行测试**
4. **提交前格式化代码**

---

## Pull Request流程

### 1. 确保代码质量

```bash
# 运行测试
pytest tests/

# 代码格式化
black code/
isort code/

# 代码检查
flake8 code/
mypy code/
```

### 2. 推送到你的Fork

```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

### 3. 创建Pull Request

1. 访问你的Fork页面
2. 点击 "New Pull Request"
3. 填写PR模板
4. 提交PR

### 4. PR模板

```markdown
## 描述
简要描述这个PR的目的和改动。

## 改动类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 性能优化
- [ ] 代码重构

## 测试
- [ ] 添加了测试用例
- [ ] 所有测试通过
- [ ] 手动测试通过

## Checklist
- [ ] 代码符合项目规范
- [ ] 更新了相关文档
- [ ] 添加了必要的注释
- [ ] 没有引入新的警告

## 相关Issue
Closes #issue_number
```

### 5. Code Review

- 响应reviewer的评论
- 根据反馈修改代码
- 保持PR最新（rebase到最新的main）

---

## 文档贡献

### Markdown 规范

- 使用标准Markdown语法
- 代码块指定语言
- 使用相对路径链接
- 保持一致的格式

**示例**:
````markdown
# 标题

简要描述。

## 章节

### 子章节

正文内容。

**代码示例**:
```python
def example():
    pass
```

**注意**: 重要提示。

**参考**: [相关文档](./relative/path.md)
````

### 文档结构

```
docs/
├── 01-模型调研与选型/
│   ├── 01-主流视觉大模型概述.md
│   └── ...
├── 02-模型微调技术/
│   └── ...
└── ...
```

### 文档内容要求

1. **准确性**: 确保技术内容正确
2. **完整性**: 包含必要的背景和上下文
3. **清晰性**: 使用简洁明了的语言
4. **示例**: 提供可运行的代码示例

---

## 问题报告

### Bug报告

使用Bug报告模板：

1. **环境信息**: OS、Python版本、依赖版本
2. **复现步骤**: 详细的步骤
3. **期望行为**: 应该发生什么
4. **实际行为**: 实际发生了什么
5. **错误信息**: 完整的错误堆栈
6. **最小复现**: 最简单的复现代码

### 功能请求

使用功能请求模板：

1. **问题描述**: 遇到了什么问题
2. **建议方案**: 你建议的解决方案
3. **替代方案**: 其他可能的方案
4. **额外上下文**: 相关的背景信息

---

## 社区准则

### 行为准则

请阅读并遵守我们的 [行为准则](CODE_OF_CONDUCT.md)。

### 尊重和包容

- 尊重不同的观点
- 接受建设性批评
- 关注对社区最有利的事情
- 对其他成员表示同理心

### 沟通

- 使用清晰、礼貌的语言
- 及时响应评论和问题
- 寻求帮助时提供足够信息
- 回答问题时保持耐心

---

## 获取帮助

### 资源

- 📖 [项目文档](docs/)
- ❓ [FAQ](docs/05-使用说明/03-常见问题FAQ.md)
- 💬 [GitHub Discussions](https://github.com/OWNER/REPO/discussions)
- 🐛 [GitHub Issues](https://github.com/OWNER/REPO/issues)

### 联系方式

- GitHub Issues: 技术问题
- GitHub Discussions: 一般讨论
- Email: [维护者邮箱]

---

## 致谢

感谢所有贡献者！你们的贡献让这个项目变得更好。

查看贡献者列表: [Contributors](https://github.com/OWNER/REPO/graphs/contributors)

---

**最后更新**: 2025-11-05  
**版本**: v1.0

