# CI/CD 工作流文档

本目录包含GitHub Actions工作流配置，用于自动化测试、代码质量检查、文档构建和发布。

## 工作流概览

### 1. 测试工作流 (`test.yml`)

**触发条件**：
- Push到main/develop/feature分支
- PR到main/develop分支

**功能**：
- ✅ 多Python版本测试（3.8, 3.9, 3.10, 3.11）
- ✅ 运行单元测试
- ✅ 生成覆盖率报告
- ✅ 上传到Codecov
- ⏱️ 集成测试（仅main分支，P2实现）

**徽章**：
```markdown
[![Tests](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/test.yml/badge.svg)](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/test.yml)
```

### 2. 代码质量检查 (`lint.yml`)

**触发条件**：
- Push到main/develop/feature分支
- PR到main/develop分支

**功能**：
- ✅ Flake8代码风格检查
- ✅ Black代码格式检查
- ✅ isort import排序检查
- ✅ MyPy类型检查
- ✅ Bandit安全检查

**徽章**：
```markdown
[![Code Quality](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/lint.yml/badge.svg)](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/lint.yml)
```

### 3. 文档构建 (`docs.yml`)

**触发条件**：
- Push到main分支
- PR到main分支

**功能**：
- ✅ 验证Markdown文件
- ✅ 检查代码块语法
- ✅ 生成文档结构
- 🔜 链接检查（规划中）
- 🔜 GitHub Pages部署（规划中）

**徽章**：
```markdown
[![Documentation](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/docs.yml/badge.svg)](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/docs.yml)
```

### 4. 发布工作流 (`release.yml`)

**触发条件**：
- 推送版本tag（例如：v1.0.0）

**功能**：
- ✅ 自动生成changelog
- ✅ 运行发布前测试
- ✅ 创建发布包
- ✅ 创建GitHub Release
- 🔜 发布到PyPI（可选）

## 使用指南

### 本地测试

在提交代码前，建议本地运行检查：

```bash
# 运行测试
pytest tests/unit/ -v

# 代码格式化
black code/ tests/
isort code/ tests/

# 代码检查
flake8 code/ tests/

# 类型检查
mypy code/ --ignore-missing-imports
```

### 创建发布

```bash
# 1. 更新版本号（如需要）
# 编辑 setup.py 或相关版本文件

# 2. 提交更改
git add .
git commit -m "chore: prepare release v1.0.0"
git push

# 3. 创建并推送tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 4. GitHub Actions会自动创建release
```

### 查看CI/CD状态

访问：https://github.com/your-org/Large-Model-Tutorial/actions

## 配置说明

### 环境变量和密钥

工作流可能需要以下secrets：

- `GITHUB_TOKEN`：自动提供，用于创建release
- `CODECOV_TOKEN`：（可选）Codecov上传token
- `PYPI_API_TOKEN`：（可选）PyPI发布token

在仓库设置中配置：Settings → Secrets and variables → Actions

### 自定义配置

#### 修改Python版本

编辑`test.yml`中的matrix：

```yaml
matrix:
  python-version: ['3.8', '3.9', '3.10', '3.11']
```

#### 跳过CI

在commit message中添加：

```
[skip ci] 或 [ci skip]
```

#### 仅运行特定工作流

使用workflow dispatch（手动触发）或修改触发条件。

## 工作流状态

### P1阶段（当前）✅
- ✅ 单元测试自动化
- ✅ 代码质量检查
- ✅ 基础文档验证
- ✅ 发布流程

### P2阶段（规划）🔜
- 🔜 集成测试（需要真实模型）
- 🔜 性能基准测试
- 🔜 文档自动部署
- 🔜 Docker镜像构建
- 🔜 依赖更新检查

## 故障排除

### 测试失败

1. 检查Python版本兼容性
2. 确认所有依赖已安装
3. 查看详细日志：Actions → 失败的workflow → 点击查看

### Lint失败

1. 本地运行相同的lint工具
2. 按提示修复问题
3. 提交修复后的代码

### 发布失败

1. 确认tag格式正确（v*.*.*）
2. 检查是否有权限
3. 查看workflow日志

## 最佳实践

1. **提交前本地测试**：`pytest tests/unit/`
2. **代码格式化**：使用`black`和`isort`
3. **小步提交**：频繁提交，保持CI绿色
4. **PR审查**：等待CI通过后再merge
5. **版本管理**：遵循语义化版本（SemVer）

## 监控和通知

### 添加徽章到README

```markdown
# Large Model Tutorial

[![Tests](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/test.yml/badge.svg)](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/test.yml)
[![Code Quality](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/lint.yml/badge.svg)](https://github.com/your-org/Large-Model-Tutorial/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/your-org/Large-Model-Tutorial/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/Large-Model-Tutorial)
```

### Slack/Email通知

在workflow中添加通知步骤（可选）。

## 参考资源

- [GitHub Actions文档](https://docs.github.com/actions)
- [pytest文档](https://docs.pytest.org/)
- [Black文档](https://black.readthedocs.io/)
- [Codecov文档](https://docs.codecov.com/)

---

*最后更新：P1阶段完成*

