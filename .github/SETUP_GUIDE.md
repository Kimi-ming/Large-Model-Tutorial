# GitHub 仓库设置指南

本文档指导如何设置GitHub仓库的Labels、Milestones和Projects。

## 目录
- [Labels设置](#labels设置)
- [Milestones设置](#milestones设置)
- [Projects看板设置](#projects看板设置)
- [创建Issues](#创建issues)

---

## Labels设置

### 方法1：手动创建（推荐新手）

1. 进入仓库页面
2. 点击 **Issues** 标签
3. 点击 **Labels**
4. 点击 **New label**
5. 按照 `.github/labels.yml` 中的配置逐个创建

### 方法2：使用GitHub CLI批量创建

```bash
# 安装GitHub CLI (如果还没有)
# https://cli.github.com/

# 登录
gh auth login

# 创建Labels（逐个执行）
gh label create "P0-MVP" --color "d73a4a" --description "最小可用版本（v0.5）必需的任务"
gh label create "P1-v1.0" --color "ff9800" --description "v1.0正式版必需的任务"
gh label create "P2-v1.5" --color "ffeb3b" --description "v1.5增强版的任务"
gh label create "P3-future" --color "4caf50" --description "未来版本的任务"
gh label create "📚教程必需" --color "2196f3" --description "学习者核心内容开发"
gh label create "🔧维护者" --color "9c27b0" --description "仓库工程化和维护内容"
gh label create "文档" --color "0075ca" --description "文档相关任务"
gh label create "代码" --color "008672" --description "代码开发任务"
gh label create "脚本" --color "1d76db" --description "脚本工具开发"
gh label create "测试" --color "d876e3" --description "测试相关任务"
gh label create "CI/CD" --color "fbca04" --description "持续集成/部署配置"
```

### 方法3：使用脚本批量创建

创建 `scripts/create_labels.sh`:

```bash
#!/bin/bash
# 从labels.yml读取并创建所有labels
# 需要安装yq工具: brew install yq (macOS) 或 apt install yq (Ubuntu)

yq eval '.[] | "gh label create \"" + .name + "\" --color \"" + .color + "\" --description \"" + .description + "\" --force"' .github/labels.yml | sh
```

---

## Milestones设置

### 手动创建步骤

1. 进入仓库页面
2. 点击 **Issues** → **Milestones**
3. 点击 **New milestone**
4. 填入以下信息：

#### Milestone 1: v0.5 MVP
- **Title**: `v0.5 MVP`
- **Due date**: （根据实际情况设定，建议4-6周）
- **Description**:
  ```
  最小可用版本，让学习者能够快速上手，完成从模型选型到基础部署的完整学习路径。
  
  包含任务：
  - 基础框架搭建
  - 至少3个模型推理示例
  - LoRA微调示例
  - NVIDIA基础部署
  - 快速开始文档
  ```

#### Milestone 2: v1.0 Release
- **Title**: `v1.0 Release`
- **Due date**: （建议在MVP完成后8-12周）
- **Description**:
  ```
  正式发布版本，提供完整的视觉大模型学习体系，覆盖主流应用场景。
  
  包含任务：
  - 所有核心教程内容
  - 至少3个实际应用场景
  - API服务和容器化部署
  - 完整文档和测试
  - 社区基础设施
  ```

#### Milestone 3: v1.5 Enhancement
- **Title**: `v1.5 Enhancement`
- **Due date**: （待定）
- **Description**:
  ```
  增强版本，根据用户反馈增强教程的广度和深度。
  
  包含任务：
  - 更多部署平台支持
  - 更多应用场景
  - 高级主题内容
  ```

#### Milestone 4: v2.0 Future
- **Title**: `v2.0 Future`
- **Due date**: （待定）
- **Description**:
  ```
  重大更新版本，探索前沿技术和特殊需求。
  ```

### 使用GitHub CLI创建

```bash
# 创建v0.5 MVP
gh api repos/:owner/:repo/milestones \
  -f title="v0.5 MVP" \
  -f description="最小可用版本，让学习者能够快速上手" \
  -f state="open"

# 创建v1.0 Release
gh api repos/:owner/:repo/milestones \
  -f title="v1.0 Release" \
  -f description="正式发布版本，提供完整的视觉大模型学习体系" \
  -f state="open"

# 创建v1.5 Enhancement
gh api repos/:owner/:repo/milestones \
  -f title="v1.5 Enhancement" \
  -f description="增强版本，根据用户反馈增强教程的广度和深度" \
  -f state="open"

# 创建v2.0 Future
gh api repos/:owner/:repo/milestones \
  -f title="v2.0 Future" \
  -f description="重大更新版本，探索前沿技术和特殊需求" \
  -f state="open"
```

---

## Projects看板设置

### 创建新Project（推荐使用Projects Beta）

1. 进入仓库页面
2. 点击 **Projects** 标签
3. 点击 **New project**
4. 选择 **Board** 模板
5. 命名为 `Large-Model-Tutorial Development`

### 配置看板列

创建以下4个列：

1. **📋 Backlog（待开始）**
   - 描述：待开始的任务
   - 自动化：无

2. **🚧 In Progress（进行中）**
   - 描述：正在进行的任务
   - 自动化：当Issue被分配时自动移到此列

3. **👀 In Review（待审查）**
   - 描述：开发完成，等待审查
   - 自动化：当PR被创建时自动移到此列

4. **✅ Done（已完成）**
   - 描述：已完成的任务
   - 自动化：当Issue或PR被关闭时自动移到此列

### 设置过滤器

#### 按优先级过滤
- P0任务: `label:"P0-MVP"`
- P1任务: `label:"P1-v1.0"`
- P2任务: `label:"P2-v1.5"`

#### 按角色过滤
- 教程内容: `label:"📚教程必需"`
- 维护任务: `label:"🔧维护者"`

#### 按里程碑过滤
- MVP: `milestone:"v0.5 MVP"`
- v1.0: `milestone:"v1.0 Release"`

---

## 创建Issues

### 第一批Issues（P0-MVP）

按照 `.github/issues-p0-mvp.md` 中的列表创建Issues。

#### 创建Issue的标准流程

1. 点击 **Issues** → **New issue**
2. 选择 **📋 任务开发** 模板
3. 填写表单：
   - **Title**: 简短清晰的标题
   - **Priority**: 选择P0-MVP
   - **Role**: 选择对应角色
   - **Type**: 选择任务类型
   - **Stage**: 选择所属阶段
   - **Description**: 复制详细描述
   - **Deliverables**: 列出交付物
   - **Acceptance**: 列出验收标准
   - **Dependencies**: 填写依赖的Issue编号
4. 添加Labels：
   - 优先级标签（如P0-MVP）
   - 角色标签（如📚教程必需）
   - 类型标签（如代码、文档）
5. 设置Milestone：v0.5 MVP
6. 分配给自己（如果开始工作）
7. 添加到Project看板的Backlog列
8. 提交Issue

### 推荐的创建顺序

**第一批（基础设施，必须最先完成）**:
1. Issue #1: 创建项目目录结构
2. Issue #2: 编写基础依赖文件
3. Issue #3: 开发环境安装脚本
4. Issue #4: 模型下载脚本
5. Issue #5: 基础工具函数库

**第二批（文档）**:
6. Issue #6: 快速开始文档
7. Issue #7: 更新README.md

**第三批（核心示例）**:
8. Issue #8-11: 模型推理示例
12. Issue #12: LoRA微调示例
13. Issue #13-14: NVIDIA部署示例

**最后（验收）**:
15. Issue #15: MVP验收测试

---

## 使用GitHub CLI批量创建Issues

创建脚本 `scripts/create_issues.sh`:

```bash
#!/bin/bash
# 批量创建P0 Issues

# Issue #1
gh issue create \
  --title "[P0] 创建项目目录结构" \
  --body "创建完整的项目目录结构，严格按照设计文档第3.1节的规定。

**交付物**:
- docs/ 目录及7个子目录
- code/ 目录及5个子目录
- 其他必要目录

**验收标准**:
- 目录结构100%符合设计文档第3章
- 所有目录使用中划线命名" \
  --label "P0-MVP,🔧维护者,脚本" \
  --milestone "v0.5 MVP"

# Issue #2
gh issue create \
  --title "[P0] 编写基础依赖文件" \
  --body "创建Python依赖管理文件和基础配置文件。

**交付物**:
- requirements.txt
- requirements-dev.txt
- environment.yml
- setup.py
- pyproject.toml
- .gitignore

**验收标准**:
- 依赖文件包含所有必需的包
- 版本号明确锁定" \
  --label "P0-MVP,📚教程必需,文档" \
  --milestone "v0.5 MVP"

# ... 继续创建其他Issues
```

---

## 工作流程

### 1. 开始工作
1. 从Backlog选择一个Issue
2. 分配给自己
3. Issue自动移到"In Progress"
4. 创建新分支: `git checkout -b feature/issue-{number}`

### 2. 开发过程
1. 按照Issue的交付物清单开发
2. 定期commit和push
3. 更新Issue进度（评论）

### 3. 提交审查
1. 完成所有交付物
2. 自测通过验收标准
3. 创建Pull Request
4. PR自动移到"In Review"
5. 在PR中关联Issue: `Closes #X`

### 4. 审查和合并
1. 代码审查
2. 测试通过
3. 合并到main分支
4. Issue和PR自动移到"Done"

---

## 注意事项

1. **Issue编号管理**: GitHub自动分配编号，不能自定义
2. **依赖关系**: 使用"阻塞"标签标记被阻塞的Issue
3. **优先级**: 严格按P0 > P1 > P2 > P3顺序
4. **验收**: 每个Issue完成后必须满足验收标准
5. **文档同步**: 代码和文档同步开发

---

## 快速启动清单

- [ ] 创建所有Labels
- [ ] 创建所有Milestones
- [ ] 创建Projects看板
- [ ] 配置看板列和自动化
- [ ] 创建前5个基础Issue（#1-#5）
- [ ] 开始第一个Issue：创建项目目录结构
- [ ] 运行 `./scripts/init_project.sh`

---

**文档版本**: v1.0  
**最后更新**: 2025-11-01

