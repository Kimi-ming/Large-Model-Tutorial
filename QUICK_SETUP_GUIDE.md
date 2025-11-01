# 🚀 快速设置指南（5分钟完成）

> **重要**: 请在GitHub网页上完成以下操作  
> **仓库地址**: https://github.com/Kimi-ming/Large-Model-Tutorial

---

## 📋 方式一：使用脚本（推荐，如果已安装GitHub CLI）

### 1. 安装GitHub CLI（如果还没有）
- Windows: 下载 https://cli.github.com/ 
- Mac: `brew install gh`
- Linux: 参考 https://github.com/cli/cli/blob/trunk/docs/install_linux.md

### 2. 登录
```bash
gh auth login
```
选择：GitHub.com → HTTPS → Login with a web browser

### 3. 运行脚本
```bash
chmod +x scripts/create_github_labels.sh
./scripts/create_github_labels.sh
```

脚本会自动创建所有11个Labels！

---

## 📋 方式二：手动创建（无需安装，5-10分钟）

### 第1步：创建Labels（11个）

**打开**: https://github.com/Kimi-ming/Large-Model-Tutorial/labels

点击 **"New label"** 按钮，然后逐个创建：

#### ⭐ 优先级标签（4个）
复制粘贴以下内容：

1. **P0-MVP**
   - Color: `#d73a4a` (红色)
   - Description: `最小可用版本（v0.5）必需的任务`

2. **P1-v1.0**
   - Color: `#ff9800` (橙色)
   - Description: `v1.0正式版必需的任务`

3. **P2-v1.5**
   - Color: `#ffeb3b` (黄色)
   - Description: `v1.5增强版的任务`

4. **P3-future**
   - Color: `#4caf50` (绿色)
   - Description: `未来版本的任务`

#### 👥 角色标签（2个）

5. **📚教程必需**
   - Color: `#2196f3` (蓝色)
   - Description: `学习者核心内容开发`

6. **🔧维护者**
   - Color: `#9c27b0` (紫色)
   - Description: `仓库工程化和维护内容`

#### 📦 类型标签（5个）

7. **文档**
   - Color: `#0075ca`
   - Description: `文档相关任务`

8. **代码**
   - Color: `#008672`
   - Description: `代码开发任务`

9. **脚本**
   - Color: `#1d76db`
   - Description: `脚本工具开发`

10. **测试**
    - Color: `#d876e3`
    - Description: `测试相关任务`

11. **CI/CD**
    - Color: `#fbca04`
    - Description: `持续集成/部署配置`

✅ **完成**: 11个Labels创建完成！

---

### 第2步：创建Milestones（4个）

**打开**: https://github.com/Kimi-ming/Large-Model-Tutorial/milestones

点击 **"New milestone"** 按钮：

#### 1️⃣ v0.5 MVP
- **Title**: `v0.5 MVP`
- **Due date**: 2025年12月15日（建议）
- **Description**: 
  ```
  最小可用版本，让学习者能够快速上手，完成从模型选型到基础部署的完整学习路径。
  
  目标：
  - 基础框架搭建
  - 至少3个模型推理示例
  - LoRA微调示例
  - NVIDIA基础部署
  - 快速开始文档
  ```

#### 2️⃣ v1.0 Release
- **Title**: `v1.0 Release`
- **Due date**: 2026年3月1日（建议）
- **Description**:
  ```
  正式发布版本，提供完整的视觉大模型学习体系，覆盖主流应用场景。
  ```

#### 3️⃣ v1.5 Enhancement
- **Title**: `v1.5 Enhancement`
- **Due date**: （留空）
- **Description**:
  ```
  增强版本，根据用户反馈增强教程的广度和深度。
  ```

#### 4️⃣ v2.0 Future
- **Title**: `v2.0 Future`
- **Due date**: （留空）
- **Description**:
  ```
  重大更新版本，探索前沿技术和特殊需求。
  ```

✅ **完成**: 4个Milestones创建完成！

---

### 第3步：创建Projects看板

**打开**: https://github.com/Kimi-ming/Large-Model-Tutorial/projects

1. 点击 **"New project"**
2. 选择 **"Board"** 模板
3. 项目名称: `Large-Model-Tutorial Development`
4. 点击 **"Create project"**

#### 配置看板列：

默认有3列，修改为以下4列：

1. **📋 Backlog** (重命名"Todo")
   - 描述: 待开始的任务

2. **🚧 In Progress** (重命名"In Progress")
   - 描述: 正在进行的任务

3. **👀 In Review** (新建)
   - 描述: 开发完成，等待审查

4. **✅ Done** (保持"Done")
   - 描述: 已完成的任务

✅ **完成**: Projects看板创建完成！

---

### 第4步：创建第一个Issue（Issue #3）

**打开**: https://github.com/Kimi-ming/Large-Model-Tutorial/issues/new/choose

选择 **"📋 任务开发"** 模板，填写：

#### 基本信息
- **Title**: `[P0] 开发环境安装脚本`

#### 表单内容

**Priority**: 选择 `P0-MVP (v0.5最小可用版本)`

**Role**: 选择 `📚教程必需 (学习者核心内容)`

**Type**: 选择 `脚本`

**Stage**: 选择 `第一阶段：基础框架搭建`

**Description**:
```
开发 `scripts/setup.sh` 环境安装脚本，帮助学习者快速搭建环境。

## 背景
学习者需要一个一键安装脚本来快速配置开发环境，避免手动安装的复杂性。

## 目标
创建一个友好的安装脚本，支持环境检查、依赖安装和验证。
```

**Deliverables**:
```
- [ ] scripts/setup.sh - Linux/Mac环境安装脚本
- [ ] Python环境检查功能（版本 >= 3.8）
- [ ] GPU驱动检查功能（CUDA/ROCm，可选）
- [ ] 依赖自动安装（pip install -r requirements.txt）
- [ ] 环境验证功能（import torch等）
- [ ] 错误处理和友好提示
- [ ] 使用说明（README或注释）
```

**Acceptance**:
```
- [ ] 能在干净的Ubuntu 20.04/22.04环境中一键完成环境搭建
- [ ] 能在macOS上正常运行
- [ ] 脚本包含详细的日志输出
- [ ] 遇到错误时有明确的提示信息和解决建议
- [ ] 执行时间合理（<10分钟，不含大包下载）
```

**Dependencies**: （留空或填写 `无，基础任务已完成`）

**Notes**: （留空）

#### 提交后添加

1. 在右侧添加 **Labels**:
   - `P0-MVP`
   - `📚教程必需`
   - `脚本`

2. 设置 **Milestone**: `v0.5 MVP`

3. 添加到 **Project**: `Large-Model-Tutorial Development` → Backlog列

4. **Assign**: 分配给自己（如果要开始工作）

✅ **完成**: Issue #3 创建完成！

---

## 🎯 后续Issues创建

参考 `.github/issues-p0-mvp.md` 文件，按照相同方式创建：

- **Issue #4**: 模型下载脚本
- **Issue #5**: 基础工具函数库  
- **Issue #6**: 快速开始文档
- **Issue #7**: 更新README.md

---

## ✅ 检查清单

完成后检查：

- [ ] Labels: 11个标签已创建
- [ ] Milestones: 4个里程碑已创建
- [ ] Projects: 看板已配置（4列）
- [ ] Issue #3: 已创建并正确标记
- [ ] 可选：创建Issue #4-#7

---

## 🚀 开始开发

完成以上设置后，就可以开始Issue #3的开发了！

```bash
# 创建功能分支
git checkout -b feature/issue-3-setup-script

# 开发 scripts/setup.sh
# ... 编写代码 ...

# 提交
git add scripts/setup.sh
git commit -m "feat: 添加环境安装脚本

- 支持Python环境检查
- 支持GPU驱动检查
- 自动安装依赖
- 环境验证功能

Closes #3"

# 推送
git push origin feature/issue-3-setup-script
```

然后在GitHub创建Pull Request → 审查 → 合并 → Issue自动关闭！

---

## 📞 需要帮助？

- 📖 详细指南: `.github/SETUP_GUIDE.md`
- 📋 Issue模板: `.github/issues-p0-mvp.md`
- 📝 TODO清单: `教程开发-TODO清单.md`

---

**预计完成时间**: 5-15分钟（手动创建Labels + Milestones + Projects + 1个Issue）

祝开发顺利！🎉

