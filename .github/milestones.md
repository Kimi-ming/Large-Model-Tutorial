# GitHub Milestones 配置

## 使用说明
在GitHub仓库中手动创建以下Milestones，或使用GitHub CLI命令。

## Milestones列表

### 1. v0.5 MVP
- **描述**: 最小可用版本，让学习者能够快速上手，完成从模型选型到基础部署的完整学习路径
- **截止日期**: 根据实际情况设定
- **包含任务**: 所有标记为 `P0-MVP` 的Issues
- **目标**:
  - 基础框架搭建完成
  - 至少3个模型推理示例可运行
  - LoRA微调完整示例可运行
  - NVIDIA平台基础部署可用
  - 快速开始文档完成

### 2. v1.0 Release
- **描述**: 正式发布版本，提供完整的视觉大模型学习体系，覆盖主流应用场景
- **截止日期**: 根据实际情况设定
- **包含任务**: 所有标记为 `P0-MVP` 和 `P1-v1.0` 的Issues
- **目标**:
  - 所有核心教程内容完成
  - 至少3个实际应用场景完成
  - API服务和容器化部署完成
  - 完整文档和测试覆盖
  - 社区基础设施就位

### 3. v1.5 Enhancement
- **描述**: 增强版本，根据用户反馈增强教程的广度和深度
- **截止日期**: 根据实际情况设定
- **包含任务**: 所有标记为 `P2-v1.5` 的Issues
- **目标**:
  - 更多部署平台支持（AMD、国产GPU、边缘设备）
  - 更多应用场景（工业质检、内容审核、安防）
  - 高级主题内容
  - 性能优化示例

### 4. v2.0 Future
- **描述**: 重大更新版本，探索前沿技术和特殊需求
- **截止日期**: 待定
- **包含任务**: 所有标记为 `P3-future` 的Issues
- **目标**:
  - 视频理解模型
  - 3D视觉模型
  - 联邦学习
  - 其他前沿技术

## GitHub CLI 创建命令示例

```bash
# 创建v0.5 MVP里程碑
gh api repos/:owner/:repo/milestones -f title="v0.5 MVP" -f description="最小可用版本，让学习者能够快速上手" -f state="open"

# 创建v1.0 Release里程碑
gh api repos/:owner/:repo/milestones -f title="v1.0 Release" -f description="正式发布版本，提供完整的视觉大模型学习体系" -f state="open"

# 创建v1.5 Enhancement里程碑
gh api repos/:owner/:repo/milestones -f title="v1.5 Enhancement" -f description="增强版本，根据用户反馈增强教程的广度和深度" -f state="open"

# 创建v2.0 Future里程碑
gh api repos/:owner/:repo/milestones -f title="v2.0 Future" -f description="重大更新版本，探索前沿技术和特殊需求" -f state="open"
```

## 手动创建步骤

1. 进入GitHub仓库页面
2. 点击 "Issues" 标签
3. 点击 "Milestones" 
4. 点击 "New milestone"
5. 填入上述信息
6. 保存

