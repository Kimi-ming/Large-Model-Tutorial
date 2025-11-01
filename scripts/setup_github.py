#!/usr/bin/env python3
"""
自动设置GitHub仓库的脚本
包括：Labels, Milestones, Issues
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta

# GitHub配置
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO_OWNER = 'Kimi-ming'
REPO_NAME = 'Large-Model-Tutorial'
BASE_URL = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}'

def get_headers():
    """获取请求头"""
    if not GITHUB_TOKEN:
        print("❌ 错误: 未找到GITHUB_TOKEN环境变量")
        print("请设置: export GITHUB_TOKEN=your_token")
        print("或访问: https://github.com/settings/tokens 创建token")
        sys.exit(1)
    
    return {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

def create_labels():
    """创建Labels"""
    print("📌 创建Labels...")
    
    labels = [
        # 优先级标签
        {'name': 'P0-MVP', 'color': 'd73a4a', 'description': '最小可用版本（v0.5）必需的任务'},
        {'name': 'P1-v1.0', 'color': 'ff9800', 'description': 'v1.0正式版必需的任务'},
        {'name': 'P2-v1.5', 'color': 'ffeb3b', 'description': 'v1.5增强版的任务'},
        {'name': 'P3-future', 'color': '4caf50', 'description': '未来版本的任务'},
        
        # 角色标签
        {'name': '📚教程必需', 'color': '2196f3', 'description': '学习者核心内容开发'},
        {'name': '🔧维护者', 'color': '9c27b0', 'description': '仓库工程化和维护内容'},
        
        # 类型标签
        {'name': '文档', 'color': '0075ca', 'description': '文档相关任务'},
        {'name': '代码', 'color': '008672', 'description': '代码开发任务'},
        {'name': '脚本', 'color': '1d76db', 'description': '脚本工具开发'},
        {'name': '测试', 'color': 'd876e3', 'description': '测试相关任务'},
        {'name': 'CI/CD', 'color': 'fbca04', 'description': '持续集成/部署配置'},
    ]
    
    created = 0
    for label in labels:
        try:
            response = requests.post(
                f'{BASE_URL}/labels',
                headers=get_headers(),
                json=label
            )
            if response.status_code == 201:
                print(f"  ✅ 创建Label: {label['name']}")
                created += 1
            elif response.status_code == 422:
                print(f"  ⚠️  Label已存在: {label['name']}")
            else:
                print(f"  ❌ 创建失败: {label['name']} - {response.json().get('message')}")
        except Exception as e:
            print(f"  ❌ 错误: {label['name']} - {e}")
    
    print(f"✅ Labels创建完成: {created}个新建")
    return created

def create_milestones():
    """创建Milestones"""
    print("\n🎯 创建Milestones...")
    
    # 计算截止日期
    today = datetime.now()
    mvp_due = (today + timedelta(weeks=6)).strftime('%Y-%m-%dT00:00:00Z')
    v10_due = (today + timedelta(weeks=20)).strftime('%Y-%m-%dT00:00:00Z')
    
    milestones = [
        {
            'title': 'v0.5 MVP',
            'description': '最小可用版本，让学习者能够快速上手，完成从模型选型到基础部署的完整学习路径。\n\n目标：\n- 基础框架搭建\n- 至少3个模型推理示例\n- LoRA微调示例\n- NVIDIA基础部署\n- 快速开始文档',
            'due_on': mvp_due,
            'state': 'open'
        },
        {
            'title': 'v1.0 Release',
            'description': '正式发布版本，提供完整的视觉大模型学习体系，覆盖主流应用场景。',
            'due_on': v10_due,
            'state': 'open'
        },
        {
            'title': 'v1.5 Enhancement',
            'description': '增强版本，根据用户反馈增强教程的广度和深度。',
            'state': 'open'
        },
        {
            'title': 'v2.0 Future',
            'description': '重大更新版本，探索前沿技术和特殊需求。',
            'state': 'open'
        }
    ]
    
    created = 0
    milestone_map = {}
    
    for milestone in milestones:
        try:
            response = requests.post(
                f'{BASE_URL}/milestones',
                headers=get_headers(),
                json=milestone
            )
            if response.status_code == 201:
                data = response.json()
                milestone_map[milestone['title']] = data['number']
                print(f"  ✅ 创建Milestone: {milestone['title']}")
                created += 1
            else:
                print(f"  ❌ 创建失败: {milestone['title']} - {response.json().get('message')}")
        except Exception as e:
            print(f"  ❌ 错误: {milestone['title']} - {e}")
    
    print(f"✅ Milestones创建完成: {created}个新建")
    return milestone_map

def create_issue(issue_data, milestone_number=None):
    """创建单个Issue"""
    try:
        if milestone_number:
            issue_data['milestone'] = milestone_number
        
        response = requests.post(
            f'{BASE_URL}/issues',
            headers=get_headers(),
            json=issue_data
        )
        
        if response.status_code == 201:
            data = response.json()
            print(f"  ✅ 创建Issue #{data['number']}: {issue_data['title']}")
            return data['number']
        else:
            print(f"  ❌ 创建失败: {issue_data['title']} - {response.json().get('message')}")
            return None
    except Exception as e:
        print(f"  ❌ 错误: {issue_data['title']} - {e}")
        return None

def create_issues(milestone_map):
    """创建Issues"""
    print("\n📋 创建Issues (前5个P0任务)...")
    
    issues = [
        {
            'title': '[P0] 开发环境安装脚本',
            'body': '''## 任务描述
开发 `scripts/setup.sh` 环境安装脚本，帮助学习者快速搭建环境。

## 背景
学习者需要一个一键安装脚本来快速配置开发环境，避免手动安装的复杂性。

## 交付物
- [ ] `scripts/setup.sh` - Linux/Mac环境安装脚本
- [ ] Python环境检查功能（版本 >= 3.8）
- [ ] GPU驱动检查功能（CUDA/ROCm，可选）
- [ ] 依赖自动安装（pip install -r requirements.txt）
- [ ] 环境验证功能（import torch等）
- [ ] 错误处理和友好提示
- [ ] 使用说明（README或注释）

## 验收标准
- [ ] 能在干净的Ubuntu 20.04/22.04环境中一键完成环境搭建
- [ ] 能在macOS上正常运行
- [ ] 脚本包含详细的日志输出
- [ ] 遇到错误时有明确的提示信息和解决建议
- [ ] 执行时间合理（<10分钟，不含大包下载）

## 依赖
无（基础任务Issue #1和#2已完成）

## 阶段
第一阶段：基础框架搭建
''',
            'labels': ['P0-MVP', '📚教程必需', '脚本']
        },
        {
            'title': '[P0] 模型下载脚本',
            'body': '''## 任务描述
开发 `scripts/download_models.sh` 模型下载脚本。

## 交付物
- [ ] `scripts/download_models.sh` - 模型下载脚本
- [ ] 支持从HuggingFace Hub下载
- [ ] 支持断点续传
- [ ] 支持镜像源配置（国内加速）
- [ ] 下载进度显示
- [ ] 支持下载CLIP、SAM、LLaVA三个模型

## 验收标准
- [ ] 能成功下载至少3个常用模型（CLIP、SAM、LLaVA）
- [ ] 下载失败时能自动重试
- [ ] 有清晰的进度提示

## 阶段
第一阶段：基础框架搭建
''',
            'labels': ['P0-MVP', '📚教程必需', '脚本']
        },
        {
            'title': '[P0] 基础工具函数库',
            'body': '''## 任务描述
开发 `code/utils/` 基础工具函数库。

## 交付物
- [ ] `code/utils/model_loader.py` - 模型加载工具
- [ ] `code/utils/data_processor.py` - 数据处理工具
- [ ] `code/utils/config_parser.py` - 配置解析工具
- [ ] `code/utils/logger.py` - 日志工具
- [ ] `code/utils/__init__.py` - 包初始化
- [ ] 每个模块包含docstring和类型注解
- [ ] 简单的单元测试

## 验收标准
- [ ] 工具函数能正常导入和使用
- [ ] 代码符合PEP 8规范
- [ ] 基础测试通过

## 阶段
第一阶段：基础框架搭建
''',
            'labels': ['P0-MVP', '📚教程必需', '代码']
        },
        {
            'title': '[P0] 快速开始文档',
            'body': '''## 任务描述
编写快速开始文档。

## 交付物
- [ ] `docs/05-使用说明/01-环境安装指南.md`
  - 系统要求
  - 安装步骤
  - 常见问题
- [ ] `docs/05-使用说明/02-快速开始.md`
  - 第一个示例：运行预训练模型推理
  - 环境验证
  - 后续学习路径指引

## 验收标准
- [ ] 文档清晰易懂
- [ ] 步骤可复现
- [ ] 包含学习目标和先修要求

## 阶段
第一阶段：基础框架搭建
''',
            'labels': ['P0-MVP', '📚教程必需', '文档']
        },
        {
            'title': '[P0] 更新README.md',
            'body': '''## 任务描述
更新项目主页README.md，提供完整的项目介绍。

## 交付物
- [ ] 教程简介
- [ ] 目标读者
- [ ] 学习路径图
- [ ] 快速开始指南
- [ ] 目录结构说明
- [ ] 贡献指南链接
- [ ] Badges（MIT License等）

## 验收标准
- [ ] README内容完整、专业
- [ ] 有吸引力，让学习者想要深入学习
- [ ] 链接都有效

## 依赖
Issue #6 (快速开始文档)

## 阶段
第一阶段：基础框架搭建
''',
            'labels': ['P0-MVP', '📚教程必需', '文档']
        }
    ]
    
    created = 0
    mvp_milestone = milestone_map.get('v0.5 MVP')
    
    for issue in issues:
        issue_num = create_issue(issue, mvp_milestone)
        if issue_num:
            created += 1
    
    print(f"✅ Issues创建完成: {created}个新建")
    return created

def main():
    """主函数"""
    print("=" * 50)
    print("  GitHub仓库自动设置")
    print("  仓库: {}/{}".format(REPO_OWNER, REPO_NAME))
    print("=" * 50)
    print()
    
    try:
        # 1. 创建Labels
        labels_created = create_labels()
        
        # 2. 创建Milestones
        milestone_map = create_milestones()
        
        # 3. 创建Issues
        issues_created = create_issues(milestone_map)
        
        # 总结
        print("\n" + "=" * 50)
        print("  ✅ 设置完成！")
        print("=" * 50)
        print(f"Labels创建: {labels_created}个")
        print(f"Milestones创建: {len(milestone_map)}个")
        print(f"Issues创建: {issues_created}个")
        print()
        print("查看结果:")
        print(f"  Labels: https://github.com/{REPO_OWNER}/{REPO_NAME}/labels")
        print(f"  Milestones: https://github.com/{REPO_OWNER}/{REPO_NAME}/milestones")
        print(f"  Issues: https://github.com/{REPO_OWNER}/{REPO_NAME}/issues")
        print()
        print("🚀 现在可以开始开发了！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

