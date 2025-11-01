#!/bin/bash
# 创建GitHub Labels的脚本
# 使用前请确保已安装并登录GitHub CLI: gh auth login

echo "=========================================="
echo "  创建GitHub Labels"
echo "=========================================="
echo ""

# 检查gh是否安装
if ! command -v gh &> /dev/null; then
    echo "❌ 错误: GitHub CLI (gh) 未安装"
    echo "请访问: https://cli.github.com/ 下载安装"
    echo ""
    echo "或者手动在GitHub网页上创建Labels:"
    echo "https://github.com/Kimi-ming/Large-Model-Tutorial/labels"
    exit 1
fi

# 检查是否登录
if ! gh auth status &> /dev/null; then
    echo "❌ 错误: 未登录GitHub CLI"
    echo "请运行: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI 已就绪"
echo ""

# 创建优先级标签
echo "📌 创建优先级标签..."
gh label create "P0-MVP" --color "d73a4a" --description "最小可用版本（v0.5）必需的任务" --force
gh label create "P1-v1.0" --color "ff9800" --description "v1.0正式版必需的任务" --force
gh label create "P2-v1.5" --color "ffeb3b" --description "v1.5增强版的任务" --force
gh label create "P3-future" --color "4caf50" --description "未来版本的任务" --force
echo "✅ 优先级标签创建完成"
echo ""

# 创建角色标签
echo "👥 创建角色标签..."
gh label create "📚教程必需" --color "2196f3" --description "学习者核心内容开发" --force
gh label create "🔧维护者" --color "9c27b0" --description "仓库工程化和维护内容" --force
echo "✅ 角色标签创建完成"
echo ""

# 创建类型标签
echo "📋 创建类型标签..."
gh label create "文档" --color "0075ca" --description "文档相关任务" --force
gh label create "代码" --color "008672" --description "代码开发任务" --force
gh label create "脚本" --color "1d76db" --description "脚本工具开发" --force
gh label create "测试" --color "d876e3" --description "测试相关任务" --force
gh label create "CI/CD" --color "fbca04" --description "持续集成/部署配置" --force
echo "✅ 类型标签创建完成"
echo ""

echo "=========================================="
echo "  所有Labels创建完成！"
echo "=========================================="
echo ""
echo "查看结果: https://github.com/Kimi-ming/Large-Model-Tutorial/labels"

