#!/bin/bash
# 项目结构初始化脚本
# 使用方式：./scripts/init_project.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  视觉大模型教程 - 项目结构初始化"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[步骤]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

# 创建docs目录结构
print_step "创建docs目录结构..."
mkdir -p docs/01-模型调研与选型
mkdir -p docs/02-模型微调技术
mkdir -p docs/03-数据集准备
mkdir -p docs/04-多平台部署
mkdir -p docs/05-使用说明
mkdir -p docs/06-实际应用场景
mkdir -p docs/07-高级主题
print_success "docs目录创建完成"

# 创建code目录结构
print_step "创建code目录结构..."
mkdir -p code/01-model-evaluation/examples
mkdir -p code/01-model-evaluation/benchmark
mkdir -p code/02-fine-tuning/full-finetuning
mkdir -p code/02-fine-tuning/lora
mkdir -p code/02-fine-tuning/qlora
mkdir -p code/02-fine-tuning/peft-methods
mkdir -p code/02-fine-tuning/tools
mkdir -p code/03-data-processing/download
mkdir -p code/03-data-processing/preprocessing
mkdir -p code/03-data-processing/augmentation
mkdir -p code/03-data-processing/custom
mkdir -p code/04-deployment/nvidia/basic
mkdir -p code/04-deployment/nvidia/onnx
mkdir -p code/04-deployment/nvidia/tensorrt
mkdir -p code/04-deployment/nvidia/triton
mkdir -p code/04-deployment/nvidia/vllm
mkdir -p code/04-deployment/huawei
mkdir -p code/04-deployment/docker
mkdir -p code/04-deployment/api-server/api
mkdir -p code/04-deployment/api-server/core
mkdir -p code/05-applications/retail
mkdir -p code/05-applications/medical
mkdir -p code/05-applications/traffic
mkdir -p code/05-applications/quality-inspection
mkdir -p code/05-applications/content-moderation
mkdir -p code/05-applications/security
mkdir -p code/utils
print_success "code目录创建完成"

# 创建notebooks目录
print_step "创建notebooks目录..."
mkdir -p notebooks
print_success "notebooks目录创建完成"

# 创建configs目录结构
print_step "创建configs目录结构..."
mkdir -p configs/models
mkdir -p configs/training
mkdir -p configs/deployment
print_success "configs目录创建完成"

# 创建scripts目录（如果不存在）
print_step "检查scripts目录..."
mkdir -p scripts
print_success "scripts目录确认"

# 创建tests目录结构
print_step "创建tests目录结构..."
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/e2e
print_success "tests目录创建完成"

# 创建assets目录结构
print_step "创建assets目录结构..."
mkdir -p assets/images
mkdir -p assets/videos
mkdir -p assets/templates
print_success "assets目录创建完成"

# 创建docker目录
print_step "创建docker目录..."
mkdir -p docker
print_success "docker目录创建完成"

# 创建.github目录（如果不存在）
print_step "检查.github目录..."
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
print_success ".github目录确认"

# 创建README文件占位符
print_step "创建README占位符..."
for dir in code/01-model-evaluation code/02-fine-tuning code/03-data-processing code/04-deployment code/05-applications; do
    if [ ! -f "$dir/README.md" ]; then
        echo "# $(basename $dir)" > "$dir/README.md"
        echo "" >> "$dir/README.md"
        echo "详细说明待补充。" >> "$dir/README.md"
    fi
done
print_success "README占位符创建完成"

# 创建__init__.py文件
print_step "创建Python包初始化文件..."
find code -type d -exec touch {}/__init__.py \; 2>/dev/null || true
print_success "Python包初始化完成"

# 显示目录结构
echo ""
echo "=========================================="
echo "  项目结构创建完成！"
echo "=========================================="
echo ""
echo "目录结构预览："
echo ""
tree -L 2 -d . 2>/dev/null || find . -type d -maxdepth 2 | grep -v "^\./\." | sort

echo ""
echo "=========================================="
echo "  下一步操作："
echo "=========================================="
echo "1. 运行 'git status' 查看新创建的目录"
echo "2. 运行 './scripts/setup.sh' 安装开发环境（脚本待创建）"
echo "3. 开始第一个任务：编写基础依赖文件"
echo ""

