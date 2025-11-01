#!/bin/bash
###############################################################################
# 视觉大模型教程 - 模型下载脚本
#
# 功能：
#   - 从HuggingFace Hub下载预训练模型
#   - 支持断点续传
#   - 支持镜像源配置（国内加速）
#   - 显示下载进度
#   - 支持批量下载
#
# 支持的模型：
#   - CLIP (openai/clip-vit-base-patch32)
#   - SAM (facebook/sam-vit-base)
#   - LLaVA (liuhaotian/llava-v1.5-7b)
#   - BLIP-2 (Salesforce/blip2-opt-2.7b)
#   - Qwen-VL (Qwen/Qwen-VL-Chat)
#
# 使用方法：
#   ./scripts/download_models.sh [选项] [模型名称]
#
# 示例：
#   # 交互式下载（推荐）
#   ./scripts/download_models.sh
#
#   # 下载指定模型
#   ./scripts/download_models.sh clip
#   ./scripts/download_models.sh sam llava
#
#   # 下载所有P0模型
#   ./scripts/download_models.sh --all-p0
#
###############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
MODELS_DIR="models"
USE_MIRROR=false
MIRROR_URL="https://hf-mirror.com"
HF_ENDPOINT="https://huggingface.co"
NON_INTERACTIVE=false

# 检查非交互模式环境变量
if [ "${CI:-false}" = "true" ] || [ "${DEBIAN_FRONTEND:-}" = "noninteractive" ] || [ -n "${FORCE_NON_INTERACTIVE:-}" ]; then
    NON_INTERACTIVE=true
fi

###############################################################################
# 工具函数
###############################################################################

print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

print_step() {
    echo -e "${BLUE}[步骤]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[信息]${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 显示帮助
show_help() {
    cat << EOF
视觉大模型教程 - 模型下载脚本

使用方法：
    ./scripts/download_models.sh [选项] [模型名称...]

选项：
    --all-p0            下载所有P0（MVP）阶段需要的模型
    --mirror            使用HuggingFace镜像源（国内推荐）
    --models-dir DIR    指定模型保存目录（默认: models/）
    --yes, -y           非交互模式（自动确认所有提示）
    --help              显示此帮助信息

支持的模型名称：
    clip                OpenAI CLIP (图文多模态，轻量级)
    sam                 Meta SAM (图像分割)
    llava               LLaVA 1.5 (视觉问答，7B)
    blip2               BLIP-2 (图像理解，2.7B)
    qwen-vl             Qwen-VL (通义千问视觉版)

示例：
    # 交互式选择下载
    ./scripts/download_models.sh

    # 下载CLIP模型
    ./scripts/download_models.sh clip

    # 下载多个模型
    ./scripts/download_models.sh clip sam llava

    # 使用镜像源下载所有P0模型（国内推荐）
    ./scripts/download_models.sh --mirror --all-p0

说明：
    - 模型文件较大（几百MB到几十GB），请确保磁盘空间充足
    - 首次下载需要较长时间，请耐心等待
    - 如果网络中断，重新运行脚本会自动断点续传
    - 国内用户建议使用 --mirror 选项加速下载

EOF
    exit 0
}

###############################################################################
# 模型配置
###############################################################################

# 定义模型信息
# 格式: 简称|HuggingFace仓库ID|描述|大小|P0标记|依赖说明
declare -A MODEL_INFO
MODEL_INFO=(
    ["clip"]="openai/clip-vit-base-patch32|OpenAI CLIP - 图文多模态模型|~600MB|P0|无额外依赖"
    ["sam"]="facebook/sam-vit-base|Meta SAM - 图像分割基础模型|~360MB|P0|需要: pip install segment-anything"
    ["llava"]="liuhaotian/llava-v1.5-7b|LLaVA 1.5 - 视觉问答模型 7B|~13GB|P0|无额外依赖"
    ["blip2"]="Salesforce/blip2-opt-2.7b|BLIP-2 - 图像理解模型 2.7B|~5.5GB|P1|无额外依赖"
    ["qwen-vl"]="Qwen/Qwen-VL-Chat|通义千问视觉版|~9GB|P1|需要: pip install transformers_stream_generator"
)

# P0模型列表（MVP阶段必需）
P0_MODELS=("clip" "sam" "llava")

###############################################################################
# 下载函数
###############################################################################

check_dependencies() {
    print_step "检查依赖..."
    
    # 检查Python
    if ! command_exists python3; then
        print_error "未找到python3，请先运行 ./scripts/setup.sh"
        exit 1
    fi
    
    # 检查huggingface_hub库
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        print_warning "未安装huggingface_hub库"
        print_step "正在安装huggingface_hub..."
        pip install -U huggingface_hub
    fi
    
    print_success "依赖检查完成"
}

setup_mirror() {
    if [ "$USE_MIRROR" = true ]; then
        print_info "使用HuggingFace镜像源: $MIRROR_URL"
        export HF_ENDPOINT="$MIRROR_URL"
    else
        # 自动检测是否在国内
        if curl -s --connect-timeout 3 http://www.google.com > /dev/null 2>&1; then
            print_info "使用官方HuggingFace源"
        else
            print_warning "检测到网络可能在国内，建议使用 --mirror 选项"
            
            if [ "$NON_INTERACTIVE" = true ]; then
                print_info "非交互模式：自动启用镜像源"
                USE_MIRROR=true
                export HF_ENDPOINT="$MIRROR_URL"
                print_success "已启用镜像源"
            else
                read -p "是否使用镜像源加速下载？[Y/n] " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    USE_MIRROR=true
                    export HF_ENDPOINT="$MIRROR_URL"
                    print_success "已启用镜像源"
                fi
            fi
        fi
    fi
}

download_model() {
    local model_key="$1"
    
    # 解析模型信息
    local info="${MODEL_INFO[$model_key]}"
    if [ -z "$info" ]; then
        print_error "未知的模型: $model_key"
        return 1
    fi
    
    IFS='|' read -r repo_id description size priority dependencies <<< "$info"
    
    print_header "下载模型: $model_key"
    print_info "仓库: $repo_id"
    print_info "描述: $description"
    print_info "大小: $size"
    if [ "$dependencies" != "无额外依赖" ]; then
        print_warning "依赖: $dependencies"
    fi
    echo ""
    
    # 创建模型目录
    local model_dir="$MODELS_DIR/$model_key"
    mkdir -p "$model_dir"
    
    # 使用Python调用huggingface_hub下载
    print_step "开始下载（支持断点续传）..."
    
    python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download
from tqdm import tqdm

try:
    print("正在连接到HuggingFace...")
    
    # 下载模型
    local_dir = "$model_dir"
    repo_id = "$repo_id"
    
    print(f"下载 {repo_id} 到 {local_dir}")
    print("提示: 如果下载中断，重新运行即可继续\n")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.py",
            "tokenizer.*", "merges.txt", "vocab.*", "*.tiktoken",
            "preprocessor_config.json", "generation_config.json"
        ],
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    
    print("\n✓ 下载完成！")
    sys.exit(0)
    
except KeyboardInterrupt:
    print("\n\n用户中断下载")
    sys.exit(130)
except Exception as e:
    print(f"\n✗ 下载失败: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "模型 $model_key 下载完成"
        print_info "保存位置: $model_dir"
        return 0
    elif [ $exit_code -eq 130 ]; then
        print_warning "下载已中断"
        return 130
    else
        print_error "模型 $model_key 下载失败"
        return 1
    fi
}

###############################################################################
# 交互式界面
###############################################################################

show_model_list() {
    echo ""
    echo "可下载的模型列表："
    echo "=================================================="
    printf "%-12s %-10s %-50s %s\n" "模型" "优先级" "描述" "大小"
    echo "--------------------------------------------------"
    
    for key in "${!MODEL_INFO[@]}"; do
        IFS='|' read -r repo_id description size priority <<< "${MODEL_INFO[$key]}"
        printf "%-12s %-10s %-50s %s\n" "$key" "[$priority]" "$description" "$size"
    done | sort
    
    echo "=================================================="
    echo ""
    echo "说明:"
    echo "  [P0] - MVP阶段必需（推荐优先下载）"
    echo "  [P1] - v1.0阶段需要"
    echo ""
}

interactive_download() {
    print_header "交互式模型下载"
    
    show_model_list
    
    echo "请选择要下载的模型（多个模型用空格分隔）："
    echo "  输入模型名称 (如: clip sam llava)"
    echo "  或输入 'p0' 下载所有P0模型"
    echo "  或输入 'all' 下载所有模型"
    echo "  或输入 'q' 退出"
    echo ""
    read -p "请输入: " -r user_input
    
    if [ -z "$user_input" ]; then
        print_warning "未选择任何模型"
        exit 0
    fi
    
    case "$user_input" in
        q|Q|quit|exit)
            print_info "退出下载"
            exit 0
            ;;
        p0|P0)
            MODELS_TO_DOWNLOAD=("${P0_MODELS[@]}")
            ;;
        all|ALL)
            MODELS_TO_DOWNLOAD=("${!MODEL_INFO[@]}")
            ;;
        *)
            read -ra MODELS_TO_DOWNLOAD <<< "$user_input"
            ;;
    esac
    
    # 显示下载计划
    echo ""
    print_info "准备下载以下模型:"
    for model in "${MODELS_TO_DOWNLOAD[@]}"; do
        echo "  - $model"
    done
    echo ""
    
    if [ "$NON_INTERACTIVE" = false ]; then
        read -p "确认下载？[Y/n] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_info "已取消"
            exit 0
        fi
    else
        print_info "非交互模式：自动确认下载"
    fi
}

###############################################################################
# 主流程
###############################################################################

main() {
    print_header "视觉大模型教程 - 模型下载工具"
    
    # 解析参数
    MODELS_TO_DOWNLOAD=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                ;;
            --all-p0)
                MODELS_TO_DOWNLOAD=("${P0_MODELS[@]}")
                shift
                ;;
            --mirror)
                USE_MIRROR=true
                shift
                ;;
            --models-dir)
                MODELS_DIR="$2"
                shift 2
                ;;
            --yes|-y)
                NON_INTERACTIVE=true
                shift
                ;;
            -*)
                print_error "未知选项: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
            *)
                MODELS_TO_DOWNLOAD+=("$1")
                shift
                ;;
        esac
    done
    
    # 检查依赖
    check_dependencies
    
    # 设置镜像源
    setup_mirror
    
    # 创建模型目录
    mkdir -p "$MODELS_DIR"
    print_info "模型保存目录: $MODELS_DIR"
    echo ""
    
    # 如果没有指定模型，进入交互模式
    if [ ${#MODELS_TO_DOWNLOAD[@]} -eq 0 ]; then
        interactive_download
    fi
    
    # 下载模型
    local success_count=0
    local fail_count=0
    local interrupted=false
    
    for model in "${MODELS_TO_DOWNLOAD[@]}"; do
        if download_model "$model"; then
            ((success_count++))
        else
            exit_code=$?
            if [ $exit_code -eq 130 ]; then
                interrupted=true
                break
            else
                ((fail_count++))
            fi
        fi
        echo ""
    done
    
    # 显示总结
    if [ "$interrupted" = true ]; then
        print_header "下载已中断"
        print_info "已下载: $success_count 个模型"
        print_info "重新运行脚本可继续未完成的下载"
        exit 130
    else
        print_header "下载完成"
        print_success "成功: $success_count 个模型"
        
        if [ $fail_count -gt 0 ]; then
            print_error "失败: $fail_count 个模型"
        fi
        
        echo ""
        print_info "模型位置: $MODELS_DIR/"
        echo ""
        print_info "下一步："
        echo "  1. 查看快速开始文档"
        echo "     cat docs/05-使用说明/02-快速开始.md"
        echo ""
        echo "  2. 运行推理示例"
        echo "     python code/01-model-evaluation/examples/clip_inference.py"
        echo ""
        
        if [ $fail_count -gt 0 ]; then
            exit 1
        fi
    fi
}

# 脚本入口
main "$@"

