#!/bin/bash
###############################################################################
# 视觉大模型教程 - 开发环境自动安装脚本
# 
# 功能：
#   - 检查系统环境（OS、Python版本）
#   - 检测GPU环境（CUDA/ROCm，可选）
#   - 安装Python依赖包
#   - 验证环境是否正确配置
#
# 支持系统：
#   - Ubuntu 20.04/22.04/24.04
#   - macOS (Intel/Apple Silicon)
#   - 其他Linux发行版（实验性支持）
#
# 使用方法：
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
#
# 参数：
#   --skip-gpu-check    跳过GPU检测（CPU-only模式）
#   --no-verify         跳过环境验证
#   --yes, -y           非交互模式
#   --help              显示帮助信息
#
###############################################################################

set -euo pipefail  # 遇到错误立即退出，未定义变量报错，管道任一命令失败则失败

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
MIN_PYTHON_VERSION="3.8"
SKIP_GPU_CHECK=false
NO_VERIFY=false
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
    echo -e "${BLUE}[信息]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
视觉大模型教程 - 开发环境安装脚本

使用方法：
    ./scripts/setup.sh [选项]

选项：
    --skip-gpu-check    跳过GPU检测（适用于CPU-only环境）
    --no-verify         跳过最终的环境验证步骤
    --yes, -y           非交互模式（自动确认所有提示）
    --help              显示此帮助信息

示例：
    # 标准安装
    ./scripts/setup.sh

    # CPU-only安装（不检测GPU）
    ./scripts/setup.sh --skip-gpu-check

    # 快速安装（跳过验证）
    ./scripts/setup.sh --no-verify

说明：
    此脚本会自动：
    1. 检查Python版本（需要 >= 3.8）
    2. 检测GPU环境（CUDA/ROCm）
    3. 安装requirements.txt中的依赖
    4. 验证PyTorch等关键包是否正确安装

EOF
    exit 0
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-gpu-check)
                SKIP_GPU_CHECK=true
                shift
                ;;
            --no-verify)
                NO_VERIFY=true
                shift
                ;;
            --yes|-y)
                NON_INTERACTIVE=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                print_error "未知参数: $1"
                echo "使用 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 版本比较函数
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

###############################################################################
# 环境检查
###############################################################################

check_os() {
    print_step "检查操作系统..."
    
    OS_TYPE=$(uname -s)
    case "$OS_TYPE" in
        Linux*)
            print_success "操作系统: Linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                print_info "发行版: $NAME $VERSION"
            fi
            ;;
        Darwin*)
            print_success "操作系统: macOS"
            MACOS_VERSION=$(sw_vers -productVersion)
            print_info "版本: $MACOS_VERSION"
            ;;
        *)
            print_warning "未识别的操作系统: $OS_TYPE"
            print_warning "脚本可能无法正常工作，建议在Ubuntu 20.04+或macOS上运行"
            ;;
    esac
}

check_python() {
    print_step "检查Python版本..."
    
    if ! command_exists python3; then
        print_error "未找到python3命令"
        print_error "请先安装Python 3.8或更高版本"
        echo ""
        echo "Ubuntu/Debian安装方法："
        echo "  sudo apt update && sudo apt install python3 python3-pip python3-venv"
        echo ""
        echo "macOS安装方法："
        echo "  brew install python@3.11"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_info "Python版本: $PYTHON_VERSION"
    
    if ! version_ge "$PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
        print_error "Python版本过低（需要 >= $MIN_PYTHON_VERSION）"
        print_error "当前版本: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python版本符合要求"
}

check_pip() {
    print_step "检查pip..."
    
    if ! command_exists pip3 && ! python3 -m pip --version >/dev/null 2>&1; then
        print_error "未找到pip"
        print_error "请先安装pip"
        echo ""
        echo "Ubuntu/Debian安装方法："
        echo "  sudo apt install python3-pip"
        echo ""
        echo "macOS安装方法："
        echo "  python3 -m ensurepip --upgrade"
        exit 1
    fi
    
    PIP_VERSION=$(python3 -m pip --version | awk '{print $2}')
    print_success "pip版本: $PIP_VERSION"
}

check_gpu() {
    if [ "$SKIP_GPU_CHECK" = true ]; then
        print_warning "跳过GPU检测（--skip-gpu-check）"
        return 0
    fi
    
    print_step "检测GPU环境..."
    
    GPU_FOUND=false
    
    # 检测NVIDIA GPU
    if command_exists nvidia-smi; then
        print_success "检测到NVIDIA GPU"
        echo ""
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | \
        while IFS=',' read -r name driver memory; do
            print_info "GPU: $name"
            print_info "驱动版本: $driver"
            print_info "显存: $memory"
        done
        echo ""
        
        # 检测CUDA
        if command_exists nvcc; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
            print_success "CUDA版本: $CUDA_VERSION"
        else
            print_warning "未检测到CUDA toolkit"
            print_info "如需使用GPU训练，请安装CUDA toolkit"
            print_info "下载地址: https://developer.nvidia.com/cuda-downloads"
        fi
        
        GPU_FOUND=true
    fi
    
    # 检测AMD GPU (ROCm)
    if command_exists rocm-smi; then
        print_success "检测到AMD GPU (ROCm)"
        rocm-smi --showproductname 2>/dev/null || true
        GPU_FOUND=true
    fi
    
    # 检测Apple Silicon
    if [[ "$OS_TYPE" == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
        print_success "检测到Apple Silicon (支持Metal加速)"
        GPU_FOUND=true
    fi
    
    if [ "$GPU_FOUND" = false ]; then
        print_warning "未检测到GPU，将以CPU模式运行"
        print_info "CPU模式仅适合学习和小规模推理，不建议用于模型训练"
    fi
    
    echo ""
}

###############################################################################
# 依赖安装
###############################################################################

detect_network_region() {
    # 检测网络区域（是否在国内）
    if curl -s --connect-timeout 3 http://www.google.com > /dev/null 2>&1; then
        return 1  # 国外
    else
        return 0  # 国内
    fi
}

install_dependencies() {
    print_step "安装Python依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "未找到requirements.txt文件"
        print_error "请确保在项目根目录下运行此脚本"
        exit 1
    fi
    
    print_info "开始安装依赖（这可能需要几分钟）..."
    echo ""
    
    # 升级pip
    print_info "升级pip..."
    python3 -m pip install --upgrade pip setuptools wheel || true
    
    # 检测网络并选择镜像源
    local pip_index=""
    if detect_network_region; then
        print_info "检测到国内网络环境，使用清华镜像源加速"
        pip_index="-i https://pypi.tuna.tsinghua.edu.cn/simple"
    else
        print_info "使用官方PyPI源"
    fi
    
    # 安装依赖
    if python3 -m pip install -r requirements.txt $pip_index; then
        print_success "依赖安装完成"
    else
        print_warning "使用主镜像源安装失败，尝试备用镜像源..."
        
        # 尝试备用镜像源
        local backup_mirrors=(
            "https://pypi.tuna.tsinghua.edu.cn/simple"
            "https://mirrors.aliyun.com/pypi/simple"
            "https://pypi.mirrors.ustc.edu.cn/simple"
        )
        
        local success=false
        for mirror in "${backup_mirrors[@]}"; do
            print_info "尝试镜像源: $mirror"
            if python3 -m pip install -r requirements.txt -i "$mirror"; then
                print_success "使用镜像源 $mirror 安装成功"
                success=true
                break
            fi
        done
        
        if [ "$success" = false ]; then
            print_error "所有镜像源都安装失败"
            print_error "请检查网络连接或手动安装依赖"
            echo ""
            echo "手动安装方法："
            echo "  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
            exit 1
        fi
    fi
}

install_dev_dependencies() {
    if [ -f "requirements-dev.txt" ]; then
        print_step "安装开发依赖（可选）..."
        
        if [ "$NON_INTERACTIVE" = true ]; then
            print_info "非交互模式：跳过开发依赖安装"
            return 0
        fi
        
        read -p "是否安装开发依赖（用于代码检查、测试等）？[y/N] " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if python3 -m pip install -r requirements-dev.txt; then
                print_success "开发依赖安装完成"
            else
                print_warning "开发依赖安装失败，跳过"
            fi
        else
            print_info "跳过开发依赖安装"
        fi
    fi
}

###############################################################################
# 环境验证
###############################################################################

verify_environment() {
    if [ "$NO_VERIFY" = true ]; then
        print_warning "跳过环境验证（--no-verify）"
        return 0
    fi
    
    print_step "验证环境配置..."
    
    # 创建临时Python脚本进行验证
    VERIFY_SCRIPT=$(cat << 'EOF'
import sys
import importlib.util

def check_package(package_name, import_name=None):
    """检查包是否可以导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, "未安装"
        
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        return True, version
    except Exception as e:
        return False, str(e)

# 关键包列表
packages = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('transformers', 'transformers'),
    ('pillow', 'PIL'),
    ('numpy', 'numpy'),
]

print("\n关键包验证：")
print("-" * 50)

all_ok = True
for pkg_name, import_name in packages:
    ok, info = check_package(pkg_name, import_name)
    status = "✓" if ok else "✗"
    print(f"{status} {pkg_name:20s} {info}")
    if not ok:
        all_ok = False

print("-" * 50)

# 检查PyTorch GPU支持
try:
    import torch
    print(f"\nPyTorch配置：")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    
    # 检查MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS可用: {torch.backends.mps.is_available()}")
except Exception as e:
    print(f"\n警告: PyTorch检查失败 - {e}")
    all_ok = False

sys.exit(0 if all_ok else 1)
EOF
)
    
    if echo "$VERIFY_SCRIPT" | python3; then
        print_success "环境验证通过！"
        return 0
    else
        print_error "环境验证失败"
        print_error "某些关键包无法正确导入"
        echo ""
        echo "建议操作："
        echo "  1. 检查pip安装日志中的错误信息"
        echo "  2. 尝试单独安装失败的包"
        echo "  3. 查看教程文档中的常见问题解答"
        return 1
    fi
}

###############################################################################
# 主流程
###############################################################################

main() {
    # 解析参数
    parse_args "$@"
    
    print_header "视觉大模型教程 - 环境安装"
    
    print_info "此脚本将帮助您快速搭建开发环境"
    print_info "预计耗时: 5-10分钟（取决于网络速度）"
    echo ""
    
    # 1. 环境检查
    check_os
    check_python
    check_pip
    check_gpu
    
    # 2. 安装依赖
    install_dependencies
    echo ""
    install_dev_dependencies
    echo ""
    
    # 3. 验证环境
    if verify_environment; then
        echo ""
        print_header "✅ 安装成功！"
        
        echo "环境已准备就绪，您现在可以："
        echo ""
        echo "  1. 查看快速开始文档:"
        echo "     cat docs/05-使用说明/02-快速开始.md"
        echo ""
        echo "  2. 下载模型:"
        echo "     ./scripts/download_models.sh"
        echo ""
        echo "  3. 运行第一个示例:"
        echo "     python code/01-model-evaluation/examples/clip_inference.py"
        echo ""
        echo "  4. 启动Jupyter Notebook:"
        echo "     jupyter notebook notebooks/"
        echo ""
        
        print_info "祝学习愉快！"
    else
        echo ""
        print_header "⚠️  安装完成但验证失败"
        
        echo "依赖包已安装，但环境验证时发现问题。"
        echo ""
        echo "建议："
        echo "  1. 查看上方的错误信息"
        echo "  2. 尝试手动导入失败的包进行调试"
        echo "  3. 查阅文档中的故障排除部分"
        echo ""
        
        return 1
    fi
}

# 脚本入口
main "$@"

