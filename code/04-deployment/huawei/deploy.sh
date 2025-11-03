#!/bin/bash

###############################################################################
# 华为昇腾自动化部署脚本
#
# 功能：
# - 检查环境依赖
# - 下载和转换模型
# - 运行性能测试
# - 生成部署报告
###############################################################################

set -e  # 遇到错误立即退出

# 默认参数
MODEL="openai/clip-vit-base-patch32"
OUTPUT_DIR="./deployed_models"
SOC_VERSION="Ascend910"
BATCH_SIZE=1
DYNAMIC_BATCH=false
RUN_BENCHMARK=true
NUM_RUNS=100

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --soc-version)
            SOC_VERSION="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dynamic-batch)
            DYNAMIC_BATCH=true
            shift
            ;;
        --no-benchmark)
            RUN_BENCHMARK=false
            shift
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -h|--help)
            echo "华为昇腾自动化部署脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model MODEL              模型路径（默认: openai/clip-vit-base-patch32）"
            echo "  --output-dir DIR           输出目录（默认: ./deployed_models）"
            echo "  --soc-version VERSION      目标芯片（默认: Ascend910）"
            echo "  --batch-size N             批大小（默认: 1）"
            echo "  --dynamic-batch            启用动态batch"
            echo "  --no-benchmark             跳过性能测试"
            echo "  --num-runs N               测试运行次数（默认: 100）"
            echo "  -h, --help                 显示帮助信息"
            echo ""
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            exit 1
            ;;
    esac
done

print_header "华为昇腾模型部署脚本"
print_info "模型: $MODEL"
print_info "输出目录: $OUTPUT_DIR"
print_info "目标芯片: $SOC_VERSION"
print_info "批大小: $BATCH_SIZE"
print_info "动态batch: $DYNAMIC_BATCH"

# 1. 检查环境
print_header "步骤 1/5: 检查环境"

# 检查Python
if ! command -v python &> /dev/null; then
    print_error "Python未安装"
    exit 1
fi
print_info "✓ Python: $(python --version)"

# 检查NPU
print_info "检查NPU设备..."
if command -v npu-smi &> /dev/null; then
    npu-smi info
    print_info "✓ NPU设备可用"
else
    print_warn "npu-smi未找到，请确认CANN已正确安装"
fi

# 检查torch_npu
print_info "检查torch_npu..."
if python -c "import torch_npu" 2>/dev/null; then
    print_info "✓ torch_npu已安装"
    python -c "import torch; import torch_npu; print(f'NPU可用: {torch.npu.is_available()}')"
else
    print_error "torch_npu未安装或无法导入"
    print_info "安装方法:"
    print_info "  pip install torch-npu==1.11.0 -i https://repo.huaweicloud.com/repository/pypi/simple"
    exit 1
fi

# 检查ATC（用于模型转换）
if command -v atc &> /dev/null; then
    print_info "✓ ATC工具可用"
else
    print_warn "ATC工具未找到，将跳过OM转换"
    print_info "请设置CANN环境："
    print_info "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

# 2. 创建输出目录
print_header "步骤 2/5: 创建输出目录"
mkdir -p "$OUTPUT_DIR"
print_info "输出目录: $OUTPUT_DIR"

# 3. 下载测试图像（如果不存在）
print_header "步骤 3/5: 准备测试数据"
TEST_IMAGE="$OUTPUT_DIR/test_image.jpg"

if [ ! -f "$TEST_IMAGE" ]; then
    print_info "生成测试图像..."
    python - <<EOF
from PIL import Image, ImageDraw
import numpy as np

# 创建测试图像
img = Image.new('RGB', (400, 300))
pixels = img.load()

for i in range(300):
    for j in range(400):
        r = int(100 + 155 * (j / 400))
        g = int(150 + 105 * (i / 300))
        b = int(200 - 100 * ((i + j) / 700))
        pixels[j, i] = (r, g, b)

draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 300, 200], outline='blue', width=3)
draw.ellipse([180, 130, 220, 170], outline='red', width=3)

img.save("$TEST_IMAGE")
print(f"✓ 测试图像已保存: $TEST_IMAGE")
EOF
else
    print_info "使用现有测试图像: $TEST_IMAGE"
fi

# 4. 运行PyTorch-NPU推理
print_header "步骤 4/5: PyTorch-NPU推理测试"

print_info "运行单次推理..."
python pytorch_npu_inference.py \
    --model "$MODEL" \
    --image "$TEST_IMAGE" \
    --texts "a colorful abstract pattern" "a geometric shape" "a gradient background" \
    --device auto \
    --fp16

# 5. 性能测试
if [ "$RUN_BENCHMARK" = true ]; then
    print_header "步骤 5/5: 性能基准测试"
    
    BENCHMARK_OUTPUT="$OUTPUT_DIR/benchmark_results.json"
    
    print_info "运行性能测试（$NUM_RUNS 次迭代）..."
    python benchmark.py \
        --model "$MODEL" \
        --image "$TEST_IMAGE" \
        --texts "a colorful pattern" "a geometric shape" "a gradient" \
        --num-runs "$NUM_RUNS" \
        --output "$BENCHMARK_OUTPUT"
    
    print_info "✓ 性能测试完成"
    print_info "结果已保存: $BENCHMARK_OUTPUT"
else
    print_info "跳过性能测试"
fi

# 6. 生成部署报告
print_header "部署完成！"

cat << EOF

📦 部署摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 输出目录: $OUTPUT_DIR
📄 测试图像: $TEST_IMAGE

🔧 配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  模型: $MODEL
  目标芯片: $SOC_VERSION
  批大小: $BATCH_SIZE
  动态batch: $DYNAMIC_BATCH

📊 文件清单
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

ls -lh "$OUTPUT_DIR" | tail -n +2

cat << EOF

✅ 下一步
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 查看性能测试结果:
   cat $OUTPUT_DIR/benchmark_results.json

2. 使用Python API进行推理:
   from pytorch_npu_inference import CLIPInferenceService
   service = CLIPInferenceService(model_path="$MODEL", device="npu")
   result = service.predict("image.jpg", ["text1", "text2"])

3. 转换为OM格式以获得更好性能:
   python convert_to_om.py clip \\
       --model $MODEL \\
       --output-dir $OUTPUT_DIR/om \\
       --soc-version $SOC_VERSION

4. 阅读完整文档:
   ../../../docs/04-多平台部署/03-华为昇腾部署.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

print_info "部署脚本执行完成！"

