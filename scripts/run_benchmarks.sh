#!/bin/bash
# 一键运行所有基准测试

set -euo pipefail

echo "========================================"
echo "  视觉大模型基准测试套件"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 环境检查
echo -e "${BLUE}[1/5] 环境检查...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# 2. 创建结果目录
echo -e "${BLUE}[2/5] 准备环境...${NC}"
mkdir -p results
mkdir -p data/test_dataset
echo -e "${GREEN}✅ 结果目录已创建${NC}"
echo ""

# 3. 数据准备提示
echo -e "${BLUE}[3/5] 测试数据准备${NC}"
if [ ! "$(ls -A data/test_dataset/*.jpg 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  测试数据不存在，请先准备测试图像：${NC}"
    echo "   方案1：手动放置测试图像到 data/test_dataset/"
    echo "   方案2：运行 python scripts/prepare_test_data.py"
    echo ""
    echo "跳过实际测试，仅生成示例报告..."
    USE_DEMO_DATA=true
else
    echo -e "${GREEN}✅ 找到测试数据${NC}"
    USE_DEMO_DATA=false
fi
echo ""

# 4. 速度测试
if [ "$USE_DEMO_DATA" = false ]; then
    echo -e "${BLUE}[4/5] 运行速度测试...${NC}"
    
    # 定义要测试的模型（可根据实际情况调整）
    MODELS=(
        "openai/clip-vit-base-patch32"
    )
    
    for model in "${MODELS[@]}"; do
        model_name=$(basename "$model" | sed 's/:/-/g')
        echo -e "  ${YELLOW}Testing $model_name...${NC}"
        
        if python code/01-model-evaluation/benchmark/speed_test.py \
            --model "$model" \
            --batch_sizes 1 2 \
            --output "results/${model_name}_speed.json" 2>&1; then
            echo -e "  ${GREEN}✅ $model_name 速度测试完成${NC}"
        else
            echo -e "  ${YELLOW}⚠️  $model_name 速度测试失败（可能是模型未下载）${NC}"
        fi
        echo ""
    done
    
    # 5. 显存测试
    echo -e "${BLUE}[5/5] 运行显存测试...${NC}"
    for model in "${MODELS[@]}"; do
        model_name=$(basename "$model" | sed 's/:/-/g')
        echo -e "  ${YELLOW}Testing $model_name...${NC}"
        
        if python code/01-model-evaluation/benchmark/memory_test.py \
            --model "$model" \
            --batch_size 1 > "results/${model_name}_memory.txt" 2>&1; then
            echo -e "  ${GREEN}✅ $model_name 显存测试完成${NC}"
        else
            echo -e "  ${YELLOW}⚠️  $model_name 显存测试失败${NC}"
        fi
        echo ""
    done
else
    echo -e "${YELLOW}[4/5] 跳过速度测试（演示模式）${NC}"
    echo -e "${YELLOW}[5/5] 跳过显存测试（演示模式）${NC}"
fi

# 6. 生成报告
echo ""
echo -e "${BLUE}生成评测报告...${NC}"
python code/01-model-evaluation/benchmark/generate_report.py \
    --results_dir results \
    --output results/benchmark_report.md

echo ""
echo "========================================"
echo -e "${GREEN}  测试完成！${NC}"
echo "========================================"
echo "结果保存在 results/ 目录："
echo ""
ls -lh results/ 2>/dev/null || echo "  (结果目录为空)"
echo ""
echo "查看报告："
echo "  cat results/benchmark_report.md"
echo ""

