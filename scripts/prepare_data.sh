#!/bin/bash
# 数据准备脚本 - 用于下载和准备训练数据
# 使用方法: ./scripts/prepare_data.sh [dataset_name]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认数据目录
DATA_DIR="./data"
MODELS_DIR="./models"

# 打印帮助信息
print_help() {
    echo "数据准备脚本"
    echo ""
    echo "用法: $0 [选项] [数据集名称]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -d, --data-dir DIR  指定数据目录（默认: ./data）"
    echo "  -l, --list          列出可用的数据集"
    echo ""
    echo "数据集:"
    echo "  dog                 狗品种数据集（用于演示）"
    echo "  coco                COCO数据集（需要手动下载）"
    echo "  imagenet            ImageNet数据集（需要手动下载）"
    echo ""
    echo "示例:"
    echo "  $0 dog             # 准备狗品种数据集"
    echo "  $0 -d /path/to/data dog  # 指定数据目录"
}

# 列出可用数据集
list_datasets() {
    echo -e "${GREEN}可用的数据集:${NC}"
    echo "  - dog: 狗品种演示数据集（自动下载）"
    echo "  - coco: COCO数据集（需要手动下载）"
    echo "  - imagenet: ImageNet数据集（需要手动下载）"
    echo ""
    echo -e "${YELLOW}提示: 大型数据集请访问官方网站手动下载${NC}"
}

# 准备狗品种数据集
prepare_dog_dataset() {
    echo -e "${GREEN}准备狗品种数据集...${NC}"
    python scripts/prepare_dog_dataset.py --output_dir "$DATA_DIR/dogs"
    echo -e "${GREEN}✓ 狗品种数据集准备完成${NC}"
}

# 准备COCO数据集
prepare_coco_dataset() {
    echo -e "${YELLOW}COCO数据集需要手动下载${NC}"
    echo ""
    echo "请访问: https://cocodataset.org/#download"
    echo ""
    echo "下载后解压到: $DATA_DIR/coco/"
    echo ""
    echo "目录结构:"
    echo "  $DATA_DIR/coco/"
    echo "    ├── train2017/"
    echo "    ├── val2017/"
    echo "    └── annotations/"
}

# 准备ImageNet数据集
prepare_imagenet_dataset() {
    echo -e "${YELLOW}ImageNet数据集需要手动下载${NC}"
    echo ""
    echo "请访问: https://image-net.org/download.php"
    echo ""
    echo "下载后解压到: $DATA_DIR/imagenet/"
    echo ""
    echo "目录结构:"
    echo "  $DATA_DIR/imagenet/"
    echo "    ├── train/"
    echo "    └── val/"
}

# 解析命令行参数
DATASET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_help
            exit 0
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -l|--list)
            list_datasets
            exit 0
            ;;
        *)
            DATASET="$1"
            shift
            ;;
    esac
done

# 如果没有指定数据集，显示帮助
if [ -z "$DATASET" ]; then
    print_help
    exit 1
fi

# 创建数据目录
mkdir -p "$DATA_DIR"

# 根据数据集名称准备数据
case "$DATASET" in
    dog|dogs)
        prepare_dog_dataset
        ;;
    coco)
        prepare_coco_dataset
        ;;
    imagenet)
        prepare_imagenet_dataset
        ;;
    *)
        echo -e "${RED}错误: 未知的数据集 '$DATASET'${NC}"
        echo ""
        list_datasets
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ 数据准备完成！${NC}"
echo "数据目录: $DATA_DIR"


