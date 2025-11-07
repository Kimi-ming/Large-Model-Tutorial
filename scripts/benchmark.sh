#!/bin/bash
# 性能基准测试脚本
# 这是 run_benchmarks.sh 的简化版本，用于快速测试

# 直接调用完整的基准测试脚本
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 启动性能基准测试..."
echo ""

# 调用完整的测试脚本
exec "$SCRIPT_DIR/run_benchmarks.sh" "$@"


