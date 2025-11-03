# 智慧零售应用

基于多模态大模型的智慧零售解决方案，包括商品识别、货架分析等功能。

## 📁 文件结构

```
retail/
├── product_recognizer.py  # 商品识别器
├── shelf_analyzer.py       # 货架分析器
├── app.py                  # FastAPI服务（待补充）
├── config.yaml             # 配置文件（待补充）
└── README.md               # 本文件
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch transformers pillow numpy opencv-python
```

### 商品识别

```bash
python product_recognizer.py \
    --image product.jpg \
    --top-k 5 \
    --threshold 0.7
```

**输出示例**：
```
📝 识别结果:
============================================================
🏆 最佳匹配:
   商品名称: 可口可乐 330ml
   SKU: SKU-001
   类别: 饮料
   品牌: 可口可乐
   价格: ¥3.5
   置信度: 96.50%
   匹配: ✅ 是
```

### 货架分析

```bash
python shelf_analyzer.py \
    --image shelf.jpg \
    --expected 可乐 雪碧 芬达 \
    --threshold 0.8 \
    --visualize \
    --output analysis.jpg
```

**输出示例**:
```
📊 货架分析结果:
============================================================
满陈率: 85.0%
总货位: 20
已占用: 17
空货位: 3

⚠️ 缺货商品:
  - 芬达

💡 建议:
  - 缺货商品：芬达
  - 有3个空货位需要补充
```

## 💡 Python API使用

### 商品识别

```python
from product_recognizer import ProductRecognizer

# 初始化识别器
recognizer = ProductRecognizer(
    model_path="openai/clip-vit-base-patch32",
    product_database="products.json",
    confidence_threshold=0.7
)

# 识别单个商品
result = recognizer.recognize("product.jpg", top_k=5)

print(f"识别结果: {result['best_match']['name']}")
print(f"置信度: {result['best_match']['confidence']:.2%}")

# 批量识别
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = recognizer.batch_recognize(images)
```

### 货架分析

```python
from shelf_analyzer import ShelfAnalyzer
from product_recognizer import ProductRecognizer

# 初始化
recognizer = ProductRecognizer()
analyzer = ShelfAnalyzer(
    product_recognizer=recognizer,
    fill_rate_threshold=0.8
)

# 分析货架
result = analyzer.analyze_shelf(
    image="shelf.jpg",
    expected_products=["可乐", "雪碧", "芬达"]
)

print(f"满陈率: {result['fill_rate']:.1%}")
print(f"缺货: {result['missing_products']}")

# 可视化
analyzer.visualize_analysis("shelf.jpg", result, "output.jpg")
```

## 🔧 自定义商品数据库

创建`products.json`文件：

```json
[
  {
    "sku": "SKU-001",
    "name": "可口可乐 330ml",
    "category": "饮料",
    "brand": "可口可乐",
    "price": 3.5,
    "description": "可口可乐经典罐装饮料 330毫升"
  },
  {
    "sku": "SKU-002",
    "name": "雪碧 330ml",
    "category": "饮料",
    "brand": "可口可乐",
    "price": 3.5,
    "description": "雪碧柠檬味汽水 330毫升"
  }
]
```

然后使用：

```bash
python product_recognizer.py \
    --image product.jpg \
    --database products.json
```

## 📊 性能参考

| 功能 | 延迟 | 准确率 | 硬件 |
|------|------|--------|------|
| 商品识别 | ~50ms | 95%+ | V100 |
| 货架分析 | ~200ms | 90%+ | V100 |

## 🔗 相关文档

- [智慧零售应用文档](../../../docs/06-行业应用/01-智慧零售应用.md)
- [CLIP模型文档](../../../docs/01-模型调研与选型/01-CLIP模型详解.md)
- [SAM模型文档](../../../docs/01-模型调研与选型/05-SAM模型详解.md)

## 📝 许可

MIT License

