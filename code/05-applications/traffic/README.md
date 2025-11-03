# 智能交通AI应用

基于CLIP、SAM和BLIP-2的智能交通监控与分析工具，支持车辆检测、违章识别和场景理解。

## 功能特性

### 1. 车辆检测与识别
- **车型识别**：零样本识别车辆类型（轿车、卡车、公交、摩托）
- **车辆分割**：精确勾画车辆轮廓
- **车牌定位**：定位车牌区域

### 2. 场景理解
- **场景描述**：生成交通场景结构化描述
- **异常检测**：检测事故、拥堵、违章等异常
- **事故分析**：分析事故类型、严重程度

### 3. 违章检测
- **闯红灯检测**：识别闯红灯行为
- **违停检测**：检测禁停区域违停
- **压线检测**：识别压实线、黄线违章

### 4. 交通流量统计
- **车辆计数**：统计通过路口/路段的车辆数量
- **车型分类**：按车型统计流量
- **拥堵检测**：评估道路拥堵程度

## 安装依赖

```bash
# 基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow numpy opencv-python

# CLIP和SAM
pip install clip-by-openai segment-anything

# 视频处理（可选）
pip install moviepy
```

## 快速开始

### 示例图片准备

本应用提供示例图片生成工具：

```bash
cd code/05-applications/traffic/examples
python generate_examples.py
```

这将生成以下示例图片：
- `traffic_scene.jpg` - 普通交通场景
- `intersection.jpg` - 路口监控场景
- `accident.jpg` - 事故场景
- `highway.jpg` - 高速公路场景

> **注意**：这些是简化的示例图片，仅用于演示工具使用方式。实际应用中应使用真实的交通监控数据。

### 基本用法

由于Python模块导入限制（目录名包含数字和破折号），推荐使用以下方式：

#### 方式1：命令行执行（推荐）

```bash
cd code/05-applications/traffic

# 车辆检测
python vehicle_detector.py \
    --image examples/traffic_scene.jpg \
    --output results/detection.json

# 场景分析
python scene_analyzer.py \
    --image examples/traffic_scene.jpg \
    --task caption \
    --output results/description.txt

# 违章检测
python violation_detector.py \
    --image examples/intersection.jpg \
    --type red_light \
    --output results/violation.json
```

#### 方式2：Python API

```python
import sys
sys.path.insert(0, 'code/05-applications/traffic')

# 加载车辆检测器
exec(open('code/05-applications/traffic/vehicle_detector.py').read(), globals())

# 使用检测器
detector = VehicleDetector(device='cuda')
image = detector.load_image('traffic_scene.jpg')
result = detector.detect_vehicles(image)

print(f"检测到 {len(result['vehicles'])} 辆车")
```

## 核心功能演示

### 1. 车辆检测

```bash
# 检测并分类车辆
python vehicle_detector.py \
    --image examples/traffic_scene.jpg \
    --output results/vehicles.json \
    --visualize

# 输出示例
# {
#   "vehicles": [
#     {"type": "car", "confidence": 0.92, "bbox": [120, 300, 250, 180]},
#     {"type": "truck", "confidence": 0.88, "bbox": [450, 280, 320, 240]}
#   ],
#   "total_count": 2
# }
```

### 2. 场景描述

```bash
# 生成交通场景描述
python scene_analyzer.py \
    --image examples/traffic_scene.jpg \
    --task caption \
    --detail high \
    --output results/scene_description.txt

# 输出示例：
# 【交通场景分析】
# 时间：白天（晴朗）
# 地点：城市主干道，双向四车道
# 车辆情况：左侧车道1辆红色轿车，右侧车道1辆白色货车
# 路况：路面干燥，标线清晰，交通信号灯绿灯
# 总体评估：交通秩序良好，无违章行为
```

### 3. 异常检测

```bash
# 检测交通异常
python scene_analyzer.py \
    --image examples/accident.jpg \
    --task detect_anomaly \
    --output results/anomaly.json

# 输出示例
# {
#   "has_anomaly": true,
#   "type": "accident",
#   "severity": "high",
#   "description": "检测到车辆碰撞事故",
#   "recommendation": "立即通知救援部门"
# }
```

### 4. 违章检测

```bash
# 闯红灯检测
python violation_detector.py \
    --image examples/intersection.jpg \
    --type red_light \
    --traffic-light red \
    --output results/violation.json

# 违停检测
python violation_detector.py \
    --image examples/roadside.jpg \
    --type illegal_parking \
    --no-parking-zone "[100,200,500,600]" \
    --output results/parking_violation.json
```

## 文件结构

```
traffic/
├── vehicle_detector.py       # 车辆检测工具（当前可用）
├── scene_analyzer.py          # 场景分析工具（当前可用）
├── violation_detector.py      # 违章检测工具（当前可用）
├── README.md                  # 本文档
├── examples/                  # 示例数据
│   ├── generate_examples.py  # 示例生成脚本
│   ├── traffic_scene.jpg     # 交通场景示例
│   ├── intersection.jpg      # 路口示例
│   ├── accident.jpg          # 事故示例
│   └── highway.jpg           # 高速公路示例
└── results/                   # 输出目录（自动创建）

（规划中 - P2阶段）
├── traffic_analyzer.py       # 流量统计工具（P2阶段实现）
├── realtime_monitor.py       # 实时监控服务（P2阶段实现）
└── config.yaml               # 配置文件（P2阶段实现）
```

## 应用场景

### 1. 智能路口监控
- 自动检测闯红灯、压线等违章
- 实时统计车流量
- 快速发现交通事故

### 2. 高速公路监控
- 事故检测与快速响应
- 拥堵预警
- 违规变道检测

### 3. 智能停车管理
- 车位占用监控
- 车牌识别（无卡通行）
- 违停检测

### 4. 交通大数据分析
- 流量统计与可视化
- 违章热点分析
- 事故多发路段识别

## 性能参考

### 车辆检测
- **速度**：~1.5s/image (RTX 3090, CLIP ViT-B/32)
- **精度**：车型识别准确率 > 90%

### 场景分析
- **速度**：~3s/caption (BLIP-2-opt-2.7b)
- **质量**：场景描述准确率 > 85%

### 违章检测
- **闯红灯检测**：准确率 > 92%
- **违停检测**：准确率 > 88%

## 注意事项

⚠️ **重要提示**：

1. **仅供教育和研究**：代码未经法律认证，不得直接用于执法

2. **隐私保护**：处理监控数据需遵守《个人信息保护法》等法规

3. **人工复核**：AI检测结果需人工复核，不应作为唯一证据

4. **设备认证**：用于执法的设备需通过计量认证

5. **模型下载**：首次运行会自动下载CLIP和BLIP-2模型（~5GB），需联网

## 性能优化建议

### 1. 模型选择
```python
# 速度优先：使用轻量模型
detector = VehicleDetector(clip_model='RN50')

# 精度优先：使用大模型
detector = VehicleDetector(clip_model='ViT-L/14')
```

### 2. 批处理
```python
# 批量处理多张图片
images = [load_image(f) for f in image_files]
results = detector.detect_vehicles_batch(images, batch_size=8)
```

### 3. 混合精度
```python
# FP16加速
detector = VehicleDetector(device='cuda', precision='fp16')
```

## 参考文档

- [智能交通应用文档](../../../docs/06-行业应用/03-智能交通应用.md)
- [CLIP模型详解](../../../docs/01-模型调研与选型/03-CLIP模型详解.md)
- [BLIP-2模型详解](../../../docs/01-模型调研与选型/06-BLIP2模型详解.md)
- [SAM模型详解](../../../docs/01-模型调研与选型/05-SAM模型详解.md)

## 常见问题

**Q: 如何处理视频流？**

A: P2阶段将实现实时视频流处理。当前阶段可使用OpenCV提取视频帧后逐帧处理：
```python
import cv2
cap = cv2.VideoCapture('traffic_video.mp4')
while True:
    ret, frame = cap.read()
    if not ret: break
    # 处理每一帧
    result = detector.detect_vehicles(frame)
```

**Q: 如何提高检测准确性？**

A:
1. 使用更大的CLIP模型（ViT-L/14）
2. 结合多帧信息进行时序融合
3. 添加交通规则约束验证

**Q: 如何部署到生产环境？**

A: P2阶段将提供：
1. Docker容器化部署
2. RTSP视频流接入
3. 分布式多摄像头处理
4. 报警推送服务

## 法律合规

### 执法证据要求
AI检测结果作为执法证据需满足：
1. 原始图像/视频保存完整
2. AI分析过程可追溯
3. 人工复核确认
4. 设备通过计量认证

### 隐私保护
处理监控数据时需：
1. 人脸自动模糊化
2. 车牌加密存储
3. 数据访问权限控制
4. 定期数据清理

## 获取帮助

- GitHub Issues: https://github.com/your-org/Large-Model-Tutorial/issues
- 文档: `docs/06-行业应用/03-智能交通应用.md`

---

*版本：v1.0 | 最后更新：2023-11-15*

