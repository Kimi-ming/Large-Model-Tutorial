#!/usr/bin/env python3
"""
生成智能交通应用示例图片
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_traffic_scene():
    """创建普通交通场景"""
    img = Image.new('RGB', (1280, 720), color=(100, 100, 100))  # 路面
    draw = ImageDraw.Draw(img)
    
    # 绘制道路标线
    for y in [240, 360, 480, 600]:
        draw.line([(0, y), (1280, y)], fill=(255, 255, 255), width=5)
    
    # 绘制车辆
    # 红色轿车
    draw.rectangle([200, 250, 350, 350], fill=(180, 0, 0), outline=(220, 0, 0), width=3)
    
    # 白色货车
    draw.rectangle([600, 370, 800, 470], fill=(220, 220, 220), outline=(200, 200, 200), width=3)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Traffic Scene - Normal", fill=(255, 255, 255), font=font)
    
    return img


def create_intersection():
    """创建路口场景"""
    img = Image.new('RGB', (1280, 720), color=(80, 80, 80))
    draw = ImageDraw.Draw(img)
    
    # 横向道路
    draw.rectangle([0, 280, 1280, 440], fill=(100, 100, 100))
    # 纵向道路  
    draw.rectangle([540, 0, 740, 720], fill=(100, 100, 100))
    
    # 停止线
    draw.line([(540, 280), (740, 280)], fill=(255, 255, 255), width=8)
    
    # 红绿灯（红灯）
    draw.ellipse([1150, 200, 1200, 250], fill=(255, 0, 0))
    
    # 车辆
    draw.rectangle([600, 150, 680, 260], fill=(0, 0, 180))
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Intersection - Red Light", fill=(255, 255, 255), font=font)
    
    return img


def create_accident():
    """创建事故场景"""
    img = Image.new('RGB', (1280, 720), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)
    
    # 两辆车碰撞
    draw.rectangle([400, 300, 550, 400], fill=(180, 0, 0), outline=(220, 0, 0), width=3)
    draw.rectangle([530, 320, 680, 420], fill=(200, 200, 200), outline=(180, 180, 180), width=3)
    
    # 碎片效果
    for _ in range(20):
        x = np.random.randint(500, 600)
        y = np.random.randint(350, 450)
        draw.ellipse([x, y, x+5, y+5], fill=(150, 150, 150))
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Accident Scene", fill=(255, 0, 0), font=font)
    
    return img


def create_highway():
    """创建高速公路场景"""
    img = Image.new('RGB', (1280, 720), color=(70, 70, 70))
    draw = ImageDraw.Draw(img)
    
    # 三车道
    for y in [200, 360, 520]:
        draw.line([(0, y), (1280, y)], fill=(255, 255, 255), width=4)
    
    # 多辆车
    positions = [(150, 210), (400, 370), (700, 530), (950, 210)]
    for x, y in positions:
        draw.rectangle([x, y, x+120, y+80], fill=(160, 160, 160), outline=(200, 200, 200), width=2)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Highway - Multi Lane", fill=(255, 255, 255), font=font)
    
    return img


def main():
    """生成所有示例图片"""
    print("生成智能交通应用示例图片...")
    
    os.makedirs(".", exist_ok=True)
    
    print("1. 生成 traffic_scene.jpg...")
    img = create_traffic_scene()
    img.save("traffic_scene.jpg")
    
    print("2. 生成 intersection.jpg...")
    img = create_intersection()
    img.save("intersection.jpg")
    
    print("3. 生成 accident.jpg...")
    img = create_accident()
    img.save("accident.jpg")
    
    print("4. 生成 highway.jpg...")
    img = create_highway()
    img.save("highway.jpg")
    
    print("\n✅ 所有示例图片生成完成！")
    print("\n注意：这些是简化的示例图片，仅用于演示工具使用方式。")
    print("实际应用中应使用真实的交通监控数据。")


if __name__ == '__main__':
    main()

