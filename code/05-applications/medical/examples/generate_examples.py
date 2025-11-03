#!/usr/bin/env python3
"""
生成医疗应用示例图片

这些是简化的示例图片，仅用于演示工具使用方式
实际应用中应使用真实的医学影像数据
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_ct_lung_example():
    """创建肺部CT示例图片"""
    # 创建512x512的灰度图像
    img = Image.new('RGB', (512, 512), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    
    # 绘制肺野轮廓（左右两肺）
    # 左肺
    draw.ellipse([80, 150, 220, 400], fill=(60, 60, 60), outline=(100, 100, 100), width=2)
    # 右肺
    draw.ellipse([292, 150, 432, 400], fill=(60, 60, 60), outline=(100, 100, 100), width=2)
    
    # 绘制结节（右肺中野）
    nodule_x, nodule_y = 350, 250
    draw.ellipse([nodule_x-15, nodule_y-15, nodule_x+15, nodule_y+15], 
                 fill=(180, 180, 180), outline=(200, 200, 200), width=1)
    
    # 绘制小结节（左肺）
    draw.ellipse([145, 220, 155, 230], fill=(160, 160, 160))
    
    # 添加标注
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Lung CT (Axial)", fill=(255, 255, 255), font=font)
    draw.text((nodule_x+20, nodule_y), "Nodule", fill=(255, 200, 0), font=font)
    
    return img


def create_chest_xray_example():
    """创建胸片示例图片"""
    img = Image.new('RGB', (512, 600), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # 绘制肺野
    # 右肺
    draw.polygon([
        (100, 150), (200, 120), (250, 150), (250, 450), (150, 500), (100, 450)
    ], fill=(80, 80, 80), outline=(120, 120, 120), width=2)
    
    # 左肺
    draw.polygon([
        (262, 150), (350, 120), (412, 150), (412, 450), (320, 500), (262, 450)
    ], fill=(80, 80, 80), outline=(120, 120, 120), width=2)
    
    # 绘制心影
    draw.ellipse([200, 350, 312, 500], fill=(100, 100, 100), outline=(130, 130, 130), width=2)
    
    # 绘制肋骨
    for y in range(180, 480, 40):
        for side in [(100, 250), (262, 412)]:
            x1, x2 = side
            draw.arc([x1, y-30, x2, y+30], 0, 180, fill=(110, 110, 110), width=1)
    
    # 添加标注
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Chest X-ray (PA)", fill=(255, 255, 255), font=font)
    draw.text((180, 550), "R", fill=(255, 255, 255), font=font)
    draw.text((320, 550), "L", fill=(255, 255, 255), font=font)
    
    return img


def create_brain_mri_example():
    """创建脑MRI示例图片"""
    img = Image.new('RGB', (512, 512), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    
    # 绘制脑部轮廓
    draw.ellipse([80, 50, 432, 462], fill=(70, 70, 70), outline=(100, 100, 100), width=3)
    
    # 绘制脑室
    draw.ellipse([220, 200, 292, 280], fill=(30, 30, 30), outline=(50, 50, 50), width=2)
    
    # 绘制病灶（右侧额叶）
    lesion_x, lesion_y = 340, 180
    draw.ellipse([lesion_x-25, lesion_y-20, lesion_x+25, lesion_y+20], 
                 fill=(150, 150, 150), outline=(180, 180, 180), width=2)
    
    # 绘制中线
    draw.line([(256, 50), (256, 462)], fill=(90, 90, 90), width=1)
    
    # 添加标注
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Brain MRI (Axial T2)", fill=(255, 255, 255), font=font)
    draw.text((lesion_x+30, lesion_y-10), "Lesion", fill=(255, 200, 0), font=font)
    draw.text((180, 480), "R", fill=(255, 255, 255), font=font)
    draw.text((320, 480), "L", fill=(255, 255, 255), font=font)
    
    return img


def create_ct_series():
    """创建CT序列（多切片）"""
    series_dir = "ct_series"
    os.makedirs(series_dir, exist_ok=True)
    
    # 生成20个切片
    for i in range(20):
        img = Image.new('RGB', (512, 512), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)
        
        # 肺野大小随切片变化
        scale = 1.0 - abs(i - 10) * 0.05
        
        # 左肺
        w, h = int(140 * scale), int(250 * scale)
        cx, cy = 150, 275
        draw.ellipse([cx-w//2, cy-h//2, cx+w//2, cy+h//2], 
                     fill=(60, 60, 60), outline=(100, 100, 100), width=2)
        
        # 右肺
        cx = 362
        draw.ellipse([cx-w//2, cy-h//2, cx+w//2, cy+h//2], 
                     fill=(60, 60, 60), outline=(100, 100, 100), width=2)
        
        # 结节在中间切片附近出现
        if 8 <= i <= 12:
            nodule_size = 15 - abs(i - 10) * 3
            draw.ellipse([350-nodule_size, 250-nodule_size, 
                         350+nodule_size, 250+nodule_size],
                        fill=(180, 180, 180))
        
        # 保存
        img.save(f"{series_dir}/slice_{i:03d}.png")
    
    print(f"生成CT序列: {series_dir}/ (20张切片)")


def main():
    """生成所有示例图片"""
    print("生成医疗应用示例图片...")
    
    # 确保examples目录存在
    os.makedirs(".", exist_ok=True)
    
    # 生成单张示例
    print("1. 生成 ct_lung.png...")
    img_ct = create_ct_lung_example()
    img_ct.save("ct_lung.png")
    
    print("2. 生成 chest_xray.png...")
    img_xray = create_chest_xray_example()
    img_xray.save("chest_xray.png")
    
    print("3. 生成 brain_mri.png...")
    img_mri = create_brain_mri_example()
    img_mri.save("brain_mri.png")
    
    # 生成CT序列
    print("4. 生成 ct_series/...")
    create_ct_series()
    
    print("\n✅ 所有示例图片生成完成！")
    print("\n注意：这些是简化的示例图片，仅用于演示工具使用方式。")
    print("实际应用中应使用真实的医学影像数据（DICOM格式）。")


if __name__ == '__main__':
    main()

