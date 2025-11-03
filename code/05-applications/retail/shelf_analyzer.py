"""
货架陈列分析器

使用SAM+CLIP进行货架满陈率分析
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Union
import cv2


class ShelfAnalyzer:
    """
    货架分析器
    
    功能：
    - 货架区域分割
    - 商品定位
    - 满陈率计算
    - 缺货检测
    """
    
    def __init__(
        self,
        product_recognizer=None,
        fill_rate_threshold: float = 0.8
    ):
        """
        初始化货架分析器
        
        Args:
            product_recognizer: 商品识别器实例
            fill_rate_threshold: 满陈率阈值
        """
        self.recognizer = product_recognizer
        self.fill_rate_threshold = fill_rate_threshold
        
        print(f"🚀 初始化货架分析器...")
        print(f"   满陈率阈值: {fill_rate_threshold}")
        print(f"✅ 初始化完成")
    
    def analyze_shelf(
        self,
        image: Union[str, Image.Image],
        expected_products: List[str] = None
    ) -> Dict:
        """
        分析货架陈列
        
        Args:
            image: 图像路径或PIL Image
            expected_products: 期望的商品列表
            
        Returns:
            分析结果字典
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 简化版：使用网格分析（实际项目中应使用SAM分割）
        grid_size = (4, 5)  # 4行5列货架
        analysis = self._grid_based_analysis(image, grid_size)
        
        # 计算满陈率
        total_slots = analysis['total_slots']
        filled_slots = analysis['filled_slots']
        empty_slots = total_slots - filled_slots
        fill_rate = filled_slots / total_slots if total_slots > 0 else 0
        
        # 检测缺货（如果提供了期望商品列表）
        missing_products = []
        if expected_products and self.recognizer:
            detected_names = [p['name'] for p in analysis['detected_products']]
            # 使用模糊匹配：检查期望商品是否包含在检测到的商品名称中
            for expected in expected_products:
                found = False
                for detected in detected_names:
                    # 支持简称匹配：如"可乐"可以匹配"可口可乐 330ml"
                    if expected in detected or detected in expected:
                        found = True
                        break
                if not found:
                    missing_products.append(expected)
        
        # 生成建议
        recommendations = []
        if fill_rate < self.fill_rate_threshold:
            recommendations.append(f"满陈率仅{fill_rate:.1%}，低于阈值{self.fill_rate_threshold:.1%}，需要补货")
        if missing_products:
            recommendations.append(f"缺货商品：{', '.join(missing_products)}")
        if empty_slots > 0:
            recommendations.append(f"有{empty_slots}个空货位需要补充")
        
        return {
            'fill_rate': fill_rate,
            'total_slots': total_slots,
            'filled_slots': filled_slots,
            'empty_slots': empty_slots,
            'detected_products': analysis['detected_products'],
            'missing_products': missing_products,
            'recommendations': recommendations,
            'alert': fill_rate < self.fill_rate_threshold,
            'grid': analysis['grid']
        }
    
    def _grid_based_analysis(self, image: Image.Image, grid_size: tuple) -> Dict:
        """
        基于网格的简化分析
        
        Args:
            image: PIL Image
            grid_size: (rows, cols)
            
        Returns:
            分析结果
        """
        rows, cols = grid_size
        width, height = image.size
        
        cell_width = width // cols
        cell_height = height // rows
        
        grid = []
        detected_products = []
        filled_count = 0
        
        for i in range(rows):
            row = []
            for j in range(cols):
                # 提取网格单元
                left = j * cell_width
                top = i * cell_height
                right = left + cell_width
                bottom = top + cell_height
                
                cell_img = image.crop((left, top, right, bottom))
                
                # 简单判断：计算亮度方差
                # 实际项目中应使用商品识别器
                np_img = np.array(cell_img)
                variance = np.var(np_img)
                
                # 方差大说明有商品（不是空白）
                is_filled = variance > 1000  # 阈值需要根据实际调整
                
                cell_info = {
                    'row': i,
                    'col': j,
                    'filled': is_filled,
                    'variance': float(variance),
                    'bbox': (left, top, right, bottom)
                }
                
                if is_filled:
                    filled_count += 1
                    # 如果有识别器，尝试识别商品
                    if self.recognizer:
                        try:
                            result = self.recognizer.recognize(cell_img, top_k=1)
                            if result['recognized']:
                                product = result['best_match'].copy()
                                product['position'] = (i, j)
                                detected_products.append(product)
                                cell_info['product'] = product['name']
                        except:
                            cell_info['product'] = "未识别"
                
                row.append(cell_info)
            grid.append(row)
        
        return {
            'grid': grid,
            'total_slots': rows * cols,
            'filled_slots': filled_count,
            'detected_products': detected_products
        }
    
    def visualize_analysis(
        self,
        image: Union[str, Image.Image],
        analysis_result: Dict,
        output_path: str = "shelf_analysis.jpg"
    ):
        """
        可视化分析结果
        
        Args:
            image: 原始图像
            analysis_result: 分析结果
            output_path: 输出路径
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        draw = ImageDraw.Draw(image)
        
        # 绘制网格和标注
        for row in analysis_result['grid']:
            for cell in row:
                bbox = cell['bbox']
                color = 'green' if cell['filled'] else 'red'
                draw.rectangle(bbox, outline=color, width=3)
                
                # 标注商品名称
                if 'product' in cell:
                    draw.text((bbox[0]+5, bbox[1]+5), cell['product'], fill='white')
        
        # 添加统计信息
        stats_text = f"满陈率: {analysis_result['fill_rate']:.1%} ({analysis_result['filled_slots']}/{analysis_result['total_slots']})"
        draw.text((10, 10), stats_text, fill='yellow')
        
        # 保存
        image.save(output_path)
        print(f"✅ 可视化结果已保存: {output_path}")


def main():
    """示例用法"""
    import argparse
    from product_recognizer import ProductRecognizer
    
    parser = argparse.ArgumentParser(description='货架陈列分析器')
    parser.add_argument('--image', type=str, required=True, help='货架图像路径')
    parser.add_argument('--expected', type=str, nargs='+', help='期望的商品列表')
    parser.add_argument('--threshold', type=float, default=0.8, help='满陈率阈值')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--output', type=str, default='shelf_analysis.jpg', help='可视化输出路径')
    
    args = parser.parse_args()
    
    # 初始化识别器（可选）
    recognizer = None
    try:
        recognizer = ProductRecognizer()
    except:
        print("⚠️ 未能初始化商品识别器，将使用简化分析")
    
    # 初始化分析器
    analyzer = ShelfAnalyzer(
        product_recognizer=recognizer,
        fill_rate_threshold=args.threshold
    )
    
    # 分析货架
    result = analyzer.analyze_shelf(
        image=args.image,
        expected_products=args.expected
    )
    
    # 打印结果
    print(f"\n📊 货架分析结果:")
    print(f"="*60)
    print(f"满陈率: {result['fill_rate']:.1%}")
    print(f"总货位: {result['total_slots']}")
    print(f"已占用: {result['filled_slots']}")
    print(f"空货位: {result['empty_slots']}")
    
    if result['detected_products']:
        print(f"\n🛒 检测到的商品:")
        for i, product in enumerate(result['detected_products'], 1):
            print(f"  {i}. {product['name']} (位置: 第{product['position'][0]+1}行第{product['position'][1]+1}列)")
    
    if result['missing_products']:
        print(f"\n⚠️ 缺货商品:")
        for product in result['missing_products']:
            print(f"  - {product}")
    
    if result['recommendations']:
        print(f"\n💡 建议:")
        for rec in result['recommendations']:
            print(f"  - {rec}")
    
    if result['alert']:
        print(f"\n🚨 警告: 满陈率低于阈值！")
    
    # 可视化
    if args.visualize:
        analyzer.visualize_analysis(args.image, result, args.output)


if __name__ == '__main__':
    main()

