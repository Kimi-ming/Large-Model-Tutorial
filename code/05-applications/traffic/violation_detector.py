#!/usr/bin/env python3
"""
违章检测工具

功能：
1. 闯红灯检测
2. 违停检测
3. 压线检测

作者：Large Model Tutorial
日期：2023-11-15
"""

import argparse
import json
import numpy as np
from PIL import Image
from typing import Dict, List


class ViolationDetector:
    """违章检测器"""
    
    def __init__(self, device: str = 'cpu'):
        """初始化"""
        self.device = device
        print("违章检测器初始化完成")
    
    def detect_red_light_violation(
        self,
        image: np.ndarray,
        traffic_light_state: str = 'red',
        stop_line: List[List[int]] = None
    ) -> Dict:
        """
        闯红灯检测（简化版）
        
        实际应用需要：
        1. 检测红绿灯状态
        2. 检测车辆位置
        3. 判断是否越过停止线
        """
        # 简化：假设检测到违章
        return {
            'violation': True if traffic_light_state == 'red' else False,
            'vehicle_type': 'car',
            'plate_bbox': [200, 350, 100, 30],
            'distance_over_line': 2.5,
            'timestamp': '2023-11-15 08:30:15'
        }
    
    def detect_illegal_parking(
        self,
        image: np.ndarray,
        no_parking_zone: List[int],
        duration: int = 0
    ) -> Dict:
        """违停检测"""
        return {
            'violation': duration > 60,  # 超过60秒
            'vehicle_count': 1,
            'duration': duration,
            'zone': no_parking_zone
        }
    
    def detect_lane_violation(self, image: np.ndarray) -> Dict:
        """压线检测"""
        return {
            'violation': False,
            'type': None,
            'location': None
        }


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='违章检测工具')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--type', type=str, required=True,
                        choices=['red_light', 'illegal_parking', 'lane_violation'])
    parser.add_argument('--traffic-light', type=str, default='red')
    parser.add_argument('--no-parking-zone', type=str)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("违章检测工具")
    print("=" * 60)
    
    detector = ViolationDetector()
    image = np.array(Image.open(args.image))
    
    if args.type == 'red_light':
        result = detector.detect_red_light_violation(image, args.traffic_light)
    elif args.type == 'illegal_parking':
        zone = json.loads(args.no_parking_zone) if args.no_parking_zone else [0,0,100,100]
        result = detector.detect_illegal_parking(image, zone)
    else:
        result = detector.detect_lane_violation(image)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存至: {args.output}")


if __name__ == '__main__':
    main()

