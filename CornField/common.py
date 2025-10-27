#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Utilities
共用工具函数
"""

import yaml
import numpy as np
from pathlib import Path


def save_detection_summary_yaml(output_path, summary_data):
    """
    保存植物检测结果为YAML格式
    
    参数:
        output_path: 输出文件路径 (Path or str)
        summary_data: 包含检测结果的字典，格式：
            {
                'method': 'density' or 'height',
                'input_points': int,
                'direction': 'x' or 'y',
                'expected_spacing_cm': float,
                'bin_size_cm': float,
                'ground_removal': bool,
                'ground_percentile': float,
                'top_percentile': float,
                'sor_applied': bool,
                'sor_k': int (optional),
                'sor_std_ratio': float (optional),
                'height_metric': str (optional, for height-based),
                'plant_count': int,
                'plants': [
                    {
                        'id': int,
                        'x': float,    # X坐标
                        'y': float,    # Y坐标
                        'height': float,    # 峰高度值
                    },
                    ...
                ]
            }
    """
    output_path = Path(output_path)
    
    # 构建YAML数据结构
    yaml_data = {
        'detection_summary': {
            'method': summary_data.get('method', 'unknown'),
            'statistics': {
                'input_points': int(summary_data.get('input_points', 0)),
                'detected_plants': int(summary_data.get('plant_count', 0)),
            },
            'parameters': {
                'direction': summary_data.get('direction', 'unknown'),
                'expected_spacing_cm': float(summary_data.get('expected_spacing_cm', 0)),
                'bin_size_cm': float(summary_data.get('bin_size_cm', 0)),
                'ground_removal': bool(summary_data.get('ground_removal', False)),
            },
            'preprocessing': {}
        }
    }
    
    # 添加地面移除参数
    if summary_data.get('ground_removal'):
        yaml_data['detection_summary']['preprocessing']['ground_removal'] = {
            'bottom_percentile': float(summary_data.get('ground_percentile', 0)),
            'top_percentile': float(summary_data.get('top_percentile', 0)),
        }
    
    # 添加SOR参数
    if summary_data.get('sor_applied'):
        yaml_data['detection_summary']['preprocessing']['statistical_outlier_removal'] = {
            'k_neighbors': int(summary_data.get('sor_k', 20)),
            'std_ratio': float(summary_data.get('sor_std_ratio', 2.0)),
        }
    
    # 添加height-based特有参数
    if summary_data.get('method') == 'height' and 'height_metric' in summary_data:
        yaml_data['detection_summary']['parameters']['height_metric'] = summary_data['height_metric']
    
    # 添加植物列表
    plants = summary_data.get('plants', [])
    if plants:
        yaml_data['detection_summary']['plants'] = []
        
        for plant in plants:
            plant_entry = {
                'id': int(plant['id']),
                'x_m': float(plant['x']),
                'y_m': float(plant['y']),
                'height_m': float(plant['height']),
            }
            yaml_data['detection_summary']['plants'].append(plant_entry)
        
        # 添加间距统计（基于检测方向）
        if len(plants) > 1:
            direction = summary_data.get('direction', 'x')
            if direction == 'x':
                # 沿x方向，计算x坐标间距
                positions = np.array([p['x'] for p in plants])
            else:
                # 沿y方向，计算y坐标间距
                positions = np.array([p['y'] for p in plants])
            
            spacings = np.diff(positions)
            
            yaml_data['detection_summary']['spacing_statistics'] = {
                'mean_cm': float(spacings.mean() * 100),
                'std_cm': float(spacings.std() * 100),
                'min_cm': float(spacings.min() * 100),
                'max_cm': float(spacings.max() * 100),
            }
    
    # 写入YAML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  检测结果已保存为YAML: {output_path}")
