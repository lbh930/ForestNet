#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Utilities
共用工具函数
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
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
    
    # print(f"  检测结果已保存为YAML: {output_path}")


def remove_ground_points(points, bottom_percentile, top_percentile):
    """
    移除地面点和顶部点（基于bounding box高度范围）
    
    参数:
        points: Nx3点云数组 [x, y, z]
        bottom_percentile: 要移除的底部百分比
        top_percentile: 要移除的顶部百分比
    
    返回:
        filtered_points: 移除后的点云
        mask: 保留点的布尔mask
    """
    if len(points) == 0:
        return points, np.ones(0, dtype=bool)
    
    heights = points[:, 2]
    z_min = heights.min()
    z_max = heights.max()
    z_range = z_max - z_min
    
    # 基于bounding box计算阈值
    lower_threshold = z_min + (z_range * bottom_percentile / 100.0)
    upper_threshold = z_max - (z_range * top_percentile / 100.0)
    
    # 保留中间范围的点
    mask = (heights > lower_threshold) & (heights < upper_threshold)
    filtered_points = points[mask]
    
    removed_bottom = np.sum(heights <= lower_threshold)
    removed_top = np.sum(heights >= upper_threshold)
    removed_total = len(points) - len(filtered_points)
    
    print(f"  地面+顶部移除 (基于bounding box):")
    print(f"    高度范围: [{z_min:.3f}, {z_max:.3f}]m (跨度 {z_range:.3f}m)")
    print(f"    下限阈值: {lower_threshold:.3f}m (底部{bottom_percentile}%)")
    print(f"    上限阈值: {upper_threshold:.3f}m (顶部{top_percentile}%)")
    print(f"    移除底部点: {removed_bottom} ({removed_bottom/len(points)*100:.1f}%)")
    print(f"    移除顶部点: {removed_top} ({removed_top/len(points)*100:.1f}%)")
    print(f"    总移除: {removed_total} ({removed_total/len(points)*100:.1f}%)")
    print(f"    剩余点数: {len(filtered_points)}")
    
    return filtered_points, mask


def statistical_outlier_removal(points, k, std_ratio):
    """
    统计离群点移除 (SOR)
    
    参数:
        points: Nx3点云数组 [x, y, z]
        k: 每个点考虑的最近邻数量
        std_ratio: 标准差倍数阈值
    
    返回:
        filtered_points: 过滤后的点云
        mask: 保留点的布尔mask
    """
    from scipy.spatial import cKDTree
    
    if len(points) < k:
        return points, np.ones(len(points), dtype=bool)
    
    print(f"  SOR去噪: k={k}, std_ratio={std_ratio}")
    
    # 构建KD树
    tree = cKDTree(points)
    
    # 计算每个点到其k个最近邻的平均距离
    distances, _ = tree.query(points, k=k+1)  # k+1因为第一个是点本身
    mean_distances = distances[:, 1:].mean(axis=1)  # 排除自己
    
    # 计算全局统计量
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    
    # 过滤离群点（距离 > mean + std_ratio * std）
    threshold = global_mean + std_ratio * global_std
    mask = mean_distances <= threshold
    
    filtered_points = points[mask]
    
    removed = len(points) - len(filtered_points)
    print(f"    移除离群点: {removed} ({removed/len(points)*100:.1f}%)")
    print(f"    剩余点数: {len(filtered_points)}")
    
    return filtered_points, mask


def visualize_3d_with_peaks(points, peak_positions, direction, output_path, verbose=False):
    """
    3D点云可视化，标记检测到的植物位置
    
    参数:
        points: Nx3点云数组
        peak_positions: 检测到的峰位置
        direction: 方向轴
        output_path: 输出文件路径
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云（根据高度着色）
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=points[:, 2], s=1, cmap='viridis', alpha=0.4)
    
    # 标记峰位置（垂直线）
    if len(peak_positions) > 0:
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        
        for pos in peak_positions:
            if direction.lower() == 'x':
                # X方向的行，绘制垂直于X的平面线
                ax.plot([pos, pos], [points[:, 1].min(), points[:, 1].max()], 
                       [z_max, z_max], 'r-', linewidth=2, alpha=0.7)
            else:  # y
                # Y方向的行，绘制垂直于Y的平面线
                ax.plot([points[:, 0].min(), points[:, 0].max()], [pos, pos],
                       [z_max, z_max], 'r-', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(f'3D Point Cloud with Detected Plants (n={len(peak_positions)})', 
                fontsize=13, fontweight='bold')
    
    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Height (m)', fontsize=10)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  3D可视化已保存: {output_path}")


def calculate_peak_heights(original_points, peak_positions, direction):
    """
    计算每个峰的实际高度（使用原始点云）
    
    高度定义：每个峰在其范围内（与左右相邻峰中点之间）的最高点与最低点的垂直距离
    
    参数:
        original_points: Nx3原始点云数组 [x, y, z]（ground removal之前）
        peak_positions: 检测到的峰位置列表（1D数组）
        direction: 'x' 或 'y'，检测方向
    
    返回:
        heights: 每个峰对应的高度值（数组）
    """
    if len(peak_positions) == 0:
        return np.array([])
    
    # 选择坐标轴
    if direction.lower() == 'x':
        coord = original_points[:, 0]  # X坐标
    else:  # y
        coord = original_points[:, 1]  # Y坐标
    
    z = original_points[:, 2]  # Z高度
    
    # 确保峰位置已排序
    sorted_peaks = np.sort(peak_positions)
    n_peaks = len(sorted_peaks)
    
    heights = np.zeros(n_peaks)
    
    # 计算每个峰的范围边界
    for i in range(n_peaks):
        peak_pos = sorted_peaks[i]
        
        # 确定左边界
        if i == 0:
            # 第一个峰：使用点云最小值到与右边峰的中点
            left_bound = coord.min()
        else:
            # 与左边峰的中点
            left_bound = (sorted_peaks[i-1] + peak_pos) / 2.0
        
        # 确定右边界
        if i == n_peaks - 1:
            # 最后一个峰：使用与左边峰的中点到点云最大值
            right_bound = coord.max()
        else:
            # 与右边峰的中点
            right_bound = (peak_pos + sorted_peaks[i+1]) / 2.0
        
        # 找到这个范围内的所有点
        mask = (coord >= left_bound) & (coord <= right_bound)
        
        if np.sum(mask) > 0:
            # 计算该范围内的最大和最小高度
            z_in_range = z[mask]
            z_max = z_in_range.max()
            z_min = z_in_range.min()
            heights[i] = z_max - z_min
        else:
            # 如果没有点（理论上不应该发生），设为0
            heights[i] = 0.0
    
    # 恢复原始peak_positions的顺序
    if not np.array_equal(peak_positions, sorted_peaks):
        # 创建排序索引映射
        sort_indices = np.argsort(peak_positions)
        unsort_indices = np.argsort(sort_indices)
        heights = heights[unsort_indices]
    
    return heights

