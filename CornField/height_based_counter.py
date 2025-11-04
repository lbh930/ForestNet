#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Height-based Plant Counter
基于高度峰检测的单株植物计数

用法:
    python height_based_counter.py input.las --direction y --output results/
    python height_based_counter.py input.las --direction x --spacing 0.25
"""

import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from common import (
    save_detection_summary_yaml,
    remove_ground_points,
    statistical_outlier_removal,
    visualize_3d_with_peaks,
    calculate_peak_heights
)


def compute_height_profile(points, direction, bin_size, verbose=False):
    """
    计算沿指定方向的高度曲线（使用最大高度、平均高度等统计量）
    
    参数:
        points: Nx3点云数组 [x, y, z]
        direction: 'x' 或 'y'，沿哪个方向计算
        bin_size: bin大小（米）
    
    返回:
        bin_centers: bin中心坐标
        max_heights: 每个bin的最大高度
        mean_heights: 每个bin的平均高度
        percentile_95_heights: 每个bin的95百分位高度
        raw_counts: 每个bin的原始点数
    """
    if verbose:
        print(f"  计算沿{direction.upper()}轴的高度曲线...")
    
    # 选择坐标轴
    if direction.lower() == 'x':
        coord = points[:, 0]  # X坐标
        height = points[:, 2]  # Z高度
    elif direction.lower() == 'y':
        coord = points[:, 1]  # Y坐标
        height = points[:, 2]  # Z高度
    else:
        raise ValueError("direction必须是 'x' 或 'y'")
    
    # 创建bins
    coord_min, coord_max = coord.min(), coord.max()
    n_bins = int(np.ceil((coord_max - coord_min) / bin_size))
    
    if n_bins < 10:
        if verbose:
            print(f"    警告: bin数量太少 ({n_bins})")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    bins = np.linspace(coord_min, coord_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 使用digitize快速分配点到bins
    bin_indices = np.digitize(coord, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # 计算每个bin的高度统计量
    max_heights = np.zeros(n_bins)
    mean_heights = np.zeros(n_bins)
    percentile_95_heights = np.zeros(n_bins)
    raw_counts = np.zeros(n_bins, dtype=np.int32)
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        raw_counts[i] = count
        
        if count > 0:
            bin_heights = height[mask]
            max_heights[i] = bin_heights.max()
            mean_heights[i] = bin_heights.mean()
            percentile_95_heights[i] = np.percentile(bin_heights, 95)
    
    if verbose:
        print(f"    Bins: {n_bins}, 范围: [{coord_min:.2f}, {coord_max:.2f}]m")
        print(f"    最大高度范围: [{max_heights.min():.3f}, {max_heights.max():.3f}]m")
        print(f"    平均高度范围: [{mean_heights.min():.3f}, {mean_heights.mean():.3f}]m")
    
    return bin_centers, max_heights, mean_heights, percentile_95_heights, raw_counts


def compute_height_profile_with_kernel(points, direction, bin_size, 
                                       kernel_length, kernel_width, 
                                       kernel_height_percentile, verbose=False):
    """
    使用3D kernel沿行中心进行卷积来计算高度曲线
    
    参数:
        points: Nx3点云数组 [x, y, z]
        direction: 'x' 或 'y'，沿哪个方向计算（行延伸方向）
        bin_size: 沿行方向的步进大小（米）
        kernel_length: kernel在行方向上的长度（米）
        kernel_width: kernel在垂直于行方向上的宽度（米）
        kernel_height_percentile: 用于提取高度的百分位数（0-100）
    
    返回:
        bin_centers: bin中心坐标
        kernel_heights: 使用kernel提取的高度曲线
        raw_counts: 每个bin中kernel内的点数
    """
    if verbose:
        print(f"  使用3D kernel计算沿{direction.upper()}轴的高度曲线...")
        print(f"    Kernel尺寸: 长度={kernel_length*100:.1f}cm, 宽度={kernel_width*100:.1f}cm")
        print(f"    高度百分位数: {kernel_height_percentile}%")
    
    # 选择坐标轴
    if direction.lower() == 'x':
        along_coord = points[:, 0]  # 沿行方向（X）
        cross_coord = points[:, 1]   # 垂直于行方向（Y）
        height = points[:, 2]        # Z高度
    elif direction.lower() == 'y':
        along_coord = points[:, 1]  # 沿行方向（Y）
        cross_coord = points[:, 0]   # 垂直于行方向（X）
        height = points[:, 2]        # Z高度
    else:
        raise ValueError("direction必须是 'x' 或 'y'")
    
    # 计算行的中心位置（在垂直于行方向上）
    row_center = cross_coord.mean()
    half_kernel_width = kernel_width / 2.0
    
    # 预先过滤垂直方向的点（只保留kernel宽度内的点）
    cross_mask = (cross_coord >= row_center - half_kernel_width) & (cross_coord <= row_center + half_kernel_width)
    along_coord_filtered = along_coord[cross_mask]
    height_filtered = height[cross_mask]
    
    if len(along_coord_filtered) == 0:
        if verbose:
            print(f"    警告: kernel宽度内没有点")
        return np.array([]), np.array([]), np.array([])
    
    # 创建沿行方向的bins
    along_min, along_max = along_coord_filtered.min(), along_coord_filtered.max()
    n_bins = int(np.ceil((along_max - along_min) / bin_size))
    
    if n_bins < 10:
        if verbose:
            print(f"    警告: bin数量太少 ({n_bins})")
        return np.array([]), np.array([]), np.array([])
    
    bins = np.linspace(along_min, along_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 对每个bin位置，使用kernel提取局部点云的高度特征
    kernel_heights = np.zeros(n_bins)
    raw_counts = np.zeros(n_bins, dtype=np.int32)
    half_kernel_length = kernel_length / 2.0
    
    for i, center_pos in enumerate(bin_centers):
        # 只需检查沿行方向的范围（垂直方向已经预过滤）
        mask = (along_coord_filtered >= center_pos - half_kernel_length) & \
               (along_coord_filtered <= center_pos + half_kernel_length)
        
        count = np.sum(mask)
        raw_counts[i] = count
        
        if count > 0:
            kernel_heights[i] = np.percentile(height_filtered[mask], kernel_height_percentile)
    
    # 对空bin进行线性插值处理（快速版本）
    zero_mask = kernel_heights == 0
    if np.any(zero_mask) and np.any(~zero_mask):
        nonzero_indices = np.nonzero(~zero_mask)[0]
        zero_indices = np.nonzero(zero_mask)[0]
        kernel_heights[zero_indices] = np.interp(zero_indices, nonzero_indices, kernel_heights[nonzero_indices])
    
    if verbose:
        print(f"    Bins: {n_bins}, 范围: [{along_min:.2f}, {along_max:.2f}]m")
        print(f"    高度范围: [{kernel_heights.min():.3f}, {kernel_heights.max():.3f}]m")
        print(f"    平均点数/kernel: {raw_counts.mean():.0f}")
    
    return bin_centers, kernel_heights, raw_counts


def detect_plants_from_height(bin_centers, heights, expected_spacing,
                              min_prominence, height_metric='max', verbose=False):
    """
    从高度曲线检测植物峰
    
    参数:
        bin_centers: bin中心坐标
        heights: 高度值（可以是max/mean/percentile）
        expected_spacing: 预期株间距（米）
        min_prominence: 最小峰突出度（相对值）
        height_metric: 使用的高度指标名称（用于显示）
    
    返回:
        peak_positions: 检测到的植物位置
        peak_heights: 对应的高度值
        smoothed_heights: 平滑后的高度曲线
    """
    if verbose:
        print(f"  检测植物峰（使用{height_metric}高度，预期株间距: {expected_spacing*100:.0f}cm）...")
    
    if len(heights) < 10:
        return np.array([]), np.array([]), heights
    
    # 计算bin大小
    bin_size = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.01
    
    # Step 1: 高斯平滑
    # kernel方法已经做了空间聚合，不需要额外平滑
    if height_metric == 'kernel':
        smoothed_heights = heights.copy()
    else:
        # 传统方法需要平滑
        sigma = 1
        smoothed_heights = gaussian_filter1d(heights, sigma=sigma)
    
    # Step 2: 峰检测
    # distance参数：预期株间距对应的bin数
    min_distance_bins = int(expected_spacing / bin_size)
    
    # prominence: 相对于局部基线的突出度
    abs_prominence = min_prominence * smoothed_heights.max()
    
    peaks, properties = find_peaks(
        smoothed_heights, 
        distance=min_distance_bins,
        prominence=abs_prominence,
        width=1  # 至少1个bin宽
    )
    
    if len(peaks) == 0:
        if verbose:
            print(f"    未检测到峰")
        return np.array([]), np.array([]), smoothed_heights
    
    # 获取峰位置和高度值
    peak_positions = bin_centers[peaks]
    peak_heights = smoothed_heights[peaks]
    
    # 计算实际平均间距
    if len(peak_positions) > 1:
        actual_spacings = np.diff(peak_positions)
        avg_spacing = actual_spacings.mean()
        std_spacing = actual_spacings.std()
        if verbose:
            print(f"    检测到 {len(peaks)} 个峰")
            print(f"    实际平均株间距: {avg_spacing*100:.1f} ± {std_spacing*100:.1f} cm")
    else:
        if verbose:
            print(f"    检测到 {len(peaks)} 个峰")
    
    return peak_positions, peak_heights, smoothed_heights


def visualize_height_profile(bin_centers, max_heights, mean_heights, 
                             percentile_95_heights, smoothed_heights,
                             peak_positions, peak_heights, raw_counts,
                             output_path, direction, height_metric, verbose=False):
    """
    可视化高度曲线和检测结果
    
    参数:
        bin_centers: bin中心坐标
        max_heights: 最大高度
        mean_heights: 平均高度
        percentile_95_heights: 95百分位高度
        smoothed_heights: 平滑后的高度（用于检测的那个）
        peak_positions: 峰位置
        peak_heights: 峰高度值
        raw_counts: 原始点数
        output_path: 输出文件路径
        direction: 方向轴
        height_metric: 使用的高度指标
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 上图：高度曲线对比
    ax1 = axes[0]
    ax1.plot(bin_centers, max_heights, 'r-', linewidth=1, alpha=0.3, label='Max Height')
    ax1.plot(bin_centers, percentile_95_heights, 'orange', linewidth=1, alpha=0.5, label='95th Percentile Height')
    ax1.plot(bin_centers, mean_heights, 'b-', linewidth=1, alpha=0.4, label='Mean Height')
    ax1.plot(bin_centers, smoothed_heights, 'g-', linewidth=2.5, 
            label=f'Smoothed {height_metric.capitalize()} Height (used for detection)')
    
    # 标记峰位置
    if len(peak_positions) > 0:
        ax1.plot(peak_positions, peak_heights, 'rx', 
                markersize=12, markeredgewidth=2.5, 
                label=f'Detected Plants (n={len(peak_positions)})')
        
        # 添加垂直虚线
        for pos in peak_positions:
            ax1.axvline(pos, color='red', linestyle='--', linewidth=1, alpha=0.3)
    
    ax1.set_xlabel(f'{direction.upper()} Coordinate (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title(f'Plant Detection from Height Profile along {direction.upper()}-axis', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 中图：使用的高度曲线（放大显示）
    ax2 = axes[1]
    # 根据height_metric选择显示哪个
    if height_metric == 'kernel':
        raw_heights = max_heights  # kernel方法中，max_heights就是kernel_heights
        metric_label = 'Kernel'
    elif height_metric == 'max':
        raw_heights = max_heights
        metric_label = 'Max'
    elif height_metric == 'mean':
        raw_heights = mean_heights
        metric_label = 'Mean'
    else:  # percentile_95
        raw_heights = percentile_95_heights
        metric_label = 'Percentile 95'
    
    ax2.plot(bin_centers, raw_heights, 'b-', linewidth=0.8, alpha=0.4, 
            label=f'Raw {metric_label} Height')
    ax2.plot(bin_centers, smoothed_heights, 'b-', linewidth=2, 
            label=f'Smoothed {metric_label} Height')
    
    # 标记峰位置
    if len(peak_positions) > 0:
        ax2.plot(peak_positions, peak_heights, 'rx', 
                markersize=12, markeredgewidth=2.5, 
                label=f'Detected Plants (n={len(peak_positions)})')
        
        # 添加垂直虚线
        for pos in peak_positions:
            ax2.axvline(pos, color='red', linestyle='--', linewidth=1, alpha=0.3)
    
    ax2.set_xlabel(f'{direction.upper()} Coordinate (m)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title(f'{metric_label} Height Profile (Detection Basis)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)
    
    # 下图：原始点数分布
    ax3 = axes[2]
    ax3.bar(bin_centers, raw_counts, width=bin_centers[1]-bin_centers[0] if len(bin_centers) > 1 else 0.01,
            color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    
    # 标记峰位置
    if len(peak_positions) > 0:
        for pos in peak_positions:
            ax3.axvline(pos, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax3.set_xlabel(f'{direction.upper()} Coordinate (m)', fontsize=12)
    ax3.set_ylabel('Point Count per Bin', fontsize=12)
    ax3.set_title('Point Distribution Histogram', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  可视化已保存: {output_path}")


def height_count_from_row(points, direction, expected_spacing, 
                          bin_size, apply_sor, sor_k, sor_std_ratio,
                          remove_ground, ground_percentile, top_percentile,
                          min_prominence, height_metric, output_dir, 
                          kernel_length=None, kernel_width=None, kernel_height_percentile=None,
                          row_center=None, row_status: str | None = None, verbose: bool = False):
    """
    从单行点云中基于高度检测并计数植物
    
    参数:
        points: Nx3点云数组 [x, y, z]
        direction: 'x' 或 'y'，沿哪个方向计算
        expected_spacing: 预期株间距（米）
        bin_size: bin大小（米）
        apply_sor: 是否应用统计离群点移除
        sor_k: SOR的k值
        sor_std_ratio: SOR的标准差倍数
        remove_ground: 是否移除地面点和顶部点
        ground_percentile: 移除底部百分比
        top_percentile: 移除顶部百分比
        min_prominence: 最小峰突出度（相对值）
        height_metric: 使用哪个高度指标 ('max', 'mean', 'percentile_95', 'kernel')
        output_dir: 可视化输出目录
        kernel_length: kernel在行方向上的长度（米），仅在height_metric='kernel'时使用
        kernel_width: kernel在垂直于行方向上的宽度（米），仅在height_metric='kernel'时使用
        kernel_height_percentile: kernel内高度的百分位数（0-100），仅在height_metric='kernel'时使用
        row_center: 行的中心坐标（用于计算完整XY坐标），如果为None则使用点云均值
    
    返回:
        plant_count: 检测到的植物数量
        peak_positions: 植物位置列表
        results_dict: 包含详细结果的字典
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"基于高度的植物计数")
        print(f"{'='*60}")
        print(f"输入点数: {len(points):,}")
        print(f"检测方向: {direction.upper()}轴")
        print(f"预期株间距: {expected_spacing*100:.0f}cm")
        print(f"高度指标: {height_metric}")
    
    # Step 1: 统计离群点移除（可选）
    if apply_sor:
        points, _ = statistical_outlier_removal(points, k=sor_k, std_ratio=sor_std_ratio)
    
    # 保存SOR后的点云（用于后续高度计算）
    sor_cleaned_points = points.copy()
    
    # Step 2: 移除地面点和顶部点
    if remove_ground:
        points, _ = remove_ground_points(points, bottom_percentile=ground_percentile, 
                                        top_percentile=top_percentile)
    
    if len(points) < 50:
        print("错误: 点数太少，无法进行分析")
        return 0, np.array([]), {}
    
    # Step 3: 计算高度曲线
    if height_metric == 'kernel':
        # 使用3D kernel卷积方法
        if kernel_length is None or kernel_width is None or kernel_height_percentile is None:
            print("错误: height_metric='kernel'时必须提供kernel_length, kernel_width, kernel_height_percentile")
            return 0, np.array([]), {}
        
        bin_centers, heights_for_detection, raw_counts = compute_height_profile_with_kernel(
            points, direction=direction, bin_size=bin_size,
            kernel_length=kernel_length, kernel_width=kernel_width,
            kernel_height_percentile=kernel_height_percentile, verbose=verbose
        )
        
        if len(heights_for_detection) == 0:
            print("错误: 无法计算kernel高度曲线")
            return 0, np.array([]), {}
        
        # 为了兼容可视化，创建虚拟的其他高度数组
        max_heights = heights_for_detection.copy()
        mean_heights = heights_for_detection.copy()
        percentile_95_heights = heights_for_detection.copy()
    else:
        # 使用传统的2D投影方法
        bin_centers, max_heights, mean_heights, percentile_95_heights, raw_counts = compute_height_profile(
            points, direction=direction, bin_size=bin_size, verbose=verbose
        )
        
        if len(max_heights) == 0:
            print("错误: 无法计算高度曲线")
            return 0, np.array([]), {}
        
        # Step 4: 选择使用哪个高度指标进行峰检测
        if height_metric == 'max':
            heights_for_detection = max_heights
        elif height_metric == 'mean':
            heights_for_detection = mean_heights
        elif height_metric == 'percentile_95':
            heights_for_detection = percentile_95_heights
        else:
            print(f"警告: 未知的height_metric '{height_metric}'，使用'max'")
            heights_for_detection = max_heights
            height_metric = 'max'
    
    # Step 5: 峰检测
    peak_positions, peak_heights, smoothed_heights = detect_plants_from_height(
        bin_centers, heights_for_detection, 
        expected_spacing=expected_spacing, 
        min_prominence=min_prominence,
        height_metric=height_metric,
        verbose=verbose
    )
    
    plant_count = len(peak_positions)
    
    # Step 6: 计算实际高度（使用SOR后但未掐头去尾的点云）
    if plant_count > 0:
        if verbose:
            print(f"  计算植物实际高度（使用SOR后的完整点云）...")
        actual_heights = calculate_peak_heights(sor_cleaned_points, peak_positions, direction)
        if verbose:
            print(f"    高度范围: [{actual_heights.min():.3f}, {actual_heights.max():.3f}]m")
            print(f"    平均高度: {actual_heights.mean():.3f}m")
    else:
        actual_heights = np.array([])
    
    # 生成可视化（如果指定了输出目录）
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 高度曲线图
        profile_path = output_dir / f"height_profile_{direction}.png"
        visualize_height_profile(
            bin_centers, max_heights, mean_heights, percentile_95_heights,
            smoothed_heights, peak_positions, peak_heights, raw_counts,
            profile_path, direction, height_metric, verbose=verbose
        )
        
        # 3D点云图
        cloud_path = output_dir / f"3d_pointcloud_{direction}.png"
        visualize_3d_with_peaks(points, peak_positions, direction, cloud_path, verbose=verbose)
        
        # 保存结果为YAML文件
        summary_path = output_dir / "detection_summary.yaml"
        
        # 确定行的中心坐标
        if row_center is None:
            # 使用点云的垂直于检测方向的均值作为行中心
            if direction == 'x':
                row_center = points[:, 0].mean()  # X方向的行，取X均值
            else:
                row_center = points[:, 1].mean()  # Y方向的行，取Y均值
        
        # 构建植物列表（包含完整XY坐标）
        plants = []
        for i, (pos, height) in enumerate(zip(peak_positions, actual_heights), 1):
            if direction == 'x':
                # 沿X方向检测，pos是X坐标，Y是行中心
                x_coord = float(pos)
                y_coord = float(row_center)
            else:
                # 沿Y方向检测，pos是Y坐标，X是行中心
                x_coord = float(row_center)
                y_coord = float(pos)
            
            plants.append({
                'id': i,
                'x': x_coord,
                'y': y_coord,
                'height': float(height),
            })
        
        # 构建summary数据
        summary_data = {
            'method': 'height',
            'input_points': len(points),
            'direction': direction,
            'expected_spacing_cm': expected_spacing * 100,
            'bin_size_cm': bin_size * 100,
            'ground_removal': remove_ground,
            'ground_percentile': ground_percentile,
            'top_percentile': top_percentile,
            'sor_applied': apply_sor,
            'sor_k': sor_k,
            'sor_std_ratio': sor_std_ratio,
            'height_metric': height_metric,
            'plant_count': plant_count,
            'plants': plants,
        }
        
        save_detection_summary_yaml(summary_path, summary_data)

        if verbose:
            print(f"\n结果已保存到: {output_dir}")

    # 返回结果
    results = {
        'plant_count': plant_count,
        'peak_positions': peak_positions,
        'peak_heights': peak_heights,
        'actual_heights': actual_heights,  # 实际计算的高度
        'bin_centers': bin_centers,
        'max_heights': max_heights,
        'mean_heights': mean_heights,
        'percentile_95_heights': percentile_95_heights,
        'smoothed_heights': smoothed_heights,
        'raw_counts': raw_counts,
        'filtered_points': points,
        'height_metric': height_metric
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ 检测完成！共检测到 {plant_count} 株植物")
        print(f"{'='*60}\n")

    # Row-level minimal status log (if provided)
    if row_status:
        print(row_status)
    
    return plant_count, peak_positions, results


def main():
    """主函数 - 用于测试"""
    if len(sys.argv) < 2:
        print("用法: python height_based_counter.py <las文件> [选项]")
        print("\n选项:")
        print("  --direction x|y     检测方向（默认: y）")
        print("  --spacing FLOAT     预期株间距（米）（默认: 0.08）")
        print("  --bin_size FLOAT    bin大小（米）（默认: 0.005）")
        print("  --prominence FLOAT  峰突出度阈值（相对值）（默认: 0.02）")
        print("  --height_metric STR 使用的高度指标: max|mean|percentile_95|kernel (默认: max)")
        print("  --kernel_length FLOAT  Kernel长度（米）（仅kernel模式，默认: 0.08）")
        print("  --kernel_width FLOAT   Kernel宽度（米）（仅kernel模式，默认: 0.15）")
        print("  --kernel_percentile INT Kernel高度百分位（仅kernel模式，默认: 90）")
        print("  --output DIR        可视化输出目录（默认: height_results）")
        print("  --no-ground         不移除地面点和顶部点")
        print("  --ground_pct FLOAT  地面移除百分比（默认: 30）")
        print("  --top_pct FLOAT     顶部移除百分比（默认: 30）")
        print("  --no-sor            不应用统计离群点移除")
        print("  --sor_k INT         SOR的k值（默认: 20）")
        print("  --sor_std FLOAT     SOR的标准差倍数（默认: 2.0）")
        print("\n示例:")
        print("  python height_based_counter.py row_points.las")
        print("  python height_based_counter.py row_points.las --direction x --spacing 0.3")
        print("  python height_based_counter.py row_points.las --height_metric percentile_95")
        print("  python height_based_counter.py row_points.las --height_metric kernel")
        print("  python height_based_counter.py row_points.las --output my_results/")
        sys.exit(1)
    
    # 解析参数
    las_path = sys.argv[1]
    
    # 默认值
    direction = 'y'
    expected_spacing = 0.08
    bin_size = 0.005
    min_prominence = 0.02
    height_metric = 'max'
    kernel_length = 0.08
    kernel_width = 0.15
    kernel_height_percentile = 90
    output_dir = 'height_results'
    remove_ground = True
    ground_percentile = 30
    top_percentile = 30
    apply_sor = True
    sor_k = 20
    sor_std_ratio = 2.0
    
    if '--direction' in sys.argv:
        idx = sys.argv.index('--direction')
        if idx + 1 < len(sys.argv):
            direction = sys.argv[idx + 1].lower()
            if direction not in ['x', 'y']:
                print("错误: direction必须是 'x' 或 'y'")
                sys.exit(1)
    
    if '--spacing' in sys.argv:
        idx = sys.argv.index('--spacing')
        if idx + 1 < len(sys.argv):
            expected_spacing = float(sys.argv[idx + 1])
    
    if '--bin_size' in sys.argv:
        idx = sys.argv.index('--bin_size')
        if idx + 1 < len(sys.argv):
            bin_size = float(sys.argv[idx + 1])
    
    if '--prominence' in sys.argv:
        idx = sys.argv.index('--prominence')
        if idx + 1 < len(sys.argv):
            min_prominence = float(sys.argv[idx + 1])
    
    if '--height_metric' in sys.argv:
        idx = sys.argv.index('--height_metric')
        if idx + 1 < len(sys.argv):
            height_metric = sys.argv[idx + 1].lower()
            if height_metric not in ['max', 'mean', 'percentile_95', 'kernel']:
                print("错误: height_metric必须是 'max', 'mean', 'percentile_95', 或 'kernel'")
                sys.exit(1)
    
    if '--kernel_length' in sys.argv:
        idx = sys.argv.index('--kernel_length')
        if idx + 1 < len(sys.argv):
            kernel_length = float(sys.argv[idx + 1])
    
    if '--kernel_width' in sys.argv:
        idx = sys.argv.index('--kernel_width')
        if idx + 1 < len(sys.argv):
            kernel_width = float(sys.argv[idx + 1])
    
    if '--kernel_percentile' in sys.argv:
        idx = sys.argv.index('--kernel_percentile')
        if idx + 1 < len(sys.argv):
            kernel_height_percentile = int(sys.argv[idx + 1])
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    if '--no-ground' in sys.argv:
        remove_ground = False
    
    if '--ground_pct' in sys.argv:
        idx = sys.argv.index('--ground_pct')
        if idx + 1 < len(sys.argv):
            ground_percentile = float(sys.argv[idx + 1])
    
    if '--top_pct' in sys.argv:
        idx = sys.argv.index('--top_pct')
        if idx + 1 < len(sys.argv):
            top_percentile = float(sys.argv[idx + 1])
    
    if '--no-sor' in sys.argv:
        apply_sor = False
    
    if '--sor_k' in sys.argv:
        idx = sys.argv.index('--sor_k')
        if idx + 1 < len(sys.argv):
            sor_k = int(sys.argv[idx + 1])
    
    if '--sor_std' in sys.argv:
        idx = sys.argv.index('--sor_std')
        if idx + 1 < len(sys.argv):
            sor_std_ratio = float(sys.argv[idx + 1])
    
    # 检查文件
    if not Path(las_path).exists():
        print(f"错误: 文件不存在: {las_path}")
        sys.exit(1)
    
    # 读取点云
    print(f"读取点云文件: {las_path}")
    las = laspy.read(las_path)
    
    # 提取坐标
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    
    points = np.column_stack([x, y, z])
    
    print(f"点云范围:")
    print(f"  X: [{x.min():.3f}, {x.max():.3f}] m")
    print(f"  Y: [{y.min():.3f}, {y.max():.3f}] m")
    print(f"  Z: [{z.min():.3f}, {z.max():.3f}] m")
    
    # 执行植物计数
    plant_count, peak_positions, results = height_count_from_row(
        points,
        direction=direction,
        expected_spacing=expected_spacing,
        bin_size=bin_size,
        apply_sor=apply_sor,
        sor_k=sor_k,
        sor_std_ratio=sor_std_ratio,
        remove_ground=remove_ground,
        ground_percentile=ground_percentile,
        top_percentile=top_percentile,
        min_prominence=min_prominence,
        height_metric=height_metric,
        kernel_length=kernel_length if height_metric == 'kernel' else None,
        kernel_width=kernel_width if height_metric == 'kernel' else None,
        kernel_height_percentile=kernel_height_percentile if height_metric == 'kernel' else None,
        output_dir=output_dir,
        verbose=True
    )
    
    # 打印结果
    if plant_count > 0:
        print("\n检测到的植物位置:")
        for i, (pos, height) in enumerate(zip(peak_positions, results['actual_heights']), 1):
            print(f"  植物 {i}: {direction.upper()} = {pos:.3f} m (实际高度: {height:.3f} m)")


if __name__ == "__main__":
    main()
