#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crop Row Splitter
简化版：从点云中检测作物行并分割保存每行的点云

用法:
    python crop_row_splitter.py input.las --tile_size 10
"""

import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, gaussian_filter, distance_transform_edt, binary_dilation
import shutil


def read_las_file(las_path):
    """读取LAS文件并返回点云坐标"""
    print(f"读取点云文件: {las_path}")
    t0 = time.time()
    las = laspy.read(las_path)
    
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    
    print(f"点云总数: {len(x):,} 个点 (耗时: {time.time()-t0:.2f}秒)")
    print(f"X范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Y范围: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Z范围: [{z.min():.3f}, {z.max():.3f}]")
    
    return x, y, z


def save_las_file(x, y, z, output_path):
    """保存点云为LAS文件"""
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([x.min(), y.min(), z.min()])
    
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    
    las.write(output_path)


def normalize_z_to_zero(x, y, z):
    """将点云Z轴标准化，使最低点位于z=0"""
    z_min = z.min()
    z_normalized = z - z_min
    print(f"\nZ轴归一化: 最小值 {z_min:.3f} -> 0.000")
    return x, y, z_normalized

def split_into_tiles(x, y, z, tile_size=10.0):
    """将点云分割成NxN米的tiles，自动跳过尺寸不足的边缘tile"""
    print(f"\n分割成 {tile_size}x{tile_size}m tiles...")
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    n_tiles_x = int(np.ceil(x_range / tile_size))
    n_tiles_y = int(np.ceil(y_range / tile_size))
    
    print(f"  田地范围: X=[{x_min:.2f}, {x_max:.2f}] ({x_range:.2f}m), "
          f"Y=[{y_min:.2f}, {y_max:.2f}] ({y_range:.2f}m)")
    print(f"  Tile网格: {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} tiles")
    
    tiles = {}
    tile_info = {}
    skipped_tiles = 0
    
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            tile_x_min = x_min + i * tile_size
            tile_x_max = tile_x_min + tile_size
            tile_y_min = y_min + j * tile_size
            tile_y_max = tile_y_min + tile_size
            
            # 计算实际tile尺寸（考虑边界）
            actual_x_size = min(tile_x_max, x_max) - tile_x_min
            actual_y_size = min(tile_y_max, y_max) - tile_y_min
            
            # 跳过尺寸不足的tile（小于标准tile尺寸的80%）
            if actual_x_size < tile_size * 0.8 or actual_y_size < tile_size * 0.5:
                skipped_tiles += 1
                continue
            
            mask = (x >= tile_x_min) & (x < tile_x_max) & \
                   (y >= tile_y_min) & (y < tile_y_max)
            
            if np.sum(mask) > 100:
                tile_x = x[mask]
                tile_y = y[mask]
                tile_z = z[mask]
                
                tile_id = (i, j)
                tiles[tile_id] = (tile_x, tile_y, tile_z)
                tile_info[tile_id] = {
                    'x_min': tile_x_min, 'x_max': tile_x_max,
                    'y_min': tile_y_min, 'y_max': tile_y_max,
                    'n_points': len(tile_x)
                }
    
    print(f"  有效tiles: {len(tiles)} / {n_tiles_x * n_tiles_y} "
          f"(跳过尺寸不足: {skipped_tiles}, 点数不足: {n_tiles_x * n_tiles_y - len(tiles) - skipped_tiles})")
    
    return tiles, tile_info


def compute_height_profile(coord, z, granularity=0.02):
    """计算沿某个轴的平均高度分布"""
    coord_min = coord.min()
    coord_max = coord.max()
    
    bins = np.arange(coord_min, coord_max + granularity, granularity)
    mean_heights, _, _ = binned_statistic(coord, z, statistic='mean', bins=bins)
    bin_centers = bins[:-1] + granularity / 2
    
    return bin_centers, mean_heights


def detect_crop_rows(centers, heights, sigma=5):
    """
    检测作物行peaks
    
    返回:
        original_peaks: peak的索引
        full_smoothed: 平滑后的完整数组
        regularity_score: 间距规律度分数（0-1，越高越规律）
    """
    # 移除NaN
    valid_mask = ~np.isnan(heights)
    valid_heights = heights[valid_mask]
    
    if len(valid_heights) < 10:
        return np.array([]), np.full_like(heights, np.nan), 0.0
    
    # 高斯平滑
    smoothed = gaussian_filter1d(valid_heights, sigma=sigma)
    
    # 检测peaks
    peaks, _ = find_peaks(smoothed, prominence=0.02, distance=10)
    
    if len(peaks) < 2:
        return np.array([]), np.full_like(heights, np.nan), 0.0
    
    # 转换回原始索引
    original_peaks = np.where(valid_mask)[0][peaks]
    
    # 同时返回平滑后的完整数组（用于可视化）
    full_smoothed = np.full_like(heights, np.nan)
    full_smoothed[valid_mask] = smoothed
    
    # 计算间距规律度
    # 使用峰之间的间距的变异系数（CV）来评估规律度
    # CV = std / mean，越小越规律
    # regularity_score = 1 / (1 + CV)，范围0-1，越大越规律
    
    if len(peaks) >= 2:
        # 计算峰间距
        peak_positions = centers[original_peaks]
        spacings = np.diff(peak_positions)
        
        if len(spacings) > 0:
            mean_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            
            if mean_spacing > 0:
                cv = std_spacing / mean_spacing  # 变异系数
                regularity_score = 1.0 / (1.0 + cv)  # 转换为0-1分数，CV越小分数越高
            else:
                regularity_score = 0.0
        else:
            regularity_score = 0.0
    else:
        regularity_score = 0.0
    
    return original_peaks, full_smoothed, regularity_score


def plot_height_profiles(x_centers, x_heights, x_peaks, x_smoothed,
                         y_centers, y_heights, y_peaks, y_smoothed,
                         chosen_direction, output_prefix):
    """绘制高度曲线图，只标记选中方向的peaks"""
    # 计算X和Y的实际范围比例
    x_range = x_centers.max() - x_centers.min() if len(x_centers) > 0 else 1
    y_range = y_centers.max() - y_centers.min() if len(y_centers) > 0 else 1
    
    # 设置子图宽度比例，使其与数据范围成正比
    width_ratios = [x_range, y_range]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': width_ratios})
    
    # X-axis height profile
    ax1 = axes[0]
    valid_x = ~np.isnan(x_heights)
    ax1.plot(x_centers[valid_x], x_heights[valid_x], 'b-', 
             linewidth=0.5, alpha=0.3, label='Raw data')
    
    valid_smooth_x = ~np.isnan(x_smoothed)
    ax1.plot(x_centers[valid_smooth_x], x_smoothed[valid_smooth_x], 'b-', 
             linewidth=1, alpha=0.8, label='Smoothed')
    
    # 只有当X是选中方向时才标记peaks
    if chosen_direction == 'x' and len(x_peaks) > 0:
        ax1.plot(x_centers[x_peaks], x_heights[x_peaks], 'rx', 
                markersize=8, markeredgewidth=2, label=f'Detected Rows (n={len(x_peaks)})')
    
    ax1.set_xlabel('X Coordinate (m)', fontsize=12)
    ax1.set_ylabel('Mean Height (m)', fontsize=12)
    title = f'X-axis: {len(x_peaks)} rows detected'
    if chosen_direction == 'x':
        title += ' [SELECTED]'
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Y-axis height profile
    ax2 = axes[1]
    valid_y = ~np.isnan(y_heights)
    ax2.plot(y_centers[valid_y], y_heights[valid_y], 'g-', 
             linewidth=0.5, alpha=0.3, label='Raw data')
    
    valid_smooth_y = ~np.isnan(y_smoothed)
    ax2.plot(y_centers[valid_smooth_y], y_smoothed[valid_smooth_y], 'g-', 
             linewidth=1, alpha=0.8, label='Smoothed')
    
    # 只有当Y是选中方向时才标记peaks
    if chosen_direction == 'y' and len(y_peaks) > 0:
        ax2.plot(y_centers[y_peaks], y_heights[y_peaks], 'rx', 
                markersize=8, markeredgewidth=2, label=f'Detected Rows (n={len(y_peaks)})')
    
    ax2.set_xlabel('Y Coordinate (m)', fontsize=12)
    ax2.set_ylabel('Mean Height (m)', fontsize=12)
    title = f'Y-axis: {len(y_peaks)} rows detected'
    if chosen_direction == 'y':
        title += ' [SELECTED]'
    ax2.set_title(title, fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    output_file = f"{output_prefix}_height_profiles.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  高度曲线图已保存: {output_file}")


def create_chm_visualization(chm_smoothed, x_min, x_max, y_min, y_max,
                            x_centers, x_peaks, y_centers, y_peaks,
                            chosen_direction, output_prefix, boundaries=None):
    """生成CHM可视化（使用预计算的平滑CHM）"""
    print("  生成CHM可视化...")
    t0 = time.time()
    
    # 计算实际宽高比
    x_range = x_max - x_min
    y_range = y_max - y_min
    aspect_ratio = y_range / x_range
    
    # 根据比例调整figure大小（基准宽度16英寸）
    fig_width = 16
    fig_height = fig_width * aspect_ratio
    
    # 限制高度范围
    if fig_height > 20:
        fig_height = 20
        fig_width = fig_height / aspect_ratio
    elif fig_height < 8:
        fig_height = 8
        fig_width = fig_height / aspect_ratio
    
    # 创建figure
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 显示CHM，设置aspect='equal'保持真实比例
    im = ax.imshow(chm_smoothed, extent=[x_min, x_max, y_min, y_max], 
                   origin='lower', cmap='terrain', aspect='equal', 
                   interpolation='bilinear', alpha=0.9)
    
    # 只叠加选中方向的rows
    if chosen_direction == 'x' and len(x_peaks) > 0:
        for peak_idx in x_peaks:
            x_pos = x_centers[peak_idx]
            ax.axvline(x=x_pos, color='blue', linewidth=1.5, alpha=0.8, linestyle='-')
        direction_label = f'X-axis rows (n={len(x_peaks)}) [SELECTED]'
        row_count_str = f'{len(x_peaks)} X-rows'
        
        # 绘制边界曲线（红色细虚线）
        if boundaries is not None and len(boundaries) > 0:
            for boundary in boundaries:
                y_along = boundary['along']
                x_min = boundary['across_min']
                x_max = boundary['across_max']
                ax.plot(x_min, y_along, 'r--', linewidth=0.5, alpha=0.6)
                ax.plot(x_max, y_along, 'r--', linewidth=0.5, alpha=0.6)
        
    elif chosen_direction == 'y' and len(y_peaks) > 0:
        for peak_idx in y_peaks:
            y_pos = y_centers[peak_idx]
            ax.axhline(y=y_pos, color='blue', linewidth=1.5, alpha=0.8, linestyle='-')
        direction_label = f'Y-axis rows (n={len(y_peaks)}) [SELECTED]'
        row_count_str = f'{len(y_peaks)} Y-rows'
        
        # 绘制边界曲线（红色细虚线）
        if boundaries is not None and len(boundaries) > 0:
            for boundary in boundaries:
                x_along = boundary['along']
                y_min = boundary['across_min']
                y_max = boundary['across_max']
                ax.plot(x_along, y_min, 'r--', linewidth=0.5, alpha=0.6)
                ax.plot(x_along, y_max, 'r--', linewidth=0.5, alpha=0.6)
        
    else:
        direction_label = 'No rows detected'
        row_count_str = '0 rows'
    
    ax.set_xlabel('X Coordinate (m)', fontsize=13)
    ax.set_ylabel('Y Coordinate (m)', fontsize=13)
    ax.set_title(f'Canopy Height Model with Detected Crop Rows\n{row_count_str}',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)', fontsize=12)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = []
    if chosen_direction == 'x':
        legend_elements.append(
            Line2D([0], [0], color='blue', linewidth=2.5, label=direction_label)
        )
        if boundaries is not None and len(boundaries) > 0:
            legend_elements.append(
                Line2D([0], [0], color='red', linewidth=0.5, linestyle='--', 
                       label=f'Row boundaries (n={len(boundaries)+1})')
            )
    elif chosen_direction == 'y':
        legend_elements.append(
            Line2D([0], [0], color='blue', linewidth=2.5, label=direction_label)
        )
        if boundaries is not None and len(boundaries) > 0:
            legend_elements.append(
                Line2D([0], [0], color='red', linewidth=0.5, linestyle='--', 
                       label=f'Row boundaries (n={len(boundaries)+1})')
            )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                 framealpha=0.9, edgecolor='black')
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    output_file = f"{output_prefix}_chm.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  CHM已保存: {output_file} (耗时: {time.time()-t0:.2f}s)")


def create_smoothed_chm(tile_x, tile_y, tile_z, resolution=0.02):
    """
    创建平滑的CHM（供可视化和边界计算共用）
    
    参数:
        tile_x, tile_y, tile_z: tile点云数据
        resolution: CHM分辨率（默认2cm）
    
    返回:
        chm_smoothed: 平滑后的CHM数组
        x_min, x_max, y_min, y_max: 边界坐标
        nx, ny: 网格尺寸
    """
    x_min, x_max = tile_x.min(), tile_x.max()
    y_min, y_max = tile_y.min(), tile_y.max()
    
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    
    # 创建CHM（使用最大高度）
    chm = np.full((ny, nx), np.nan)
    x_idx = ((tile_x - x_min) / resolution).astype(int)
    y_idx = ((tile_y - y_min) / resolution).astype(int)
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)
    
    for i in range(len(tile_x)):
        if np.isnan(chm[y_idx[i], x_idx[i]]) or tile_z[i] > chm[y_idx[i], x_idx[i]]:
            chm[y_idx[i], x_idx[i]] = tile_z[i]
    
    # 填充并平滑CHM
    mask = ~np.isnan(chm)
    if np.sum(mask) > 0:
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        chm_filled = chm[tuple(indices)]
        chm_smoothed = gaussian_filter(chm_filled, sigma=2.5, mode='nearest')
        dilated_mask = binary_dilation(mask, iterations=5)
        chm_smoothed[~dilated_mask] = np.nan
    else:
        chm_smoothed = chm
    
    return chm_smoothed, x_min, x_max, y_min, y_max, nx, ny


def calculate_row_boundaries_curves(row_positions, chm_smoothed, x_min, y_min, nx, ny, 
                                    row_direction, resolution=0.02, granularity=0.1):
    """
    计算每个row的边界曲线（基于平滑CHM在垂直方向上找最低点）
    
    参数:
        row_positions: 排序后的row中心位置列表
        chm_smoothed: 平滑后的CHM数组
        x_min, y_min: CHM的起始坐标
        nx, ny: CHM的网格尺寸
        row_direction: 'x' 或 'y'
        resolution: CHM分辨率（默认2cm）
        granularity: 沿row方向的采样间隔（默认10cm）
    
    返回:
        boundary_curves: [{'along': [...], 'across_min': [...], 'across_max': [...]}, ...]
            对于每个row，返回沿着row方向的坐标和对应的边界位置
    """
    if len(row_positions) == 0:
        return []
    
    n_rows = len(row_positions)
    
    # 计算along方向的范围和采样点
    if row_direction == 'x':
        along_min = y_min
        along_max = y_min + ny * resolution
    else:
        along_min = x_min
        along_max = x_min + nx * resolution
    
    along_samples = np.arange(along_min, along_max + granularity, granularity)
    
    # 为每对相邻rows计算分割曲线
    split_curves = []  # 存储每个间隙的分割曲线
    
    for i in range(n_rows - 1):
        pos1 = row_positions[i]
        pos2 = row_positions[i + 1]
        
        split_positions = []
        valid_along = []
        
        # 对于沿着row方向的每个采样点
        for along_pos in along_samples:
            # 确定沿着方向的像素索引范围
            if row_direction == 'x':
                # X方向的row，沿Y延伸
                y_idx_center = int((along_pos - y_min) / resolution)
                if y_idx_center < 0 or y_idx_center >= ny:
                    continue
                
                # 在X方向（垂直于row）上找最低点
                x_start_idx = int((pos1 - x_min) / resolution)
                x_end_idx = int((pos2 - x_min) / resolution)
                x_start_idx = max(0, x_start_idx)
                x_end_idx = min(nx, x_end_idx)
                
                if x_end_idx <= x_start_idx:
                    continue
                
                # 从平滑CHM中提取这一行
                chm_slice = chm_smoothed[y_idx_center, x_start_idx:x_end_idx]
                
            else:
                # Y方向的row，沿X延伸
                x_idx_center = int((along_pos - x_min) / resolution)
                if x_idx_center < 0 or x_idx_center >= nx:
                    continue
                
                # 在Y方向（垂直于row）上找最低点
                y_start_idx = int((pos1 - y_min) / resolution)
                y_end_idx = int((pos2 - y_min) / resolution)
                y_start_idx = max(0, y_start_idx)
                y_end_idx = min(ny, y_end_idx)
                
                if y_end_idx <= y_start_idx:
                    continue
                
                # 从平滑CHM中提取这一列
                chm_slice = chm_smoothed[y_start_idx:y_end_idx, x_idx_center]
            
            # 在平滑CHM切片中找最低点
            valid_mask = ~np.isnan(chm_slice)
            if np.sum(valid_mask) > 3:
                valid_heights = chm_slice[valid_mask]
                min_height_idx = np.argmin(valid_heights)
                
                # 转换回坐标
                valid_indices = np.where(valid_mask)[0]
                min_idx_in_slice = valid_indices[min_height_idx]
                
                if row_direction == 'x':
                    min_pos = x_min + (x_start_idx + min_idx_in_slice) * resolution
                else:
                    min_pos = y_min + (y_start_idx + min_idx_in_slice) * resolution
                
                split_positions.append(min_pos)
                valid_along.append(along_pos)
        
        if len(split_positions) > 0:
            split_curves.append({
                'along': np.array(valid_along),
                'split': np.array(split_positions)
            })
        else:
            # 回退到直线
            split_curves.append({
                'along': np.array([along_min, along_max]),
                'split': np.array([(pos1 + pos2) / 2, (pos1 + pos2) / 2])
            })
    
    # 为每个row构建边界
    boundary_curves = []
    for i in range(n_rows):
        if n_rows == 1:
            # 只有一个row，固定宽度
            boundary_curves.append({
                'along': np.array([along_min, along_max]),
                'across_min': np.array([row_positions[0] - 0.2, row_positions[0] - 0.2]),
                'across_max': np.array([row_positions[0] + 0.2, row_positions[0] + 0.2])
            })
        elif i == 0:
            # 第一个row
            right_curve = split_curves[0]
            along_pts = right_curve['along']
            across_max = right_curve['split']
            across_min = 2 * row_positions[0] - across_max  # 对称
            boundary_curves.append({
                'along': along_pts,
                'across_min': across_min,
                'across_max': across_max
            })
        elif i == n_rows - 1:
            # 最后一个row
            left_curve = split_curves[-1]
            along_pts = left_curve['along']
            across_min = left_curve['split']
            across_max = 2 * row_positions[-1] - across_min  # 对称
            boundary_curves.append({
                'along': along_pts,
                'across_min': across_min,
                'across_max': across_max
            })
        else:
            # 中间的row
            left_curve = split_curves[i - 1]
            right_curve = split_curves[i]
            
            # 使用交集的along坐标
            along_pts = np.intersect1d(left_curve['along'], right_curve['along'])
            if len(along_pts) == 0:
                along_pts = left_curve['along']
            
            # 插值获取对应的边界值
            across_min = np.interp(along_pts, left_curve['along'], left_curve['split'])
            across_max = np.interp(along_pts, right_curve['along'], right_curve['split'])
            
            boundary_curves.append({
                'along': along_pts,
                'across_min': across_min,
                'across_max': across_max
            })
    
    return boundary_curves


def split_and_save_rows(tile_x, tile_y, tile_z, row_centers, row_direction, 
                        output_dir, tile_id, boundary_curves):
    """
    分割并保存每个row的点云（使用曲线边界）
    
    参数:
        tile_x, tile_y, tile_z: tile点云
        row_centers: row中心位置（已排序）
        row_direction: 'x' 或 'y'
        output_dir: 输出目录
        tile_id: tile ID
        boundary_curves: 边界曲线列表
    """
    print(f"  分割 {len(row_centers)} 个 {row_direction.upper()}-方向的rows...")
    
    # 确定坐标方向
    if row_direction == 'x':
        along_coord = tile_y
        across_coord = tile_x
    else:
        along_coord = tile_x
        across_coord = tile_y
    
    # 保存每个row
    for i, (row_pos, boundary) in enumerate(zip(row_centers, boundary_curves), 1):
        along_pts = boundary['along']
        across_min = boundary['across_min']
        across_max = boundary['across_max']
        
        # 向量化：先过滤沿着方向的范围
        range_mask = (along_coord >= along_pts[0]) & (along_coord <= along_pts[-1])
        
        # 对范围内的点进行插值（向量化）
        valid_along = along_coord[range_mask]
        valid_across = across_coord[range_mask]
        
        # 向量化插值
        min_bounds = np.interp(valid_along, along_pts, across_min)
        max_bounds = np.interp(valid_along, along_pts, across_max)
        
        # 向量化判断
        in_boundary = (valid_across >= min_bounds) & (valid_across <= max_bounds)
        
        # 构建完整mask
        mask = np.zeros(len(tile_x), dtype=bool)
        mask[range_mask] = in_boundary
        
        row_point_count = np.sum(mask)
        
        if row_point_count < 10:
            print(f"    Row {i}: 点数太少 ({row_point_count}), 跳过")
            continue
        
        row_x = tile_x[mask]
        row_y = tile_y[mask]
        row_z = tile_z[mask]
        
        # 保存LAS文件
        row_filename = f"row_{row_direction}{i:02d}_at_{row_pos:.2f}m.las"
        row_path = output_dir / row_filename
        save_las_file(row_x, row_y, row_z, row_path)
        
        avg_width = np.mean(across_max - across_min)
        print(f"    Row {i}: 位置={row_pos:.2f}m, 平均宽度={avg_width:.2f}m, 点数={row_point_count:,}")


def process_single_tile(tile_id, tile_x, tile_y, tile_z, tile_info, output_dir):
    """处理单个tile：检测rows并分割保存"""
    i, j = tile_id
    print(f"\n{'='*50}")
    print(f"处理 Tile ({i}, {j}) - {tile_info['n_points']:,} 点")
    print(f"{'='*50}")
    
    # 创建tile输出目录
    tile_dir = output_dir / f"tile_{i}_{j}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存tile完整点云
    tile_las_path = tile_dir / f"tile_{i}_{j}_full.las"
    save_las_file(tile_x, tile_y, tile_z, tile_las_path)
    print(f"  已保存完整tile: {tile_las_path.name}")
    
    granularity = 0.02  # 2cm
    
    # X轴分析
    print("  检测X轴方向的rows...")
    x_centers, x_heights = compute_height_profile(tile_x, tile_z, granularity)
    x_peaks, x_smoothed, x_regularity = detect_crop_rows(x_centers, x_heights, sigma=int(0.05/granularity))
    x_row_count = len(x_peaks)
    print(f"    -> 检测到 {x_row_count} 个X-方向rows, 规律度={x_regularity:.3f}")
    
    # Y轴分析
    print("  检测Y轴方向的rows...")
    y_centers, y_heights = compute_height_profile(tile_y, tile_z, granularity)
    y_peaks, y_smoothed, y_regularity = detect_crop_rows(y_centers, y_heights, sigma=int(0.05/granularity))
    y_row_count = len(y_peaks)
    print(f"    -> 检测到 {y_row_count} 个Y-方向rows, 规律度={y_regularity:.3f}")
    
    # 选择间距最规律的方向（规律度分数最高）
    if x_row_count == 0 and y_row_count == 0:
        print("  警告: 未检测到任何rows")
        return {
            'tile_id': tile_id,
            'direction': None,
            'row_count': 0,
            'regularity': 0.0
        }

    if x_row_count <= 3:
        print(f"  X方向row过少 ({x_row_count} ≤ 3)，忽略该方向")
        x_row_count = 0
    if y_row_count <= 3:
        print(f"  Y方向row过少 ({y_row_count} ≤ 3)，忽略该方向")
        y_row_count = 0

    # 若两个方向都无效
    if x_row_count == 0 and y_row_count == 0:
        print("  跳过：两个方向都未检测到足够的rows")
        return {
            'tile_id': tile_id,
            'direction': None,
            'row_count': 0,
            'regularity': 0.0
        }
    
    # 如果只有一个方向检测到rows，选择那个方向
    if x_row_count == 0:
        chosen_direction = 'y'
        row_positions = y_centers[y_peaks]
        row_count = y_row_count
        chosen_regularity = y_regularity
    elif y_row_count == 0:
        chosen_direction = 'x'
        row_positions = x_centers[x_peaks]
        row_count = x_row_count
        chosen_regularity = x_regularity
    else:
        # 两个方向都有rows，选择规律度高的
        if x_regularity >= y_regularity:
            chosen_direction = 'x'
            row_positions = x_centers[x_peaks]
            row_count = x_row_count
            chosen_regularity = x_regularity
            print(f"  → X方向规律度更高 ({x_regularity:.3f} vs {y_regularity:.3f})")
        else:
            chosen_direction = 'y'
            row_positions = y_centers[y_peaks]
            row_count = y_row_count
            chosen_regularity = y_regularity
            print(f"  → Y方向规律度更高 ({y_regularity:.3f} vs {x_regularity:.3f})")
    
    print(f"\n  选择 {chosen_direction.upper()}-方向进行分割 ({row_count} rows, 规律度={chosen_regularity:.3f})")
    
    # 生成可视化
    output_prefix = str(tile_dir / f"tile_{i}_{j}")
    
    print("  生成高度曲线图...")
    plot_height_profiles(x_centers, x_heights, x_peaks, x_smoothed,
                        y_centers, y_heights, y_peaks, y_smoothed,
                        chosen_direction, output_prefix)
    
    # 创建平滑CHM（只计算一次）
    print("  创建平滑CHM...")
    chm_smoothed, x_min, x_max, y_min, y_max, nx, ny = create_smoothed_chm(
        tile_x, tile_y, tile_z, resolution=granularity
    )
    
    # 计算边界曲线（使用预计算的CHM）
    print("  计算边界曲线...")
    boundary_curves = calculate_row_boundaries_curves(
        row_positions, chm_smoothed, x_min, y_min, nx, ny, 
        chosen_direction, resolution=granularity, granularity=granularity
    )
    
    # 生成CHM可视化（使用预计算的CHM）
    print("  生成CHM可视化...")
    create_chm_visualization(chm_smoothed, x_min, x_max, y_min, y_max,
                            x_centers, x_peaks, y_centers, y_peaks,
                            chosen_direction, output_prefix, boundary_curves)
    
    # 分割并保存rows
    split_and_save_rows(tile_x, tile_y, tile_z, row_positions, 
                       chosen_direction, tile_dir, tile_id, boundary_curves)
    
    return {
        'tile_id': tile_id,
        'direction': chosen_direction,
        'row_count': row_count,
        'regularity': chosen_regularity
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python crop_row_splitter.py <las文件路径> [选项]")
        print("示例: python crop_row_splitter.py input.las --tile_size 10")
        sys.exit(1)
    
    las_path = sys.argv[1]
    
    # 解析参数
    tile_size = 10.0  # 默认10x10m
    if '--tile_size' in sys.argv:
        idx = sys.argv.index('--tile_size')
        if idx + 1 < len(sys.argv):
            tile_size = float(sys.argv[idx + 1])
    
    if not Path(las_path).exists():
        print(f"错误: 文件不存在: {las_path}")
        sys.exit(1)
    
    print("="*60)
    print("作物行分割器 - Row Point Cloud Splitter")
    print("="*60)
    total_t0 = time.time()
    
    # 创建输出目录
    output_base = Path("row_split_output")
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取点云
    x, y, z = read_las_file(las_path)
    
    # 2. 标准化Z轴
    x, y, z = normalize_z_to_zero(x, y, z)

    # 4. 分割成tiles
    tiles, tile_info = split_into_tiles(x, y, z, tile_size=tile_size)
    
    # 5. 处理每个tile
    all_stats = []
    
    for tile_id, (tile_x, tile_y, tile_z) in tiles.items():
        stats = process_single_tile(tile_id, tile_x, tile_y, tile_z, 
                                    tile_info[tile_id], output_base)
        all_stats.append(stats)
    
    # 6. 生成总结报告
    print("\n" + "="*60)
    print("处理总结")
    print("="*60)
    
    total_rows = sum(s['row_count'] for s in all_stats)
    x_tiles = sum(1 for s in all_stats if s['direction'] == 'x')
    y_tiles = sum(1 for s in all_stats if s['direction'] == 'y')
    
    print(f"处理tiles: {len(all_stats)}")
    print(f"总分割rows: {total_rows}")
    print(f"X-方向tiles: {x_tiles}")
    print(f"Y-方向tiles: {y_tiles}")
    print(f"输出目录: {output_base.absolute()}")
    
    # 保存summary
    summary_file = output_base / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Crop Row Splitting Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input file: {las_path}\n")
        f.write(f"Tile size: {tile_size}x{tile_size}m\n")
        f.write(f"Total tiles processed: {len(all_stats)}\n")
        f.write(f"Total rows split: {total_rows}\n")
        f.write(f"Selection criterion: Peak spacing regularity (higher is better)\n\n")
        
        for stats in all_stats:
            i, j = stats['tile_id']
            direction = stats['direction'] or 'none'
            regularity = stats.get('regularity', 0.0)
            f.write(f"Tile ({i},{j}): direction={direction.upper()}, "
                   f"rows={stats['row_count']}, regularity={regularity:.3f}\n")
    
    total_time = time.time() - total_t0
    print(f"\n✓ 处理完成！总耗时: {total_time:.2f}秒")
    print("="*60)


if __name__ == "__main__":
    main()
