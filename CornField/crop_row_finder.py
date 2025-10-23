#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crop Row Finder
独立脚本：从空中扫描的田地点云中提取作物行信息

用法:
    python crop_row_finder.py soybean.las
    python crop_row_finder.py soybean.las --downsample 0.02  # 2cm降采样加速
    python crop_row_finder.py soybean.las --tile_size 20  # 20x20m tile处理
"""

import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.stats import binned_statistic
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d, gaussian_filter, distance_transform_edt, binary_dilation
import os
import shutil


def read_las_file(las_path):
    """读取LAS文件并返回点云坐标"""
    print(f"读取点云文件: {las_path}")
    t0 = time.time()
    las = laspy.read(las_path)
    
    # 提取xyz坐标（使用视图避免复制）
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    
    print(f"点云总数: {len(x):,} 个点 (耗时: {time.time()-t0:.2f}秒)")
    print(f"X范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Y范围: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Z范围: [{z.min():.3f}, {z.max():.3f}]")
    
    return x, y, z


def downsample_voxel_grid(x, y, z, voxel_size=0.02):
    """
    体素网格降采样 - 极大提升处理速度
    
    参数:
        voxel_size: 体素大小（米），默认2cm
    """
    print(f"\n体素网格降采样 (体素大小: {voxel_size*100:.1f}cm)...")
    t0 = time.time()
    
    # 计算体素索引
    voxel_x = np.floor(x / voxel_size).astype(np.int32)
    voxel_y = np.floor(y / voxel_size).astype(np.int32)
    voxel_z = np.floor(z / voxel_size).astype(np.int32)
    
    # 使用字典快速去重（每个体素保留一个点）
    voxel_dict = {}
    for i in range(len(x)):
        key = (voxel_x[i], voxel_y[i], voxel_z[i])
        if key not in voxel_dict:
            voxel_dict[key] = i
    
    # 提取降采样后的点
    indices = list(voxel_dict.values())
    x_down = x[indices]
    y_down = y[indices]
    z_down = z[indices]
    
    reduction = (1 - len(indices)/len(x)) * 100
    print(f"  降采样完成: {len(x):,} -> {len(indices):,} 点 (减少 {reduction:.1f}%)")
    print(f"  耗时: {time.time()-t0:.2f}秒")
    
    return x_down, y_down, z_down


def normalize_z_to_zero(x, y, z):
    """将点云Z轴标准化，使最低点位于z=0"""
    t0 = time.time()
    z_min = z.min()
    z_normalized = z - z_min
    
    print(f"\nZ轴标准化: (耗时: {time.time()-t0:.3f}秒)")
    print(f"  原始Z最小值: {z_min:.3f}")
    print(f"  标准化后Z范围: [{z_normalized.min():.3f}, {z_normalized.max():.3f}]")
    
    return x, y, z_normalized


def remove_ground_points(x, y, z, percentile=15):
    """移除地面点：删除高度范围底部15%的点"""
    t0 = time.time()
    z_threshold = np.percentile(z, percentile)
    mask = z > z_threshold
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    
    removed_count = len(x) - len(x_filtered)
    print(f"  移除地面点: (耗时: {time.time()-t0:.3f}秒)")
    print(f"    高度阈值 (底部{percentile}%): {z_threshold:.3f}")
    print(f"    移除点数: {removed_count:,} ({removed_count/len(x)*100:.1f}%)")
    print(f"    剩余点数: {len(x_filtered):,}")
    
    return x_filtered, y_filtered, z_filtered


def split_into_tiles(x, y, z, tile_size=10.0):
    """
    将点云分割成NxN米的tiles
    
    参数:
        x, y, z: 点云坐标
        tile_size: tile大小（米）
    
    返回:
        tiles: 字典，键为(tile_x, tile_y)，值为(x, y, z)点云数据
        tile_info: tile的边界信息
    """
    print(f"\n分割成 {tile_size}x{tile_size}m tiles...")
    t0 = time.time()
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # 计算tile数量
    n_tiles_x = int(np.ceil((x_max - x_min) / tile_size))
    n_tiles_y = int(np.ceil((y_max - y_min) / tile_size))
    
    print(f"  田地范围: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"  Tile网格: {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} tiles")
    
    tiles = {}
    tile_info = {}
    
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            # 计算tile边界
            tile_x_min = x_min + i * tile_size
            tile_x_max = tile_x_min + tile_size
            tile_y_min = y_min + j * tile_size
            tile_y_max = tile_y_min + tile_size
            
            # 筛选该tile内的点
            mask = (x >= tile_x_min) & (x < tile_x_max) & \
                   (y >= tile_y_min) & (y < tile_y_max)
            
            if np.sum(mask) > 100:  # 至少100个点才保留
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
    
    print(f"  有效tiles: {len(tiles)} / {n_tiles_x * n_tiles_y}")
    print(f"  耗时: {time.time()-t0:.2f}秒")
    
    return tiles, tile_info


def compute_height_profile_along_axis(coord, z, granularity=0.01):
    """
    沿某个轴计算平均高度分布 - 使用scipy优化版本
    
    参数:
        coord: x或y坐标数组
        z: z坐标数组
        granularity: 颗粒度 (单位: 米，默认1cm=0.01m)
    
    返回:
        bin_centers: 每个bin的中心坐标
        mean_heights: 每个bin的平均高度
        point_counts: 每个bin的点数
    """
    t0 = time.time()
    
    coord_min = coord.min()
    coord_max = coord.max()
    
    # 创建bins
    bins = np.arange(coord_min, coord_max + granularity, granularity)
    
    # 使用scipy的binned_statistic - 比循环快10-100倍
    mean_heights, _, _ = binned_statistic(coord, z, statistic='mean', bins=bins)
    point_counts, _, _ = binned_statistic(coord, z, statistic='count', bins=bins)
    
    # bin中心位置
    bin_centers = bins[:-1] + granularity / 2
    
    print(f"    -> 计算完成，耗时: {time.time()-t0:.3f}秒")
    
    return bin_centers, mean_heights, point_counts


def filter_and_find_peaks(centers, heights, sigma=5):
    """
    自适应检测作物行peaks - 自动忽略无规律区域
    
    策略:
    1. 高斯滤波平滑曲线
    2. 找到所有可能的peaks
    3. 分析局部规律性（sliding window）
    4. 计算全局主导间距
    5. 只在高规律性区域内提取符合间距规律的peaks
    
    参数:
        centers: bin中心坐标
        heights: 平均高度
        sigma: 高斯滤波标准差
    
    返回:
        peak_indices: peak的索引
        smoothed_heights: 平滑后的高度
        regular_mask: 规律区间的mask
    """
    # 移除NaN值
    valid_mask = ~np.isnan(heights)
    valid_centers = centers[valid_mask]
    valid_heights = heights[valid_mask]
    
    if len(valid_heights) < 10:
        return np.array([]), valid_heights, np.ones(len(valid_heights), dtype=bool)
    
    # 1. 高斯滤波平滑
    smoothed = gaussian_filter1d(valid_heights, sigma=sigma)
    
    # 2. 找到所有peaks
    peaks, peak_props = find_peaks(smoothed, prominence=0.02)
    
    if len(peaks) < 3:
        print(f"    -> Too few peaks ({len(peaks)}), cannot determine spacing pattern")
        return np.array([]), smoothed, np.ones(len(smoothed), dtype=bool)
    
    # 3. 计算局部规律性分数（每个区域的变异系数）
    window_size = 200  # 2m窗口
    regularity_score = np.zeros(len(smoothed))
    
    for i in range(len(smoothed)):
        start = max(0, i - window_size // 2)
        end = min(len(smoothed), i + window_size // 2)
        
        # 计算窗口内的标准差和均值
        local_signal = smoothed[start:end]
        local_std = np.std(local_signal)
        local_mean = np.mean(local_signal)
        
        # 计算变异系数（标准化的波动度）
        if local_mean > 0.01:  # 避免除以0
            cv = local_std / local_mean
            # 规律性分数：高变异系数 = 有规律的波动
            regularity_score[i] = cv
        else:
            regularity_score[i] = 0
    
    # 4. 确定规律性阈值（使用智能自适应方法）
    # 分析规律性分数的分布特征
    regularity_mean = np.mean(regularity_score)
    regularity_std = np.std(regularity_score)
    regularity_median = np.median(regularity_score)
    
    # 如果大部分区域规律性高（中位数接近均值），使用更低的阈值保留更多
    # 如果规律性差异大（高标准差），使用更严格的阈值
    coefficient_of_variation = regularity_std / regularity_mean if regularity_mean > 0 else 1.0
    
    if coefficient_of_variation < 0.5:  # 分布集中，大部分区域规律性接近
        # 使用均值的50%作为阈值，保留约80-90%的区域
        regularity_threshold = regularity_mean * 0.5
        retention_note = "High uniformity detected, keeping ~80-90% of regions"
    elif coefficient_of_variation < 0.8:  # 中等分布
        # 使用30百分位，保留约70%
        regularity_threshold = np.percentile(regularity_score, 30)
        retention_note = "Medium uniformity, keeping ~70% of regions"
    else:  # 分布分散，规律性差异大
        # 使用30百分位，保留约70%
        regularity_threshold = np.percentile(regularity_score, 70)
        retention_note = "Low uniformity, keeping ~30% of regions"
    
    regular_region_mask = regularity_score > regularity_threshold
    retention_rate = np.sum(regular_region_mask) / len(regular_region_mask) * 100
    
    print(f"    -> Regularity CV: {coefficient_of_variation:.2f} ({retention_note})")
    print(f"    -> Actual retention: {retention_rate:.1f}% of signal")
    
    # 5. 只保留规律区域内的peaks
    peaks_in_regular = peaks[regular_region_mask[peaks]]
    
    if len(peaks_in_regular) < 3:
        print(f"    -> Too few peaks in regular regions ({len(peaks_in_regular)})")
        return np.array([]), smoothed, regular_region_mask
    
    # 6. 计算规律区域内的主导间距
    peak_spacings = np.diff(peaks_in_regular)
    
    if len(peak_spacings) > 0:
        # 使用中位数而非众数，更稳健
        dominant_spacing = np.median(peak_spacings)
        
        # 计算间距的标准差，用于自适应容忍度
        spacing_std = np.std(peak_spacings)
        spacing_tolerance = max(0.3, spacing_std / dominant_spacing)  # 至少30%容忍度
        
        min_spacing = dominant_spacing * (1 - spacing_tolerance)
        max_spacing = dominant_spacing * (1 + spacing_tolerance)
        
        print(f"    -> Dominant peak spacing: {dominant_spacing:.1f} bins ({dominant_spacing*0.01:.3f}m)")
        print(f"    -> Acceptable range: {min_spacing:.1f}-{max_spacing:.1f} bins")
        print(f"    -> Regularity threshold: {regularity_threshold:.4f}")
    else:
        return np.array([]), smoothed, regular_region_mask
    
    # 7. 识别规律序列：连续的peaks间距符合规律 + 在规律区域内
    regular_peaks = []
    regular_sequences = []  # 记录每个规律序列
    i = 0
    
    while i < len(peaks_in_regular):
        # 尝试从当前peak开始找一个规律序列
        sequence = [peaks_in_regular[i]]
        j = i + 1
        
        while j < len(peaks_in_regular):
            spacing = peaks_in_regular[j] - sequence[-1]
            if min_spacing <= spacing <= max_spacing:
                sequence.append(peaks_in_regular[j])
                j += 1
            else:
                break
        
        # 如果找到至少3个连续规律的peaks，保留它们
        if len(sequence) >= 3:
            regular_sequences.append(sequence)
            regular_peaks.extend(sequence)
            i = j
        else:
            i += 1
    
    regular_peaks = np.array(regular_peaks)
    
    # 8. 计算每个peak的局部prominence来进一步过滤
    if len(regular_peaks) > 0:
        final_peaks = []
        for peak in regular_peaks:
            # 计算peak周围的局部prominence
            left = max(0, peak - 50)
            right = min(len(smoothed), peak + 50)
            local_min = np.min(smoothed[left:right])
            prominence = smoothed[peak] - local_min
            
            # 只保留prominence足够大的peaks
            if prominence > 0.03:  # 3cm以上的显著性
                final_peaks.append(peak)
        
        regular_peaks = np.array(final_peaks)
    
    print(f"    -> Found {len(peaks)} total peaks")
    print(f"    -> {len(peaks_in_regular)} in regular regions")
    print(f"    -> {len(regular_peaks)} with consistent spacing ({len(regular_sequences)} sequences)")
    
    # 返回原始索引空间的peaks
    original_peaks = np.where(valid_mask)[0][regular_peaks] if len(regular_peaks) > 0 else np.array([])
    
    # 构建完整的smoothed数组（包含NaN）
    full_smoothed = np.full_like(heights, np.nan)
    full_smoothed[valid_mask] = smoothed
    
    # 构建规律区间mask（基于规律序列的范围）
    regular_mask = np.zeros_like(heights, dtype=bool)
    if len(regular_sequences) > 0:
        for sequence in regular_sequences:
            # 标记从序列第一个peak到最后一个peak的区域
            seq_start = sequence[0]
            seq_end = sequence[-1]
            original_start = np.where(valid_mask)[0][seq_start]
            original_end = np.where(valid_mask)[0][seq_end]
            regular_mask[original_start:original_end+1] = True
    
    return original_peaks, full_smoothed, regular_mask


def plot_height_profiles(x_centers, x_heights, x_peaks, x_smoothed,
                         y_centers, y_heights, y_peaks, y_smoothed,
                         output_prefix):
    """Plot height profiles along X and Y axes with detected peaks"""
    t0 = time.time()
    
    # Create figure with 1 row, 2 columns (no count plots)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # X-axis height profile with peaks
    ax1 = axes[0]
    valid_x = ~np.isnan(x_heights)
    
    # Plot raw data (light)
    ax1.plot(x_centers[valid_x], x_heights[valid_x], 'b-', 
             linewidth=0.5, alpha=0.3, label='Raw data')
    
    # Plot smoothed data
    valid_smooth_x = ~np.isnan(x_smoothed)
    ax1.plot(x_centers[valid_smooth_x], x_smoothed[valid_smooth_x], 'b-', 
             linewidth=1, alpha=0.8, label='Smoothed')
    
    # Plot peaks as small crosses
    if len(x_peaks) > 0:
        ax1.plot(x_centers[x_peaks], x_heights[x_peaks], 'rx', 
                markersize=8, markeredgewidth=2, label=f'Peaks (n={len(x_peaks)})')
    
    ax1.set_xlabel('X Coordinate (m)', fontsize=12)
    ax1.set_ylabel('Mean Height (m)', fontsize=12)
    ax1.set_title(f'Crop Row Detection along X-axis ({len(x_peaks)} rows detected)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Y-axis height profile with peaks
    ax2 = axes[1]
    valid_y = ~np.isnan(y_heights)
    
    # Plot raw data (light)
    ax2.plot(y_centers[valid_y], y_heights[valid_y], 'g-', 
             linewidth=0.5, alpha=0.3, label='Raw data')
    
    # Plot smoothed data
    valid_smooth_y = ~np.isnan(y_smoothed)
    ax2.plot(y_centers[valid_smooth_y], y_smoothed[valid_smooth_y], 'g-', 
             linewidth=1, alpha=0.8, label='Smoothed')
    
    # Plot peaks as small crosses
    if len(y_peaks) > 0:
        ax2.plot(y_centers[y_peaks], y_heights[y_peaks], 'rx', 
                markersize=8, markeredgewidth=2, label=f'Peaks (n={len(y_peaks)})')
    
    ax2.set_xlabel('Y Coordinate (m)', fontsize=12)
    ax2.set_ylabel('Mean Height (m)', fontsize=12)
    ax2.set_title(f'Crop Row Detection along Y-axis ({len(y_peaks)} rows detected)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure (no display)
    output_file = f"{output_prefix}_crop_rows.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    print(f"  Image saved: {output_file} (Time: {time.time()-t0:.2f}s)")


def create_chm_with_crop_rows(x, y, z, x_centers, x_peaks, y_centers, y_peaks, 
                               resolution, output_prefix, plant_records=None):
    """
    生成CHM（Canopy Height Model）并叠加检测到的作物行和单株植物位置
    
    参数:
        x, y, z: 点云坐标
        x_centers, y_centers: bin中心坐标
        x_peaks, y_peaks: 检测到的peak索引
        resolution: CHM分辨率（米）
        output_prefix: 输出文件前缀
        plant_records: 可选，单株植物位置记录列表
    """
    print("  生成CHM可视化...")
    t0 = time.time()
    
    # 1. 创建2D网格
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # 计算网格大小
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    
    print(f"    CHM grid size: {nx} x {ny} (resolution: {resolution*100:.1f}cm)")
    
    # 2. 创建CHM（使用最大高度）
    chm = np.full((ny, nx), np.nan)
    
    # 将点分配到网格
    x_idx = ((x - x_min) / resolution).astype(int)
    y_idx = ((y - y_min) / resolution).astype(int)
    
    # 防止越界
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)
    
    # 对每个网格取最大高度
    for i in range(len(x)):
        if np.isnan(chm[y_idx[i], x_idx[i]]) or z[i] > chm[y_idx[i], x_idx[i]]:
            chm[y_idx[i], x_idx[i]] = z[i]
    
    # 3. 填充空白并平滑（仅用于可视化）
    print("    应用插值和平滑...")
    mask = ~np.isnan(chm)
    valid_count = np.sum(mask)
    
    if valid_count > 0:
        # 使用最近邻插值填充小空洞
        # 找到最近的有效值来填充NaN
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        chm_filled = chm[tuple(indices)]
        
        # 对填充后的数据应用高斯平滑
        chm_smoothed = gaussian_filter(chm_filled, sigma=2.5, mode='nearest')
        
        # 可选：只在原始有效区域及其邻近区域显示
        # 创建扩展的mask（稍微扩大有效区域）
        dilated_mask = binary_dilation(mask, iterations=5)
        
        # 在扩展区域外的地方恢复NaN（避免过度外推）
        chm_smoothed[~dilated_mask] = np.nan
        
        valid_after = np.sum(~np.isnan(chm_smoothed))
        print(f"    CHM cells: {valid_count:,} -> {valid_after:,} (增加 {valid_after-valid_count:,} 个插值点)")
    else:
        chm_smoothed = chm
        print("    Warning: No valid CHM data found!")
    
    # 4. 创建高分辨率figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # 5. 显示平滑后的CHM（使用更好的colormap）
    im = ax.imshow(chm_smoothed, extent=[x_min, x_max, y_min, y_max], 
                   origin='lower', cmap='terrain', aspect='auto', 
                   interpolation='bilinear', alpha=0.9)
    
    # 6. 叠加X轴检测的作物行（垂直红线）
    if len(x_peaks) > 0:
        for peak_idx in x_peaks:
            x_pos = x_centers[peak_idx]
            ax.axvline(x=x_pos, color='red', linewidth=1.2, alpha=0.8, linestyle='-')
    
    # 7. 叠加Y轴检测的作物行（水平青色线）
    if len(y_peaks) > 0:
        for peak_idx in y_peaks:
            y_pos = y_centers[peak_idx]
            ax.axhline(y=y_pos, color='cyan', linewidth=1.2, alpha=0.8, linestyle='-')
    
    # 8. 设置标签和标题
    ax.set_xlabel('X Coordinate (m)', fontsize=13)
    ax.set_ylabel('Y Coordinate (m)', fontsize=13)
    ax.set_title(f'Canopy Height Model with Detected Crop Rows (Smoothed)\n'
                 f'X-axis: {len(x_peaks)} rows (red) | Y-axis: {len(y_peaks)} rows (cyan)',
                 fontsize=14, fontweight='bold', pad=15)
    
    # 9. 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)', fontsize=12)
    
    # 10. 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2.5, label=f'X-axis rows (n={len(x_peaks)})'),
        Line2D([0], [0], color='cyan', linewidth=2.5, label=f'Y-axis rows (n={len(y_peaks)})')
    ]
    
    # 【新增】标记单株植物位置
    if plant_records is not None and len(plant_records) > 0:
        plant_x = [p['plant_x'] for p in plant_records]
        plant_y = [p['plant_y'] for p in plant_records]
        ax.scatter(plant_x, plant_y, c='red', marker='x', s=15, linewidths=1.2, 
                  alpha=0.9, zorder=10)
        legend_elements.append(
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                   markersize=8, label=f'Individual plants (n={len(plant_records)})')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
              framealpha=0.9, edgecolor='black')
    
    # 11. 添加网格（细微）
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # 12. 保存高分辨率图像
    output_file = f"{output_prefix}_chm_with_rows.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')  # 600 DPI for high quality
    plt.close(fig)
    
    print(f"    CHM已保存: {output_file} (耗时: {time.time()-t0:.2f}s)")


def detect_individual_plants_in_section(section_coord, section_z, section_weights, 
                                        coord_name='Y', granularity=0.01):
    """
    在切面上检测单株植物位置（类似于作物行检测的方法）
    
    参数:
        section_coord: 切面横坐标（Y或X）
        section_z: 高度
        section_weights: 高斯权重
        coord_name: 坐标轴名称（用于打印）
        granularity: 高度分布计算粒度（米），默认0.01m（1cm）
    
    返回:
        plant_positions: 检测到的植物位置列表
        plant_heights: 对应的植物高度列表
        profile_centers: 高度曲线中心坐标
        profile_heights: 高度曲线值
    """
    if len(section_coord) < 20:
        return [], [], [], []

    # 1. 沿切面方向计算加权高度分布
    # granularity由参数传入，与主程序设置保持一致
    coord_min = section_coord.min()
    coord_max = section_coord.max()
    
    n_bins = int((coord_max - coord_min) / granularity) + 1
    if n_bins < 10:
        return [], [], [], []
    
    bin_edges = np.linspace(coord_min, coord_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 加权binned_statistic: 使用高斯权重
    weighted_heights = section_z * section_weights
    
    # 计算每个bin的加权平均高度
    heights_sum, _, _ = binned_statistic(section_coord, weighted_heights, 
                                         statistic='sum', bins=bin_edges)
    weights_sum, _, _ = binned_statistic(section_coord, section_weights,
                                         statistic='sum', bins=bin_edges)
    
    # 避免除零
    valid_mask = weights_sum > 1e-6
    profile_heights = np.zeros_like(heights_sum)
    profile_heights[valid_mask] = heights_sum[valid_mask] / weights_sum[valid_mask]

    # 2. 高斯平滑（sigma根据粒度自适应：保持~4cm的平滑效果）
    # sigma = 4cm / granularity，例如granularity=0.01m时，sigma=4
    sigma_bins = int(0.04 / granularity)
    profile_smoothed = gaussian_filter1d(profile_heights, sigma=sigma_bins)
    
    # 3. 初步检测所有peak候选（prominence > 2cm）
    peaks, properties = find_peaks(profile_smoothed, prominence=0.02)

    if len(peaks) == 0:
        return [], [], bin_centers, profile_smoothed
    
    # 4. 使用最大堆（按高度排序）+ 最小间距约束筛选peaks
    import heapq
    
    # 获取每个peak的位置和高度
    peak_data = []
    for peak_idx in peaks:
        position = bin_centers[peak_idx]
        height = profile_smoothed[peak_idx]
        # 使用负高度构建最大堆（Python的heapq是最小堆）
        peak_data.append((-height, position, peak_idx))
    
    # 构建最大堆
    heapq.heapify(peak_data)
    
    # 贪心选择peaks，确保间距≥12cm
    min_spacing = 0.12  # 12cm最小间距
    selected_positions = []
    selected_heights = []
    
    while peak_data:
        # 弹出当前最高的peak
        neg_height, position, peak_idx = heapq.heappop(peak_data)
        height = -neg_height
        
        # 检查与已选择peaks的间距
        too_close = False
        for selected_pos in selected_positions:
            if abs(position - selected_pos) < min_spacing:
                too_close = True
                break
        
        # 如果间距满足要求，选择该peak
        if not too_close:
            selected_positions.append(position)
            selected_heights.append(height)
    
    # 按位置排序（从左到右），同时保持高度对应
    sorted_indices = np.argsort(selected_positions)
    plant_positions = [selected_positions[i] for i in sorted_indices]
    plant_heights = [selected_heights[i] for i in sorted_indices]
    
    return plant_positions, plant_heights, bin_centers, profile_smoothed


def create_cross_section_views(tile_x, tile_y, tile_z, x_centers, x_peaks, 
                                y_centers, y_peaks, voxel_size, output_dir, granularity=0.01):
    """
    为每条检测到的作物行生成垂直切面视图（高斯加权版 + 单株植物检测）
    
    参数:
        tile_x, tile_y, tile_z: tile点云坐标
        x_centers, y_centers: bin中心坐标
        x_peaks, y_peaks: 检测到的peak索引
        voxel_size: 体素大小，用于确定切面厚度
        output_dir: 输出目录
        granularity: 高度分布计算粒度（米），默认0.01m
    """
    print("  生成切面视图（高斯加权）...")
    t0 = time.time()
    
    # 高斯参数设置
    # sigma选择: voxel_size * 1.5 (在这个距离内点有显著贡献)
    # cutoff: 3*sigma (99.7%的高斯分布范围)
    sigma = voxel_size * 2
    cutoff_distance = 3 * sigma
    
    # 上限约束
    cutoff_distance = min(cutoff_distance, 0.18)  # 最大18cm
    
    print(f"    高斯sigma: {sigma*100:.2f}cm, cutoff: {cutoff_distance*100:.1f}cm")
    
    # 创建切面输出目录
    cross_section_dir = output_dir / "cross_sections"
    cross_section_dir.mkdir(exist_ok=True)
    
    total_sections = 0
    
    # 优化：预先对坐标排序以加速范围查询
    x_sort_idx = np.argsort(tile_x)
    y_sort_idx = np.argsort(tile_y)
    
    tile_x_sorted = tile_x[x_sort_idx]
    tile_y_sorted_for_x = tile_y[x_sort_idx]
    tile_z_sorted_for_x = tile_z[x_sort_idx]
    
    tile_y_sorted = tile_y[y_sort_idx]
    tile_x_sorted_for_y = tile_x[y_sort_idx]
    tile_z_sorted_for_y = tile_z[y_sort_idx]
    
    # 1. 处理X轴方向的行（垂直线，生成YZ平面切面）
    if len(x_peaks) > 0:
        print(f"    生成 {len(x_peaks)} 条X轴作物行的切面...")
        for idx, peak_idx in enumerate(x_peaks):
            x_pos = x_centers[peak_idx]
            
            # 使用二分查找快速定位范围
            x_min_search = x_pos - cutoff_distance
            x_max_search = x_pos + cutoff_distance
            
            start_idx = np.searchsorted(tile_x_sorted, x_min_search, side='left')
            end_idx = np.searchsorted(tile_x_sorted, x_max_search, side='right')
            
            n_points = end_idx - start_idx
            if n_points < 10:  # 至少10个点
                continue
            
            # 提取候选点
            candidate_x = tile_x_sorted[start_idx:end_idx]
            candidate_y = tile_y_sorted_for_x[start_idx:end_idx]
            candidate_z = tile_z_sorted_for_x[start_idx:end_idx]
            
            # 计算每个点到切面的距离
            distances = np.abs(candidate_x - x_pos)
            
            # 计算高斯权重
            weights = np.exp(-(distances**2) / (2 * sigma**2))
            
            # 只保留权重 > 0.01 的点（99%以下cutoff）
            valid_mask = weights > 0.01
            section_y = candidate_y[valid_mask]
            section_z = candidate_z[valid_mask]
            section_weights = weights[valid_mask]
            
            n_valid = len(section_y)
            if n_valid < 10:
                continue
            
            # 【新增】检测单株植物位置（在降采样之前用完整数据）
            plant_positions, plant_heights, profile_centers, profile_smoothed = \
                detect_individual_plants_in_section(section_y, section_z, section_weights, 'Y', granularity)
            
            # 计算每个植物位置的平均高度（用于在点云上标记）
            # 注意：plant_heights已经从profile获得，但我们用更精确的加权平均覆盖
            plant_heights_refined = []
            if len(plant_positions) > 0:
                for plant_y, profile_h in zip(plant_positions, plant_heights):
                    # 找到该位置附近的点（范围为2倍粒度）
                    nearby_mask = np.abs(section_y - plant_y) < (2 * granularity)
                    if np.any(nearby_mask):
                        # 加权平均高度
                        weights_nearby = section_weights[nearby_mask]
                        z_nearby = section_z[nearby_mask]
                        avg_height = np.average(z_nearby, weights=weights_nearby)
                        plant_heights_refined.append(avg_height)
                    else:
                        # 使用profile高度
                        plant_heights_refined.append(profile_h)
                plant_heights = plant_heights_refined
            
            # 智能降采样（保持权重高的点）
            if n_valid > 5000:
                # 根据权重概率采样，权重高的点更可能被保留
                sample_prob = section_weights / section_weights.sum()
                sample_idx = np.random.choice(n_valid, 5000, replace=False, p=sample_prob)
                section_y = section_y[sample_idx]
                section_z = section_z[sample_idx]
                section_weights = section_weights[sample_idx]
                n_valid = 5000
            
            # 生成YZ切面图（2行：上图点云，下图高度曲线）
            fig, axes = plt.subplots(2, 1, figsize=(12, 9), 
                                     gridspec_kw={'height_ratios': [2, 1]})
            
            # 上图：点云切面
            ax_cloud = axes[0]
            scatter = ax_cloud.scatter(section_y, section_z, c=section_z, 
                                      s=section_weights * 3 + 0.5,
                                      alpha=np.clip(section_weights * 0.8, 0.1, 0.8),
                                      cmap='viridis', edgecolors='none')
            
            # 标记检测到的植物位置（垂直虚线 + 红叉）
            if len(plant_positions) > 0:
                for plant_y, plant_h in zip(plant_positions, plant_heights):
                    # 垂直虚线
                    ax_cloud.axvline(plant_y, color='red', linestyle='--', 
                                    linewidth=1, alpha=0.4)
                    # 红色叉叉标记
                    ax_cloud.plot(plant_y, plant_h, 'rx', markersize=12, 
                                 markeredgewidth=2.5, label='_nolegend_')
            
            ax_cloud.set_ylabel('Height (m)', fontsize=11)
            ax_cloud.set_title(f'Cross Section - X-axis Row #{idx+1} (X={x_pos:.2f}m) - '
                              f'{len(plant_positions)} plants detected',
                              fontsize=12, fontweight='bold')
            ax_cloud.grid(True, alpha=0.3)
            ax_cloud.set_aspect('equal', adjustable='box')
            
            # 添加colorbar
            cbar = plt.colorbar(scatter, ax=ax_cloud, pad=0.02)
            cbar.set_label('Height (m)', fontsize=10)
            
            # 下图：高度曲线
            ax_profile = axes[1]
            if len(profile_centers) > 0:
                # 绘制原始和平滑曲线
                ax_profile.plot(profile_centers, profile_smoothed, 'b-', 
                               linewidth=2, label='Smoothed Height Profile', zorder=1)
                
                # 标记植物位置（用之前计算好的高度）
                if len(plant_positions) > 0:
                    # 用红色叉叉标记
                    ax_profile.plot(plant_positions, plant_heights, 'rx', 
                                   markersize=10, markeredgewidth=2.5,
                                   label=f'{len(plant_positions)} Individual Plants', zorder=3)
                    # 添加垂直虚线以便对齐上下图
                    for plant_y in plant_positions:
                        ax_profile.axvline(plant_y, color='red', linestyle='--',
                                          linewidth=1, alpha=0.3, zorder=0)
                
                ax_profile.set_xlabel('Y Coordinate (m)', fontsize=11)
                ax_profile.set_ylabel('Avg Height (m)', fontsize=11)
                ax_profile.grid(True, alpha=0.3)
                ax_profile.legend(loc='upper right', fontsize=9)
            
            ax_profile.set_xlim(ax_cloud.get_xlim())
            
            plt.tight_layout()
            
            # 保存
            output_file = cross_section_dir / f"x_row_{idx+1:03d}_at_{x_pos:.2f}m.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            total_sections += 1
    
    # 2. 处理Y轴方向的行（水平线，生成XZ平面切面）
    if len(y_peaks) > 0:
        print(f"    生成 {len(y_peaks)} 条Y轴作物行的切面...")
        for idx, peak_idx in enumerate(y_peaks):
            y_pos = y_centers[peak_idx]
            
            # 使用二分查找快速定位范围
            y_min_search = y_pos - cutoff_distance
            y_max_search = y_pos + cutoff_distance
            
            start_idx = np.searchsorted(tile_y_sorted, y_min_search, side='left')
            end_idx = np.searchsorted(tile_y_sorted, y_max_search, side='right')
            
            n_points = end_idx - start_idx
            if n_points < 10:  # 至少10个点
                continue
            
            # 提取候选点
            candidate_x = tile_x_sorted_for_y[start_idx:end_idx]
            candidate_y = tile_y_sorted[start_idx:end_idx]
            candidate_z = tile_z_sorted_for_y[start_idx:end_idx]
            
            # 计算每个点到切面的距离
            distances = np.abs(candidate_y - y_pos)
            
            # 计算高斯权重
            weights = np.exp(-(distances**2) / (2 * sigma**2))
            
            # 只保留权重 > 0.01 的点
            valid_mask = weights > 0.01
            section_x = candidate_x[valid_mask]
            section_z = candidate_z[valid_mask]
            section_weights = weights[valid_mask]
            
            n_valid = len(section_x)
            if n_valid < 10:
                continue
            
            # 【新增】检测单株植物位置
            plant_positions, plant_heights, profile_centers, profile_smoothed = \
                detect_individual_plants_in_section(section_x, section_z, section_weights, 'X', granularity)
            
            # 计算每个植物位置的平均高度
            plant_heights_refined = []
            if len(plant_positions) > 0:
                for plant_x, profile_h in zip(plant_positions, plant_heights):
                    # 找到该位置附近的点（±2cm范围）
                    nearby_mask = np.abs(section_x - plant_x) < 0.02
                    if np.any(nearby_mask):
                        # 加权平均高度
                        weights_nearby = section_weights[nearby_mask]
                        z_nearby = section_z[nearby_mask]
                        avg_height = np.average(z_nearby, weights=weights_nearby)
                        plant_heights_refined.append(avg_height)
                    else:
                        # 使用profile高度
                        plant_heights_refined.append(profile_h)
                plant_heights = plant_heights_refined
            
            # 智能降采样（保持权重高的点）
            if n_valid > 5000:
                sample_prob = section_weights / section_weights.sum()
                sample_idx = np.random.choice(n_valid, 5000, replace=False, p=sample_prob)
                section_x = section_x[sample_idx]
                section_z = section_z[sample_idx]
                section_weights = section_weights[sample_idx]
                n_valid = 5000
            
            # 生成XZ切面图（2行：上图点云，下图高度曲线）
            fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                                     gridspec_kw={'height_ratios': [2, 1]})
            
            # 上图：点云切面
            ax_cloud = axes[0]
            scatter = ax_cloud.scatter(section_x, section_z, c=section_z, 
                                      s=section_weights * 3 + 0.5,
                                      alpha=np.clip(section_weights * 0.8, 0.1, 0.8),
                                      cmap='viridis', edgecolors='none')
            
            # 标记检测到的植物位置（垂直虚线 + 红叉）
            if len(plant_positions) > 0:
                for plant_x, plant_h in zip(plant_positions, plant_heights):
                    # 垂直虚线
                    ax_cloud.axvline(plant_x, color='red', linestyle='--',
                                    linewidth=1, alpha=0.4)
                    # 红色叉叉标记
                    ax_cloud.plot(plant_x, plant_h, 'rx', markersize=12,
                                 markeredgewidth=2.5, label='_nolegend_')
            
            ax_cloud.set_ylabel('Height (m)', fontsize=11)
            ax_cloud.set_title(f'Cross Section - Y-axis Row #{idx+1} (Y={y_pos:.2f}m) - '
                              f'{len(plant_positions)} plants detected',
                              fontsize=12, fontweight='bold')
            ax_cloud.grid(True, alpha=0.3)
            ax_cloud.set_aspect('equal', adjustable='box')
            
            # 添加colorbar
            cbar = plt.colorbar(scatter, ax=ax_cloud, pad=0.02)
            cbar.set_label('Height (m)', fontsize=10)
            
            # 下图：高度曲线
            ax_profile = axes[1]
            if len(profile_centers) > 0:
                # 绘制原始和平滑曲线
                ax_profile.plot(profile_centers, profile_smoothed, 'b-',
                               linewidth=2, label='Smoothed Height Profile', zorder=1)
                
                # 标记植物位置（用之前计算好的高度）
                if len(plant_positions) > 0:
                    # 用红色叉叉标记
                    ax_profile.plot(plant_positions, plant_heights, 'rx',
                                   markersize=10, markeredgewidth=2.5,
                                   label=f'{len(plant_positions)} Individual Plants', zorder=3)
                    # 添加垂直虚线以便对齐上下图
                    for plant_x in plant_positions:
                        ax_profile.axvline(plant_x, color='red', linestyle='--',
                                          linewidth=1, alpha=0.3, zorder=0)
                
                ax_profile.set_xlabel('X Coordinate (m)', fontsize=11)
                ax_profile.set_ylabel('Avg Height (m)', fontsize=11)
                ax_profile.grid(True, alpha=0.3)
                ax_profile.legend(loc='upper right', fontsize=9)
            
            ax_profile.set_xlim(ax_cloud.get_xlim())
            
            plt.tight_layout()
            
            # 保存
            output_file = cross_section_dir / f"y_row_{idx+1:03d}_at_{y_pos:.2f}m.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            total_sections += 1
    
    print(f"    生成 {total_sections} 个切面视图 (耗时: {time.time()-t0:.2f}s)")
    
    return total_sections


def create_cross_section_views_with_plant_detection(tile_x, tile_y, tile_z, x_centers, x_peaks,
                                                     y_centers, y_peaks, voxel_size, output_dir,
                                                     tile_id, granularity=0.01):
    """
    生成切面视图并检测单株植物，保存植物位置到CSV
    
    参数:
        granularity: 高度分布计算粒度（米），默认0.01m
    
    返回:
        plant_records: 列表，每个元素为 {'row_type', 'row_id', 'row_pos', 'plant_id', 'plant_pos'}
    """
    # 调用原函数生成切面
    n_sections = create_cross_section_views(tile_x, tile_y, tile_z, x_centers, x_peaks,
                                            y_centers, y_peaks, voxel_size, output_dir, granularity)
    
    # 收集植物位置信息
    plant_records = []
    
    # 高斯参数（与create_cross_section_views保持一致）
    sigma = voxel_size * 2
    cutoff_distance = min(3 * sigma, 0.18)
    
    # 预排序
    x_sort_idx = np.argsort(tile_x)
    y_sort_idx = np.argsort(tile_y)
    
    tile_x_sorted = tile_x[x_sort_idx]
    tile_y_sorted_for_x = tile_y[x_sort_idx]
    tile_z_sorted_for_x = tile_z[x_sort_idx]
    
    tile_y_sorted = tile_y[y_sort_idx]
    tile_x_sorted_for_y = tile_x[y_sort_idx]
    tile_z_sorted_for_y = tile_z[y_sort_idx]
    
    # X轴方向的行
    for idx, peak_idx in enumerate(x_peaks):
        x_pos = x_centers[peak_idx]
        
        x_min_search = x_pos - cutoff_distance
        x_max_search = x_pos + cutoff_distance
        
        start_idx = np.searchsorted(tile_x_sorted, x_min_search, side='left')
        end_idx = np.searchsorted(tile_x_sorted, x_max_search, side='right')
        
        n_points = end_idx - start_idx
        if n_points < 10:
            continue
        
        candidate_x = tile_x_sorted[start_idx:end_idx]
        candidate_y = tile_y_sorted_for_x[start_idx:end_idx]
        candidate_z = tile_z_sorted_for_x[start_idx:end_idx]
        
        distances = np.abs(candidate_x - x_pos)
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        
        valid_mask = weights > 0.01
        section_y = candidate_y[valid_mask]
        section_z = candidate_z[valid_mask]
        section_weights = weights[valid_mask]
        
        if len(section_y) < 10:
            continue
        
        # 检测植物
        plant_positions, plant_heights, _, _ = detect_individual_plants_in_section(
            section_y, section_z, section_weights, 'Y', granularity)
        
        # 记录（保存高度）
        for plant_id, (plant_y, plant_h) in enumerate(zip(plant_positions, plant_heights), 1):
            plant_records.append({
                'tile_id': f"{tile_id[0]}_{tile_id[1]}",
                'row_type': 'X',
                'row_id': idx + 1,
                'row_pos': x_pos,
                'plant_id': plant_id,
                'plant_coord': plant_y,
                'plant_x': x_pos,
                'plant_y': plant_y,
                'plant_height': plant_h
            })
    
    # Y轴方向的行
    for idx, peak_idx in enumerate(y_peaks):
        y_pos = y_centers[peak_idx]
        
        y_min_search = y_pos - cutoff_distance
        y_max_search = y_pos + cutoff_distance
        
        start_idx = np.searchsorted(tile_y_sorted, y_min_search, side='left')
        end_idx = np.searchsorted(tile_y_sorted, y_max_search, side='right')
        
        n_points = end_idx - start_idx
        if n_points < 10:
            continue
        
        candidate_x = tile_x_sorted_for_y[start_idx:end_idx]
        candidate_y = tile_y_sorted[start_idx:end_idx]
        candidate_z = tile_z_sorted_for_y[start_idx:end_idx]
        
        distances = np.abs(candidate_y - y_pos)
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        
        valid_mask = weights > 0.01
        section_x = candidate_x[valid_mask]
        section_z = candidate_z[valid_mask]
        section_weights = weights[valid_mask]
        
        if len(section_x) < 10:
            continue
        
        # 检测植物
        plant_positions, plant_heights, _, _ = detect_individual_plants_in_section(
            section_x, section_z, section_weights, 'X', granularity)
        
        # 记录（保存高度）
        for plant_id, (plant_x, plant_h) in enumerate(zip(plant_positions, plant_heights), 1):
            plant_records.append({
                'tile_id': f"{tile_id[0]}_{tile_id[1]}",
                'row_type': 'Y',
                'row_id': idx + 1,
                'row_pos': y_pos,
                'plant_id': plant_id,
                'plant_coord': plant_x,
                'plant_x': plant_x,
                'plant_y': y_pos,
                'plant_height': plant_h
            })
    
    # 【去重】合并X轴和Y轴检测结果，使用0.15m最小间距去重，优先保留更高的植物
    if len(plant_records) > 0:
        print(f"    检测到 {len(plant_records)} 个植物位置（去重前）...")
        
        import heapq
        
        # 构建最大堆（按高度优先）
        plant_heap = []
        for record in plant_records:
            x = record['plant_x']
            y = record['plant_y']
            h = record['plant_height']
            # 负高度构建最大堆
            heapq.heappush(plant_heap, (-h, x, y, record))
        
        # 贪心去重
        min_spacing = 0.15  # 15cm最小间距
        deduplicated = []
        selected_positions = []
        
        while plant_heap:
            neg_h, x, y, record = heapq.heappop(plant_heap)
            h = -neg_h
            
            # 检查与已选择植物的2D距离
            too_close = False
            for sel_x, sel_y in selected_positions:
                distance = np.sqrt((x - sel_x)**2 + (y - sel_y)**2)
                if distance < min_spacing:
                    too_close = True
                    break
            
            if not too_close:
                selected_positions.append((x, y))
                deduplicated.append(record)
        
        plant_records = deduplicated
        print(f"    去重后保留 {len(plant_records)} 个植物位置（最小间距15cm）")
    
    return n_sections, plant_records


def process_single_tile(tile_id, tile_x, tile_y, tile_z, tile_info, 
                        output_dir, chm_resolution):
    """
    处理单个tile
    
    参数:
        tile_id: tile ID (i, j)
        tile_x, tile_y, tile_z: tile点云坐标
        tile_info: tile信息字典
        output_dir: 输出目录
        chm_resolution: CHM分辨率，同时用作所有分析的统一粒度
    """
    i, j = tile_id
    print(f"\n{'='*50}")
    print(f"处理 Tile ({i}, {j}) - {tile_info['n_points']:,} 点")
    print(f"{'='*50}")
    
    # 创建tile输出目录
    tile_dir = output_dir / f"tile_{i}_{j}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    # 统一使用chm_resolution作为分析粒度
    granularity = chm_resolution
    print(f"  使用分析粒度: {granularity*100:.1f}cm")
    
    # X轴分析
    print("  计算X轴高度分布...")
    x_centers, x_heights, x_counts = compute_height_profile_along_axis(
        tile_x, tile_z, granularity=granularity
    )
    
    # Y轴分析
    print("  计算Y轴高度分布...")
    y_centers, y_heights, y_counts = compute_height_profile_along_axis(
        tile_y, tile_z, granularity=granularity
    )
    
    # 检测peaks（sigma根据粒度自适应：保持~5cm的平滑效果）
    print("  检测作物行...")
    sigma_bins = int(0.05 / granularity)  # 5cm / granularity
    x_peaks, x_smoothed, x_clarity = filter_and_find_peaks(
        x_centers, x_heights, sigma=sigma_bins
    )
    y_peaks, y_smoothed, y_clarity = filter_and_find_peaks(
        y_centers, y_heights, sigma=sigma_bins
    )
    
    # 生成可视化
    output_prefix = str(tile_dir / f"tile_{i}_{j}")
    
    print("  生成高度曲线图...")
    plot_height_profiles(x_centers, x_heights, x_peaks, x_smoothed,
                        y_centers, y_heights, y_peaks, y_smoothed,
                        output_prefix)
    
    # 先生成切面视图并检测单株植物（获取plant_records）
    n_cross_sections, plant_records = create_cross_section_views_with_plant_detection(
        tile_x, tile_y, tile_z, 
        x_centers, x_peaks, y_centers, y_peaks,
        chm_resolution, tile_dir, tile_id, granularity
    )
    
    print("  生成CHM可视化（含植物位置）...")
    create_chm_with_crop_rows(tile_x, tile_y, tile_z, 
                              x_centers, x_peaks, y_centers, y_peaks,
                              chm_resolution, output_prefix, plant_records)
    
    # 保存植物位置CSV
    if len(plant_records) > 0:
        import csv
        csv_path = tile_dir / "individual_plants.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['tile_id', 'row_type', 'row_id', 
                                                   'row_pos', 'plant_id', 'plant_coord',
                                                   'plant_x', 'plant_y', 'plant_height'])
            writer.writeheader()
            writer.writerows(plant_records)
        print(f"  已保存 {len(plant_records)} 个植物位置到: {csv_path}")
    
    # 保存统计信息
    stats = {
        'tile_id': tile_id,
        'bounds': tile_info,
        'n_points': tile_info['n_points'],
        'x_rows': len(x_peaks),
        'y_rows': len(y_peaks),
        'total_rows': len(x_peaks) + len(y_peaks),
        'cross_sections': n_cross_sections,
        'detected_plants': len(plant_records)
    }
    
    return stats


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python crop_row_finder.py <las文件路径> [选项]")
        print("示例: python crop_row_finder.py soybean.las")
        print("      python crop_row_finder.py soybean.las --downsample 0.02")
        print("      python crop_row_finder.py soybean.las --tile_size 20")
        sys.exit(1)
    
    las_path = sys.argv[1]
    
    # 解析参数
    downsample_size = None
    if '--downsample' in sys.argv:
        idx = sys.argv.index('--downsample')
        if idx + 1 < len(sys.argv):
            downsample_size = float(sys.argv[idx + 1])
    
    tile_size = 10.0  # 默认10x10m
    if '--tile_size' in sys.argv:
        idx = sys.argv.index('--tile_size')
        if idx + 1 < len(sys.argv):
            tile_size = float(sys.argv[idx + 1])
    
    if not Path(las_path).exists():
        print(f"错误: 文件不存在: {las_path}")
        sys.exit(1)
    
    print("="*60)
    print("作物行识别 - Tile分割处理")
    print("="*60)
    total_t0 = time.time()
    
    # 创建输出目录
    output_base = Path("visualization")
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取点云
    x, y, z = read_las_file(las_path)
    
    # 2. 可选：体素降采样加速
    if downsample_size is not None:
        x, y, z = downsample_voxel_grid(x, y, z, voxel_size=downsample_size)
    
    # 3. 标准化Z轴到z=0
    x, y, z = normalize_z_to_zero(x, y, z)
    
    # 4. 移除地面点（底部15%）
    print("\n预处理...")
    x, y, z = remove_ground_points(x, y, z, percentile=15)
    
    # 5. 分割成tiles
    tiles, tile_info = split_into_tiles(x, y, z, tile_size=tile_size)
    
    # 6. 处理每个tile
    chm_resolution = downsample_size if downsample_size is not None else 0.02
    all_stats = []
    
    for tile_id, (tile_x, tile_y, tile_z) in tiles.items():
        stats = process_single_tile(tile_id, tile_x, tile_y, tile_z, 
                                     tile_info[tile_id], output_base, 
                                     chm_resolution)
        all_stats.append(stats)
    
    # 7. 生成总结报告
    print("\n" + "="*60)
    print("处理总结")
    print("="*60)
    total_x_rows = sum(s['x_rows'] for s in all_stats)
    total_y_rows = sum(s['y_rows'] for s in all_stats)
    total_cross_sections = sum(s['cross_sections'] for s in all_stats)
    total_plants = sum(s.get('detected_plants', 0) for s in all_stats)
    print(f"处理tiles: {len(all_stats)}")
    print(f"总检测行数: X轴={total_x_rows}, Y轴={total_y_rows}")
    print(f"总切面视图: {total_cross_sections}")
    print(f"总检测植物: {total_plants}")
    print(f"输出目录: {output_base.absolute()}")
    
    # 保存summary
    summary_file = output_base / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Crop Row Detection & Individual Plant Detection Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input file: {las_path}\n")
        f.write(f"Tile size: {tile_size}x{tile_size}m\n")
        f.write(f"Downsample size: {downsample_size if downsample_size else 'None'}m\n")
        f.write(f"Total tiles processed: {len(all_stats)}\n")
        f.write(f"Total cross sections: {total_cross_sections}\n")
        f.write(f"Total individual plants detected: {total_plants}\n\n")
        for stats in all_stats:
            i, j = stats['tile_id']
            f.write(f"Tile ({i},{j}): X_rows={stats['x_rows']}, Y_rows={stats['y_rows']}, "
                   f"cross_sections={stats['cross_sections']}, "
                   f"plants={stats.get('detected_plants', 0)}\n")
    
    total_time = time.time() - total_t0
    print(f"\n✓ 处理完成！总耗时: {total_time:.2f}秒")
    print("="*60)


if __name__ == "__main__":
    main()
