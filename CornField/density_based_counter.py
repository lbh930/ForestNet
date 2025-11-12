#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Density-based Plant Counter
基于密度峰检测的单株植物计数

用法:
    python density_based_counter.py input.las --direction y --output results/
    python density_based_counter.py input.las --direction x --spacing 0.25
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


def compute_density_profile(points, direction, bin_size, verbose=False):
    """
    计算沿指定方向的密度曲线
    
    参数:
        points: Nx3点云数组 [x, y, z]
        direction: 'x' 或 'y'，沿哪个方向计算
        bin_size: bin大小（米）
    
    返回:
        bin_centers: bin中心坐标
        density: 每个bin的点密度（加权高度）
        raw_counts: 每个bin的原始点数
    """
    if verbose:
        print(f"  计算沿{direction.upper()}轴的密度曲线...")
    
    # 选择坐标轴
    if direction.lower() == 'x':
        coord = points[:, 0]  # X坐标
        height = points[:, 2]  # Z高度作为权重
    elif direction.lower() == 'y':
        coord = points[:, 1]  # Y坐标
        height = points[:, 2]  # Z高度作为权重
    else:
        raise ValueError("direction必须是 'x' 或 'y'")
    
    # 创建bins
    coord_min, coord_max = coord.min(), coord.max()
    n_bins = int(np.ceil((coord_max - coord_min) / bin_size))
    
    if n_bins < 10:
        if verbose:
            print(f"    警告: bin数量太少 ({n_bins})")
        return np.array([]), np.array([]), np.array([])
    
    bins = np.linspace(coord_min, coord_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 计算每个bin的密度（使用高度加权）
    density = np.zeros(n_bins)
    raw_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (coord >= bins[i]) & (coord < bins[i+1])
        raw_counts[i] = np.sum(mask)
        
        if raw_counts[i] > 0:
            # 密度 = 点数 * 平均高度（高的植物权重更大）
            density[i] = raw_counts[i] * height[mask].mean()
    
    if verbose:
        print(f"    Bins: {n_bins}, 范围: [{coord_min:.2f}, {coord_max:.2f}]m")
        print(f"    密度范围: [{density.min():.3f}, {density.max():.3f}]")
    
    return bin_centers, density, raw_counts


def detect_plants_from_density(bin_centers, density, expected_spacing, 
                               min_prominence, smooth_sigma=1.0, verbose=False):
    """
    从密度曲线检测植物峰
    
    参数:
        bin_centers: bin中心坐标
        density: 密度值
        expected_spacing: 预期株间距（米）
        min_prominence: 最小峰突出度（相对值）
        smooth_sigma: 高斯平滑的sigma值（越大越平滑）
    
    返回:
        peak_positions: 检测到的植物位置
        peak_densities: 对应的密度值
        smoothed_density: 平滑后的密度曲线
    """
    if verbose:
        print(f"  检测植物峰（预期株间距: {expected_spacing*100:.0f}cm, sigma={smooth_sigma:.1f}）...")
    
    if len(density) < 10:
        return np.array([]), np.array([]), density
    
    # 计算bin大小
    bin_size = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.01
    
    # Step 1: 高斯平滑
    smoothed_density = gaussian_filter1d(density, sigma=smooth_sigma)
    
    # Step 2: 峰检测
    # distance参数：预期株间距对应的bin数
    min_distance_bins = int(expected_spacing / bin_size)
    
    # prominence: 相对于局部基线的突出度
    abs_prominence = min_prominence * smoothed_density.max()
    
    peaks, properties = find_peaks(
        smoothed_density, 
        distance=min_distance_bins,
        prominence=abs_prominence,
        width=1  # 至少1个bin宽
    )
    
    if len(peaks) == 0:
        if verbose:
            print(f"    未检测到峰")
        return np.array([]), np.array([]), smoothed_density
    
    # 获取峰位置和密度值
    peak_positions = bin_centers[peaks]
    peak_densities = smoothed_density[peaks]
    
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
    
    return peak_positions, peak_densities, smoothed_density


def visualize_density_profile(bin_centers, density, smoothed_density, 
                              peak_positions, peak_densities, raw_counts,
                              output_path, direction, verbose=False):
    """
    可视化密度曲线和检测结果
    
    参数:
        bin_centers: bin中心坐标
        density: 原始密度
        smoothed_density: 平滑后的密度
        peak_positions: 峰位置
        peak_densities: 峰密度值
        raw_counts: 原始点数
        output_path: 输出文件路径
        direction: 方向轴
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    # 只显示平滑后的密度曲线
    ax1.plot(bin_centers, smoothed_density, 'b-', linewidth=3, label='Point Density')
    
    # 标记峰位置
    if len(peak_positions) > 0:
        ax1.plot(peak_positions, peak_densities, 'rx', 
                markersize=18, markeredgewidth=3.75, 
                label=f'Detected Plants (n={len(peak_positions)})')
        
        # 添加垂直虚线
        for pos in peak_positions:
            ax1.axvline(pos, color='red', linestyle='--', linewidth=1.5, alpha=0.3)
    
    ax1.set_xlabel(f'{direction.upper()} Coordinate (m)', fontsize=25)
    ax1.set_ylabel('Density (weighted by height)', fontsize=25)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=25)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  可视化已保存: {output_path}")


def density_count_from_row(points, direction, expected_spacing, 
                           bin_size, apply_sor, sor_k, sor_std_ratio,
                           remove_ground, ground_percentile, top_percentile,
                           min_prominence, smooth_sigma=1.0, output_dir=None, row_center=None,
                           row_status: str | None = None, verbose: bool = False):
    """
    从单行点云中基于密度检测并计数植物
    
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
        output_dir: 可视化输出目录
        row_center: 行的中心坐标（用于计算完整XY坐标），如果为None则使用点云均值
    
    返回:
        plant_count: 检测到的植物数量
        peak_positions: 植物位置列表
        results_dict: 包含详细结果的字典
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"基于密度的植物计数")
        print(f"{'='*60}")
        print(f"输入点数: {len(points):,}")
        print(f"检测方向: {direction.upper()}轴")
        print(f"预期株间距: {expected_spacing*100:.0f}cm")
    
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
    
    # Step 3: 计算密度曲线
    bin_centers, density, raw_counts = compute_density_profile(
        points, direction=direction, bin_size=bin_size, verbose=verbose
    )
    
    if len(density) == 0:
        print("错误: 无法计算密度曲线")
        return 0, np.array([]), {}
    
    # Step 4: 峰检测
    peak_positions, peak_densities, smoothed_density = detect_plants_from_density(
        bin_centers, density, expected_spacing=expected_spacing, 
        min_prominence=min_prominence, smooth_sigma=smooth_sigma, verbose=verbose
    )
    
    plant_count = len(peak_positions)
    
    # Step 5: 计算实际高度（使用SOR后但未掐头去尾的点云）
    if plant_count > 0 and verbose:
        print(f"  计算植物实际高度（使用SOR后的完整点云）...")
        actual_heights = calculate_peak_heights(sor_cleaned_points, peak_positions, direction)
        if verbose:
            print(f"    高度范围: [{actual_heights.min():.3f}, {actual_heights.max():.3f}]m")
            print(f"    平均高度: {actual_heights.mean():.3f}m")
    else:
        actual_heights = calculate_peak_heights(sor_cleaned_points, peak_positions, direction) if plant_count > 0 else np.array([])
    
    # 生成可视化（如果指定了输出目录）
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 密度曲线图
        profile_path = output_dir / f"density_profile_{direction}.png"
        visualize_density_profile(
            bin_centers, density, smoothed_density, 
            peak_positions, peak_densities, raw_counts,
            profile_path, direction, verbose=verbose
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
            'method': 'density',
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
        'peak_densities': peak_densities,
        'actual_heights': actual_heights,  # 实际计算的高度
        'bin_centers': bin_centers,
        'density': density,
        'smoothed_density': smoothed_density,
        'raw_counts': raw_counts,
        'filtered_points': points
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ 检测完成！共检测到 {plant_count} 株植物")
        print(f"{'='*60}\n")

    # Row-level minimal status log (if provided)
    if row_status and verbose:
        print(row_status)
    
    return plant_count, peak_positions, results


def main():
    """主函数 - 用于测试"""
    if len(sys.argv) < 2:
        print("用法: python density_based_counter.py <las文件> [选项]")
        print("\n选项:")
        print("  --direction x|y     检测方向（默认: y）")
        print("  --spacing FLOAT     预期株间距（米）（默认: 0.05）")
        print("  --bin_size FLOAT    bin大小（米）（默认: 0.005）")
        print("  --prominence FLOAT  峰突出度阈值（相对值）（默认: 0.03）")
        print("  --output DIR        可视化输出目录（默认: density_results）")
        print("  --no-ground         不移除地面点和顶部点")
        print("  --ground_pct FLOAT  地面移除百分比（默认: 30）")
        print("  --top_pct FLOAT     顶部移除百分比（默认: 30）")
        print("  --no-sor            不应用统计离群点移除")
        print("  --sor_k INT         SOR的k值（默认: 20）")
        print("  --sor_std FLOAT     SOR的标准差倍数（默认: 2.0）")
        print("\n示例:")
        print("  python density_based_counter.py row_points.las")
        print("  python density_based_counter.py row_points.las --direction x --spacing 0.3")
        print("  python density_based_counter.py row_points.las --output my_results/")
        sys.exit(1)
    
    # 解析参数
    las_path = sys.argv[1]
    
    # 默认值
    direction = 'y'
    expected_spacing = 0.08
    bin_size = 0.005
    min_prominence = 0.02
    output_dir = 'density_results'
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
    plant_count, peak_positions, results = density_count_from_row(
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
