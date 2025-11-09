#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leaf Area Density (LAD) Estimator
叶面积密度估算器 - 基于体素化和Beer-Lambert定律

用法:
    python leaf_density_estimator.py input_row.las [--vx 0.05] [--vy 0.05] [--vz 0.03]
"""

import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
import time
import yaml


def read_las_file(las_path):
    """读取LAS文件并返回点云坐标"""
    print(f"读取点云文件: {las_path}")
    t0 = time.time()
    las = laspy.read(las_path)
    
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    
    print(f"  点云总数: {len(x):,} 个点 (耗时: {time.time()-t0:.2f}秒)")
    print(f"  X范围: [{x.min():.3f}, {x.max():.3f}]m (跨度: {x.max()-x.min():.3f}m)")
    print(f"  Y范围: [{y.min():.3f}, {y.max():.3f}]m (跨度: {y.max()-y.min():.3f}m)")
    print(f"  Z范围: [{z.min():.3f}, {z.max():.3f}]m (跨度: {z.max()-z.min():.3f}m)")
    
    return x, y, z


def remove_ground_simple(x, y, z, bottom_percentile=10.0):
    """
    简单地移除底部百分比的点作为地面
    
    参数:
        x, y, z: 点云坐标
        bottom_percentile: 移除底部的百分比（默认10%）
    
    返回:
        x, y, z: 过滤后的点云
    """
    z_min = z.min()
    z_max = z.max()
    z_range = z_max - z_min
    
    # 计算阈值
    threshold = z_min + (z_range * bottom_percentile / 100.0)
    
    # 过滤
    mask = z > threshold
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    
    removed = len(z) - len(z_filtered)
    print(f"\n地面移除:")
    print(f"  Z范围: [{z_min:.3f}, {z_max:.3f}]m (跨度: {z_range:.3f}m)")
    print(f"  底部{bottom_percentile}%阈值: {threshold:.3f}m")
    print(f"  移除点数: {removed} ({removed/len(z)*100:.1f}%)")
    print(f"  剩余点数: {len(z_filtered)}")
    
    return x_filtered, y_filtered, z_filtered


def voxelize_point_cloud(x, y, z, vx=0.05, vy=0.05, vz=0.03):
    """
    将点云体素化
    
    参数:
        x, y, z: 点云坐标
        vx, vy, vz: 体素尺寸 (m)
    
    返回:
        voxel_grid: 3D布尔数组，True表示该体素有点
        voxel_counts: 3D数组，每个体素的点数
        grid_info: 字典，包含网格信息
    """
    print(f"\n体素化点云: vx={vx}m, vy={vy}m, vz={vz}m")
    t0 = time.time()
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    # 计算网格尺寸
    nx = int(np.ceil((x_max - x_min) / vx))
    ny = int(np.ceil((y_max - y_min) / vy))
    nz = int(np.ceil((z_max - z_min) / vz))
    
    print(f"  网格尺寸: {nx} x {ny} x {nz} = {nx*ny*nz:,} 体素")
    
    # 将点映射到体素索引
    ix = ((x - x_min) / vx).astype(int)
    iy = ((y - y_min) / vy).astype(int)
    iz = ((z - z_min) / vz).astype(int)
    
    # 边界处理
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    
    # 创建体素网格
    voxel_grid = np.zeros((nx, ny, nz), dtype=bool)
    voxel_counts = np.zeros((nx, ny, nz), dtype=int)
    
    # 填充体素
    for i in range(len(x)):
        voxel_grid[ix[i], iy[i], iz[i]] = True
        voxel_counts[ix[i], iy[i], iz[i]] += 1
    
    occupied_voxels = np.sum(voxel_grid)
    print(f"  占用体素: {occupied_voxels:,} ({occupied_voxels/(nx*ny*nz)*100:.2f}%)")
    print(f"  体素化耗时: {time.time()-t0:.2f}秒")
    
    grid_info = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_min': z_min, 'z_max': z_max,
        'nx': nx, 'ny': ny, 'nz': nz,
        'vx': vx, 'vy': vy, 'vz': vz
    }
    
    return voxel_grid, voxel_counts, grid_info


def compute_gap_probability(voxel_grid):
    """
    计算每个高度层的穿透率 P_gap(z)
    
    简化实现：对每个高度层，计算水平面内有植被的体素比例
    P_gap(z) = 1 - (有植被的体素数 / 总体素数)
    
    参数:
        voxel_grid: 3D布尔数组
    
    返回:
        z_indices: 高度层索引
        p_gap: 每层的穿透率
        occupancy: 每层的占用率 (1 - p_gap)
    """
    print("\n计算穿透率 P_gap(z)...")
    
    nx, ny, nz = voxel_grid.shape
    z_indices = np.arange(nz)
    
    # 对每个高度层，计算占用率
    occupancy = np.zeros(nz)
    
    for k in range(nz):
        # 该层的水平切片
        layer = voxel_grid[:, :, k]
        # 占用的体素数 / 总体素数
        occupancy[k] = np.sum(layer) / (nx * ny)
    
    # 穿透率 = 1 - 占用率
    p_gap = 1.0 - occupancy
    
    print(f"  高度层数: {nz}")
    print(f"  平均占用率: {np.mean(occupancy):.3f}")
    print(f"  平均穿透率: {np.mean(p_gap):.3f}")
    
    return z_indices, p_gap, occupancy


def compute_lad_beer_lambert(p_gap, vz, G=0.5):
    """
    使用Beer-Lambert定律反演LAD
    
    简化形式：
    LAD(z) ≈ -1/G * d(ln P_gap) / dz
    
    离散化：
    LAD(z_k) ≈ -1/G * (ln P_gap[k+1] - ln P_gap[k]) / vz
    
    参数:
        p_gap: 每层的穿透率
        vz: 垂直体素尺寸 (m)
        G: 投影函数 (默认0.5，水平叶假设)
    
    返回:
        lad: 叶面积密度 [m²/m³]
    """
    print(f"\n计算LAD (Beer-Lambert法, G={G})...")
    
    nz = len(p_gap)
    lad = np.zeros(nz)
    
    # 避免log(0)
    p_gap_safe = np.maximum(p_gap, 1e-6)
    ln_p_gap = np.log(p_gap_safe)
    
    # 向上差分 (从底到顶)
    for k in range(nz - 1):
        d_ln_p = ln_p_gap[k + 1] - ln_p_gap[k]
        lad[k] = -d_ln_p / (G * vz)
    
    # 最后一层使用前一层的值
    if nz > 1:
        lad[-1] = lad[-2]
    
    # 限制LAD为非负值（物理意义）
    lad = np.maximum(lad, 0.0)
    
    # 统计
    valid_lad = lad[lad > 0]
    if len(valid_lad) > 0:
        print(f"  LAD范围: [{valid_lad.min():.4f}, {valid_lad.max():.4f}] m²/m³")
        print(f"  LAD平均: {valid_lad.mean():.4f} m²/m³")
        print(f"  LAD中位数: {np.median(valid_lad):.4f} m²/m³")
    else:
        print(f"  警告: 没有有效的LAD值")
    
    return lad


def plot_vertical_profiles(z_indices, p_gap, occupancy, lad, grid_info, output_dir):
    """
    绘制垂直剖面图
    """
    print("\n生成垂直剖面图...")
    
    vz = grid_info['vz']
    z_min = grid_info['z_min']
    
    # 转换索引到实际高度
    z_actual = z_min + (z_indices + 0.5) * vz  # 体素中心高度
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 占用率
    ax1 = axes[0]
    ax1.plot(occupancy, z_actual, 'b-', linewidth=2)
    ax1.fill_betweenx(z_actual, 0, occupancy, alpha=0.3, color='blue')
    ax1.set_xlabel('Occupancy Rate', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('Voxel Occupancy by Height', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # 2. 穿透率
    ax2 = axes[1]
    ax2.plot(p_gap, z_actual, 'g-', linewidth=2)
    ax2.fill_betweenx(z_actual, 0, p_gap, alpha=0.3, color='green')
    ax2.set_xlabel('Gap Probability P_gap', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Gap Probability Profile', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # 3. LAD
    ax3 = axes[2]
    ax3.plot(lad, z_actual, 'r-', linewidth=2)
    ax3.fill_betweenx(z_actual, 0, lad, alpha=0.3, color='red')
    ax3.set_xlabel('LAD (m²/m³)', fontsize=12)
    ax3.set_ylabel('Height (m)', fontsize=12)
    ax3.set_title('Leaf Area Density Profile', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "vertical_profiles.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  已保存: {output_file.name}")


def plot_voxel_visualization(voxel_grid, grid_info, output_dir):
    """
    绘制体素化可视化（3D或切片）
    """
    print("\n生成体素可视化...")
    
    nx, ny, nz = voxel_grid.shape
    vz = grid_info['vz']
    z_min = grid_info['z_min']
    
    # 创建高度切片可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 选择6个高度层均匀分布
    layer_indices = np.linspace(0, nz-1, 6, dtype=int)
    
    for idx, k in enumerate(layer_indices):
        ax = axes[idx]
        layer = voxel_grid[:, :, k]
        
        # 显示该层
        im = ax.imshow(layer.T, origin='lower', cmap='Greens', interpolation='nearest')
        
        z_height = z_min + (k + 0.5) * vz
        ax.set_title(f'Height Layer: {z_height:.2f}m (k={k})', fontsize=11, fontweight='bold')
        ax.set_xlabel('X voxel index', fontsize=10)
        ax.set_ylabel('Y voxel index', fontsize=10)
        
        # 显示占用率
        occupancy_rate = np.sum(layer) / (nx * ny)
        ax.text(0.02, 0.98, f'Occupancy: {occupancy_rate:.2%}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    output_file = output_dir / "voxel_layers.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  已保存: {output_file.name}")


def save_summary_yaml(output_path, summary_data):
    """
    保存LAD计算结果为YAML格式
    """
    output_path = Path(output_path)
    
    yaml_data = {
        'leaf_density_summary': {
            'input_file': summary_data['input_file'],
            'statistics': {
                'input_points': int(summary_data['input_points']),
                'filtered_points': int(summary_data['filtered_points']),
                'occupied_voxels': int(summary_data['occupied_voxels']),
                'total_voxels': int(summary_data['total_voxels']),
            },
            'parameters': {
                'voxel_size': {
                    'vx_m': float(summary_data['vx']),
                    'vy_m': float(summary_data['vy']),
                    'vz_m': float(summary_data['vz']),
                },
                'ground_removal': {
                    'bottom_percentile': float(summary_data['bottom_percentile']),
                },
                'beer_lambert': {
                    'projection_function_G': float(summary_data['G']),
                },
            },
            'results': {
                'lad_statistics': {
                    'mean_m2_m3': float(summary_data['lad_mean']),
                    'median_m2_m3': float(summary_data['lad_median']),
                    'max_m2_m3': float(summary_data['lad_max']),
                    'std_m2_m3': float(summary_data['lad_std']),
                },
                'lai_estimate': {
                    'total_m2_m2': float(summary_data['lai_total']),
                    'description': 'Leaf Area Index estimated by integrating LAD over height',
                },
            },
            'height_layers': summary_data['height_layers'],
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  已保存: {output_path.name}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python leaf_density_estimator.py <las文件或文件夹路径> [选项]")
        print("示例: python leaf_density_estimator.py row_x01.las --vx 0.05 --vy 0.05 --vz 0.03")
        print("      python leaf_density_estimator.py ./tile_0_0/ --vx 0.05")
        print("\n选项:")
        print("  --vx <float>   水平体素尺寸X (默认: 0.05m)")
        print("  --vy <float>   水平体素尺寸Y (默认: 0.05m)")
        print("  --vz <float>   垂直体素尺寸 (默认: 0.03m)")
        print("  --G <float>    投影函数 (默认: 0.5)")
        print("  --ground <float>  地面移除百分比 (默认: 10%)")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # 解析参数
    vx = 0.05  # 默认5cm
    vy = 0.05
    vz = 0.03  # 默认3cm
    G = 0.5
    ground_percentile = 10.0
    
    if '--vx' in sys.argv:
        idx = sys.argv.index('--vx')
        if idx + 1 < len(sys.argv):
            vx = float(sys.argv[idx + 1])
    
    if '--vy' in sys.argv:
        idx = sys.argv.index('--vy')
        if idx + 1 < len(sys.argv):
            vy = float(sys.argv[idx + 1])
    
    if '--vz' in sys.argv:
        idx = sys.argv.index('--vz')
        if idx + 1 < len(sys.argv):
            vz = float(sys.argv[idx + 1])
    
    if '--G' in sys.argv:
        idx = sys.argv.index('--G')
        if idx + 1 < len(sys.argv):
            G = float(sys.argv[idx + 1])
    
    if '--ground' in sys.argv:
        idx = sys.argv.index('--ground')
        if idx + 1 < len(sys.argv):
            ground_percentile = float(sys.argv[idx + 1])
    
    if not input_path.exists():
        print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)
    
    print("="*60)
    print("叶面积密度估算器 - LAD Estimator")
    print("="*60)
    total_t0 = time.time()
    
    # 判断输入是文件还是文件夹
    if input_path.is_file():
        # 单个文件模式
        las_files = [input_path]
        output_base_dir = input_path.parent / "leaf_density_results"
    elif input_path.is_dir():
        # 文件夹模式：查找所有.las文件
        las_files = sorted(input_path.glob("*.las"))
        if len(las_files) == 0:
            print(f"错误: 文件夹中没有找到.las文件: {input_path}")
            sys.exit(1)
        output_base_dir = input_path / "leaf_density_results"
    else:
        print(f"错误: 无效的输入路径: {input_path}")
        sys.exit(1)
    
    print(f"\n找到 {len(las_files)} 个.las文件")
    for las_file in las_files:
        print(f"  - {las_file.name}")
    
    # 创建输出目录
    if output_base_dir.exists():
        import shutil
        shutil.rmtree(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base_dir}")
    
    # 处理所有文件并收集LAI
    all_lai_values = []
    all_results = []
    
    for file_idx, las_path in enumerate(las_files, 1):
        print(f"\n{'='*60}")
        print(f"处理文件 {file_idx}/{len(las_files)}: {las_path.name}")
        print(f"{'='*60}")
        
        # 为每个文件创建子目录
        file_output_dir = output_base_dir / las_path.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 读取点云
            x, y, z = read_las_file(las_path)
            n_input = len(x)
            
            # 2. 移除地面
            x, y, z = remove_ground_simple(x, y, z, bottom_percentile=ground_percentile)
            n_filtered = len(x)
            
            # 3. 体素化
            voxel_grid, voxel_counts, grid_info = voxelize_point_cloud(x, y, z, vx, vy, vz)
            
            # 4. 计算穿透率
            z_indices, p_gap, occupancy = compute_gap_probability(voxel_grid)
            
            # 5. 计算LAD
            lad = compute_lad_beer_lambert(p_gap, vz, G=G)
            
            # 6. 计算LAI (Leaf Area Index)
            lai_total = np.sum(lad) * vz
            all_lai_values.append(lai_total)
            
            print(f"\n叶面积指数 (LAI):")
            print(f"  LAI = {lai_total:.3f} m²/m²")
            
            # 7. 统计
            valid_lad = lad[lad > 0]
            lad_mean = valid_lad.mean() if len(valid_lad) > 0 else 0.0
            lad_median = np.median(valid_lad) if len(valid_lad) > 0 else 0.0
            lad_max = valid_lad.max() if len(valid_lad) > 0 else 0.0
            lad_std = valid_lad.std() if len(valid_lad) > 0 else 0.0
            
            # 8. 可视化
            plot_vertical_profiles(z_indices, p_gap, occupancy, lad, grid_info, file_output_dir)
            plot_voxel_visualization(voxel_grid, grid_info, file_output_dir)
            
            # 9. 准备高度层详细数据
            z_actual = grid_info['z_min'] + (z_indices + 0.5) * vz
            height_layers = []
            for i, (z_val, lad_val, occ_val, pgap_val) in enumerate(zip(z_actual, lad, occupancy, p_gap)):
                height_layers.append({
                    'layer_index': int(i),
                    'height_m': float(z_val),
                    'lad_m2_m3': float(lad_val),
                    'occupancy': float(occ_val),
                    'gap_probability': float(pgap_val),
                })
            
            # 10. 保存单个文件的YAML摘要
            summary_data = {
                'input_file': str(las_path.name),
                'input_points': n_input,
                'filtered_points': n_filtered,
                'occupied_voxels': int(np.sum(voxel_grid)),
                'total_voxels': int(voxel_grid.size),
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'G': G,
                'bottom_percentile': ground_percentile,
                'lad_mean': lad_mean,
                'lad_median': lad_median,
                'lad_max': lad_max,
                'lad_std': lad_std,
                'lai_total': lai_total,
                'height_layers': height_layers,
            }
            
            yaml_path = file_output_dir / "leaf_summary.yaml"
            save_summary_yaml(yaml_path, summary_data)
            
            # 记录结果
            all_results.append({
                'filename': las_path.name,
                'lai': lai_total,
                'lad_mean': lad_mean,
                'lad_max': lad_max,
                'input_points': n_input,
                'filtered_points': n_filtered,
            })
            
            print(f"✓ {las_path.name} 处理完成")
            
        except Exception as e:
            print(f"✗ 处理 {las_path.name} 时出错: {e}")
            continue
    
    # 11. 计算汇总统计
    if len(all_lai_values) > 0:
        mean_lai = np.mean(all_lai_values)
        median_lai = np.median(all_lai_values)
        std_lai = np.std(all_lai_values)
        min_lai = np.min(all_lai_values)
        max_lai = np.max(all_lai_values)
        
        print("\n" + "="*60)
        print("汇总统计 - 所有文件")
        print("="*60)
        print(f"成功处理文件数: {len(all_lai_values)} / {len(las_files)}")
        print(f"\nLAI统计:")
        print(f"  平均LAI: {mean_lai:.3f} m²/m²")
        print(f"  中位数LAI: {median_lai:.3f} m²/m²")
        print(f"  标准差: {std_lai:.3f} m²/m²")
        print(f"  范围: [{min_lai:.3f}, {max_lai:.3f}] m²/m²")
        
        print(f"\n各文件LAI详情:")
        for result in all_results:
            print(f"  {result['filename']:40s} LAI={result['lai']:.3f} m²/m²")
        
        # 保存汇总YAML
        summary_yaml_data = {
            'aggregate_summary': {
                'total_files_processed': len(all_lai_values),
                'total_files_found': len(las_files),
                'parameters': {
                    'voxel_size': {'vx_m': vx, 'vy_m': vy, 'vz_m': vz},
                    'projection_function_G': G,
                    'ground_removal_percentile': ground_percentile,
                },
                'lai_statistics': {
                    'mean_m2_m2': float(mean_lai),
                    'median_m2_m2': float(median_lai),
                    'std_m2_m2': float(std_lai),
                    'min_m2_m2': float(min_lai),
                    'max_m2_m2': float(max_lai),
                },
                'individual_files': all_results,
            }
        }
        
        aggregate_yaml_path = output_base_dir / "aggregate_summary.yaml"
        with open(aggregate_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary_yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"\n汇总结果已保存: {aggregate_yaml_path}")
        
    print(f"\n输出目录: {output_base_dir.absolute()}")
    print(f"总耗时: {time.time()-total_t0:.2f}秒")
    print("="*60)


if __name__ == "__main__":
    main()
