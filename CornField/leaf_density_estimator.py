#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leaf Area Density (LAD) Estimator
基于体素化和Beer-Lambert定律的叶面积密度估算器
运行方式与输出目录与旧版完全一致
"""

import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
import time
import yaml
import shutil


def read_las_file(las_path):
    print(f"读取点云文件: {las_path}")
    t0 = time.time()
    las = laspy.read(las_path)
    x, y, z = np.asarray(las.x, np.float32), np.asarray(las.y, np.float32), np.asarray(las.z, np.float32)
    print(f"  点云总数: {len(x):,} 个点 (耗时: {time.time()-t0:.2f}s)")
    return x, y, z


def remove_ground_percentile(x, y, z, bottom_percentile=15.0):
    """简单的全局percentile地面移除（用于单行点云）"""
    z_min, z_max = z.min(), z.max()
    threshold = z_min + (z_max - z_min) * (bottom_percentile / 100.0)
    mask = z > threshold
    print(f"\n地面移除: 移除底部 {bottom_percentile}% 点 (Z阈值={threshold:.3f}m)")
    return x[mask], y[mask], z[mask]


def remove_ground_adaptive(x, y, z, grid_size=1.0, bottom_percentile=15.0):
    """自适应地面移除：在每个grid内移除底部percentile的点"""
    print(f"\n自适应地面移除: 网格={grid_size}m, 移除每格底部{bottom_percentile}%")
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    nx = int(np.ceil((x_max - x_min) / grid_size))
    ny = int(np.ceil((y_max - y_min) / grid_size))
    
    print(f"  XY范围: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"  网格数: {nx}×{ny} = {nx*ny}个")
    
    # 计算每个点所属的网格
    ix = np.clip(((x - x_min) / grid_size).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / grid_size).astype(int), 0, ny - 1)
    
    mask = np.zeros(len(x), dtype=bool)
    non_empty_grids = 0
    
    for i in range(nx):
        for j in range(ny):
            grid_mask = (ix == i) & (iy == j)
            n_points = np.sum(grid_mask)
            
            if n_points > 0:
                non_empty_grids += 1
                z_grid = z[grid_mask]
                z_min_grid = z_grid.min()
                z_max_grid = z_grid.max()
                
                # 计算该网格的阈值
                threshold = z_min_grid + (z_max_grid - z_min_grid) * (bottom_percentile / 100.0)
                
                # 标记高于阈值的点
                grid_indices = np.where(grid_mask)[0]
                keep_mask = z_grid > threshold
                mask[grid_indices[keep_mask]] = True
    
    print(f"  非空网格: {non_empty_grids}/{nx*ny}")
    print(f"  移除点数: {len(x) - np.sum(mask):,} / {len(x):,} ({(1 - np.sum(mask)/len(x))*100:.1f}%)")
    
    return x[mask], y[mask], z[mask]


def voxelize_point_cloud(x, y, z, vx=0.01, vy=0.01, vz=0.01):
    print(f"\n体素化点云: vx={vx}m, vy={vy}m, vz={vz}m")
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    nx = int(np.ceil((x_max - x_min) / vx))
    ny = int(np.ceil((y_max - y_min) / vy))
    nz = int(np.ceil((z_max - z_min) / vz))
    print(f"  网格尺寸: {nx}×{ny}×{nz}")

    ix = np.clip(((x - x_min) / vx).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / vy).astype(int), 0, ny - 1)
    iz = np.clip(((z - z_min) / vz).astype(int), 0, nz - 1)

    voxel_grid = np.zeros((nx, ny, nz), dtype=bool)
    voxel_grid[ix, iy, iz] = True
    occupied_voxels = np.sum(voxel_grid)
    print(f"  占用体素: {occupied_voxels:,} ({occupied_voxels/(nx*ny*nz)*100:.2f}%)")

    return voxel_grid, {'vx': vx, 'vy': vy, 'vz': vz,
                        'z_min': z_min, 'z_max': z_max, 'nz': nz}


def compute_gap_probability(voxel_grid):
    nx, ny, nz = voxel_grid.shape
    occupancy = np.zeros(nz)
    for k in range(nz):
        occupancy[k] = np.sum(voxel_grid[:, :, k]) / (nx * ny)

    p_gap = np.ones(nz)
    for k in range(nz - 2, -1, -1):
        p_gap[k] = p_gap[k + 1] * (1.0 - occupancy[k])

    p_gap = np.clip(p_gap, 1e-6, 1.0)
    
    # 调试输出
    print(f"\n[调试] Gap Probability 计算:")
    print(f"  Occupancy 范围: [{occupancy.min():.6f}, {occupancy.max():.6f}]")
    print(f"  Occupancy 非零层数: {np.sum(occupancy > 0)}/{nz}")
    print(f"  Occupancy 平均值(非零): {occupancy[occupancy > 0].mean():.6f}" if np.any(occupancy > 0) else "  Occupancy 全为0!")
    print(f"  P_gap 范围: [{p_gap.min():.6f}, {p_gap.max():.6f}]")
    print(f"  P_gap 前5层: {p_gap[:5]}")
    print(f"  P_gap 后5层: {p_gap[-5:]}")
    
    return occupancy, p_gap


def compute_lad_beer_lambert(p_gap, vz, G=0.5):
    """Beer-Lambert定律: LAD = -1/G × d(ln P_gap)/dz
    
    k=0是底层，k=nz-1是顶层，z向上增加
    p_gap[k]是从上往下看到k层的透射率，从底到顶递增
    
    正确的物理意义：
    - LAD是叶面积密度（正值）
    - d(ln P_gap)/dz = (ln P[k+1] - ln P[k])/dz > 0 (向上递增)
    - 但Beer定律描述的是消光，所以加负号
    - 实际上应该用 -d(ln P)/dz，即 (ln P[k] - ln P[k+1])/dz
    """
    nz = len(p_gap)
    lad = np.zeros(nz)
    
    # 对每一层k，计算 LAD[k] = 1/G × (ln P[k+1] - ln P[k]) / vz
    # 注意：这里不用负号，因为我们要的是"消光系数"的绝对值
    for k in range(nz - 1):
        if p_gap[k] > 0 and p_gap[k+1] > 0:
            lad[k] = (1.0 / G) * (np.log(p_gap[k+1]) - np.log(p_gap[k])) / vz
    
    lad[lad < 0] = 0  # 确保非负
    lad[-1] = lad[-2] if nz > 1 else 0  # 填充顶层
    
    # 调试输出
    print(f"\n[调试] LAD 计算:")
    print(f"  log(P_gap) 范围: [{np.log(np.clip(p_gap, 1e-10, None)).min():.6f}, {np.log(p_gap).max():.6f}]")
    print(f"  LAD 范围: [{lad.min():.6f}, {lad.max():.6f}]")
    print(f"  LAD 非零层数: {np.sum(lad > 0)}/{nz}")
    print(f"  LAD 平均值(非零): {lad[lad > 0].mean():.6f}" if np.any(lad > 0) else "  LAD 全为0!")
    print(f"  LAD 前5层(底): {lad[:5]}")
    print(f"  LAD 后5层(顶): {lad[-5:]}")
    
    return lad


def plot_vertical_profiles(occupancy, p_gap, lad, grid_info, output_dir):
    vz, z_min, nz = grid_info['vz'], grid_info['z_min'], grid_info['nz']
    z_actual = z_min + (np.arange(nz) + 0.5) * vz

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(occupancy, z_actual, 'b-', linewidth=2)
    axes[0].fill_betweenx(z_actual, 0, occupancy, color='blue', alpha=0.3)
    axes[0].set_title('Occupancy Rate by Height'); axes[0].set_xlabel('Occupancy'); axes[0].set_ylabel('Height (m)')
    axes[1].plot(p_gap, z_actual, 'g-', linewidth=2)
    axes[1].fill_betweenx(z_actual, 0, p_gap, color='green', alpha=0.3)
    axes[1].set_title('Gap Probability'); axes[1].set_xlabel('P_gap'); axes[1].set_ylabel('Height (m)')
    axes[2].plot(lad, z_actual, 'r-', linewidth=2)
    axes[2].fill_betweenx(z_actual, 0, lad, color='red', alpha=0.3)
    axes[2].set_title('Leaf Area Density'); axes[2].set_xlabel('LAD (m²/m³)'); axes[2].set_ylabel('Height (m)')
    plt.tight_layout()
    out_path = output_dir / "vertical_profiles.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_path.name}")


def save_summary_yaml(output_path, info):
    yaml_data = {
        'leaf_density_summary': {
            'input_file': info['input_file'],
            'parameters': {
                'voxel_size_m': info['voxel_size'],
                'projection_function_G': info['G'],
                'ground_removal_percentile': info['ground_percentile'],
            },
            'results': {
                'LAI_m2_m2': info['lai'],
                'LAD_mean_m2_m3': info['lad_mean'],
                'LAD_max_m2_m3': info['lad_max'],
            }
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
    print(f"  已保存: {output_path.name}")


def main():
    if len(sys.argv) < 2:
        print("用法: python leaf_density_estimator.py <las文件或文件夹路径>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    vx = vy = vz = 0.005
    G = 0.5
    ground_percentile = 15.0

    if '--vx' in sys.argv:
        vx = float(sys.argv[sys.argv.index('--vx') + 1])
    if '--vy' in sys.argv:
        vy = float(sys.argv[sys.argv.index('--vy') + 1])
    if '--vz' in sys.argv:
        vz = float(sys.argv[sys.argv.index('--vz') + 1])
    if '--G' in sys.argv:
        G = float(sys.argv[sys.argv.index('--G') + 1])
    if '--ground' in sys.argv:
        ground_percentile = float(sys.argv[sys.argv.index('--ground') + 1])

    if not input_path.exists():
        print(f"错误: {input_path} 不存在")
        sys.exit(1)

    # 只处理名字包含'full'的.las文件
    if input_path.is_file():
        if 'full' not in input_path.stem.lower():
            print(f"错误: 文件名必须包含'full': {input_path.name}")
            sys.exit(1)
        las_files = [input_path]
    else:
        las_files = [f for f in input_path.glob("*.las") if 'full' in f.stem.lower()]
    
    if not las_files:
        print("未找到任何包含'full'的 .las 文件")
        sys.exit(1)
    
    print(f"找到 {len(las_files)} 个'full'文件:")
    for f in las_files:
        print(f"  - {f.name}")

    output_dir = (input_path.parent if input_path.is_file() else input_path) / "leaf_density_results"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for las_file in las_files:
        print(f"\n{'='*60}")
        print(f"==== 处理文件 {las_file.name} ====")
        print('='*60)
        
        # 读取原始点云
        las_original = laspy.read(las_file)
        x, y, z = read_las_file(las_file)
        
        # 使用自适应地面移除
        x_clean, y_clean, z_clean = remove_ground_adaptive(x, y, z, grid_size=1.0, bottom_percentile=ground_percentile)
        
        # 保存去除地面后的点云（不包含'full'）
        ground_removed_name = las_file.stem.replace('_full', '') + '_ground_removed.las'
        ground_removed_path = las_file.parent / ground_removed_name
        
        # 创建新的LAS文件
        header = laspy.LasHeader(point_format=las_original.header.point_format, version=las_original.header.version)
        header.offsets = las_original.header.offsets
        header.scales = las_original.header.scales
        
        las_clean = laspy.LasData(header)
        las_clean.x = x_clean
        las_clean.y = y_clean
        las_clean.z = z_clean
        
        las_clean.write(ground_removed_path)
        print(f"  已保存: {ground_removed_path.name} ({len(x_clean):,} 点)")
        
        # 继续处理LAD/LAI
        voxel_grid, grid_info = voxelize_point_cloud(x_clean, y_clean, z_clean, vx, vy, vz)
        occupancy, p_gap = compute_gap_probability(voxel_grid)
        lad = compute_lad_beer_lambert(p_gap, vz, G)
        lai_total = np.sum(lad) * vz
        
        # 调试输出
        print(f"\n[调试] LAI 计算:")
        print(f"  LAI = sum(LAD) × vz = {np.sum(lad):.6f} × {vz} = {lai_total:.6f}")

        file_out = output_dir / las_file.stem.replace('_full', '')
        file_out.mkdir(parents=True, exist_ok=True)
        plot_vertical_profiles(occupancy, p_gap, lad, grid_info, file_out)

        info = {
            'input_file': las_file.name,
            'voxel_size': {'vx': vx, 'vy': vy, 'vz': vz},
            'G': G,
            'ground_percentile': ground_percentile,
            'lai': float(lai_total),
            'lad_mean': float(np.mean(lad[lad > 0])),
            'lad_max': float(np.max(lad)),
        }
        save_summary_yaml(file_out / "leaf_summary.yaml", info)
        print(f"✓ {las_file.name} 完成 (LAI={lai_total:.3f} m²/m²)")

    print(f"\n输出目录: {output_dir.resolve()}")
    print("所有文件处理完成。")


if __name__ == "__main__":
    main()
