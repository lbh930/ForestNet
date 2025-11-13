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
import matplotlib.ticker as mticker
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

    # 全局字体
    plt.rcParams.update({'font.size': 32})

    # 1) Occupancy 图
    fig1, ax1 = plt.subplots(figsize=(12, 12))  # 方形画布
    ax1.plot(occupancy, z_actual, 'b-', linewidth=3)
    ax1.fill_betweenx(z_actual, 0, occupancy, color='blue', alpha=0.3)
    ax1.set_xlabel('Occupancy', fontsize=32)
    ax1.set_ylabel('Height (m)', fontsize=32)
    ax1.tick_params(axis='both', labelsize=28)
    # 减少X轴刻度密度并格式化为两位小数
    try:
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    except Exception:
        pass
    # 使坐标框架尽量正方（若Matplotlib版本支持）
    try:
        ax1.set_box_aspect(1)
    except Exception:
        pass
    plt.tight_layout()
    out1 = output_dir / "vertical_occupancy.png"
    fig1.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  已保存: {out1.name}")

    # 2) P_gap 图
    fig2, ax2 = plt.subplots(figsize=(12, 12))  # 方形画布
    ax2.plot(p_gap, z_actual, 'g-', linewidth=3)
    ax2.fill_betweenx(z_actual, 0, p_gap, color='green', alpha=0.3)
    ax2.set_xlabel('P_gap', fontsize=32)
    ax2.set_ylabel('Height (m)', fontsize=32)
    ax2.tick_params(axis='both', labelsize=28)
    try:
        ax2.set_box_aspect(1)
    except Exception:
        pass
    plt.tight_layout()
    out2 = output_dir / "vertical_pgap.png"
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  已保存: {out2.name}")

    # 3) LAD 图
    fig3, ax3 = plt.subplots(figsize=(12, 12))  # 方形画布
    ax3.plot(lad, z_actual, 'r-', linewidth=3)
    ax3.fill_betweenx(z_actual, 0, lad, color='red', alpha=0.3)
    ax3.set_xlabel('LAD (m²/m³)', fontsize=32)
    ax3.set_ylabel('Height (m)', fontsize=32)
    ax3.tick_params(axis='both', labelsize=28)
    try:
        ax3.set_box_aspect(1)
    except Exception:
        pass
    plt.tight_layout()
    out3 = output_dir / "vertical_lad.png"
    fig3.savefig(out3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"  已保存: {out3.name}")


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
    if len(sys.argv) < 2 or ('-h' in sys.argv) or ('--help' in sys.argv):
        print("用法: python leaf_density_estimator.py <las文件或文件夹路径> [选项]")
        print("选项:")
        print("  --vx <float>          体素尺寸X (默认 0.005 m)")
        print("  --vy <float>          体素尺寸Y (默认 0.005 m)")
        print("  --vz <float>          体素尺寸Z (默认 0.005 m)")
        print("  --G <float>           投影函数G (默认 0.5)")
        print("  --ground <percent>    自适应地面移除百分比 (默认 15)")
        print("  --visualize_voxel     使用Open3D渲染体素化可视化")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    vx = vy = vz = 0.005
    G = 0.5
    ground_percentile = 15.0
    visualize_voxel = ('--visualize_voxel' in sys.argv)

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

        # 可选：Open3D 体素可视化
        if visualize_voxel:
            try:
                import open3d as o3d
                print("\n使用 Open3D 渲染体素可视化窗口 (关闭窗口以继续)…")
                # 将几何居中到原点，避免UTM等大坐标导致初始视角很远
                cx = 0.5 * (x_clean.min() + x_clean.max())
                cy = 0.5 * (y_clean.min() + y_clean.max())
                cz = 0.5 * (z_clean.min() + z_clean.max())
                x_vis = x_clean - cx
                y_vis = y_clean - cy
                z_vis = z_clean - cz

                # 构建点云
                pcd = o3d.geometry.PointCloud()
                pts = np.vstack([x_vis, y_vis, z_vis]).T.astype(np.float64)
                pcd.points = o3d.utility.Vector3dVector(pts)

                # 可选着色：按相对高度着色，提升可读性
                z_norm = (z_vis - z_vis.min()) / max(1e-9, (z_vis.max() - z_vis.min()))
                colors = np.stack([0.2 + 0.8 * z_norm, 0.6 * (1 - z_norm), 1.0 - 0.5 * z_norm], axis=1)
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

                # 基于点云生成体素网格（统一体素大小使用 vx）
                voxel_size = float(vx)
                vgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

                # 将体素网格转为三角网格（采样一部分体素以保障性能），以获得真实的明暗着色
                vox_list = vgrid.get_voxels()
                n_vox = len(vox_list)
                max_vox_mesh = 150000  # 控制最大转网格体素数量，避免卡顿
                if n_vox == 0:
                    print("  [可视化] 体素为空，跳过渲染。")
                    continue
                if n_vox > max_vox_mesh:
                    sel_idx = np.random.choice(n_vox, max_vox_mesh, replace=False)
                    sel_voxels = [vox_list[i] for i in sel_idx]
                    print(f"  [可视化] 体素数 {n_vox:,}，随机采样 {max_vox_mesh:,} 个用于着色渲染…")
                else:
                    sel_voxels = vox_list
                    print(f"  [可视化] 体素数 {n_vox:,}，全部用于着色渲染…")

                mesh_vox = o3d.geometry.TriangleMesh()
                half = voxel_size * 0.5
                base_box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
                base_box.compute_vertex_normals()

                # 根据体素中心复制 box 并平移
                z_min_c = z_vis.min(); z_max_c = z_vis.max(); z_rng = max(1e-9, (z_max_c - z_min_c))
                for v in sel_voxels:
                    center = vgrid.get_voxel_center_coordinate(v.grid_index)
                    box_i = o3d.geometry.TriangleMesh(base_box)
                    box_i.translate(center - np.array([half, half, half]))
                    # 按高度着色（有光照的材质基色）
                    zn = (center[2] - z_min_c) / z_rng
                    color = np.array([0.2 + 0.8 * zn, 0.6 * (1 - zn), 1.0 - 0.5 * zn])
                    box_i.paint_uniform_color(color)
                    mesh_vox += box_i

                mesh_vox.compute_vertex_normals()

                # 坐标系
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(voxel_size * 5, 0.1))

                # 使用自定义可视化器以设置初始相机：看向原点并放大到合理比例
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=f"Voxelized View: {las_file.name}", width=1280, height=800)
                vis.add_geometry(mesh_vox)
                vis.add_geometry(axis)
                vis.update_renderer()

                # 相机控制：lookat到体素中心（接近原点），适度缩放
                vc = vis.get_view_control()
                try:
                    center = mesh_vox.get_axis_aligned_bounding_box().get_center()
                except Exception:
                    center = np.array([0.0, 0.0, 0.0])
                # 设置视角参数：保持默认front/up，只调整lookat与zoom，避免空白/过远
                vc.set_lookat(center.tolist())
                vc.set_zoom(0.7)

                # 渲染选项：开启光照，确保体素为明暗着色（shaded）
                ro = vis.get_render_option()
                try:
                    ro.light_on = True
                except Exception:
                    pass
                # 适度的背景颜色可增强对比
                try:
                    ro.background_color = np.array([1.0, 1.0, 1.0])
                except Exception:
                    pass

                # 进入交互
                vis.run()
                vis.destroy_window()
            except ImportError:
                print("警告: 未安装 open3d，跳过体素可视化。请在conda环境中安装 open3d 后重试。")
            except Exception as e:
                print(f"警告: Open3D 可视化失败，已跳过。错误: {e}")
        
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
