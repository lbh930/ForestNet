#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize Detection Results
可视化植物检测结果

用法:
    python visualize_result.py input.las results_dir/ output.png
"""

import sys
import yaml
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation


def read_las_file(las_path):
    """读取LAS文件"""
    print(f"读取点云: {las_path}")
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    print(f"  点数: {len(x):,}")
    return x, y, z


def create_chm(x, y, z, resolution=0.02):
    """创建冠层高度模型（CHM）"""
    print(f"生成CHM (分辨率: {resolution*100:.1f}cm)...")
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    
    print(f"  网格大小: {nx} x {ny}")
    
    # 创建CHM（最大高度）
    chm = np.full((ny, nx), np.nan)
    
    x_idx = ((x - x_min) / resolution).astype(int)
    y_idx = ((y - y_min) / resolution).astype(int)
    
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)
    
    for i in range(len(x)):
        if np.isnan(chm[y_idx[i], x_idx[i]]) or z[i] > chm[y_idx[i], x_idx[i]]:
            chm[y_idx[i], x_idx[i]] = z[i]
    
    # 填充空白
    mask = ~np.isnan(chm)
    if np.sum(mask) > 0:
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        chm_filled = chm[tuple(indices)]
        
        # 平滑
        chm_smoothed = gaussian_filter(chm_filled, sigma=2.5, mode='nearest')
        
        # 限制填充范围（避免过度外推）
        dilated_mask = binary_dilation(mask, iterations=5)
        chm_smoothed[~dilated_mask] = np.nan
    else:
        chm_smoothed = chm
    
    extent = [x_min, x_max, y_min, y_max]
    print(f"  CHM范围: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
    
    return chm_smoothed, extent


def find_all_detection_summaries(results_dir):
    """递归查找所有detection_summary.yaml文件"""
    results_dir = Path(results_dir)
    yaml_files = list(results_dir.rglob("detection_summary.yaml"))
    print(f"\n找到 {len(yaml_files)} 个检测结果文件")
    return yaml_files


def load_plant_positions(yaml_files):
    """从所有YAML文件中加载植物位置"""
    all_plants = []
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'detection_summary' in data and 'plants' in data['detection_summary']:
                plants = data['detection_summary']['plants']
                
                for plant in plants:
                    # 直接读取x和y坐标
                    x = plant.get('x_m')
                    y = plant.get('y_m')
                    height = plant.get('height_m', 0)
                    
                    if x is not None and y is not None:
                        all_plants.append({'x': x, 'y': y, 'height': height})
                
                print(f"  {yaml_file.parent.name}: {len(plants)} 株植物")
        
        except Exception as e:
            print(f"  警告: 无法读取 {yaml_file}: {e}")
    
    return all_plants


def visualize_chm_with_plants(chm, extent, plants, output_path):
    """可视化CHM并标记植物位置"""
    print(f"\n生成可视化...")
    
    # 计算点云的实际宽高比
    x_range = extent[1] - extent[0]  # x_max - x_min
    y_range = extent[3] - extent[2]  # y_max - y_min
    aspect_ratio = y_range / x_range
    
    # 根据比例调整figure大小（基准宽度16英寸）
    fig_width = 16
    fig_height = fig_width * aspect_ratio
    
    # 限制高度范围，避免过于极端的比例
    if fig_height > 20:
        fig_height = 20
        fig_width = fig_height / aspect_ratio
    elif fig_height < 8:
        fig_height = 8
        fig_width = fig_height / aspect_ratio
    
    print(f"  点云比例: {x_range:.2f}m x {y_range:.2f}m (宽高比 1:{aspect_ratio:.2f})")
    print(f"  图片尺寸: {fig_width:.1f} x {fig_height:.1f} 英寸")
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 显示CHM，设置aspect='equal'保持真实比例
    im = ax.imshow(chm, extent=extent, origin='lower', 
                   cmap='terrain', aspect='equal', 
                   interpolation='bilinear', alpha=0.9)
    
    # 标记植物位置
    if plants:
        x_plants = [p['x'] for p in plants]
        y_plants = [p['y'] for p in plants]
        
        ax.plot(x_plants, y_plants, 'rx', markersize=2, 
               markeredgewidth=0.25, alpha=0.8, label=f'Detected Plants (n={len(plants)})')
        print(f"  标记了 {len(plants)} 株植物")
    else:
        print(f"  警告: 没有植物坐标可以标记")
    
    ax.set_xlabel('X Coordinate (m)', fontsize=13)
    ax.set_ylabel('Y Coordinate (m)', fontsize=13)
    ax.set_title('Canopy Height Model with Detected Plants', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)', fontsize=12)
    
    if plants:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ 可视化已保存: {output_path}")


def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python visualize_result.py <las文件> <结果目录> <输出图片>")
        print("\n示例:")
        print("  python visualize_result.py corn.las row_split_output/corn/ output.png")
        sys.exit(1)
    
    las_path = Path(sys.argv[1])
    results_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    
    # 检查输入
    if not las_path.exists():
        print(f"错误: LAS文件不存在: {las_path}")
        sys.exit(1)
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        sys.exit(1)
    
    print("="*60)
    print("植物检测结果可视化")
    print("="*60)
    
    # 1. 读取点云
    x, y, z = read_las_file(las_path)
    
    # 2. 创建CHM
    chm, extent = create_chm(x, y, z, resolution=0.02)
    
    # 3. 加载所有检测结果
    yaml_files = find_all_detection_summaries(results_dir)
    
    if not yaml_files:
        print("\n警告: 未找到任何检测结果文件")
        print("将只生成CHM图")
    
    plants = load_plant_positions(yaml_files) if yaml_files else []
    
    print(f"\n总计: {len(plants)} 株植物")
    
    # 计算统计信息
    if plants:
        # 计算面积（从extent）
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        area_m2 = x_range * y_range
        
        # 计算密度
        density = len(plants) / area_m2
        
        # 计算平均高度
        heights = [p['height'] for p in plants]
        avg_height = np.mean(heights)
        std_height = np.std(heights)
        min_height = np.min(heights)
        max_height = np.max(heights)
        
        print(f"\n统计信息:")
        print(f"  田地面积: {area_m2:.2f} m²")
        print(f"  作物密度: {density:.2f} 株/m²")
        print(f"  平均高度: {avg_height:.3f} ± {std_height:.3f} m")
        print(f"  高度范围: [{min_height:.3f}, {max_height:.3f}] m")
    
    # 4. 可视化
    visualize_chm_with_plants(chm, extent, plants, output_path)
    
    print("="*60)


if __name__ == "__main__":
    main()
