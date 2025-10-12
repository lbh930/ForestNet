#!/usr/bin/env python3
# crop_row.py
# 读取 .las 文件，生成 CHM (Canopy Height Model) 并绘制出来
# 使用方法: python crop_row.py soybean.las

import argparse
import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import os

try:
    from skimage.filters import frangi
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.exposure import rescale_intensity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[警告] scikit-image未安装，行检测功能将被禁用")

def read_las_xyz(path):
    """读取LAS文件到Nx3浮点数组."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"LAS文件不存在: {path}")
    
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    print(f"[信息] 读取了 {len(pts)} 个点")
    return pts

def normalize_z_values(points):
    """将Z值标准化，使最小值为0."""
    z_min = points[:, 2].min()
    points[:, 2] -= z_min
    print(f"[信息] Z值标准化: 最小值 {z_min:.3f} -> 0.0")
    return points

def keep_top_height_points(points, top_percentage=8.0):
    """保留最高百分比的点."""
    original_count = len(points)
    
    if original_count == 0:
        return points
    
    # 计算高度阈值（保留最高的百分比）
    height_threshold = np.percentile(points[:, 2], 100 - top_percentage)
    
    # 保留高于阈值的点
    mask = points[:, 2] >= height_threshold
    filtered_points = points[mask]
    
    kept_count = len(filtered_points)
    kept_percentage = (kept_count / original_count) * 100
    
    print(f"[信息] 保留最高 {top_percentage}% 的点")
    print(f"[信息] 高度阈值: {height_threshold:.3f}m")
    print(f"[信息] 保留了 {kept_count} 个点 ({kept_percentage:.1f}%), 移除 {original_count - kept_count} 个点")
    
    return filtered_points

def remove_outliers(points, grid_res=0.05, min_neighbors=3, search_radius=0.2):
    """
    移除离群点（outliers）：基于邻域密度过滤稀疏点
    - grid_res: 网格分辨率，用于初步分组
    - min_neighbors: 搜索半径内最少邻居数量
    - search_radius: 搜索半径（米）
    """
    original_count = len(points)
    
    if original_count == 0:
        return points
    
    print(f"[信息] 开始离群点移除：搜索半径 {search_radius}m, 最少邻居 {min_neighbors} 个")
    
    # 构建KDTree（只使用XY坐标）
    tree = cKDTree(points[:, :2])
    
    # 查找每个点的邻居数量
    neighbor_counts = tree.query_ball_point(points[:, :2], r=search_radius, return_length=True)
    
    # 保留邻居数量足够的点（包括自己）
    mask = neighbor_counts >= (min_neighbors + 1)  # +1 因为包括自己
    filtered_points = points[mask]
    
    removed_count = original_count - len(filtered_points)
    removal_percentage = (removed_count / original_count) * 100
    
    print(f"[信息] 离群点移除完成：移除了 {removed_count} 个点 ({removal_percentage:.1f}%)")
    print(f"[信息] 保留了 {len(filtered_points)} 个点")
    
    return filtered_points

def make_grid_meta(points, res=0.01):
    """计算网格元数据."""
    xy = points[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    
    # 添加小的epsilon以确保最大点落在最后一个网格内
    eps = 1e-9
    nx = max(1, int(np.ceil((xmax - xmin) / res)))
    ny = max(1, int(np.ceil((ymax - ymin) / res)))
    
    meta = {
        "xmin": float(xmin),
        "xmax": float(xmin + nx*res + eps),
        "ymin": float(ymin),
        "ymax": float(ymin + ny*res + eps),
        "res": float(res),
        "nx": nx,
        "ny": ny
    }
    
    print(f"[信息] 网格: {nx} x {ny} 像素, 分辨率: {res}m/像素")
    print(f"[信息] 范围: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}]")
    
    return meta

def bin_points_maxz_count(points, meta):
    """将点分配到网格中，计算每个网格的最大Z值和点数."""
    x = points[:, 0]
    y = points[:, 1] 
    z = points[:, 2]
    
    res = meta["res"]
    xmin = meta["xmin"]
    ymin = meta["ymin"]
    nx, ny = meta["nx"], meta["ny"]

    # 计算网格索引
    ix = np.floor((x - xmin) / res).astype(np.int64)
    iy = np.floor((y - ymin) / res).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # 初始化网格
    count = np.zeros((ny, nx), dtype=np.float64)
    maxz = np.full((ny, nx), -np.inf, dtype=np.float64)

    # 累积计算
    np.add.at(count, (iy, ix), 1.0)
    np.maximum.at(maxz, (iy, ix), z)

    return maxz, count

def create_chm(maxz, count, smooth_sigma=1.0):
    """创建冠层高度模型 (CHM)."""
    # 创建原始CHM (无数据的地方设为NaN)
    chm_raw = np.full_like(maxz, np.nan, dtype=np.float64)
    mask = count > 0
    chm_raw[mask] = maxz[mask]
    
    valid_cells = mask.sum()
    total_cells = mask.size
    coverage = valid_cells / total_cells * 100
    
    print(f"[信息] CHM覆盖率: {valid_cells}/{total_cells} ({coverage:.1f}%)")
    
    # 平滑CHM
    if smooth_sigma > 0:
        # 使用标准化卷积进行平滑
        mask_float = mask.astype(np.float64)
        vals = np.where(mask, maxz, 0.0)
        
        gv = gaussian_filter(vals, sigma=smooth_sigma, mode="nearest")
        gm = gaussian_filter(mask_float, sigma=smooth_sigma, mode="nearest")
        
        chm_smooth = np.full_like(gv, np.nan, dtype=np.float64)
        valid = gm > 1e-9
        chm_smooth[valid] = gv[valid] / gm[valid]
        
        print(f"[信息] 应用高斯平滑 (σ={smooth_sigma})")
        return chm_smooth, mask
    else:
        return chm_raw, mask

def detect_crop_rows_fast(chm, mask, sigma_range=(1, 2), beta=1.0, gamma=50):
    """
    使用Frangi血管滤波快速检测作物行
    - sigma_range: 滤波器尺度范围（像素）
    - beta: 线性度参数
    - gamma: 噪声抑制参数
    """
    if not SKIMAGE_AVAILABLE:
        print("[错误] 需要安装scikit-image才能使用行检测功能")
        return None, None
    
    print(f"[信息] 开始Frangi行检测: sigma={sigma_range}, beta={beta}, gamma={gamma}")
    
    # 预处理CHM
    chm_work = np.nan_to_num(chm, nan=0.0)
    
    # 只处理有效区域
    chm_work = chm_work * mask.astype(np.float64)
    
    # 归一化到[0,1]范围以提高性能
    if np.max(chm_work) > 0:
        chm_work = rescale_intensity(chm_work, in_range='image', out_range=(0, 1))
    
    # 应用Frangi滤波器（快速版本，使用较少的sigma值）
    sigma_min, sigma_max = sigma_range
    frangi_result = frangi(chm_work, 
                          sigmas=np.linspace(sigma_min, sigma_max, 3),  # 只用3个尺度以提高速度
                          beta=beta, 
                          gamma=gamma)
    
    # 归一化结果
    if np.max(frangi_result) > 0:
        frangi_result = rescale_intensity(frangi_result, in_range='image', out_range=(0, 1))
    
    # 快速阈值化和骨架提取
    threshold = np.percentile(frangi_result[frangi_result > 0], 85) if np.any(frangi_result > 0) else 0.5
    binary_rows = frangi_result > threshold
    
    # 移除小对象（快速清理）
    binary_rows = remove_small_objects(binary_rows, min_size=50)
    
    # 骨架化得到行中心线
    skeleton = skeletonize(binary_rows)
    
    print(f"[信息] 行检测完成，阈值: {threshold:.3f}")
    
    return frangi_result, skeleton

def plot_chm(chm, meta, mask, detect_rows=False, output_path=None, title="Canopy Height Model"):
    """绘制CHM."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 计算显示范围
    xmin, xmax = meta["xmin"], meta["xmax"]
    ymin, ymax = meta["ymin"], meta["ymax"]
    extent = [xmin, xmax, ymin, ymax]
    
    # CHM可视化
    valid_heights = chm[mask & np.isfinite(chm)]
    if len(valid_heights) > 0:
        vmin, vmax = np.percentile(valid_heights, [2, 98])
        mean_height = np.mean(valid_heights)
        max_height = np.max(valid_heights)
    else:
        vmin, vmax = 0, 1
        mean_height = 0
        max_height = 0
    
    im = ax.imshow(chm, extent=extent, origin='lower', 
                   cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'{title}\n高度范围: {vmin:.2f} - {vmax:.2f} m | 平均: {mean_height:.2f} m | 最大: {max_height:.2f} m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='高度 (m)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    # 行检测叠加
    if detect_rows and SKIMAGE_AVAILABLE:
        frangi_result, skeleton = detect_crop_rows_fast(chm, mask)
        if skeleton is not None:
            # 叠加骨架线
            skeleton_y, skeleton_x = np.nonzero(skeleton)
            if len(skeleton_x) > 0:
                # 转换像素坐标到世界坐标
                res = meta["res"]
                world_x = meta["xmin"] + (skeleton_x + 0.5) * res
                world_y = meta["ymin"] + (skeleton_y + 0.5) * res
                ax.scatter(world_x, world_y, c='red', s=0.5, alpha=0.8, marker='.')
                print(f"[信息] 叠加了 {len(skeleton_x)} 个行检测点")
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"[信息] CHM图像已保存到: {output_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)

def print_chm_stats(chm, mask):
    """打印CHM统计信息."""
    valid_heights = chm[mask & np.isfinite(chm)]
    
    if len(valid_heights) == 0:
        print("[警告] 没有有效的高度数据")
        return
    
    print("\n=== CHM 统计信息 ===")
    print(f"有效像素数量: {len(valid_heights)}")
    print(f"最小高度: {np.min(valid_heights):.3f} m")
    print(f"最大高度: {np.max(valid_heights):.3f} m")
    print(f"平均高度: {np.mean(valid_heights):.3f} m")
    print(f"中位数高度: {np.median(valid_heights):.3f} m")
    print(f"标准差: {np.std(valid_heights):.3f} m")
    print(f"第95百分位: {np.percentile(valid_heights, 95):.3f} m")

def main():
    parser = argparse.ArgumentParser(description='从LAS文件生成并显示冠层高度模型(CHM)')
    parser.add_argument('las_file', help='输入的LAS文件路径')
    parser.add_argument('--res', type=float, default=0.01, 
                       help='网格分辨率 (米/像素), 默认: 0.01')
    parser.add_argument('--smooth', type=float, default=1.0, 
                       help='高斯平滑参数 (像素), 默认: 1.0, 设为0禁用平滑')
    parser.add_argument('--top_percentage', type=float, default=8.0,
                       help='保留最高百分比的点, 默认: 8.0, 设为0禁用过滤')
    parser.add_argument('--remove_outliers', action='store_true', default=False,
                       help='启用离群点移除')
    parser.add_argument('--outlier_radius', type=float, default=0.2,
                       help='离群点检测搜索半径 (米), 默认: 0.2')
    parser.add_argument('--min_neighbors', type=int, default=10,
                       help='离群点检测最少邻居数, 默认: 3')
    parser.add_argument('--detect_rows', action='store_true', default=True,
                       help='启用作物行检测并叠加到图像上')
    parser.add_argument('--output', type=str, 
                       help='输出图像文件路径 (可选)')
    parser.add_argument('--title', type=str, default="Canopy Height Model",
                       help='图像标题')
    
    args = parser.parse_args()
    
    try:
        print(f"[信息] 正在处理文件: {args.las_file}")
        
        # 1. 读取LAS文件
        points = read_las_xyz(args.las_file)
        
        # 2. 标准化Z值
        points = normalize_z_values(points)
        
        # 3. 移除离群点 (可选)
        if args.remove_outliers:
            points = remove_outliers(points, 
                                   search_radius=args.outlier_radius,
                                   min_neighbors=args.min_neighbors)
        
        # 4. 保留最高百分比的点
        if args.top_percentage > 0:
            points = keep_top_height_points(points, args.top_percentage)
        
        # 5. 创建网格元数据
        meta = make_grid_meta(points, res=args.res)
        
        # 6. 将点分配到网格
        maxz, count = bin_points_maxz_count(points, meta)
        
        # 7. 创建CHM
        chm, mask = create_chm(maxz, count, smooth_sigma=args.smooth)
        
        # 8. 打印统计信息
        print_chm_stats(chm, mask)
        
        # 9. 绘制并保存CHM
        output_path = args.output
        if output_path is None:
            # 自动生成输出文件名
            base_name = os.path.splitext(os.path.basename(args.las_file))[0]
            output_path = f"{base_name}_chm.png"
        
        plot_chm(chm, meta, mask, detect_rows=args.detect_rows, output_path=output_path, title=args.title)
        
        print("\n[信息] 处理完成!")
        
    except Exception as e:
        print(f"[错误] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()