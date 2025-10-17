#!/usr/bin/env python3
# crop_row.py
# 读取 .las 文件，生成 CHM (Canopy Height Model) 并绘制出来
# 使用方法: python crop_row.py soybean.las

import argparse
import sys
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
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

def make_grid_meta(points, res=0.02):
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

def find_chm_peaks(chm, mask, meta, percentile=None, min_peak_distance=2, smooth_sigma=1):
    """
    从CHM中找到峰值点，使用堆和最小间距筛选
    - percentile: 高度百分位阈值，只保留高于此百分位的点作为峰值
    - min_peak_distance: 峰值点之间的最小距离（米）
    - 4: 用于找峰值的高斯平滑核标准差（默认4像素）
    """
    import heapq
    
    print(f"[信息] 开始查找CHM峰值点，阈值: {percentile}百分位, 最小间距: {min_peak_distance}m, 平滑核: {smooth_sigma}")
    
    # 对CHM进行平滑处理用于找峰值
    chm_smoothed = chm.copy()
    chm_smoothed[~mask] = 0  # 无效区域设为0
    chm_smoothed = gaussian_filter(chm_smoothed, sigma=smooth_sigma)
    chm_smoothed[~mask] = np.nan  # 恢复无效区域为nan
    
    # 先在原始CHM上计算高度阈值（这样更准确）
    valid_heights_original = chm[mask & np.isfinite(chm)]
    if len(valid_heights_original) == 0:
        print("[警告] 没有有效的高度数据")
        return np.array([]), np.array([])
    
    # 如果percentile为None或0，则不使用高度阈值限制
    if percentile is None or percentile <= 0:
        height_threshold = -np.inf  # 不设阈值，所有有效点都可以考虑
        print(f"[信息] 不使用高度阈值限制，考虑所有有效区域的局部最大值")
    else:
        # 使用原始CHM的百分位作为阈值
        height_threshold = np.percentile(valid_heights_original, percentile)
        print(f"[信息] 高度阈值: {height_threshold:.3f}m (原始CHM的{percentile}百分位)")
    
    # 使用局部最大值检测找峰值（在平滑后的CHM上）
    # 先准备用于最大值滤波的数据：无效区域设为-inf，这样不会被选为最大值
    chm_for_maxfilter = chm_smoothed.copy()
    chm_for_maxfilter[~mask | ~np.isfinite(chm_smoothed)] = -np.inf
    
    # 计算局部最大值（使用3x3窗口）
    local_max = maximum_filter(chm_for_maxfilter, size=3)
    
    # 峰值条件：1) 是局部最大值 2) 在有效区域 3) 高于阈值（如果设置了）
    peak_mask = (chm_for_maxfilter == local_max) & mask & np.isfinite(chm_smoothed) & (chm_smoothed >= height_threshold) & (local_max > -np.inf)
    candidate_y, candidate_x = np.nonzero(peak_mask)
    candidate_heights = chm_smoothed[candidate_y, candidate_x]
    
    if height_threshold == -np.inf:
        print(f"[信息] 初步找到 {len(candidate_x)} 个候选峰值点 (无高度限制)")
    else:
        print(f"[信息] 初步找到 {len(candidate_x)} 个候选峰值点 (阈值: {height_threshold:.3f}m)")
    
    if len(candidate_x) == 0:
        return np.array([]), np.array([])
    
    # 转换为世界坐标
    res = meta["res"]
    candidate_world_x = meta["xmin"] + (candidate_x + 0.5) * res
    candidate_world_y = meta["ymin"] + (candidate_y + 0.5) * res
    
    # 创建最大堆（使用负高度实现）
    # 堆元素格式: (-height, idx)
    heap = [(-candidate_heights[i], i) for i in range(len(candidate_x))]
    heapq.heapify(heap)
    
    # 选中的峰值点
    selected_indices = []
    selected_positions = []  # 存储世界坐标
    
    min_dist_sq = min_peak_distance ** 2  # 平方距离，避免重复计算平方根
    
    # 贪心选择峰值点
    while heap:
        neg_height, idx = heapq.heappop(heap)
        height = -neg_height
        
        pos_x = candidate_world_x[idx]
        pos_y = candidate_world_y[idx]
        
        # 检查与已选峰值点的距离
        too_close = False
        for sel_x, sel_y in selected_positions:
            dist_sq = (pos_x - sel_x)**2 + (pos_y - sel_y)**2
            if dist_sq < min_dist_sq:
                too_close = True
                break
        
        if not too_close:
            selected_indices.append(idx)
            selected_positions.append((pos_x, pos_y))
    
    # 提取选中的峰值点
    if len(selected_indices) > 0:
        selected_indices = np.array(selected_indices)
        peak_x = candidate_x[selected_indices]
        peak_y = candidate_y[selected_indices]
    else:
        peak_x = np.array([])
        peak_y = np.array([])
    
    print(f"[信息] 经过最小间距筛选后，保留 {len(peak_x)} 个峰值点")
    
    return peak_y, peak_x

def check_line_validity(p1, p2, chm, mask, meta, min_valid_ratio=0.3):
    """
    检查两点之间的线段是否有足够的有效CHM覆盖
    - p1, p2: 像素坐标 (y, x)
    - min_valid_ratio: 线段上至少需要多少比例的有效像素（默认30%）
    - 返回: True如果线段有足够的有效覆盖
    """
    y1, x1 = p1
    y2, x2 = p2
    
    # 使用Bresenham算法获取线段上的所有像素
    num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.linspace(x1, x2, num_points).astype(int)
    ys = np.linspace(y1, y2, num_points).astype(int)
    
    # 检查边界
    ny, nx = chm.shape
    
    # 统计有效点的数量
    valid_count = 0
    total_count = 0
    
    for x, y in zip(xs, ys):
        # 跳过超出边界的点
        if x < 0 or x >= nx or y < 0 or y >= ny:
            continue
        
        total_count += 1
        
        # 检查是否在有效区域
        if mask[y, x] and np.isfinite(chm[y, x]):
            valid_count += 1
    
    # 如果没有有效点，返回False
    if total_count == 0:
        return False
    
    # 计算有效比例
    valid_ratio = valid_count / total_count
    
    # 要求至少有min_valid_ratio的像素是有效的
    return valid_ratio >= min_valid_ratio

def line_to_line_distance(line1, line2):
    """
    计算两条线段之间的最小距离
    - line1, line2: ((y1,x1), (y2,x2)) 格式
    """
    p1, p2 = line1
    p3, p4 = line2
    
    # 计算所有点对之间的距离，取最小值
    distances = []
    for p_a in [p1, p2]:
        for p_b in [p3, p4]:
            dist = np.sqrt((p_a[0] - p_b[0])**2 + (p_a[1] - p_b[1])**2)
            distances.append(dist)
    
    return min(distances)

def detect_crop_rows_ransac(chm, mask, meta, 
                            min_dist_peak=3.0, 
                            num_lines=10, 
                            min_line_spacing=0.5,
                            peak_percentile=1,
                            peak_smooth_sigma=1,
                            max_attempts=10000):
    """
    基于峰值点和RANSAC的作物行检测
    - min_dist_peak: 两个峰值点之间的最小距离(米)
    - num_lines: 要检测的线段数量
    - min_line_spacing: 线段之间的最小间距(米)
    - peak_percentile: 峰值点高度百分位阈值
    - peak_smooth_sigma: 找峰值时的平滑核标准差
    - max_attempts: 最大尝试次数
    """
    print(f"[信息] 开始基于峰值的行检测")
    print(f"[信息] 参数: min_dist_peak={min_dist_peak}m, num_lines={num_lines}, min_spacing={min_line_spacing}m")
    
    # 1. 找到峰值点（使用堆和最小间距筛选，在平滑后的CHM上找）
    peak_y, peak_x = find_chm_peaks(chm, mask, meta, percentile=peak_percentile, 
                                     min_peak_distance=2, smooth_sigma=peak_smooth_sigma)
    
    if len(peak_x) < 2:
        print("[警告] 峰值点数量不足")
        return []
    
    # 转换为世界坐标
    res = meta["res"]
    peak_world_x = meta["xmin"] + (peak_x + 0.5) * res
    peak_world_y = meta["ymin"] + (peak_y + 0.5) * res
    
    # 计算最小像素距离
    min_pixel_dist = int(min_dist_peak / res)
    min_spacing_pixels = int(min_line_spacing / res)
    
    valid_lines = []  # 存储有效的线段 [(p1, p2), ...]
    
    print(f"[信息] 开始随机采样线段...")
    
    for attempt in range(max_attempts):
        if len(valid_lines) >= num_lines:
            print(f"[信息] 达到目标线段数量: {num_lines}")
            break
        
        # 2. 随机选择两个峰值点
        if len(peak_x) < 2:
            print("[警告] 峰值点数量不足")
            break
            
        idx1, idx2 = np.random.choice(len(peak_x), 2, replace=False)
        p1 = (peak_y[idx1], peak_x[idx1])
        p2 = (peak_y[idx2], peak_x[idx2])
        
        # 检查距离是否满足要求
        pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if pixel_dist < min_pixel_dist or pixel_dist > min_pixel_dist*3:
            continue
        
        # 3. 检查线段是否有足够的有效覆盖（至少30%的像素有效）
        if not check_line_validity(p1, p2, chm, mask, meta, min_valid_ratio=0.3):
            continue
        
        # 4. 检查与已有线段的间距
        line_valid = True
        new_line = (p1, p2)
        for existing_line in valid_lines:
            if line_to_line_distance(new_line, existing_line) < min_spacing_pixels:
                line_valid = False
                break
        
        if line_valid:
            valid_lines.append(new_line)
            print(f"[信息] 找到第 {len(valid_lines)} 条有效线段")
    
    print(f"[信息] 共找到 {len(valid_lines)} 条有效行线段")
    
    # 返回线段和峰值点
    return valid_lines, (peak_y, peak_x)

def project_points_to_lines(original_points, row_lines, meta, distance_threshold=0.1):
    """
    将原始点云投影到行中线的垂直平面上
    - original_points: 原始点云(预处理前)
    - row_lines: 检测到的行线段列表
    - distance_threshold: 距离中线的阈值(米)，默认10cm
    """
    if len(row_lines) == 0:
        print("[警告] 没有检测到行线段")
        return []
    
    print(f"[信息] 开始投影点云到 {len(row_lines)} 条行中线")
    
    res = meta["res"]
    projections = []  # 存储每条线的投影点
    
    for i, line in enumerate(row_lines):
        p1, p2 = line
        
        # 转换像素坐标到世界坐标
        x1 = meta["xmin"] + (p1[1] + 0.5) * res
        y1 = meta["ymin"] + (p1[0] + 0.5) * res
        x2 = meta["xmin"] + (p2[1] + 0.5) * res
        y2 = meta["ymin"] + (p2[0] + 0.5) * res
        
        # 计算线的方向向量
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx**2 + dy**2)
        
        if line_length < 1e-6:
            continue
        
        # 单位方向向量
        ux = dx / line_length
        uy = dy / line_length
        
        # 对每个原始点计算到线的距离和投影
        px = original_points[:, 0]
        py = original_points[:, 1]
        
        # 点到线段起点的向量
        vx = px - x1
        vy = py - y1
        
        # 投影长度
        proj_length = vx * ux + vy * uy
        
        # 投影点坐标
        proj_x = x1 + proj_length * ux
        proj_y = y1 + proj_length * uy
        
        # 计算点到投影点的距离
        dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        # 筛选在阈值范围内的点
        valid_mask = dist <= distance_threshold
        
        if valid_mask.sum() > 0:
            # 保存投影点坐标
            line_projections = {
                'line_id': i,
                'line_start': (x1, y1),
                'line_end': (x2, y2),
                'proj_x': proj_x[valid_mask],
                'proj_y': proj_y[valid_mask],
                'proj_length': proj_length[valid_mask],  # 沿线的位置
                'num_points': valid_mask.sum()
            }
            projections.append(line_projections)
            print(f"[信息] 行 {i+1}: {valid_mask.sum()} 个点在{distance_threshold}m范围内")
    
    return projections

def plot_projections(projections, output_path):
    """
    绘制点云在行中线垂直平面上的投影
    """
    if len(projections) == 0:
        print("[警告] 没有投影数据可绘制")
        return
    
    print(f"[信息] 绘制 {len(projections)} 条行的投影")
    
    # 创建多个子图，每行一个
    n_rows = len(projections)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 3*n_rows))
    
    # 如果只有一条行，确保axes是数组
    if n_rows == 1:
        axes = [axes]
    
    for i, proj in enumerate(projections):
        ax = axes[i]
        
        # 沿线的位置 vs 垂直于线的位置（这里为0）
        proj_length = proj['proj_length']
        
        # 绘制投影点的分布
        ax.scatter(proj_length, np.zeros_like(proj_length), 
                  s=0.1, alpha=0.3, c='blue')
        
        ax.set_title(f"行 {proj['line_id']+1}: {proj['num_points']} 个投影点")
        ax.set_xlabel('沿行方向位置 (m)')
        ax.set_ylabel('垂直于行方向')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, 0.5])
        
        # 添加密度统计
        if len(proj_length) > 0:
            length_range = proj_length.max() - proj_length.min()
            if length_range > 0:
                density = len(proj_length) / length_range
                ax.text(0.02, 0.98, f'密度: {density:.1f} 点/米', 
                       transform=ax.transAxes, va='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[信息] 投影图已保存到: {output_path}")

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
    if detect_rows:
        result = detect_crop_rows_ransac(chm, mask, meta)
        row_lines, (peak_y, peak_x) = result
        
        # 绘制峰值点
        if len(peak_x) > 0:
            res = meta["res"]
            peak_world_x = meta["xmin"] + (peak_x + 0.5) * res
            peak_world_y = meta["ymin"] + (peak_y + 0.5) * res
            ax.scatter(peak_world_x, peak_world_y, c='cyan', s=10, alpha=0.6, 
                      marker='o', edgecolors='blue', linewidths=0.5, label=f'峰值点 ({len(peak_x)})')
            print(f"[信息] 可视化了 {len(peak_x)} 个峰值点")
        
        # 绘制检测到的行线段
        if len(row_lines) > 0:
            res = meta["res"]
            for line in row_lines:
                p1, p2 = line
                # 转换像素坐标到世界坐标
                x1 = meta["xmin"] + (p1[1] + 0.5) * res
                y1 = meta["ymin"] + (p1[0] + 0.5) * res
                x2 = meta["xmin"] + (p2[1] + 0.5) * res
                y2 = meta["ymin"] + (p2[0] + 0.5) * res
                ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.8, label='行线段' if line == row_lines[0] else '')
            print(f"[信息] 叠加了 {len(row_lines)} 条行线段")
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"[信息] CHM图像已保存到: {output_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    # 返回检测结果
    if detect_rows:
        return row_lines
    return None

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
    parser.add_argument('--res', type=float, default=0.02, 
                       help='网格分辨率 (米/像素), 默认: 0.02')
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
        original_points = points.copy()  # 保存原始点云用于后续投影
        
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
        
        row_lines = plot_chm(chm, meta, mask, detect_rows=args.detect_rows, output_path=output_path, title=args.title)
        
        # 10. 如果检测到行，进行点云投影
        if row_lines is not None and len(row_lines) > 0:
            projections = project_points_to_lines(original_points, row_lines, meta)
            
            # 绘制投影图
            if len(projections) > 0:
                plot_projections(projections, output_path.replace('_chm.png', '_projections.png'))
        
        print("\n[信息] 处理完成!")
        
    except Exception as e:
        print(f"[错误] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()