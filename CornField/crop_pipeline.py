#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crop Pipeline
=============
完整的作物密度计算流程：
1. 对输入点云 (.las) 进行作物行检测与分割；
2. 对每个 tile 的每一行，使用 density-based 或 height-based 方法计数；
3. 汇总所有 tile 的结果，计算平均作物密度（株/m²）。

用法:
    python crop_pipeline.py input.las --method density
    python crop_pipeline.py input.las --method height
"""

import sys
import yaml
import numpy as np
from pathlib import Path
import importlib.util
import laspy
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ------------------------------
# 工具函数
# ------------------------------

def load_module_from_path(path, name):
    """动态加载外部Python脚本"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(config_path="config.yaml"):
    """读取配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def read_las_points(las_path):
    """读取LAS点云为numpy数组"""
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    return np.column_stack([x, y, z])


# ------------------------------
# 并行处理函数
# ------------------------------

def process_tile_parallel(args):
    """处理单个 tile 的并行函数"""
    import re
    (tile_id, tile_data, tile_info_single, params, output_base,
     splitter_path, density_counter_path, height_counter_path) = args
    
    # 从params中获取method
    method = params['method']
    
    # 在子进程中重新加载模块
    splitter = load_module_from_path(splitter_path, "crop_row_splitter")
    density_counter = load_module_from_path(density_counter_path, "density_based_counter")
    height_counter = load_module_from_path(height_counter_path, "height_based_counter")
    
    tile_x, tile_y, tile_z = tile_data
    
    # 处理单个 tile（传递row detection参数）
    stats = splitter.process_single_tile(
        tile_id, tile_x, tile_y, tile_z, 
        tile_info_single, output_base,
        row_detection_smooth_sigma=params.get("row_detection_smooth_sigma", 0.05),
        row_detection_prominence=params.get("row_detection_prominence", 0.02)
    )
    if stats['row_count'] == 0:
        return (tile_id, 0, [])

    tile_dir = output_base / f"tile_{tile_id[0]}_{tile_id[1]}"
    rows = sorted(list(tile_dir.glob("row_*.las")))

    tile_plant_count = 0
    tile_heights = []

    for idx, row_path in enumerate(rows, 1):
        row_name = row_path.stem
        points = read_las_points(row_path)
        if len(points) < 50:
            continue

        # 为每个row建立独立输出目录
        row_output_dir = tile_dir / "count_results" / row_name
        row_output_dir.mkdir(parents=True, exist_ok=True)

        # 计数方向应与行方向垂直
        count_direction = 'y' if stats['direction'] == 'x' else 'x'
        
        # 从文件名中提取行中心坐标
        match = re.search(r'at_([-\d.]+)m', row_name)
        row_center = float(match.group(1)) if match else None

        # 行状态文案
        row_status = f"tile: {tile_id} row ({idx}/{len(rows)}) completed"

        if method == 'density':
            plant_count, _, results = density_counter.density_count_from_row(
                points,
                direction=count_direction,
                expected_spacing=params["expected_spacing"],
                bin_size=params["bin_size"],
                apply_sor=params["apply_sor"],
                sor_k=params["sor_k"],
                sor_std_ratio=params["sor_std_ratio"],
                remove_ground=params["remove_ground"],
                ground_percentile=params["ground_percentile"],
                top_percentile=params["top_percentile"],
                min_prominence=params["min_prominence"],
                smooth_sigma=params.get("smooth_sigma", 1.0),
                output_dir=row_output_dir,
                row_center=row_center,
                row_status=row_status,
                verbose=False
            )
        else:
            plant_count, _, results = height_counter.height_count_from_row(
                points,
                direction=count_direction,
                expected_spacing=params["expected_spacing"],
                bin_size=params["bin_size"],
                apply_sor=params["apply_sor"],
                sor_k=params["sor_k"],
                sor_std_ratio=params["sor_std_ratio"],
                remove_ground=params["remove_ground"],
                ground_percentile=params["ground_percentile"],
                top_percentile=params["top_percentile"],
                min_prominence=params["min_prominence"],
                height_metric=params["height_metric"],
                kernel_length=params.get("kernel_length"),
                kernel_width=params.get("kernel_width"),
                kernel_height_percentile=params.get("kernel_height_percentile"),
                output_dir=row_output_dir,
                row_center=row_center,
                row_status=row_status,
                verbose=False
            )

        # 累加计数
        tile_plant_count += plant_count
        
        # 收集高度信息
        if plant_count > 0 and results:
            heights = results.get('actual_heights')
            if (heights is None or len(heights) == 0):
                heights = results.get('peak_densities') if method == 'density' else results.get('peak_heights')
            if heights is not None and len(heights) > 0:
                tile_heights.extend(heights)

    print(f"[Tile {tile_id}] 检测完成，共检测到 {tile_plant_count} 株植物。")
    return (tile_id, tile_plant_count, tile_heights)


# ------------------------------
# 主流程
# ------------------------------

def main():
    if len(sys.argv) < 3:
        print("用法: python crop_pipeline.py <las文件路径> --crop <corn|soybean>")
        sys.exit(1)
    
    las_path = Path(sys.argv[1])
    if not las_path.exists():
        print(f"错误: 文件不存在: {las_path}")
        sys.exit(1)
    
    # 获取作物类型
    if '--crop' in sys.argv:
        idx = sys.argv.index('--crop')
        if idx + 1 < len(sys.argv):
            crop_type = sys.argv[idx + 1].lower()
            if crop_type not in ['corn', 'soybean']:
                print("错误: crop 必须是 'corn' 或 'soybean'")
                sys.exit(1)
        else:
            print("错误: 未指定crop参数")
            sys.exit(1)
    else:
        print("错误: 需要 --crop 参数")
        sys.exit(1)
    
    # 加载配置
    config = load_config()
    params = config[crop_type]
    method = params['method']  # 从作物配置中获取方法
    
    print("="*60)
    print(f"Crop Pipeline 启动")
    print(f"作物类型: {crop_type.upper()}")
    print(f"检测方法: {method}-based counting")
    print("="*60)

    t0 = time.time()

    # 创建以输入文件名命名的输出目录
    las_name = las_path.stem  # 获取不带扩展名的文件名
    output_base = Path("row_split_output") / las_name
    
    # 如果目录已存在，清除其内容
    if output_base.exists():
        print(f"\n[清理] 删除已存在的输出目录: {output_base}")
        shutil.rmtree(output_base)
    
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"[输出] 结果将保存到: {output_base}")

    # 加载外部脚本
    splitter_path = Path("crop_row_splitter.py")
    density_counter_path = Path("density_based_counter.py")
    height_counter_path = Path("height_based_counter.py")
    
    splitter = load_module_from_path(splitter_path, "crop_row_splitter")
    density_counter = load_module_from_path(density_counter_path, "density_based_counter")
    height_counter = load_module_from_path(height_counter_path, "height_based_counter")

    # Step 1: 作物行分割
    print("\n[1] 执行作物行分割...")
    print(f"  Row detection参数: smooth_sigma={params.get('row_detection_smooth_sigma', 0.05):.3f}m, "
          f"prominence={params.get('row_detection_prominence', 0.02):.3f}")
    x, y, z = splitter.read_las_file(las_path)
    x, y, z = splitter.normalize_z_to_zero(x, y, z)
    tiles, tile_info = splitter.split_into_tiles(x, y, z, tile_size=params["tile_size"])

    # Step 2: 并行处理所有 tiles
    # 从配置中获取进程数
    max_workers = config.get('parallel', {}).get('max_workers')
    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count()
    else:
        max_workers = min(max_workers, os.cpu_count())  # 不超过实际核心数
    
    print(f"\n[并行] 开始处理 {len(tiles)} 个 tiles (使用 {max_workers} 个进程)...")
    tasks = [
        (tile_id, (tile_x, tile_y, tile_z), tile_info[tile_id], params,
         output_base, splitter_path, density_counter_path, height_counter_path)
        for tile_id, (tile_x, tile_y, tile_z) in tiles.items()
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tile_parallel, t): t[0] for t in tasks}
        for future in as_completed(futures):
            try:
                tile_id, tile_plants, heights = future.result()
                results.append((tile_id, tile_plants, heights))
            except Exception as e:
                print(f"[错误] Tile {futures[future]} 执行失败: {e}")

    # Step 3: 汇总结果
    total_plants = sum(r[1] for r in results)
    total_area = len(results) * (params["tile_size"] ** 2)
    all_heights = [h for r in results for h in r[2]]
    
    # 用于记录每个 tile 的汇总
    tile_summaries = [(r[0], r[1]) for r in results]

    # Step 3: 汇总密度
    avg_density = total_plants / total_area if total_area > 0 else 0.0
    
    # 计算平均高度
    if all_heights:
        avg_height = np.mean(all_heights)
        std_height = np.std(all_heights)
        min_height = np.min(all_heights)
        max_height = np.max(all_heights)
    else:
        avg_height = std_height = min_height = max_height = 0.0

    print("\n" + "="*60)
    print("处理完成 ✅")
    print(f"总检测作物数: {total_plants}")
    print(f"总面积: {total_area:.2f} m²")
    print(f"平均作物密度: {avg_density:.3f} 株/m²")
    if all_heights:
        print(f"平均高度: {avg_height:.3f} ± {std_height:.3f} m")
        print(f"高度范围: [{min_height:.3f}, {max_height:.3f}] m")
    print(f"总耗时: {time.time() - t0:.2f}s")
    print("="*60)

    # 输出summary文件
    summary_path = output_base / "crop_density_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Crop Density Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Input file: {las_path}\n")
        f.write(f"Crop type: {crop_type}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Total plants: {total_plants}\n")
        f.write(f"Total area: {total_area:.2f} m²\n")
        f.write(f"Average density: {avg_density:.3f} plants/m²\n")
        if all_heights:
            f.write(f"Average height: {avg_height:.3f} ± {std_height:.3f} m\n")
            f.write(f"Height range: [{min_height:.3f}, {max_height:.3f}] m\n")
        f.write(f"Total time: {time.time() - t0:.2f}s\n\n")
        f.write("Per Tile Summary:\n")
        for tid, cnt in tile_summaries:
            f.write(f"  Tile {tid}: {cnt} plants\n")

    print(f"结果已写入 {summary_path}")


if __name__ == "__main__":
    main()
