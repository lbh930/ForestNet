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
# 主流程
# ------------------------------

def main():
    if len(sys.argv) < 3:
        print("用法: python crop_pipeline.py <las文件路径> --method <density|height>")
        sys.exit(1)
    
    las_path = Path(sys.argv[1])
    if not las_path.exists():
        print(f"错误: 文件不存在: {las_path}")
        sys.exit(1)
    
    # 获取计数方式
    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1].lower()
            if method not in ['density', 'height']:
                print("错误: method 必须是 'density' 或 'height'")
                sys.exit(1)
        else:
            print("错误: 未指定method参数")
            sys.exit(1)
    else:
        print("错误: 需要 --method 参数")
        sys.exit(1)
    
    # 加载配置
    config = load_config()
    params = config[method]
    print("="*60)
    print(f"Crop Pipeline 启动 ({method}-based counting)")
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
    splitter = load_module_from_path("crop_row_splitter.py", "crop_row_splitter")
    density_counter = load_module_from_path("density_based_counter.py", "density_based_counter")
    height_counter = load_module_from_path("height_based_counter.py", "height_based_counter")

    # Step 1: 作物行分割
    print("\n[1] 执行作物行分割...")
    x, y, z = splitter.read_las_file(las_path)
    x, y, z = splitter.normalize_z_to_zero(x, y, z)
    tiles, tile_info = splitter.split_into_tiles(x, y, z, tile_size=params["tile_size"])

    total_plants = 0
    total_area = 0.0

    # Step 2: 处理每个 tile
    for tile_id, (tile_x, tile_y, tile_z) in tiles.items():
        stats = splitter.process_single_tile(tile_id, tile_x, tile_y, tile_z, 
                                             tile_info[tile_id], output_base)
        if stats['row_count'] == 0:
            continue

        tile_dir = output_base / f"tile_{tile_id[0]}_{tile_id[1]}"
        chosen_dir = tile_dir.glob("row_*.las")

        for row_path in chosen_dir:
            row_name = row_path.stem  # 比如 "row_y01_at_12.34m"
            points = read_las_points(row_path)
            if len(points) < 50:
                continue

            # 为每个row建立独立输出目录
            row_output_dir = tile_dir / "count_results" / row_name
            row_output_dir.mkdir(parents=True, exist_ok=True)

            if stats['direction'] == 'x':
                count_direction = 'y'
            else:
                count_direction = 'x'

            # ✅ 修正：计数方向应与行方向垂直
            if stats['direction'] == 'x':
                count_direction = 'y'
            else:
                count_direction = 'x'

            if method == 'density':
                plant_count, _, _ = density_counter.density_count_from_row(
                    points,
                    direction=count_direction,   # ✅ 使用反方向
                    expected_spacing=params["expected_spacing"],
                    bin_size=params["bin_size"],
                    apply_sor=params["apply_sor"],
                    sor_k=params["sor_k"],
                    sor_std_ratio=params["sor_std_ratio"],
                    remove_ground=params["remove_ground"],
                    ground_percentile=params["ground_percentile"],
                    top_percentile=params["top_percentile"],
                    min_prominence=params["min_prominence"],
                    output_dir=row_output_dir
                )
            else:
                plant_count, _, _ = height_counter.height_count_from_row(
                    points,
                    direction=count_direction,   # ✅ 使用反方向
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
                    output_dir=row_output_dir
                )


        # Tile 面积
        total_area += params["tile_size"] ** 2

    # Step 3: 汇总密度
    avg_density = total_plants / total_area if total_area > 0 else 0.0

    print("\n" + "="*60)
    print("处理完成 ✅")
    print(f"总检测作物数: {total_plants}")
    print(f"总面积: {total_area:.2f} m²")
    print(f"平均作物密度: {avg_density:.3f} 株/m²")
    print(f"总耗时: {time.time() - t0:.2f}s")
    print("="*60)

    # 输出summary文件
    with open("crop_density_summary.txt", "w", encoding="utf-8") as f:
        f.write("Crop Density Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Input file: {las_path}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Total plants: {total_plants}\n")
        f.write(f"Total area: {total_area:.2f} m²\n")
        f.write(f"Average density: {avg_density:.3f} plants/m²\n")
        f.write(f"Total time: {time.time() - t0:.2f}s\n")

    print("结果已写入 crop_density_summary.txt")


if __name__ == "__main__":
    main()
