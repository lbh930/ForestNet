# Crop Density Pipeline 使用说明

## 概述

`crop_pipeline.py` 是一个完整的作物密度计算流程，它整合了以下三个模块：

1. **crop_row_splitter.py** - 检测并分割作物行
2. **density_based_counter.py** - 基于密度峰的植物计数
3. **height_based_counter.py** - 基于高度峰的植物计数

## 工作流程

```
输入LAS文件
    ↓
Z轴归一化
    ↓
分割成Tiles (NxN米)
    ↓
对每个Tile:
    ├─ 检测作物行方向 (X或Y)
    ├─ 分割每个作物行
    ├─ 对每行计数植物 (density或height方法)
    └─ 计算该Tile的密度
    ↓
计算所有Tiles的平均密度
    ↓
输出结果报告
```

## 使用方法

### 基本用法

```bash
# 使用density-based方法
python crop_pipeline.py corn.las --method density

# 使用height-based方法
python crop_pipeline.py corn.las --method height
```

### 使用自定义配置文件

```bash
python crop_pipeline.py corn.las --method density --config my_config.yaml
```

## 配置文件说明 (config.yaml)

配置文件分为4个部分：

### 1. 通用配置 (general)

- `tile_size`: Tile分割大小（米），默认10.0
- `save_visualizations`: 是否保存可视化结果
- `output_dir_prefix`: 输出目录前缀

### 2. Row Splitting 配置 (row_splitting)

控制如何检测和分割作物行：

- `granularity`: 高度曲线采样粒度（米），默认0.02
- `smooth_sigma_bins`: 高斯平滑的sigma值
- `peak_prominence`: Peak检测的突出度阈值
- `peak_min_distance_bins`: Peak之间的最小距离
- `row_boundary_shrink`: Row边界收缩比例（0-1）
- `chm_resolution`: CHM可视化分辨率

### 3. Density-based 配置 (density_based)

当使用 `--method density` 时的参数：

- `expected_spacing`: 预期株间距（米），默认0.08
- `bin_size`: Bin大小（米），默认0.005
- `min_prominence`: 峰突出度阈值（0-1）
- `remove_ground`: 是否移除地面点
- `ground_percentile`: 移除底部百分比
- `top_percentile`: 移除顶部百分比
- `apply_sor`: 是否应用统计离群点移除
- `sor_k`: SOR的k值
- `sor_std_ratio`: SOR的标准差倍数

### 4. Height-based 配置 (height_based)

当使用 `--method height` 时的参数：

- `expected_spacing`: 预期株间距（米）
- `bin_size`: Bin大小（米）
- `min_prominence`: 峰突出度阈值
- `height_metric`: 高度指标类型 (max/mean/percentile_95)
- 其他参数同density-based

## 输出结果

运行后会在当前目录创建输出文件夹：

```
crop_density_output_density/  (或 crop_density_output_height/)
├── pipeline_summary.json     # JSON格式的总结报告
├── pipeline_summary.txt      # 文本格式的总结报告
├── tile_0_0/
│   └── tile_summary.json     # 该tile的详细结果
├── tile_0_1/
│   └── tile_summary.json
└── ...
```

### 主要结果指标

1. **总植物数** (total_plants): 检测到的植物总数
2. **总面积** (total_area_m2): 所有tile的总面积（平方米）
3. **平均密度** (average_density_per_m2): 总植物数 / 总面积（株/平方米）
4. **Tile密度分布**: 各个tile的密度统计（最小、最大、平均、标准差）

### 示例输出

```
Overall Statistics:
  Total tiles: 25
  Total area: 2500.00 m²
  Total rows: 120
  Total plants: 3456
  Average density: 1.38 plants/m²

Tile Density Distribution:
  Min: 0.85 plants/m²
  Max: 1.92 plants/m²
  Mean: 1.42 plants/m²
  Std: 0.23 plants/m²
```

## 方法选择建议

### Density-based 方法
- **优势**: 适合密度变化明显的数据
- **适用**: 植物密集、高度变化较小的场景
- **计算**: 密度 = 点数 × 平均高度

### Height-based 方法
- **优势**: 适合高度变化明显的数据
- **适用**: 植物高度差异大、有明显峰值的场景
- **计算**: 直接使用高度统计量（max/mean/percentile_95）

## 调试和优化

如果结果不理想，可以调整以下参数：

1. **检测不到作物行**:
   - 减小 `row_splitting.peak_prominence`
   - 增大 `row_splitting.smooth_sigma_bins`

2. **植物计数过多**:
   - 增大 `expected_spacing`
   - 增大 `min_prominence`
   - 增大 `ground_percentile` 和 `top_percentile`

3. **植物计数过少**:
   - 减小 `expected_spacing`
   - 减小 `min_prominence`
   - 减小 `ground_percentile` 和 `top_percentile`

4. **噪声太多**:
   - 启用 `apply_sor: true`
   - 增大 `sor_std_ratio`

## 依赖库

```bash
pip install numpy scipy laspy pyyaml matplotlib
```

## 注意事项

1. 输入文件必须是LAS格式的点云文件
2. 点云应该已经过基本的预处理（去噪、配准等）
3. 配置文件必须是有效的YAML格式
4. 输出目录会自动清空重建，注意备份重要数据
5. 处理大文件时可能需要较长时间，请耐心等待

## 示例工作流

```bash
# 1. 准备数据
# 确保corn.las文件存在

# 2. 检查配置文件
# 编辑config.yaml，调整参数

# 3. 运行density方法
python crop_pipeline.py corn.las --method density

# 4. 查看结果
# 打开 crop_density_output_density/pipeline_summary.txt

# 5. 如果结果不满意，尝试height方法
python crop_pipeline.py corn.las --method height

# 6. 比较两种方法的结果
# 选择更合适的方法
```

## 常见问题

**Q: 为什么有些tile没有检测到行？**
A: 可能是该tile的点云太少或没有明显的行结构，可以检查tile_summary.json查看详情。

**Q: 密度结果是否包括空地？**
A: 是的，密度 = 总植物数 / 总面积，包括了所有tile的面积。

**Q: 两种方法的结果差异很大怎么办？**
A: 这说明数据的特征更适合某一种方法，建议可视化检查中间结果，选择更准确的方法。

**Q: 如何处理非矩形田地？**
A: 当前pipeline按矩形tile分割，对于非矩形田地，边缘tile可能包含非作物区域，会影响密度计算。

## 技术支持

如有问题，请检查：
1. 输入文件格式是否正确
2. 配置文件语法是否正确
3. 依赖库是否完整安装
4. 查看terminal输出的详细日志
