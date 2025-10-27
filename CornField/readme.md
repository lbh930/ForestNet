## Measure corn density from corn field lidar point cloud.

conda env create -f environment_windows.yml
conda activate corn-density-env

python corn_density_pipeline.py cloud0.las --res 0.025 --bw 0.1 --min_peak_dist 0.2 --fig_dpi 320 --save_peaks_csv

python corn_density_pipeline.py soybean.las --res 0.01 --bw 0.05 --min_peak_dist 0.15 --fig_dpi 640 --save_peaks_csv

python corn_density_pipeline.py corn.las --res 0.01 --bw 0.05 --min_peak_dist 0.2 --fig_dpi 640 --save_peaks_csv

## New pipeline based on row extraction:
cd C:/ForestNet/Cornfield
conda activate corn-density-env
python crop_row.py soybean.las


## New new pipeline based on row extraction:
cd C:/ForestNet/Cornfield
conda activate corn-density-env
python crop_row_finder.py soybean.las
python crop_row_finder.py morrow_plots.las
python crop_row_splitter.py soybean.las  
python density_based_counter.py row_split_output\tile_0_4\row_y08_at_25.75m.las --direction x --output row_split_output\tile_0_4\row_y08_density

## New New New pipeline
cd C:/ForestNet/Cornfield
conda activate corn-density-env

# Density-based方法
python crop_pipeline.py data/ifarm_corn.las --method density

# Height-based方法
python crop_pipeline.py data/ifarm_soybean_2.las --method height

# 使用自定义配置
python crop_pipeline.py corn.las --method density --config my_config.yaml

# Visualize Result
# 基本用法
python visualize_result.py corn.las row_split_output/corn/ result.png

# 完整路径示例
python visualize_result.py data/ifarm_soybean_2.las row_split_output\ifarm_soybean_2 visualization.png


