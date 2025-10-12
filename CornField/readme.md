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





