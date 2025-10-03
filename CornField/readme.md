## Measure corn density from corn field lidar point cloud.

conda env create -f environment_windows.yml
conda activate corn-density-env

python corn_density_pipeline.py cloud0.las --res 0.025 --bw 0.1 --min_peak_dist 0.2 --thr_rel 0.25 --thr_q 0.4 --fig_dpi 640 --save_peaks_csv





