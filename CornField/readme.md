## Measure corn density from corn field lidar point cloud.

conda env create -f environment_windows.yml
conda activate corn-density-env

python corn_density_pipeline.py cloud0.las --voxel 0.03 --nb_neighbors 20 --std_ratio 2.0 --ransac_dist 0.02 --ground_band 0.10 --grid_cell 0.5 --kde_bw 0.4 --kde_res 0.25 --row_spacing 0.75 --save_intermediate_ply --row_band 0.22 --xbin 0.03 --smooth_sigma 1.0 --peak_min_spacing 0.16 --peak_prominence 2.5 --row_height_min 0.1

