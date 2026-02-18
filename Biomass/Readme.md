# Volumetric Biomass Density Proxy + Height Profile (UAV LiDAR)

## Minimal run (`config.yaml` + `example_forest.las`)
`config.yaml` is already set to `example_forest.las`, so you can run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py --config config.yaml
```

## Generate Minecraft-style voxel block density images (separate command)
This command reads `outputs/run_001/*voxel_density_points_per_cubic_meter.npy` and only renders voxel blocks:

```bash
.venv/bin/python run_voxel_block_visualization.py --config config.yaml
```

Optional fast mode if you only need one image:

```bash
.venv/bin/python run_voxel_block_visualization.py --config config.yaml --voxel-density-grid-source-mode observed
```

This repo implements a lightweight pipeline to turn a single UAV LiDAR scan into:
1) a 3D volumetric density proxy field, and
2) a height-resolved “mass distribution” profile,
with an optional occlusion correction that compensates for the exponential drop in returns under canopy.

We are not sure if we are going to have ground truth. So we need a stable, interpretable proxy that supports:
- separating near-ground understory vs overstory mass distribution
- wildfire-relevant analysis (low-lying fuels vs canopy fuels)

## Problem framing (what we can honestly claim)
- Output is a visibility-corrected volumetric density proxy (relative units).
- Validation is via sanity checks and stability tests (subsampling invariance, flightline consistency, counterfactual slicing).
- Absolute mass calibration is future work unless we later add field plots / allometry / destructive samples.

Existing biomass and fuels work typically uses LiDAR to capture canopy geometry and then relies on field plots, allometric equations, or national forest inventories to obtain ground-truth biomass or canopy fuel loading, sometimes fitting height-layer profiles from those measurements. Our goal is different: with only a drone LiDAR scan and no external ground truth, we aim to produce a visibility-corrected 3D volumetric density proxy and a height-resolved profile that separates near-ground understory from overstory structure, enabling fast, low-cost stratified fuel assessment for wildfire-relevant analysis.

## Potential Ground Truth Retrieval Method
Some papers say they cut down part of the canopy and weight them. 
A few other papers look into NFI (which stands for national forest inventory) for pre-measured values and use allometry equations to estimate biomass.
Some Hand-measure data (trunk diameter) and use allometry.
So I think if we were capturing new data, we can measure trunk diameter manually, and get tree heights from point cloud, then we have 2 parameters (d, h) to plug into allometry equations

## Inputs
- UAV LiDAR point cloud for a targeted area (LAS/LAZ/PLY/etc).
- Optional: RGB imagery (only for visualization/weak qualitative checks; not required).

Assumptions:
- Point cloud has reasonable georeferencing.
- We can estimate ground and normalize heights (z relative to ground).

## Outputs
- `rho_xyz`: 3D voxel grid where each voxel stores an observed local density proxy (points per m^3) and the corrected version.
- `profile_z`: vertical profile over height bins (sum over x,y for each z bin), both observed and corrected.
- Optional: 2D density maps (collapse along z, or per-height-slab maps).
- Figures for stability tests.

## Pipeline overview
### 1) Preprocess
- Crop AOI if needed.
- Ground removal / DTM estimation.
- Height normalization: z_norm = z - ground_z(x,y).
- Optional: range normalization if you have variable flight geometry (keep simple if time is short).

### 2) Voxelize
- Choose voxel size:
  - start with 0.25–1.0 m depending on point density; keep it fixed across experiments.
- Build a 3D grid over AOI and height range.
- For each voxel v:
  - `N_v` = number of points in voxel
  - `rho_obs_v = N_v / voxel_volume`  (points per m^3)

Also compute a simple occupancy flag:
- `occ_v = 1 if N_v > 0 else 0` (or use a threshold)

### 3) Build vertical profile
- Pick height binning (often same as voxel z size).
- For each height layer k:
  - `profile_obs[k] = sum_{x,y} rho_obs[x,y,k]`
Optionally normalize:
- `profile_obs_norm = profile_obs / sum(profile_obs)` for shape-only comparisons.

### 4) Occlusion correction (exponential transmission per vertical column)
Goal: compensate for canopy occlusion causing fewer returns at lower heights.

For each x,y column:
- Compute cumulative occlusion above height k:
  - `O[k] = sum_{i < k} occ[i]`  (simple and stable)
- Transmission:
  - `T[k] = exp(-alpha * O[k])`
- Corrected density per voxel:
  - `rho_cor[k] = clamp(rho_obs[k] / max(T[k], eps), max_gain)`
Where:
- `eps` avoids blow-ups (e.g., 1e-3)
- `max_gain` caps amplification (e.g., 5–10x)

Then recompute:
- `profile_cor[k] = sum_{x,y} rho_cor[x,y,k]`

### 5) Choosing alpha without ground truth
Pick `alpha` by minimizing profile drift under subsampling:
- For a grid of alpha values:
  - Randomly subsample points at p = 0.25, 0.5, 0.75 (repeat a few seeds)
  - Compute `profile_cor` for each run
  - Score stability (e.g., correlation between profiles, or KL divergence)
- Choose alpha that maximizes stability.

This makes alpha a robustness hyperparameter, not a “physical” calibration.

## Validation (no ground truth)
We rely on falsifiable sanity checks:

1) Subsampling invariance
- Profile shape should be stable when you randomly drop points.

2) Rigid transform invariance
- Translating/rotating the point cloud should not change the profile.

3) Counterfactual slicing
- If you keep only points with z < 2 m, the profile should concentrate near ground.
- If you keep only points with z > 10 m, the profile should shift upward.

4) Sensitivity to voxel size
- Show results for 2–3 voxel sizes; profiles should be qualitatively consistent.

5) Occlusion correction ablation
- Compare observed vs corrected; corrected should be more stable under subsampling, especially for dense canopy scenes.

## Repo structure suggestion
- `src/io/` point cloud loading, cropping
- `src/ground/` ground estimation + height normalization
- `src/voxel/` voxel grid + feature accumulation
- `src/occlusion/` transmission model + correction
- `src/profile/` profile + map projections
- `src/experiments/` subsampling, ablations, plots
- `configs/` voxel size, alpha grid, eps, max_gain, height ranges
- `notebooks/` quick visualization

## Notes on scope (for paper)
- We produce a height-resolved volumetric density proxy from UAV LiDAR alone.
- Focus is understory vs overstory separation via profiles and derived metrics (e.g., near-ground fraction).
- Absolute kg/m^3 requires external calibration (NFI/field plots/destructive sampling), which we do not assume.
