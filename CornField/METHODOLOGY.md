# Methodology

## 1. Overview

Our canopy parameter estimation pipeline employs a three-stage approach for individual plant detection and density estimation from LiDAR point clouds: (1) crop row detection and segmentation, (2) point cloud preprocessing and noise filtering, and (3) individual plant detection using either density-based or height-based methods. The pipeline is designed to handle different crop species with crop-specific parameter configurations.

## 2. Crop Row Detection and Segmentation

### 2.1 Point Cloud Normalization
The input LiDAR point cloud is first normalized by setting the ground level to z=0 through percentile-based ground estimation, ensuring consistent height measurements across the field.

### 2.2 Spatial Tiling
To enable parallel processing and handle large-scale datasets, the normalized point cloud is divided into square tiles of configurable size (default: 10m × 10m). Each tile is processed independently to detect crop rows and extract individual plant parameters.

### 2.3 Canopy Height Model (CHM) Generation
For each tile, we generate a smoothed Canopy Height Model with a spatial resolution of 2cm:

```
CHM(x,y) = max{z | (x,y,z) ∈ points within grid cell}
```

The CHM is then smoothed using a Gaussian filter (σ=3 pixels) to reduce noise while preserving row structures.

### 2.4 Row Direction Detection
Crop rows are detected by projecting the CHM onto both x and y axes and analyzing the resulting height profiles:

```
H_x(x) = mean{CHM(x,y) | y ∈ [y_min, y_max]}
H_y(y) = mean{CHM(x,y) | x ∈ [x_min, x_max]}
```

Each profile is smoothed using Gaussian filtering with crop-specific parameters:
- **Smooth sigma** (σ_row): Controls the spatial extent of smoothing, set according to expected row spacing (e.g., 10cm for corn, 5cm for soybean)
- **Prominence threshold** (p_row): Minimum peak height relative to surrounding baseline for row detection

The row direction is determined by comparing the regularity scores of detected peaks in both projections, calculated as:

```
R = 1 / (1 + CV(Δd))
```

where CV(Δd) is the coefficient of variation of inter-peak distances, and higher R indicates more regular spacing characteristic of planted rows.

### 2.5 Row Boundary Delineation
Once the row direction is determined, individual row boundaries are computed using a distance transform approach:

1. Create a binary mask from detected row centers
2. Apply Euclidean distance transform
3. Define boundaries as midpoints between adjacent rows or at half-distance from field edges

This approach handles curved rows and variable row spacing naturally.

## 3. Point Cloud Preprocessing

### 3.1 Statistical Outlier Removal (SOR)
For crops requiring noise filtering (configurable per crop type), we apply Statistical Outlier Removal:

```
For each point p:
  d̄_p = mean{||p - q|| | q ∈ KNN(p, k)}
  σ_global = std{d̄_q | q ∈ all points}
  Remove p if d̄_p > d̄_global + α·σ_global
```

where k is the neighborhood size (default: 20) and α is the threshold multiplier (default: 2.0). This step effectively removes isolated noise points while preserving the plant structure.

### 3.2 Ground and Canopy Filtering
To focus on the main plant body and improve detection accuracy, we apply percentile-based filtering:

- **Ground removal**: Remove points below the p_ground percentile of z-values (e.g., 20th percentile)
- **Top canopy removal** (optional): Remove points above the (100 - p_top) percentile to eliminate outliers from sensor noise or overhanging vegetation

Importantly, while these filtered point clouds are used for peak detection, the actual plant height measurements are computed from the SOR-cleaned point cloud that retains the full vertical extent, ensuring accurate height estimation.

## 4. Individual Plant Detection

### 4.1 Density-Based Method

The density-based approach is suitable for crops with uniform distribution and dense canopy cover (e.g., mature corn).

#### 4.1.1 Density Profile Computation
For each row, we compute a 1D density profile along the direction perpendicular to the row:

```
D(x_i) = Σ{h_j | (x_j, y_j, z_j) ∈ bin_i} / N_i
```

where:
- x_i is the center of bin i with width Δx (default: 1cm)
- h_j = z_j - z_min is the relative height of point j
- N_i is the number of points in bin i
- The density is height-weighted to emphasize taller vegetation

#### 4.1.2 Signal Smoothing and Peak Detection
The raw density profile is smoothed using Gaussian filtering:

```
D_smooth(x) = (D * G_σ)(x)
```

where σ is crop-specific (e.g., σ=2.0 for corn, accounting for ~25cm plant spacing). Peaks are then detected using the scipy `find_peaks` algorithm with:
- **Distance constraint**: Minimum peak separation = expected plant spacing
- **Prominence threshold**: Minimum peak prominence above local baseline

### 4.2 Height-Based Method

The height-based approach is more robust for crops with smaller spacing or variable canopy density (e.g., soybean).

#### 4.2.1 Standard Height Metrics
We offer four height aggregation methods for computing the 1D height profile H(x):

1. **Maximum height**: `H(x_i) = max{z_j | j ∈ bin_i}`
2. **Mean height**: `H(x_i) = mean{z_j | j ∈ bin_i}`
3. **95th percentile**: `H(x_i) = percentile_95{z_j | j ∈ bin_i}`
4. **Kernel convolution** (recommended, described below)

#### 4.2.2 3D Kernel Convolution Method
To better capture local height features and reduce noise sensitivity, we developed a kernel-based sampling method that performs spatial aggregation in 3D:

**Algorithm**:
```
For each sampling position x_i along the row:
  1. Define 3D kernel window:
     K_i = {(x,y,z) | |x - x_i| ≤ L/2, |y - y_c| ≤ W/2}
     where:
       L = kernel_length (along row direction, e.g., 6cm for soybean)
       W = kernel_width (perpendicular to row, e.g., 20cm)
       y_c = row center coordinate
  
  2. Extract points within kernel:
     P_i = {p | p ∈ preprocessed_points ∩ K_i}
  
  3. Compute robust height statistic:
     H(x_i) = percentile_q{z | (x,y,z) ∈ P_i}
     where q is configurable (e.g., 85th percentile for soybean)
```

**Key advantages**:
- **Local spatial context**: Each height sample aggregates information from a 3D neighborhood rather than a thin vertical column
- **Noise robustness**: Percentile-based aggregation is less sensitive to outliers than maximum height
- **Configurable resolution**: Kernel size can be tuned to match plant morphology (smaller kernels for densely spaced plants)
- **Preservation of peak locations**: Unlike global smoothing, kernel convolution maintains spatial resolution for accurate plant positioning

**Parameter selection**:
The kernel parameters are crop-specific and determined by plant spacing and canopy characteristics:
- **Kernel length** (L): ~0.5-1.0× expected plant spacing (e.g., 6cm for 10cm soybean spacing)
- **Kernel width** (W): ~1-2× row width to capture full plant canopy perpendicular to row
- **Height percentile** (q): 85-95th percentile to balance between capturing peak height and rejecting noise

#### 4.2.3 Peak Detection from Height Profile
The height profile H(x) is processed for peak detection:

1. **Optional smoothing**: For non-kernel methods, apply Gaussian smoothing with σ=1
2. **Peak detection**: Apply `find_peaks` with distance and prominence constraints
3. **Peak refinement**: Filter peaks based on minimum prominence threshold

**Note**: When using the kernel method, we skip the additional Gaussian smoothing step since the kernel convolution already provides spatial smoothing. This prevents over-smoothing that could merge closely spaced plants.

## 5. Plant Height Measurement

Once plant locations are identified, we measure the actual height of each plant using the SOR-cleaned point cloud that preserves the full vertical extent:

```
For each detected plant at position x_p:
  1. Define plant region:
     R_p = [x_p - Δx/2, x_p + Δx/2]
     where Δx is the distance to adjacent plants or field boundary
  
  2. Extract points in region:
     P_p = {(x,y,z) | x ∈ R_p, y ∈ row_bounds}
  
  3. Compute height:
     h_p = max{z | (x,y,z) ∈ P_p} - min{z | (x,y,z) ∈ P_p}
```

This ensures height measurements reflect the true vertical extent from ground to canopy top, unaffected by the preprocessing filters used for detection.

## 6. Density Estimation

The final crop density is computed by aggregating results across all tiles and rows:

```
ρ = N_total / A_total

where:
  N_total = Σ{plants detected in all rows}
  A_total = n_tiles × tile_area
```

Statistical summaries are also computed for plant heights across the field, including mean, standard deviation, and range.

## 7. Parallel Processing

To efficiently process large-scale point clouds, the pipeline implements tile-based parallelization using Python's `ProcessPoolExecutor`. Each tile is processed independently in a separate process, with results aggregated at the end. The number of parallel workers is configurable (default: use all available CPU cores).

## 8. Crop-Specific Configuration

All parameters are externalized in a YAML configuration file, allowing easy adaptation to different crop species without code modification. Key configurable parameters include:

- **Method selection**: density vs. height-based detection
- **Preprocessing**: SOR parameters, ground/canopy removal percentiles
- **Row detection**: smoothing sigma, prominence threshold
- **Plant detection**: expected spacing, bin size, minimum prominence
- **Height method**: metric type, kernel dimensions and percentile
- **Density method**: smoothing sigma

Example configurations are provided for corn (wide spacing, tall canopy, density method) and soybean (narrow spacing, shorter canopy, height-kernel method).
