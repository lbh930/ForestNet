# Leaf Area Density (LAD) Estimation Methodology

## 1. Overview

The Leaf Area Density (LAD) estimator quantifies the vertical distribution of leaf area within crop canopies using LiDAR point clouds. This method implements a voxel-based approach combined with the Beer-Lambert law to derive LAD profiles and estimate Leaf Area Index (LAI).

**Key output**: LAD(z) [m²/m³] - the one-sided leaf area per unit volume at height z

## 2. Preprocessing

### 2.1 Point Cloud Input
The method operates on individual crop row point clouds, typically generated from the row splitting stage of the main pipeline.

### 2.2 Ground Point Removal
A simple percentile-based filter removes ground points:

```
z_threshold = z_min + (z_max - z_min) × (bottom_percentile / 100)
points_retained = {(x,y,z) | z > z_threshold}
```

**Default parameter**: `bottom_percentile = 10%`

This removes the lowest-elevation points assumed to represent ground returns, ensuring subsequent calculations focus on the canopy structure.

## 3. Voxelization

### 3.1 3D Grid Construction
The filtered point cloud is discretized into a regular 3D voxel grid:

```
Grid dimensions:
  nx = ⌈(x_max - x_min) / vx⌉
  ny = ⌈(y_max - y_min) / vy⌉
  nz = ⌈(z_max - z_min) / vz⌉
  
Voxel mapping:
  For each point (x, y, z):
    ix = ⌊(x - x_min) / vx⌋
    iy = ⌊(y - y_min) / vy⌋
    iz = ⌊(z - z_min) / vz⌋
    
    voxel_grid[ix, iy, iz] = True
    voxel_counts[ix, iy, iz] += 1
```

**Default voxel dimensions**:
- Horizontal: vx = vy = 0.05 m (5 cm)
- Vertical: vz = 0.03 m (3 cm)

The finer vertical resolution captures more detailed canopy stratification, while coarser horizontal resolution balances computational efficiency with spatial detail.

### 3.2 Voxel Grid Outputs
- **voxel_grid**: Boolean 3D array indicating occupied voxels (True if ≥1 point)
- **voxel_counts**: Integer 3D array storing point counts per voxel

## 4. Gap Probability Estimation

### 4.1 Layer-wise Occupancy
For each horizontal layer (height z_k), we compute the voxel occupancy rate:

```
For height layer k (at elevation z_k):
  layer_slice = voxel_grid[:, :, k]  # Extract 2D horizontal slice
  
  occupancy(k) = (number of occupied voxels in layer) / (total voxels in layer)
               = Σ(layer_slice) / (nx × ny)
```

### 4.2 Gap Probability
The gap probability P_gap(z) represents the probability that a downward-traveling laser beam penetrates through the canopy without hitting vegetation at height z:

```
P_gap(z_k) = 1 - occupancy(k)
```

**Physical interpretation**:
- P_gap = 1.0: No vegetation (complete penetration)
- P_gap = 0.0: Dense vegetation (no penetration)
- Intermediate values indicate partial canopy coverage

This simplified formulation assumes:
1. Each occupied voxel represents an opaque vegetation element
2. Voxel occupancy is proportional to vegetation presence
3. Gap probability reflects cumulative interception by vegetation layers above

## 5. LAD Inversion via Beer-Lambert Law

### 5.1 Theoretical Foundation
The Beer-Lambert law for canopy radiation transfer relates gap probability to cumulative leaf area:

```
P_gap(z) = exp(-G × LAI_cumulative(z))

where:
  LAI_cumulative(z) = ∫[z to z_top] LAD(z') dz'
  G = leaf projection function (typically 0.5 for random leaf orientation)
```

Taking the natural logarithm and differentiating with respect to height:

```
ln(P_gap(z)) = -G × LAI_cumulative(z)

d(ln P_gap) / dz = -G × d(LAI_cumulative) / dz = -G × LAD(z)

Therefore:
  LAD(z) = -(1/G) × d(ln P_gap) / dz
```

### 5.2 Discrete Implementation
We apply finite differences to compute the vertical derivative:

```
For each height layer k:
  ln_P_gap[k] = ln(max(P_gap[k], ε))  # ε = 1e-6 to avoid ln(0)
  
  LAD[k] = -(1/G) × (ln_P_gap[k+1] - ln_P_gap[k]) / vz

Physical constraint:
  LAD[k] = max(LAD[k], 0)  # Ensure non-negative values
```

**Note**: The derivative is computed upward (from ground to canopy top) because P_gap decreases with increasing height in vegetated canopies.

### 5.3 Parameter Selection

**Projection function G**:
- Default: G = 0.5 (spherical leaf angle distribution)
- Can be adjusted for specific crops:
  - G ≈ 0.5: Random/spherical orientation (typical for many crops)
  - G < 0.5: Horizontal leaf tendency (e.g., some soybeans)
  - G > 0.5: Vertical leaf tendency (e.g., some grasses)

## 6. Leaf Area Index (LAI) Calculation

The total Leaf Area Index is obtained by vertical integration of LAD:

```
LAI = ∫[0 to z_max] LAD(z) dz

Discrete approximation:
  LAI = Σ[k=0 to nz-1] LAD[k] × vz
```

**Units**: LAI [m²/m²] - one-sided leaf area per unit ground area

## 7. Output Metrics

### 7.1 LAD Statistics
For each processed row, we compute:
- **Mean LAD**: Average across all height layers with LAD > 0
- **Median LAD**: Robust central tendency measure
- **Maximum LAD**: Peak vegetation density (typically mid-canopy)
- **Standard deviation**: Variability of LAD across height

### 7.2 Vertical Profiles
The method generates complete vertical profiles of:
1. **Voxel occupancy** (ratio, 0-1): Proportion of occupied voxels per layer
2. **Gap probability** (ratio, 0-1): Canopy penetration probability
3. **LAD** (m²/m³): Leaf area density

### 7.3 Height Layer Information
For each vertical layer k, we store:
- Height (m): `z = z_min + (k + 0.5) × vz` (voxel center)
- LAD (m²/m³): Computed leaf area density
- Occupancy: Voxel occupancy rate
- P_gap: Gap probability

## 8. Assumptions and Limitations

### 8.1 Key Assumptions
1. **Homogeneous horizontal distribution**: LAD computed from horizontal layer averages assumes uniform distribution within each layer
2. **Binary voxel occupancy**: Voxels are either occupied or empty (no partial occupancy)
3. **Single-pass approximation**: Simplifies multiple scattering effects
4. **Constant G-function**: Assumes uniform leaf angle distribution across the canopy

### 8.2 Applicability
This method is most accurate for:
- ✅ Row crops with relatively uniform canopy structure
- ✅ Dense LiDAR point clouds (high point density)
- ✅ Single-species canopies
- ✅ Vertical canopy profiling applications

### 8.3 Limitations
- ⚠️ Underestimates LAD in sparse canopies where voxel resolution exceeds leaf size
- ⚠️ Sensitive to ground removal threshold in short canopies
- ⚠️ Does not account for clumping effects at sub-voxel scales
- ⚠️ G-function simplification may introduce bias for crops with non-random leaf angles

## 9. Computational Workflow

```
Input: Row point cloud (.las)
  ↓
[Ground removal] (percentile-based filtering)
  ↓
[Voxelization] (3D discretization)
  ↓
[Occupancy calculation] (layer-wise aggregation)
  ↓
[Gap probability] (P_gap = 1 - occupancy)
  ↓
[LAD inversion] (Beer-Lambert with finite differences)
  ↓
[LAI integration] (vertical summation)
  ↓
Output: LAD(z) profile, LAI value, statistics
```

## 10. Visualization Outputs

The pipeline generates:

1. **Vertical profiles plot**: Three-panel figure showing:
   - Voxel occupancy vs. height
   - Gap probability vs. height
   - LAD vs. height

2. **Voxel layer visualization**: Horizontal slices at multiple heights showing spatial distribution of occupied voxels

3. **YAML summary**: Machine-readable output containing:
   - Processing parameters (voxel sizes, G-function)
   - LAD statistics (mean, median, max, std)
   - LAI estimate
   - Complete height layer data

## 11. Parameter Configuration

| Parameter | Default | Description | Tuning Guidelines |
|-----------|---------|-------------|-------------------|
| vx, vy | 0.05 m | Horizontal voxel size | Smaller: more detail, more computation |
| vz | 0.03 m | Vertical voxel size | Smaller: finer LAD resolution |
| G | 0.5 | Projection function | 0.4-0.6 for most crops |
| bottom_percentile | 10% | Ground removal threshold | Increase for shorter crops |

## 12. Integration with Main Pipeline

The LAD estimator operates independently on row-split point clouds generated by the main crop detection pipeline:

```
Main Pipeline → Row Splitting → Individual Row .las files
                                         ↓
                                 LAD Estimator (this module)
                                         ↓
                              LAD profiles + LAI estimates
```

This modular design allows:
- Selective LAD estimation on specific rows or crops
- Independent parameter tuning for LAD vs. plant detection
- Flexible integration into various analysis workflows

## 13. Future Enhancements

Potential improvements to the methodology:
- **Adaptive G-function**: Height-dependent leaf angle distribution
- **Multi-return processing**: Utilize LiDAR return intensity and multiple returns
- **Clumping correction**: Account for non-random spatial distribution (Ω factor)
- **Woody-to-leaf separation**: Distinguish stem/branch returns from foliage
- **Temporal LAD tracking**: Monitor canopy development across growth stages
