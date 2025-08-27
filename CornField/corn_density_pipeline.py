# corn_density_pipeline.py
# Goal: decoupled steps with short English comments + per-step visualizations.

import argparse
import json
import numpy as np
import laspy
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans, DBSCAN
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks  # add
from scipy.ndimage import gaussian_filter1d  # add (与 ndimage 已导入不同函数)
from scipy.spatial.transform import Rotation as R

# -------------------- I/O utils --------------------

def read_las_xyz(path):
    """Read LAS to Nx3 float64 array."""
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

def np_to_o3d(points):
    """Numpy -> Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def o3d_to_np(pcd):
    """Open3D -> Numpy Nx3."""
    return np.asarray(pcd.points)

# -------------------- Viz helpers --------------------

def plot_xy(points, title, out_png, s=0.5):
    """Scatter XY for quick look, save PNG."""
    if points.size == 0:
        print(f"[warn] empty points for {title}")
        # still create an empty plot for consistency
    plt.figure(figsize=(6,6), dpi=150)
    if points.size:
        plt.scatter(points[:,0], points[:,1], s=s)
    plt.gca().set_aspect('equal', 'box')
    plt.title(title)
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_heatmap(grid, meta, title, out_png):
    """Save a 2D heatmap for grid/KDE results."""
    plt.figure(figsize=(6,6), dpi=150)
    extent = [meta["xmin"], meta["xmax"], meta["ymin"], meta["ymax"]]
    plt.imshow(grid, origin='lower', extent=extent, aspect='equal')
    plt.title(title); plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    cbar = plt.colorbar(); cbar.set_label("Intensity / Count")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_rows(xprime, yprime, row_ids, labels, title, out_png):
    """Visualize row assignment and DBSCAN clusters."""
    plt.figure(figsize=(7,5), dpi=150)
    # color by row id
    if len(row_ids):
        for rid in np.unique(row_ids):
            idx = (row_ids == rid)
            plt.scatter(xprime[idx], yprime[idx], s=1.0, label=f"row {rid}")
    else:
        plt.scatter(xprime, yprime, s=1.0, label="points")
    # highlight clusters (labels >= 0)
    if len(labels):
        core = (labels >= 0)
        plt.scatter(xprime[core], yprime[core], s=2.0, alpha=0.6)
    plt.gca().set_aspect('equal', 'box')
    plt.title(title); plt.xlabel("x' (m)"); plt.ylabel("y' (m)")
    plt.legend(markerscale=6, fontsize=8, loc='best')
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_row_hist(xbins_centers, counts_smooth, peak_xs, title, out_png):
    """Plot 1D histogram along row (x') with detected peaks."""
    plt.figure(figsize=(7,3), dpi=150)
    plt.plot(xbins_centers, counts_smooth, linewidth=1.2, label="smoothed hist")
    if len(peak_xs):
        plt.scatter(peak_xs, np.interp(peak_xs, xbins_centers, counts_smooth),
                    s=12, marker='x', label="peaks")
    plt.title(title)
    plt.xlabel("x' (m)"); plt.ylabel("count")
    plt.tight_layout(); plt.legend(fontsize=8)
    plt.savefig(out_png); plt.close()


# -------------------- Step 1: read --------------------

def step1_read(las_path):
    """Step1: read raw LAS."""
    pts = read_las_xyz(las_path)
    plot_xy(pts, "Step1: Raw XY", "step1_raw_xy.png")
    return pts

# -------------------- Step 2: denoise --------------------

def statistical_denoise(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal."""
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                              std_ratio=std_ratio)
    return clean

def voxel_downsample(pcd, voxel=0.03):
    """Voxel downsample for speed."""
    if voxel and voxel > 0:
        return pcd.voxel_down_sample(voxel)
    return pcd

def step2_denoise(points, voxel=0.03, nb_neighbors=20, std_ratio=2.0):
    """Step2: downsample + denoise."""
    pcd = np_to_o3d(points)
    pcd = voxel_downsample(pcd, voxel)
    pcd = statistical_denoise(pcd, nb_neighbors, std_ratio)
    pts = o3d_to_np(pcd)
    plot_xy(pts, "Step2: Denoised XY", "step2_denoised_xy.png")
    return pts

# -------------------- Step 3: RANSAC plane --------------------

def ransac_plane(points, dist_thresh=0.02, ransac_n=3, max_iter=2000):
    """Fit ground plane by RANSAC."""
    pcd = np_to_o3d(points)
    model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                       ransac_n=ransac_n,
                                       num_iterations=max_iter)
    a,b,c,d = model
    inlier = pcd.select_by_index(inliers)
    outlier = pcd.select_by_index(inliers, invert=True)
    in_np = o3d_to_np(inlier)
    out_np = o3d_to_np(outlier)
    # Visualize inliers/outliers
    plot_xy(in_np,  "Step3: Ground inliers XY",  "step3_ground_inliers_xy.png")
    plot_xy(out_np, "Step3: Non-ground (pre-band) XY", "step3_nonground_preband_xy.png")
    return (a,b,c,d), in_np, out_np

def signed_distance(points, plane):
    """Signed distance to plane ax+by+cz+d=0."""
    a,b,c,d = plane
    n = np.sqrt(a*a+b*b+c*c) + 1e-12
    return (points[:,0]*a + points[:,1]*b + points[:,2]*c + d) / n

# -------------------- Step 4: remove ground band --------------------

def step4_remove_ground_band(nonground_preband, plane, band=0.10):
    """Remove |dist|<=band."""
    dist = signed_distance(nonground_preband, plane)
    keep = np.abs(dist) > band
    kept = nonground_preband[keep]
    removed = nonground_preband[~keep]
    plot_xy(removed, "Step4: Removed band XY", "step4_removed_band_xy.png")
    plot_xy(kept,   "Step4: Kept after band XY", "step4_kept_after_band_xy.png")
    return kept

# -------------------- Step 5A: grid density --------------------

def grid_density_xy(points, cell=0.5):
    """Uniform XY grid counting."""
    if points.size == 0:
        return np.zeros((1,1)), {
            "xmin":0,"xmax":1,"ymin":0,"ymax":1,"cell":cell,"nx":1,"ny":1
        }
    xy = points[:,:2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))
    grid = np.zeros((ny, nx), dtype=int)
    ix = np.clip(((xy[:,0]-xmin)/cell).astype(int), 0, nx-1)
    iy = np.clip(((xy[:,1]-ymin)/cell).astype(int), 0, ny-1)
    for xg, yg in zip(ix, iy):
        grid[yg, xg] += 1
    meta = {"xmin":float(xmin),"xmax":float(xmax),
            "ymin":float(ymin),"ymax":float(ymax),
            "cell":float(cell),"nx":nx,"ny":ny}
    return grid, meta

def step5a_grid(points, cell=0.5):
    """Step5A: grid heatmap."""
    grid, meta = grid_density_xy(points, cell=cell)
    plot_heatmap(grid, meta, f"Step5A: Grid density (cell={cell}m)", "step5a_grid_density.png")
    return grid, meta

# -------------------- Step 5B: KDE intensity --------------------

def kde_intensity_xy(points, bandwidth=0.4, res=0.25):
    """KDE-based 2D intensity."""
    if points.size == 0:
        return np.zeros((1,1)), {
            "xmin":0,"xmax":1,"ymin":0,"ymax":1,"grid_res":res,"bandwidth":bandwidth,"nx":1,"ny":1
        }
    xy = points[:,:2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xs = np.arange(xmin, xmax + res, res)
    ys = np.arange(ymin, ymax + res, res)
    xx, yy = np.meshgrid(xs, ys)
    sample = np.vstack([xx.ravel(), yy.ravel()]).T
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(xy)
    logd = kde.score_samples(sample)
    dens = np.exp(logd).reshape(len(ys), len(xs))
    meta = {"xmin":float(xmin),"xmax":float(xmax),
            "ymin":float(ymin),"ymax":float(ymax),
            "grid_res":float(res),"bandwidth":float(bandwidth),
            "nx":len(xs),"ny":len(ys)}
    return dens, meta

def step5b_kde(points, bandwidth=0.4, res=0.25):
    """Fast KDE via grid count + Gaussian blur."""
    if points.size == 0:
        dens = np.zeros((1,1)); meta = {"xmin":0,"xmax":1,"ymin":0,"ymax":1,"grid_res":res,"bandwidth":bandwidth,"nx":1,"ny":1}
        plot_heatmap(dens, meta, f"Step5B: KDE (fast, bw={bandwidth}, res={res}m)", "step5b_kde_intensity.png")
        return dens, meta

    # 1) grid count (reuse Step5A logic)
    grid, meta = grid_density_xy(points, cell=res)

    # 2) gaussian blur; sigma in pixels = bandwidth / res
    sigma_pix = max(0.5, bandwidth / res)  # avoid sigma too small
    dens = gaussian_filter(grid.astype(float), sigma=sigma_pix, mode="nearest")

    dens = dens / (2.0 * np.pi * (bandwidth**2))  # optional scaling

    plot_heatmap(dens, meta, f"Step5B: KDE (fast, bw={bandwidth}, res={res}m)", "step5b_kde_intensity.png")
    return dens, meta

# -------------------- Step 5C: row-aware density --------------------

def pca_xy(points):
    """Rotate XY so rows align with x' axis."""
    xy = points[:,:2]
    mean = xy.mean(axis=0, keepdims=True)
    zero = xy - mean
    cov = np.cov(zero.T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    angle = np.arctan2(major[1], major[0])
    rot = R.from_euler('z', -angle).as_matrix()[:2,:2]
    xy_rot = zero @ rot.T
    return xy_rot[:,0], xy_rot[:,1]

def estimate_num_rows(yprime, approx_row_spacing):
    """Estimate row count from span / spacing."""
    span = yprime.max() - yprime.min() if len(yprime) else 0.0
    if span <= 0: return 1
    n = int(np.round(span / max(approx_row_spacing, 1e-3)))
    return max(1, n)

def step5c_row_density_peaks(points,
                             row_spacing=0.75,
                             row_band=0.12,
                             xbin=0.05,
                             smooth_sigma=1.5,
                             peak_min_spacing=0.18,
                             peak_prominence=5.0,
                             height_min=None,
                             example_rows_to_plot=12):
    """
    Row-wise stem counting via 1D peak detection along x'.
    - row_spacing: prior row spacing (m), for plants/m^2 conversion.
    - row_band: half-width in y' (m) to keep near the row center (suppress leaves).
    - xbin: histogram bin size along x' (m).
    - smooth_sigma: Gaussian smoothing sigma in bins.
    - peak_min_spacing: minimal peak separation in meters (≈ plant spacing).
    - peak_prominence: required prominence for peaks (robustness).
    - height_min: optional min height above low z to filter out low clutter (m).
    - example_rows_to_plot: save hist figures for first N rows.
    """
    P = points
    if P.size == 0:
        return [], {"avg_linear_density_stems_per_m": 0.0,
                    "avg_density_plants_per_m2": 0.0,
                    "row_spacing_m": row_spacing}

    # optional height filter
    if height_min is not None:
        z0 = np.percentile(P[:,2], 5)
        P = P[P[:,2] > (z0 + height_min)]
    if P.size == 0:
        return [], {"avg_linear_density_stems_per_m": 0.0,
                    "avg_density_plants_per_m2": 0.0,
                    "row_spacing_m": row_spacing}

    # PCA align rows
    xprime, yprime = pca_xy(P)

    # KMeans split rows
    n_rows = estimate_num_rows(yprime, row_spacing)
    if n_rows > 1:
        km = KMeans(n_clusters=n_rows, n_init="auto", random_state=0)
        row_ids = km.fit_predict(yprime.reshape(-1,1))
        row_centers = np.sort(km.cluster_centers_.ravel())
    else:
        row_ids = np.zeros(len(yprime), dtype=int)
        row_centers = np.array([np.median(yprime)])

    results = []
    # distance in bins for peak separation
    min_dist_bins = max(1, int(np.round(peak_min_spacing / max(xbin, 1e-6))))

    # per-row processing
    for rid in np.unique(row_ids):
        mask_row = (row_ids == rid)
        xr = xprime[mask_row]
        yr = yprime[mask_row]

        if len(xr) < 2:
            continue

        # keep a tight band around row center to suppress leaves
        y_med = np.median(yr)
        band_mask = np.abs(yr - y_med) <= row_band
        xr_band = xr[band_mask]

        if len(xr_band) < 2:
            # fallback to full row points
            xr_band = xr

        # histogram along x'
        xmin, xmax = xr_band.min(), xr_band.max()
        if xmax - xmin < xbin*3:
            continue
        nbins = int(np.ceil((xmax - xmin) / xbin))
        bins = np.linspace(xmin, xmax, nbins+1)
        counts, edges = np.histogram(xr_band, bins=bins)
        centers = 0.5*(edges[:-1]+edges[1:])

        # smooth histogram to form "stem signal"
        counts_smooth = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)

        # detect peaks
        peaks_idx, _ = find_peaks(counts_smooth,
                                  distance=min_dist_bins,
                                  prominence=peak_prominence)
        plants = int(len(peaks_idx))

        # row length in x'
        row_length = float(xmax - xmin)
        linear_density = (plants / row_length) if row_length > 0 else 0.0

        # save per-row hist figure for first few rows
        if rid < example_rows_to_plot:
            peak_xs = centers[peaks_idx] if plants > 0 else np.array([])
            plot_row_hist(centers, counts_smooth, peak_xs,
                          title=f"Row {rid}: 1D hist & peaks",
                          out_png=f"step5c_row{rid:02d}_hist.png")

        results.append({
            "row_index": int(rid),
            "row_length_m": row_length,
            "plants_count": plants,
            "linear_density_stems_per_m": linear_density
        })

    # sort rows by index
    results = sorted(results, key=lambda d: d["row_index"])

    # compute averages
    avg_lin = float(np.mean([r["linear_density_stems_per_m"] for r in results])) if results else 0.0
    avg_plants_m2 = (avg_lin / row_spacing) if row_spacing > 0 else 0.0

    # quick overview scatter for sanity
    # (reuse your row scatter but label-only; not necessary to draw peaks there)
    # we keep your existing plot_rows for DBSCAN; for peaks, scatter per-row is enough
    # If you want: you can still call plot_xy on the kept points already saved.

    meta = {"avg_linear_density_stems_per_m": avg_lin,
            "avg_density_plants_per_m2": avg_plants_m2,
            "row_spacing_m": row_spacing}
    return results, meta

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las_path", help=".las path (units in meters expected)")
    ap.add_argument("--voxel", type=float, default=0.03, help="voxel size (m)")
    ap.add_argument("--nb_neighbors", type=int, default=20, help="denoise neighbors")
    ap.add_argument("--std_ratio", type=float, default=2.0, help="denoise std ratio")
    ap.add_argument("--ransac_dist", type=float, default=0.02, help="RANSAC thresh (m)")
    ap.add_argument("--ground_band", type=float, default=0.10, help="remove |dist|<=band (m)")
    ap.add_argument("--grid_cell", type=float, default=0.5, help="grid cell (m)")
    ap.add_argument("--kde_bw", type=float, default=0.4, help="KDE bandwidth (m)")
    ap.add_argument("--kde_res", type=float, default=0.25, help="KDE grid (m)")
    ap.add_argument("--row_spacing", type=float, default=0.75, help="row spacing prior (m)")
    ap.add_argument("--dbscan_eps", type=float, default=0.10, help="DBSCAN eps (m)")
    ap.add_argument("--dbscan_min", type=int, default=20, help="DBSCAN min samples")
    ap.add_argument("--row_height_min", type=float, default=None, help="min height above low z (m)")
    ap.add_argument("--save_intermediate_ply", action="store_true", help="save PLYs")
    ap.add_argument("--row_band", type=float, default=0.12, help="half band in y' around row center (m)")
    ap.add_argument("--xbin", type=float, default=0.05, help="histogram bin along x' (m)")
    ap.add_argument("--smooth_sigma", type=float, default=1.5, help="gaussian sigma (in bins) for hist smoothing")
    ap.add_argument("--peak_min_spacing", type=float, default=0.18, help="min spacing between peaks (m)")
    ap.add_argument("--peak_prominence", type=float, default=5.0, help="peak prominence for find_peaks")

    args = ap.parse_args()

    # Step 1
    print ("[info] Step 1: Read LASs")
    raw = step1_read(args.las_path)
    if raw.size == 0:
        print("[error] no points read from LAS")
        return
    print(f"[info]   read {len(raw)} points")

    # Step 2
    print ("[info] Step 2: Denoise")
    den = step2_denoise(raw, voxel=args.voxel, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    if den.size == 0:    
        print("[error] no points left after denoising")
        return
    print(f"[info]   {len(den)} points after denoising")

    # Step 3
    print ("[info] Step 3: RANSAC plane")
    plane, ground_inliers, nonground_preband = ransac_plane(den, dist_thresh=args.ransac_dist)
    print(f"[info]   plane: {plane[0]:.4f}x + {plane[1]:.4f}y + {plane[2]:.4f}z + {plane[3]:.4f} = 0")
    print(f"[info]   {len(ground_inliers)} ground inliers, {len(nonground_preband)} non-ground pre-band")
    if nonground_preband.size == 0:
        print("[error] no non-ground points left after RANSAC")
        return  

    # Optionally save PLYs
    if args.save_intermediate_ply:
        o3d.io.write_point_cloud("step3_ground_inliers.ply", np_to_o3d(ground_inliers))
        o3d.io.write_point_cloud("step3_nonground_preband.ply", np_to_o3d(nonground_preband))

    # Step 4
    print ("[info] Step 4: Remove ground band")
    kept = step4_remove_ground_band(nonground_preband, plane, band=args.ground_band)
    print(f"[info]   {len(kept)} points kept after band removal")

    if args.save_intermediate_ply:
        o3d.io.write_point_cloud("step4_kept_after_band.ply", np_to_o3d(kept))

    if kept.size == 0:
        print("[error] no points left after ground band removal")
        return

    # Step 5A
    print ("[info] Step 5A: Grid density")
    grid, grid_meta = step5a_grid(kept, cell=args.grid_cell)


    # Step 5B
    print ("[info] Step 5B: KDE intensity")
    dens, kde_meta = step5b_kde(kept, bandwidth=args.kde_bw, res=args.kde_res)


    # Step 5C (peaks-based)
    print ("[info] Step 5C: Row-wise peak counting")
    row_stats, row_meta = step5c_row_density_peaks(
        kept,
        row_spacing=args.row_spacing,
        row_band=args.row_band,
        xbin=args.xbin,
        smooth_sigma=args.smooth_sigma,
        peak_min_spacing=args.peak_min_spacing,
        peak_prominence=args.peak_prominence,
        height_min=args.row_height_min,
        example_rows_to_plot=12
    )
    print(f"[info]   {len(row_stats)} rows detected")
    print(f"[info]   average linear density: {row_meta['avg_linear_density_stems_per_m']:.3f} stems/m")
    print(f"[info]   average plants per m^2: {row_meta['avg_density_plants_per_m2']:.3f} (using row_spacing={row_meta['row_spacing_m']} m)")
    print(f"[info]   row spacing used: {row_meta['row_spacing_m']} m")

    # Save summary
    summary = {
        "plane": {"a":plane[0], "b":plane[1], "c":plane[2], "d":plane[3]},
        "grid_meta": grid_meta,
        "kde_meta": kde_meta,
        "rows": row_stats,
        "avg_linear_density_stems_per_m": row_meta["avg_linear_density_stems_per_m"],
        "avg_density_plants_per_m2": row_meta["avg_density_plants_per_m2"],
        "row_spacing_m": row_meta["row_spacing_m"]
    }

    with open("summary_density.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[done] Figures saved:")
    print(" - step1_raw_xy.png")
    print(" - step2_denoised_xy.png")
    print(" - step3_ground_inliers_xy.png")
    print(" - step3_nonground_preband_xy.png")
    print(" - step4_removed_band_xy.png")
    print(" - step4_kept_after_band_xy.png")
    print(" - step5a_grid_density.png")
    print(" - step5b_kde_intensity.png")
    print(" - step5c_rows_clusters.png")
    print("Summary: summary_density.json")

if __name__ == "__main__":
    main()
