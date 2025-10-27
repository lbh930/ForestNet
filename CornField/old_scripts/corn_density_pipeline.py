# corn_density_pipeline.py
# Goal: minimal pipeline using height map peaks (no denoise, no ground removal).
# Steps:
# 1) Read LAS -> shift Z so min(Z)=0
# 2) Grid accumulation at fine resolution: per-cell sumZ and count
# 3) Raw mean-height map = sumZ / count
# 4) Fast "kernel" smoothing: Gaussian(sumZ) / Gaussian(count)
# 5) Peak detection on smoothed mean-height map (top-down)
# 6) plants/m^2 = #peaks / covered_area; save visuals + CSV + JSON
# 7) NEW: Corn height evaluation from peak heights (avg/median/p10/p90), histogram + CSV

import argparse
import json
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.linear_model import RANSACRegressor
from skimage.feature import blob_log
import numpy as np
from scipy.spatial import cKDTree

def blob_peaks(H, valid_mask, meta,
               min_radius_m=0.05, max_radius_m=0.20,
               threshold_rel=0.02,
               min_peak_dist_m=0.15,
               min_height_rel=0.2):
    """
    Detect approximately circular canopy blobs using Laplacian of Gaussian.
    Adds:
      - min_peak_dist_m: suppress peaks closer than this distance (m)
      - min_height_rel: ignore blobs below Hmax * this ratio
    Returns peaks_xy, peaks_rc, extra info.
    """
    res = meta["res"]
    sigma_min = min_radius_m / res / np.sqrt(2)
    sigma_max = max_radius_m / res / np.sqrt(2)
    n_sigma = 8

    # Work image (NaN -> 0)
    H_work = np.nan_to_num(H * valid_mask, nan=0.0)
    Hmax = np.nanmax(H_work)
    if not np.isfinite(Hmax) or Hmax <= 0:
        return np.empty((0, 2)), (np.array([]), np.array([])), {}

    # Mask out low-height regions
    low_thr = Hmax * min_height_rel
    H_masked = H_work.copy()
    H_masked[H_masked < low_thr] = 0.0

    # LoG blob detection
    blobs = blob_log(H_masked, min_sigma=sigma_min, max_sigma=sigma_max,
                     num_sigma=n_sigma, threshold=threshold_rel)

    if blobs.size == 0:
        return np.empty((0, 2)), (np.array([]), np.array([])), {}

    ys, xs, sigmas = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    px = meta["xmin"] + (xs + 0.5) * res
    py = meta["ymin"] + (ys + 0.5) * res
    peaks_xy = np.column_stack([px, py])
    radii_m = sigmas * np.sqrt(2) * res

    # 1️⃣ 过滤不合理半径
    ok = (radii_m >= min_radius_m) & (radii_m <= max_radius_m)
    peaks_xy = peaks_xy[ok]
    ys, xs = ys[ok].astype(int), xs[ok].astype(int)

    # 2️⃣ 按高度强度排序 (高的优先保留)
    heights = H[ys, xs]
    order = np.argsort(-heights)
    peaks_xy = peaks_xy[order]
    ys, xs = ys[order], xs[order]

    # 3️⃣ 最小间距过滤 (基于KDTree)
    if len(peaks_xy) > 1 and min_peak_dist_m > 0:
        tree = cKDTree(peaks_xy)
        keep_mask = np.ones(len(peaks_xy), dtype=bool)
        for i, p in enumerate(peaks_xy):
            if not keep_mask[i]:
                continue
            dists, idxs = tree.query(p, k=len(peaks_xy), distance_upper_bound=min_peak_dist_m)
            close_idxs = idxs[(dists < min_peak_dist_m) & (idxs != i)]
            keep_mask[close_idxs] = False
        peaks_xy = peaks_xy[keep_mask]
        ys, xs = ys[keep_mask], xs[keep_mask]

    blob_meta = {
        "avg_radius_m": float(np.mean(radii_m[ok])) if ok.any() else None,
        "Hmax_m": float(Hmax),
        "min_height_rel": float(min_height_rel),
        "min_height_abs_m": float(low_thr),
        "min_peak_dist_m": float(min_peak_dist_m)
    }

    return peaks_xy, (ys, xs), blob_meta


# -------------------- I/O --------------------

def read_las_xyz(path):
    """Read LAS to Nx3 float64 array."""
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    return pts

# -------------------- Grid + Meta --------------------

def make_grid_meta(points, res):
    """Compute XY bounds and grid shape for given resolution (meters per pixel)."""
    xy = points[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    # Expand tiny epsilon so max point falls inside last bin
    eps = 1e-9
    nx = max(1, int(np.ceil((xmax - xmin) / res)))
    ny = max(1, int(np.ceil((ymax - ymin) / res)))
    meta = {
        "xmin": float(xmin),
        "xmax": float(xmin + nx*res + eps),
        "ymin": float(ymin),
        "ymax": float(ymin + ny*res + eps),
        "res": float(res),
        "nx": nx,
        "ny": ny
    }
    return meta

def bin_points_maxz_count(points, meta):
    """Accumulate maxZ and counts per grid cell using integer indexing."""
    x = points[:, 0]; y = points[:, 1]; z = points[:, 2]
    res = meta["res"]; xmin = meta["xmin"]; ymin = meta["ymin"]
    nx, ny = meta["nx"], meta["ny"]

    ix = np.floor((x - xmin) / res).astype(np.int64)
    iy = np.floor((y - ymin) / res).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    count = np.zeros((ny, nx), dtype=np.float64)
    maxz  = np.full((ny, nx), -np.inf, dtype=np.float64)

    np.add.at(count, (iy, ix), 1.0)
    np.maximum.at(maxz, (iy, ix), z)

    return maxz, count

def bin_points_sumz_count(points, meta):
    """Accumulate sumZ and counts per grid cell using integer indexing (fast, robust)."""
    x = points[:, 0]; y = points[:, 1]; z = points[:, 2]
    res = meta["res"]; xmin = meta["xmin"]; ymin = meta["ymin"]
    nx, ny = meta["nx"], meta["ny"]

    ix = np.floor((x - xmin) / res).astype(np.int64)
    iy = np.floor((y - ymin) / res).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    count = np.zeros((ny, nx), dtype=np.float64)
    sumz  = np.zeros((ny, nx), dtype=np.float64)

    np.add.at(count, (iy, ix), 1.0)
    np.add.at(sumz,  (iy, ix), z)

    return sumz, count

# -------------------- Height Maps --------------------

def raw_max_height(maxz, count):
    """Raw per-cell max height; NaN where count==0."""
    H = np.full_like(maxz, np.nan, dtype=np.float64)
    mask = count > 0
    H[mask] = maxz[mask]
    return H, mask

def smoothed_max_height(maxz, count, sigma_pix):
    """
    Smoothed max height via normalized convolution:
    Hs = gaussian(maxz * mask) / gaussian(mask)
    This avoids diluting values in empty cells.
    """
    if sigma_pix < 0.5:
        sigma_pix = 0.5
    mask = (count > 0).astype(np.float64)
    vals = np.where(mask > 0, maxz, 0.0)
    gv = gaussian_filter(vals, sigma=sigma_pix, mode="nearest")
    gm = gaussian_filter(mask, sigma=sigma_pix, mode="nearest")
    Hs = np.full_like(gv, np.nan, dtype=np.float64)
    valid = gm > 1e-9
    Hs[valid] = gv[valid] / gm[valid]
    return Hs, valid, gv, gm

def raw_mean_height(sumz, count):
    """Raw per-cell mean height; NaN where count==0."""
    H = np.full_like(sumz, np.nan, dtype=np.float64)
    mask = count > 0
    H[mask] = sumz[mask] / count[mask]
    return H, mask

def smoothed_mean_height(sumz, count, sigma_pix):
    """
    Fast kernel-like smoothing:
    smoothed_mean = gaussian(sumz) / gaussian(count)
    Returns mean map and mask of valid pixels (where smoothed count is > tiny).
    """
    if sigma_pix < 0.5:
        sigma_pix = 0.5  # avoid degenerate kernels
    gz = gaussian_filter(sumz,  sigma=sigma_pix, mode="nearest")
    gc = gaussian_filter(count, sigma=sigma_pix, mode="nearest")
    Hs = np.full_like(gz, np.nan, dtype=np.float64)
    valid = gc > 1e-9
    Hs[valid] = gz[valid] / gc[valid]
    return Hs, valid, gz, gc

# -------------------- Ground Removal (RANSAC) --------------------

def remove_ground_ransac(P, sample_n=500000, residual_threshold=0.1,
                         out_sample_las="sampled_ransac_ground.las",
                         out_nonground_las="nonground_downsampled.las",
                         seed=42):
    """
    Detect and remove ground plane via RANSAC regression (color-preserving).
    - P: Nx3 or Nx6 [x,y,z,(r,g,b)]
    - out_sample_las: will contain ONLY sampled GROUND points (no non-ground)
    - out_nonground_las: downsampled NON-GROUND points
    """
    import numpy as np
    import laspy
    from sklearn.linear_model import RANSACRegressor

    np.random.seed(seed)
    N = len(P)
    has_color = (P.shape[1] >= 6)
    sample_n = min(sample_n, N)

    # ---- 1) Subsample for RANSAC fitting ----
    idx = np.random.choice(N, sample_n, replace=False)
    X = P[idx, :2]
    y = P[idx, 2]

    # ---- 2) Fit plane with RANSAC ----
    ransac = RANSACRegressor(
        min_samples=0.3,
        residual_threshold=residual_threshold,
        max_trials=100,
        random_state=seed
    )
    ransac.fit(X, y)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # ---- 3) All-point inlier mask (ground) ----
    z_pred = a * P[:, 0] + b * P[:, 1] + c
    residuals = np.abs(P[:, 2] - z_pred)
    inliers = residuals < residual_threshold  # True = ground

    print(f"[info]   ground plane: z = {a:.4f}*x + {b:.4f}*y + {c:.4f}")
    print(f"[info]   ground inliers: {inliers.sum()}/{len(P)} "
          f"({inliers.sum()/len(P)*100:.1f}%)")

    # ---- helper: build LAS with optional RGB ----
    def make_las(points, classifications):
        header = laspy.LasHeader(version="1.2", point_format=3)  # supports RGB
        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.classification = classifications
        if has_color:
            rgb = np.clip(points[:, 3:6], 0, 65535).astype(np.uint16)
            las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        return las

    # ---- 4) SAVE *ONLY* SAMPLED GROUND POINTS ----
    sampled_pts = P[idx]
    sampled_pred_z = a * sampled_pts[:, 0] + b * sampled_pts[:, 1] + c
    sampled_residuals = np.abs(sampled_pts[:, 2] - sampled_pred_z)
    sampled_inliers = sampled_residuals < residual_threshold  # sampled ground mask

    sampled_ground = sampled_pts[sampled_inliers]
    if sampled_ground.shape[0] > 0:
        cls_ground = np.full(sampled_ground.shape[0], 2, dtype=np.uint8)  # ground=2
        make_las(sampled_ground, cls_ground).write(out_sample_las)
        print(f"[info]   saved sampled ground only: {sampled_ground.shape[0]} -> {out_sample_las}")
    else:
        print(f"[warn]   no sampled ground points; skip writing {out_sample_las}")

    # ---- 5) Remove ground & SAVE non-ground downsample ----
    P_out = P[~inliers]  # non-ground only
    print(f"[info]   remaining non-ground points: {len(P_out)}")

    keep_n = min(4000000, len(P_out))
    if keep_n > 0:
        keep_idx = np.random.choice(len(P_out), keep_n, replace=False)
        P_vis = P_out[keep_idx]
        cls_nonground = np.ones(keep_n, dtype=np.uint8)  # non-ground=1
        make_las(P_vis, cls_nonground).write(out_nonground_las)
        print(f"[info]   saved downsampled non-ground: {keep_n} -> {out_nonground_las}")

    return P_out, (a, b, c)





# -------------------- Visualization --------------------

def plot_heatmap(array, meta, title, out_png, cmap="viridis", vmin=None, vmax=None, dpi=300):
    """Save a georeferenced heatmap (XY extent), high DPI."""
    plt.figure(figsize=(8, 8), dpi=dpi)
    extent = [meta["xmin"], meta["xmax"], meta["ymin"], meta["ymax"]]
    plt.imshow(array, origin='lower', extent=extent, aspect='equal',
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    cbar = plt.colorbar()
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_peaks_on_height(H, meta, peaks_xy, title, out_png, dpi=320):
    """Overlay peak points on the height map."""
    plt.figure(figsize=(8, 8), dpi=dpi)
    extent = [meta["xmin"], meta["xmax"], meta["ymin"], meta["ymax"]]
    plt.imshow(H, origin='lower', extent=extent, aspect='equal', cmap="viridis")
    if len(peaks_xy):
        plt.scatter(peaks_xy[:,0], peaks_xy[:,1], s=2, marker='x', c='w', linewidths=0.2)
    plt.title(title)
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    cbar = plt.colorbar(); cbar.set_label("Height (m)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_height_histogram(heights, out_png, dpi=300, bins=30):
    """Plot and save histogram of per-plant heights."""
    plt.figure(figsize=(7,5), dpi=dpi)
    plt.hist(heights[~np.isnan(heights)], bins=bins)
    plt.xlabel("Plant height (m)")
    plt.ylabel("Count")
    plt.title("Corn height distribution")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# -------------------- Peak Detection --------------------

def local_max_peaks(H, valid_mask, meta, min_peak_distance_m=0.18,
                    rel_threshold=0.1, q_threshold=None, min_smoothed_count=0.5):
    """
    Find local maxima (NMS) on height map.
    - min_peak_distance_m: NMS window in meters (typical plant spacing ~0.15-0.30)
    - rel_threshold: keep peaks with H >= rel_threshold * max(H in valid)
    - q_threshold: optional percentile (0..1), e.g., 0.6; final thr = max(rel_thr, q_thr)
    - min_smoothed_count: require smoothed count >= this to avoid spurious peaks

    Returns peaks_xy, peaks_rc, and thresholds used.
    """
    res = meta["res"]
    # prepare working mask and values
    H_work = H.copy()
    H_work[~valid_mask] = np.nan

    # global stats over valid region
    H_valid = H_work[np.isfinite(H_work)]
    if H_valid.size == 0:
        return np.empty((0,2)), (np.array([]), np.array([])), {"thr": np.nan, "win_px": 0}

    Hmax = float(np.nanmax(H_valid))
    thr_rel = rel_threshold * Hmax
    thr_q = -np.inf
    if q_threshold is not None:
        thr_q = float(np.nanquantile(H_valid, q_threshold))
    thr = max(thr_rel, thr_q)

    # NMS window in pixels
    win_px = max(3, int(round(min_peak_distance_m / max(res, 1e-9))))

    # maximum_filter ignores NaN => temporarily fill with -inf
    H_filled = H_work.copy()
    H_filled[~np.isfinite(H_filled)] = -np.inf
    max_f = maximum_filter(H_filled, size=win_px, mode="nearest")
    is_peak = (H_filled == max_f)

    # apply thresholds
    is_peak &= (H_filled >= thr)

    ys, xs = np.where(is_peak)

    # convert (row,col) -> world XY
    xmin, ymin, res = meta["xmin"], meta["ymin"], meta["res"]
    px = xmin + (xs + 0.5) * res
    py = ymin + (ys + 0.5) * res
    peaks_xy = np.vstack([px, py]).T

    return peaks_xy, (ys, xs), {"thr": thr, "win_px": win_px, "Hmax": Hmax}

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las_path", help=".las path (units in meters expected)")
    # Grid + smoothing
    ap.add_argument("--res", type=float, default=0.05, help="grid resolution (m/pixel)")
    ap.add_argument("--bw", type=float, default=0.15, help="Gaussian bandwidth (m) for smoothing (converted to sigma in pixels)")
    # Peak detection
    ap.add_argument("--min_peak_dist", type=float, default=0.18, help="minimum peak spacing (m)")
    ap.add_argument("--thr_rel", type=float, default=0.25, help="relative threshold (0..1 of max height)")
    ap.add_argument("--thr_q", type=float, default=None, help="optional quantile threshold (0..1), e.g., 0.6")
    # Output options
    ap.add_argument("--fig_dpi", type=int, default=320, help="PNG figure DPI")
    ap.add_argument("--save_peaks_csv", action="store_true", help="save peaks as CSV")

    args = ap.parse_args()

    # 1) Read
    print("[info] Step 1: Read LAS")
    P = read_las_xyz(args.las_path)
    if P.size == 0:
        print("[error] no points read from LAS")
        return
    print(f"[info]   points: {len(P)}")
    # Shift Z so min=0
    zmin = float(P[:,2].min())
    P[:,2] -= zmin
    print(f"[info]   z-shift applied: minZ -> 0. (original minZ={zmin:.3f})")

    # --- Ground removal ---
    #print("[info] Step 1.5: RANSAC ground removal")
    #P, plane = remove_ground_ransac(P)


    # 2) Grid + accumulation (maxZ + count)
    print("[info] Step 2: Grid accumulation (maxZ + count)")
    meta = make_grid_meta(P, args.res)
    print(f"[info]   grid: {meta['ny']} x {meta['nx']}  res={meta['res']} m")
    print(f"[info]   extent X[{meta['xmin']:.2f}, {meta['xmax']:.2f}], Y[{meta['ymin']:.2f}, {meta['ymax']:.2f}]")
    maxz, count = bin_points_maxz_count(P, meta)
    covered_cells = int((count > 0).sum())
    coverage = covered_cells / (meta["nx"] * meta["ny"]) * 100.0
    print(f"[info]   covered cells: {covered_cells} ({coverage:.1f}%)")

    # 3) Raw max-height map
    print("[info] Step 3: Raw max-height map")
    H_raw, mask_raw = raw_max_height(maxz, count)
    vmax_raw = float(np.nanquantile(H_raw, 0.98)) if np.isfinite(H_raw).any() else None
    plot_heatmap(H_raw, meta, "Step3: Raw max height (m)", "step3_max_height_raw.png",
                 vmin=0.0, vmax=vmax_raw, dpi=args.fig_dpi)

    # 4) Smoothed max-height via normalized convolution
    print("[info] Step 4: Smoothed max-height via Gaussian(normalized)")
    sigma_pix = max(0.5, args.bw / max(meta["res"], 1e-9))
    print(f"[info]   smoothing: bandwidth={args.bw} m  => sigma={sigma_pix:.2f} px")
    H_smooth, mask_smooth, gv, gm = smoothed_max_height(maxz, count, sigma_pix)
    vmax_sm = float(np.nanquantile(H_smooth, 0.98)) if np.isfinite(H_smooth).any() else None
    plot_heatmap(H_smooth, meta, "Step4: Smoothed max height (m)", "step4_max_height_smooth.png",
                 vmin=0.0, vmax=vmax_sm, dpi=args.fig_dpi)

    '''
    # 5) Peak detection on height map
    print("[info] Step 5: Peak detection on smoothed height map")
    peaks_xy, (ys, xs), thr_meta = local_max_peaks(
        H_smooth, mask_smooth, meta,
        min_peak_distance_m=args.min_peak_dist,
        rel_threshold=args.thr_rel,
        q_threshold=args.thr_q
    )
    print(f"[info]   NMS window: {thr_meta['win_px']} px  (~{thr_meta['win_px']*meta['res']:.2f} m)")
    print(f"[info]   Hmax(valid)={thr_meta['Hmax']:.3f} m  threshold={thr_meta['thr']:.3f} m")
    print(f"[info]   peaks detected: {len(peaks_xy)}")
    '''

    peaks_xy, (ys, xs), blob_meta = blob_peaks(
        H_smooth, mask_smooth, meta,
        min_radius_m=0.06,  # 冠幅下限
        max_radius_m=0.25,  # 冠幅上限
        threshold_rel=0.03,
        min_peak_dist_m=args.min_peak_dist,
        min_height_rel=args.thr_rel  # 冠高下限

    )
    print(f"[info]   blob-based peaks detected: {len(peaks_xy)} "
        f"(avg radius={blob_meta.get('avg_radius_m'):.3f} m)")


    # Visualization with peaks
    plot_peaks_on_height(H_smooth, meta, peaks_xy,
                         title=f"Step5: Height peaks (bw={args.bw}m, minDist={args.min_peak_dist}m)",
                         out_png="step5_height_peaks.png", dpi=args.fig_dpi)

    # 6) Plants per m^2 from peaks and covered area
    area_m2 = covered_cells * (meta["res"] ** 2)
    plants_per_m2 = (len(peaks_xy) / area_m2) if area_m2 > 0 else 0.0
    print(f"[info] Step 6: Density estimate")
    print(f"[info]   covered area: {area_m2:.2f} m^2")
    print(f"[info]   plants per m^2: {plants_per_m2:.3f}")

    # -------- NEW: Corn height evaluation (per-plant) --------
    print("[info] Step 7: Corn height evaluation (per-plant stats)")
    if len(peaks_xy):
        peak_heights = H_smooth[ys, xs]  # height proxy at plant peaks
        # Basic stats
        h_valid = peak_heights[np.isfinite(peak_heights)]
        if h_valid.size > 0:
            h_mean   = float(np.mean(h_valid))
            h_std    = float(np.std(h_valid, ddof=0))
            h_median = float(np.median(h_valid))
            h_p10    = float(np.quantile(h_valid, 0.10))
            h_p90    = float(np.quantile(h_valid, 0.90))
            print(f"[info]   plants used: {h_valid.size}")
            print(f"[info]   avg height: {h_mean:.3f} m  (median={h_median:.3f} m, P10={h_p10:.3f}, P90={h_p90:.3f})")
            # Save histogram + CSV
            hist_png = "corn_height_hist.png"
            plot_height_histogram(h_valid, hist_png, dpi=300, bins=30)
            heights_csv = "corn_heights.csv"
            # Save positions + heights for downstream QA
            out_ph = np.column_stack([peaks_xy, peak_heights])
            np.savetxt(heights_csv, out_ph, delimiter=",", header="x_m,y_m,height_m", comments="")
        else:
            h_mean = h_std = h_median = h_p10 = h_p90 = None
            hist_png = None
            heights_csv = None
            print("[warn]   no valid peak heights to evaluate.")
    else:
        peak_heights = np.array([])
        h_mean = h_std = h_median = h_p10 = h_p90 = None
        hist_png = None
        heights_csv = None
        print("[warn]   no peaks detected; skip height evaluation.")

    # Optionally save peaks (legacy switch keeps behavior)
    peaks_file = None
    if args.save_peaks_csv and len(peaks_xy):
        peaks_file = "peaks_xy.csv"
        out = np.column_stack([peaks_xy, H_smooth[ys, xs]])
        np.savetxt(peaks_file, out, delimiter=",", header="x_m,y_m,height_m", comments="")
        print(f"[info]   saved peaks to: {peaks_file}")

    # 8) Save summary JSON
    summary = {
        "grid_meta": meta,
        "params": {
            "res": args.res,
            "bandwidth_m": args.bw,
            "min_peak_dist_m": args.min_peak_dist,
            "thr_rel": args.thr_rel,
            "thr_q": args.thr_q
        },
        "stats": {
            "points": int(len(P)),
            "z_shift_min_applied": float(zmin),
            "covered_cells": covered_cells,
            "covered_area_m2": float(area_m2),
            "peaks_count": int(len(peaks_xy)),
            "plants_per_m2": float(plants_per_m2),
            # new blob-based metadata
            "Hmax_valid_m": None,
            "threshold_used_m": None,
            "blob_avg_radius_m": blob_meta.get("avg_radius_m"),
            "corn_height": {
                "n_peaks_used": int(len(peaks_xy)),
                "avg_m": h_mean,
                "std_m": h_std,
                "median_m": h_median,
                "p10_m": h_p10,
                "p90_m": h_p90,
                "method": "sample H_smooth at blob centers (LoG peaks)"
            }
        },
        "outputs": {
            "max_height_raw_png": "step3_max_height_raw.png",
            "max_height_smooth_png": "step4_max_height_smooth.png",
            "height_peaks_png": "step5_height_peaks.png",
            "peaks_csv": peaks_file,
            # NEW exports
            "corn_height_hist_png": hist_png,
            "corn_heights_csv": heights_csv
        }
    }
    with open("summary_height_peaks.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[done] Figures saved:")
    print(" - step3_max_height_raw.png")
    print(" - step4_max_height_smooth.png")
    print(" - step5_height_peaks.png")
    if hist_png:
        print(f" - {hist_png}")
    if peaks_file:
        print(f" - {peaks_file}")
    if heights_csv:
        print(f" - {heights_csv}")
    print("Summary: summary_height_peaks.json")

if __name__ == "__main__":
    main()
