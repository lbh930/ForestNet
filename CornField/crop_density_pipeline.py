
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop Row & Density Estimator (UAV point cloud, .las)
----------------------------------------------------
Usage (quick start):
    python crop_density_pipeline.py input.las

Recommended (with parameters):
    python crop_density_pipeline.py input.las \
        --voxel_size 0.03 \
        --k_neighbors 30 \
        --grow_radius 0.15 \
        --angle_threshold_deg 18 \
        --slice_radius 0.08 \
        --peak_min_distance 0.18 \
        --min_height_fraction 0.25 \
        --peak_prominence 0.05 \
        --max_rows_to_plot 12

Outputs:
  - CSV: <input_basename>_row_density.csv
  - Plots in ./figs/: overview + per-row (centerline, slices, height profile, peaks)

Dependencies:
  numpy, scipy, scikit-learn, laspy, matplotlib
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import laspy
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------------
# Utility structures
# ------------------------
@dataclass
class Params:
    voxel_size: float = 0.03
    k_neighbors: int = 30
    grow_radius: float = 0.15
    angle_threshold_deg: float = 18.0
    slice_radius: float = 0.08
    peak_min_distance: float = 0.18
    min_height_fraction: float = 0.25
    peak_prominence: float = 0.05
    max_rows_to_plot: int = 12
    downsample_max_points: int = 2_000_000  # safety cap

@dataclass
class RowResult:
    row_id: int
    num_points: int
    row_length_m: float
    num_crops: int
    crops_per_meter: float
    plot_paths: Tuple[str, str]  # (xy_plot, profile_plot)


# ------------------------
# Helpers
# ------------------------
class Timer:
    def __init__(self, label: str):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"[info] {self.label} ...", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.start
        print(f"[info] {self.label} done in {dt:.2f}s", flush=True)


def load_las_points(las_path: str) -> np.ndarray:
    with Timer("Load LAS"):
        las = laspy.read(las_path)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        print(f"[info] points: {xyz.shape[0]:,}")
        return xyz


def simple_ground_filter(xyz: np.ndarray, ground_quantile: float = 0.05, margin: float = 0.05) -> np.ndarray:
    """Remove ground by cutting off the lowest Z quantile + margin (meters). Fast & simple."""
    with Timer("Simple ground filter"):
        z = xyz[:, 2]
        z_thresh = np.quantile(z, ground_quantile) + margin
        keep = z > z_thresh
        print(f"[info] ground z-thresh ~ {z_thresh:.3f} m; kept {keep.sum():,} / {len(z)} points")
        return xyz[keep]


def voxel_downsample_xy(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Grid/voxel downsample in XY only (preserve Z via average)."""
    if voxel_size <= 0:
        return xyz
    with Timer(f"XY voxel downsample (voxel={voxel_size}m)"):
        # Map to voxel indices in XY
        xy = xyz[:, :2]
        min_xy = xy.min(axis=0)
        ij = np.floor((xy - min_xy) / voxel_size).astype(np.int32)
        # Hash each cell
        keys = ij[:, 0].astype(np.int64) << 32 | (ij[:, 1].astype(np.int64) & 0xFFFFFFFF)
        # Aggregate per cell: average XYZ
        order = np.argsort(keys)
        keys_sorted = keys[order]
        xyz_sorted = xyz[order]

        unique_idx = np.r_[True, keys_sorted[1:] != keys_sorted[:-1]]
        cell_starts = np.where(unique_idx)[0]
        cell_ends = np.r_[cell_starts[1:], len(keys_sorted)]

        ds_points = []
        for s, e in zip(cell_starts, cell_ends):
            block = xyz_sorted[s:e]
            ds_points.append(block.mean(axis=0))
        ds = np.vstack(ds_points)
        print(f"[info] downsampled: {len(xyz):,} -> {len(ds):,}")
        return ds


def estimate_local_directions(xy: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Local 2D PCA direction for each point."""
    with Timer(f"Estimate local directions (k={k_neighbors})"):
        tree = cKDTree(xy)
        dists, idxs = tree.query(xy, k=min(k_neighbors, len(xy)))
        dirs = np.zeros_like(xy)
        for i in range(len(xy)):
            neighborhood = xy[idxs[i]]
            pca = PCA(n_components=2)
            pca.fit(neighborhood)
            v = pca.components_[0]  # principal axis in 2D
            # enforce sign consistency locally by aligning to previous if close, else keep
            dirs[i] = v
        return dirs


def region_grow_rows(xy: np.ndarray, dirs: np.ndarray, grow_radius: float, angle_threshold_deg: float) -> np.ndarray:
    """Direction-aware region growing: connect points within radius and similar direction."""
    with Timer(f"Region growing (radius={grow_radius}m, angle<={angle_threshold_deg}°)"):
        n = len(xy)
        tree = cKDTree(xy)
        visited = np.zeros(n, dtype=bool)
        labels = -np.ones(n, dtype=np.int32)
        current_label = 0
        cos_thresh = math.cos(math.radians(angle_threshold_deg))

        for seed in range(n):
            if visited[seed]: 
                continue
            stack = [seed]
            visited[seed] = True
            labels[seed] = current_label
            while stack:
                j = stack.pop()
                # neighbors within radius
                nbrs = tree.query_ball_point(xy[j], r=grow_radius)
                if len(nbrs) == 0:
                    continue
                # direction alignment (abs cosine similarity)
                vj = dirs[j] / (np.linalg.norm(dirs[j]) + 1e-9)
                nbrs = [k for k in nbrs if not visited[k] and abs(np.dot(vj, dirs[k] / (np.linalg.norm(dirs[k]) + 1e-9))) >= cos_thresh]
                for k in nbrs:
                    visited[k] = True
                    labels[k] = current_label
                    stack.append(k)
            current_label += 1

        # Remove tiny clusters (noise)
        counts = np.bincount(labels[labels >= 0])
        keep_mask = np.zeros_like(labels, dtype=bool)
        kept = 0
        for lbl, cnt in enumerate(counts):
            if cnt >= 100:  # min points per row
                keep_mask |= labels == lbl
                kept += 1
        labels_filtered = -np.ones_like(labels)
        remap = {}
        new_id = 0
        for i in range(len(labels)):
            if keep_mask[i]:
                old = labels[i]
                if old not in remap:
                    remap[old] = new_id
                    new_id += 1
                labels_filtered[i] = remap[old]
        print(f"[info] rows found (>=100 pts): {new_id}")
        return labels_filtered


def order_points_along_row(xy_row: np.ndarray, k_graph: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Order row points along a polyline using:
      - kNN graph in 2D
      - MST on the graph
      - endpoints = degree-1 nodes on MST
      - distances from one endpoint give an ordering
    Returns:
      ordered_points (M,2), order_indices (M,)
    """
    n = len(xy_row)
    if n <= 2:
        order = np.arange(n)
        return xy_row[order], order

    # Build kNN graph
    tree = cKDTree(xy_row)
    dists, idxs = tree.query(xy_row, k=min(k_graph+1, n))  # include self
    # build symmetric CSR matrix
    rows, cols, data = [], [], []
    for i in range(n):
        for j, dist in zip(idxs[i][1:], dists[i][1:]):  # skip self
            rows.append(i); cols.append(j); data.append(dist)
            rows.append(j); cols.append(i); data.append(dist)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))

    # MST
    mst = minimum_spanning_tree(G)
    mst = mst + mst.T  # make undirected

    # Degree and endpoints
    deg = np.array(mst.getnnz(axis=0)).flatten()
    end_candidates = np.where(deg == 1)[0]
    if len(end_candidates) < 2:
        # fallback: pick farthest pair
        # approximate by picking farthest from an arbitrary node
        d0, _ = shortest_path(mst, indices=[0], directed=False, return_predecessors=True)
        a = np.nanargmax(d0[0])
        da, _ = shortest_path(mst, indices=[a], directed=False, return_predecessors=True)
        b = np.nanargmax(da[0])
        start = int(a)
    else:
        start = int(end_candidates[0])

    # distances from start -> order
    dist_from_start, predecessors = shortest_path(mst, indices=[start], directed=False, return_predecessors=True)
    dist_from_start = dist_from_start[0]
    order = np.argsort(dist_from_start)
    ordered_points = xy_row[order]
    return ordered_points, order


def resample_polyline_along_arclength(polyline_xy: np.ndarray, num_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Resample polyline into N equally spaced points along arc-length."""
    if len(polyline_xy) < 2:
        return polyline_xy, np.array([0.0] * len(polyline_xy))
    seg = np.hypot(np.diff(polyline_xy[:, 0]), np.diff(polyline_xy[:, 1]))
    arc = np.r_[0.0, np.cumsum(seg)]
    total = arc[-1] if len(arc) > 0 else 0.0
    if total <= 0:
        return polyline_xy[:1], np.array([0.0])
    t = np.linspace(0, total, num_samples)
    # interpolate separately for x and y
    x = np.interp(t, arc, polyline_xy[:, 0])
    y = np.interp(t, arc, polyline_xy[:, 1])
    return np.column_stack([x, y]), t


def compute_height_profile(sample_xy: np.ndarray, all_xyz: np.ndarray, slice_radius: float) -> np.ndarray:
    """At each sample point, average Z of points within slice_radius."""
    tree = cKDTree(all_xyz[:, :2])
    heights = np.full(len(sample_xy), np.nan, dtype=float)
    for i, p in enumerate(sample_xy):
        idxs = tree.query_ball_point(p, r=slice_radius)
        if len(idxs) > 0:
            heights[i] = np.mean(all_xyz[idxs, 2])
    return heights


def detect_crops_from_height_profile(heights: np.ndarray, min_height_fraction: float,
                                     peak_min_distance: float, sampling_step_m: float,
                                     peak_prominence: float) -> Tuple[np.ndarray, dict]:
    """Find peaks as crops, with height threshold and spacing constraint."""
    h = np.copy(heights)
    # replace NaN by local interpolation
    if np.any(np.isnan(h)):
        n = len(h)
        xi = np.arange(n)
        good = ~np.isnan(h)
        if good.sum() >= 2:
            h = np.interp(xi, xi[good], h[good])
        else:
            h = np.nan_to_num(h, nan=0.0)

    if len(h) == 0:
        return np.array([], dtype=int), {}

    h_max = float(np.max(h)) if len(h) > 0 else 0.0
    height_thresh = h_max * max(0.0, min(1.0, min_height_fraction))
    # convert min distance meters to samples
    min_dist_samples = max(1, int(round(peak_min_distance / max(sampling_step_m, 1e-6))))
    peaks, props = find_peaks(h, height=height_thresh, distance=min_dist_samples, prominence=peak_prominence)
    return peaks, props


def save_overview_plot(xy: np.ndarray, labels: np.ndarray, out_path: str):
    plt.figure(figsize=(8, 6), dpi=150)
    if labels.max() >= 0:
        # draw by row id (matplotlib default colors)
        for lbl in range(labels.max() + 1):
            mask = labels == lbl
            plt.scatter(xy[mask, 0], xy[mask, 1], s=1, label=f"row {lbl}", alpha=0.8)
        plt.legend(loc='best', markerscale=4, fontsize=6)
    else:
        plt.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.8)
    plt.gca().set_aspect('equal', 'box')
    plt.title("Estimated Rows (XY)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_row_plots(row_id: int,
                   row_xy: np.ndarray,
                   ordered_xy: np.ndarray,
                   sampled_xy: np.ndarray,
                   heights: np.ndarray,
                   peaks_idx: np.ndarray,
                   out_xy_path: str,
                   out_profile_path: str):
    # XY plot (row points + ordered polyline + sampled points)
    plt.figure(figsize=(7, 6), dpi=150)
    plt.scatter(row_xy[:, 0], row_xy[:, 1], s=2, alpha=0.6, label="row points")
    if len(ordered_xy) > 0:
        plt.plot(ordered_xy[:, 0], ordered_xy[:, 1], lw=1.0, label="centerline (MST)")
    if len(sampled_xy) > 0:
        plt.scatter(sampled_xy[:, 0], sampled_xy[:, 1], s=6, alpha=0.9, label="slice points")
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"Row {row_id} - XY")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc='best', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_xy_path)
    plt.close()

    # Height profile with peaks
    plt.figure(figsize=(8, 4), dpi=150)
    x = np.arange(len(heights))
    plt.plot(x, heights, lw=1.2, label="height profile")
    if len(peaks_idx) > 0:
        plt.scatter(peaks_idx, heights[peaks_idx], s=14, label="detected crops")
    plt.title(f"Row {row_id} - Height Profile & Peaks")
    plt.xlabel("Sample index (along arc-length)")
    plt.ylabel("Mean Z (m)")
    plt.legend(loc='best', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_profile_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="UAV Crop Row & Density (LAS)")
    parser.add_argument("las_path", type=str, help=".las file path")
    parser.add_argument("--voxel_size", type=float, default=Params.voxel_size)
    parser.add_argument("--k_neighbors", type=int, default=Params.k_neighbors)
    parser.add_argument("--grow_radius", type=float, default=Params.grow_radius)
    parser.add_argument("--angle_threshold_deg", type=float, default=Params.angle_threshold_deg)
    parser.add_argument("--slice_radius", type=float, default=Params.slice_radius)
    parser.add_argument("--peak_min_distance", type=float, default=Params.peak_min_distance)
    parser.add_argument("--min_height_fraction", type=float, default=Params.min_height_fraction)
    parser.add_argument("--peak_prominence", type=float, default=Params.peak_prominence)
    parser.add_argument("--max_rows_to_plot", type=int, default=Params.max_rows_to_plot)
    parser.add_argument("--no_ground_filter", action="store_true", help="Skip simple ground removal")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    p = Params(
        voxel_size=args.voxel_size,
        k_neighbors=args.k_neighbors,
        grow_radius=args.grow_radius,
        angle_threshold_deg=args.angle_threshold_deg,
        slice_radius=args.slice_radius,
        peak_min_distance=args.peak_min_distance,
        min_height_fraction=args.min_height_fraction,
        peak_prominence=args.peak_prominence,
        max_rows_to_plot=args.max_rows_to_plot
    )

    las_path = args.las_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    xyz = load_las_points(las_path)

    # Safety downsample cap (random) to avoid pathological slowdowns
    if xyz.shape[0] > p.downsample_max_points:
        with Timer(f"Safety random downsample to {p.downsample_max_points:,} points"):
            idx = np.random.choice(xyz.shape[0], size=p.downsample_max_points, replace=False)
            xyz = xyz[idx]

    # 2) Simple ground removal (fast)
    if not args.no_ground_filter:
        xyz = simple_ground_filter(xyz, ground_quantile=0.05, margin=0.05)

    # 3) XY voxel downsample for speed
    xyz_ds = voxel_downsample_xy(xyz, p.voxel_size)
    xy_ds = xyz_ds[:, :2]

    # 4) Local directions
    dirs = estimate_local_directions(xy_ds, p.k_neighbors)

    # 5) Direction-aware region growing to get curvy rows
    labels = region_grow_rows(xy_ds, dirs, p.grow_radius, p.angle_threshold_deg)

    # Save overview
    overview_png = str((figs_dir / "rows_overview.png").resolve())
    save_overview_plot(xy_ds, labels, overview_png)

    # 6) For each row, build centerline via MST ordering, resample, slice height, detect peaks
    row_ids = np.unique(labels[labels >= 0])
    results: List[RowResult] = []
    csv_lines = ["row_id,num_points,row_length_m,num_crops,crops_per_meter"]
    plotted = 0

    for row_id in row_ids:
        row_mask = labels == row_id
        row_xy = xy_ds[row_mask]
        row_xyz = xyz_ds[row_mask]
        if len(row_xy) < 100:
            continue

        with Timer(f"Row {row_id}: centerline ordering via MST"):
            ordered_xy, order_idx = order_points_along_row(row_xy, k_graph=8)

        # Compute total row length
        if len(ordered_xy) >= 2:
            seg = np.hypot(np.diff(ordered_xy[:, 0]), np.diff(ordered_xy[:, 1]))
            row_length = float(np.sum(seg))
        else:
            row_length = 0.0

        # Resample along arc-length
        sample_xy, arc = resample_polyline_along_arclength(ordered_xy, num_samples=200)
        sampling_step_m = (arc[1] - arc[0]) if len(arc) >= 2 else 1e-6

        # Height profile from original (non-downsampled) neighbors for better Z
        heights = compute_height_profile(sample_xy, xyz, p.slice_radius)

        # Peak detection (crops)
        peaks_idx, props = detect_crops_from_height_profile(
            heights,
            min_height_fraction=p.min_height_fraction,
            peak_min_distance=p.peak_min_distance,
            sampling_step_m=sampling_step_m,
            peak_prominence=p.peak_prominence,
        )
        num_crops = int(len(peaks_idx))
        crops_per_meter = float(num_crops / max(row_length, 1e-6))

        # Save per-row plots (limit count)
        xy_plot_path = str((figs_dir / f"row_{row_id:03d}_xy.png").resolve())
        profile_plot_path = str((figs_dir / f"row_{row_id:03d}_profile.png").resolve())
        if plotted < p.max_rows_to_plot:
            save_row_plots(row_id, row_xy, ordered_xy, sample_xy, heights, peaks_idx, xy_plot_path, profile_plot_path)
            plotted += 1
        else:
            xy_plot_path, profile_plot_path = "", ""

        results.append(RowResult(
            row_id=row_id,
            num_points=int(len(row_xy)),
            row_length_m=row_length,
            num_crops=num_crops,
            crops_per_meter=crops_per_meter,
            plot_paths=(xy_plot_path, profile_plot_path),
        ))
        print(f"[info] Row {row_id}: points={len(row_xy)}, length={row_length:.2f} m, crops={num_crops}, density={crops_per_meter:.2f} crops/m")

        csv_lines.append(f"{row_id},{len(row_xy)},{row_length:.3f},{num_crops},{crops_per_meter:.6f}")

    # 7) Write CSV
    base = Path(las_path).stem
    csv_path = str((out_dir / f"{base}_row_density.csv").resolve())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # 8) Final summary
    n_rows = len(results)
    if n_rows == 0:
        print("[warn] No rows detected. Try relaxing thresholds or checking input.")
    else:
        mean_density = float(np.mean([r.crops_per_meter for r in results]))
        total_crops = int(np.sum([r.num_crops for r in results]))
        total_length = float(np.sum([r.row_length_m for r in results]))
        print("\n========== SUMMARY ==========")
        print(f"rows: {n_rows}")
        print(f"total length: {total_length:.1f} m")
        print(f"total crops: {total_crops}")
        print(f"mean density: {mean_density:.3f} crops/m")
        print(f"overview plot: {overview_png}")
        print(f"csv: {csv_path}")
        print("=============================\n")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=120)
    main()
