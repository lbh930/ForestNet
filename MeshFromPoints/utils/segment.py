import numpy as np, open3d as o3d


def preprocess(pc: o3d.geometry.PointCloud, voxel=0.03):
    n_raw = len(pc.points)
    pc = pc.voxel_down_sample(voxel)
    n_vox = len(pc.points)
    pc, _ = pc.remove_statistical_outlier(16, 1.5)
    n_clean = len(pc.points)
    print(f"[pre] denoise   raw {n_raw} → voxel {n_vox} → clean {n_clean}")
    return pc


def split_trunk_crown(pts: np.ndarray, base_xy: np.ndarray, r_est: float):
    z = pts[:, 2]
    slice_h, width_th = 0.5, 6.0 * r_est
    crown_start = None
    for z0 in np.arange(z.min(), z.max(), slice_h):
        idx = (z >= z0) & (z < z0 + slice_h)
        if idx.sum() < 20:
            continue
        sl = pts[idx]
        if max(sl[:, 0].ptp(), sl[:, 1].ptp()) > width_th:
            crown_start = z0
            break
    if crown_start is not None:
        print(f"[split] crown starts @ {crown_start:.2f} m")
        return z < crown_start
    rad = np.linalg.norm(pts[:, :2] - base_xy, axis=1)
    return rad <= r_est * 1.5