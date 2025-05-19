

from __future__ import annotations
import math
import numpy as np, trimesh, open3d as o3d
from .segment import preprocess, split_trunk_crown
from .allometry import est_dbh_radius
from .io_utils import pts_to_o3d


def fit_cylinder(pts: np.ndarray, distance=0.02):
    pc = pts_to_o3d(pts)
    try:
        m, inl = pc.segment_cylinder(distance, 0.6, 0.02, 0.3, 3, 5000)
    except AttributeError:
        return None, np.array([], int), np.inf
    proj = m.project_to_cylinder_coordinate(pts[inl])
    rmse = np.sqrt(((proj[:, 0] - m.radius) ** 2).mean())
    info = dict(radius=m.radius, axis=m.axis_vector / np.linalg.norm(m.axis_vector), point=m.axis_point)
    return info, inl, rmse


def ellipsoid_from_bbox(pts):
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) * 0.5, np.clip((hi - lo) * 0.5, 1e-3, None)


def ellipsoid_mesh(cen, radii, sub=3):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    m.apply_scale(radii)
    m.apply_translation(cen)
    return m


def build_tree_mesh(pts: np.ndarray, voxel=0.03):
    if pts.shape[0] < 100:
        return [], None

    # --- preprocess
    pc = preprocess(pts_to_o3d(pts), voxel)
    p = np.asarray(pc.points)
    z = p[:, 2]
    H = float(np.percentile(z, 98))

    # --- radius guess
    r_guess = est_dbh_radius(H)

    # --- split trunk / crown
    base_xy = p[:, :2].mean(0)
    trunk_mask = split_trunk_crown(p, base_xy, r_guess)
    trunk_pts, crown_pts = p[trunk_mask], p[~trunk_mask]

    # --- trunk fit
    cyl, _, rmse = fit_cylinder(trunk_pts) if trunk_pts.shape[0] > 500 else (None, None, np.inf)
    if cyl and rmse < 0.03:
        r0, axis = cyl["radius"], cyl["axis"]
        base = cyl["point"] - axis * cyl["point"][2]
    else:
        r0, axis = r_guess, np.array([0.0, 0.0, 1.0])
        base = np.hstack((base_xy, 0.0))

    # --- crown dims
    cen, radii = ellipsoid_from_bbox(crown_pts)

    # --- trunk mesh for preview
    h_tr = max(cen[2], 0.5)
    trunk_mesh = trimesh.creation.cylinder(r0, h_tr, sections=32)
    if np.linalg.norm(np.cross([0.0, 0.0, 1.0], axis)) > 1e-6:
        rot = trimesh.transformations.rotation_matrix(
            math.acos(np.clip(np.dot([0.0, 0.0, 1.0], axis), -1, 1)), np.cross([0.0, 0.0, 1.0], axis)
        )
        trunk_mesh.apply_transform(rot)
    trunk_mesh.apply_translation(base + axis * h_tr * 0.5)

    canopy_mesh = ellipsoid_mesh(cen, radii)

    # --- pack params for STEP solid
    param = (base, float(r0), float(h_tr), radii.astype(float))

    return [trunk_mesh, canopy_mesh], param
