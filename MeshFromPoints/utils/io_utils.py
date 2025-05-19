import laspy, open3d as o3d
import numpy as np


def read_las_points(path: str):
    """Return Nx3 xyz points + las object."""
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    print(f"[io] loaded {path} → {pts.shape[0]} pts")
    return pts, las


def pts_to_o3d(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc
