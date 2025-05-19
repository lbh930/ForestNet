"""
Example:
    python build_forest.py --input cloud.laz --output forest.obj --voxel 0.03
"""
import argparse, time
from pathlib import Path
import numpy as np, trimesh

from utils.io_utils import read_las_points
from utils.mesh_tree import build_tree_mesh
from utils.export_stp import export_trees_to_step
tree_params = []  # [(base, r, h, ellip_radii), ...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".las/.laz file")
    ap.add_argument("--output", default="forest.obj", help="mesh path")
    ap.add_argument("--voxel", type=float, default=0.03)
    args = ap.parse_args()

    # ---------- load + global shift ----------------
    pts_all, las = read_las_points(args.input)
    min_z = pts_all[:, 2].min()
    cen_x = (pts_all[:, 0].min() + pts_all[:, 0].max()) * 0.5
    cen_y = (pts_all[:, 1].min() + pts_all[:, 1].max()) * 0.5

    shift = np.array([-cen_x, -cen_y, -min_z], np.float32)
    pts_all += shift
    print(
        f"[init] shift applied  dz={-min_z:.3f}  dxy=({-cen_x:.3f},{-cen_y:.3f}) → ground=0, centred"
    )

    # ---------- tree logic -------------------------
    if "treeID" not in las.point_format.dimension_names:
        raise SystemExit("[err] file lacks treeID dimension")

    tree_ids = las["treeID"]
    unique = np.unique(tree_ids)
    unique = unique[unique != 0]
    print(f"[info] {len(unique)} trees detected")

    scene = trimesh.Scene()
    for i, tid in enumerate(unique, 1):
        mask = tree_ids == tid
        meshes, param = build_tree_mesh(pts_all[mask], voxel=args.voxel)
        if param:
            tree_params.append(param)
        for m in meshes:
            scene.add_geometry(m)
            
        if i % 10 == 0:
            print(f"[progress] {i}/{len(unique)} done")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)
    print(f"[✓] saved {args.output}")
    
    # export to STEP (solid model) as well
    export_trees_to_step(tree_params, args.output.replace(".obj", ".stp"))
    print(f"[✓] saved {args.output.replace('.obj', '.stp')}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[done] elapsed {time.time()-t0:.1f}s")
