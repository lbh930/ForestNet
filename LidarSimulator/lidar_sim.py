import math
import argparse
import numpy as np
import trimesh
import random

try:
    # Prefer Embree if available (fast)
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    embree_loaded = True
except ImportError:
    from trimesh.ray.ray_triangle import RayMeshIntersector
    embree_loaded = False

try:
    import laspy
    las_loaded = True
except ImportError:
    las_loaded = False

from utils import generate_rays  # keeps original util

# runtime info -----------------------------------------------------------------
if embree_loaded:
    print("Embree loaded! Using hardware‑accelerated ray tracing.")
else:
    print("Falling back to pure Python ray tracer.")

if las_loaded:
    print("laspy loaded, will write LAS.")
else:
    print("laspy not found, will fall back to PLY.")

# helpers ----------------------------------------------------------------------

def save_ply(points, tree_ids, ply_path):
    """Write XYZ + treeID to a PLY file."""
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint treeID\n")
        f.write("end_header\n")
        for p, tid in zip(points, tree_ids):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {tid}\n")


def write_las(points, tree_ids, las_path):
    """Write to LAS if laspy is present."""
    header = laspy.header.LasHeader(point_format=3, version="1.4")
    
    tree_ids = tree_ids.astype(np.int16)

    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las["treeID"] = tree_ids
    las.write(las_path)
    
# ---------------------------------------------------------------------------
# group by the "g" mark in obj file
# ---------------------------------------------------------------------------
def _parse_face_groups(obj_path: str):
    """
    Return:
        face_to_gid : (N_faces,) np.int32
        n_groups    : int
    """
    mapping = []
    id_by_name = {}
    current_id = 0
    with open(obj_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("g "):
                name = line[2:].strip()
                # assign a new ID to the group name
                current_id = id_by_name.setdefault(name, len(id_by_name))
            elif line.startswith("f "):
                mapping.append(current_id)

    return np.asarray(mapping, dtype=np.int32), len(id_by_name)

def load_mesh_with_labels(obj_path: str):
    mesh = trimesh.load(
        obj_path,
        force="mesh",      
        process=False,
        maintain_order=True,
    )

    face_to_tree, n_tree = _parse_face_groups(obj_path)
    if n_tree == 0:
        # if no g group, all set to 0
        face_to_tree = np.zeros(len(mesh.faces), dtype=np.int32)
        n_tree = 1

    print(f"Found {n_tree} sub‑meshes (trees) via OBJ 'g' groups.")
    return mesh, face_to_tree

# grid origins -----------------------------------------------------------------
def generate_origins(bounds, spacing, height_offset):
    """Create an XY grid of scanner positions *inside* the mesh bounds."""
    (xmin, ymin, _), (xmax, ymax, zmax) = bounds
    z_scan = zmax + height_offset
    # first/last grid lines that stay inside the bounds
    x_start = math.ceil(xmin / spacing) * spacing          # >= xmin
    x_end   = math.floor(xmax / spacing) * spacing         # <= xmax
    y_start = math.ceil(ymin / spacing) * spacing          # >= ymin
    y_end   = math.floor(ymax / spacing) * spacing         # <= ymax
    # build grid
    xs = np.arange(x_start, x_end + 1e-6, spacing, dtype=np.float32)
    ys = np.arange(y_start, y_end + 1e-6, spacing, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    origins = np.stack([gx.ravel(),
                        gy.ravel(),
                        np.full(gx.size, z_scan, dtype=np.float32)], axis=1)
    #print (origins)
    return origins, z_scan


# main -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Airborne LiDAR simulator (multi‑ray)")
    p.add_argument("obj", help="input OBJ with per‑tree groups")
    p.add_argument("--spacing", type=float, default=2, help="grid spacing (m)")
    p.add_argument("--height", type=float, default=10.0, help="scan height above top (m)")
    p.add_argument("--h_fov", type=float, default=70, help="horizontal FOV (deg)")
    p.add_argument("--v_fov", type=float, default=70, help="vertical FOV (deg)")
    p.add_argument("--h_steps", type=int, default=200, help="horizontal samples")
    p.add_argument("--v_steps", type=int, default=150, help="vertical samples")
    p.add_argument("--out", default="sim_scan.las", help="output path")
    p.add_argument("--max_range", type=float, default=200.0, help="range filter (m)")
    args = p.parse_args()

    print("Loading mesh…")
    mesh, face_to_tree = load_mesh_with_labels(args.obj)
    if mesh.is_empty:
        print("Mesh empty, abort.")
        return
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print("Bounds:", mesh.bounds)

    print("Building intersector…")
    intersector = RayMeshIntersector(mesh)

    print("Generating origins…")
    origins_grid, z_scan = generate_origins(mesh.bounds, args.spacing, args.height)
    print(f"Scan height: {z_scan:.2f} m | origins: {len(origins_grid)}")

    print("Precomputing ray pattern…")
    _, dir_pattern = generate_rays(
        camera_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        camera_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        horizontal_fov_deg=args.h_fov,
        vertical_fov_deg=args.v_fov,
        horizontal_samples=args.h_steps,
        vertical_samples=args.v_steps,
    )
    rays_per_origin = len(dir_pattern)
    total_rays = len(origins_grid) * rays_per_origin
    print(f"origins: {len(origins_grid)} | rays/origin: {rays_per_origin} | total rays: {total_rays}")

    print("Tiling rays…")  # kept for compatibility, actual tiling is per‑origin now

    print("Intersecting rays…")
    # ---------------------------------------------------------------------
    # Modified strategy: cast rays per origin to keep memory low.
    # ---------------------------------------------------------------------
    pts, tids = [], []
    for oi, origin in enumerate(origins_grid):
        # one origin -> duplicate for all rays
        o_batch = np.repeat(origin[np.newaxis, :], rays_per_origin, axis=0)
        d_batch = dir_pattern  # already float32

        locs, idx_ray, idx_tri = intersector.intersects_location(
            ray_origins=o_batch, ray_directions=d_batch, multiple_hits=False
        )

        # filter + collect
        for loc, rid, tri in zip(locs, idx_ray, idx_tri):
            dist = np.linalg.norm(loc - o_batch[rid])
            if dist > args.max_range:
                continue  # skip out-of-range returns
            hit_prob = np.exp(-dist / 200.0)  # farther = less likely to hit
            if random.random() > hit_prob:
                continue  # simulate missed detection
            std_err = 0.00013 * dist  # 2cm error @150m according to Zenmuse L2 spec
            noise = np.random.normal(0, std_err, size=3)
            beam_div_h = 0.0002  # 0.2 mrad horizontal according to Zenmuse L2 spec
            beam_div_v = 0.0006  # 0.6 mrad vertical according to Zenmuse L2 spec
            spot_size_h = dist * beam_div_h
            spot_size_v = dist * beam_div_v
            spot_noise = np.array([
                np.random.normal(0, spot_size_h / 2),
                np.random.normal(0, spot_size_v / 2),
                0.0
            ])
            noisy_loc = loc + noise + spot_noise
            
            #print sample noise amount
            #if random.random() < 0.0001:
            #    #print total noise in cm
            #    noise = np.linalg.norm(noise + spot_noise) * 100
            #    print (f"Sample noise: {noise:.2f} cm")
                
            pts.append(noisy_loc)
            tids.append(face_to_tree[tri])



        # print progress every 10 origins
        if (oi + 1) % 10 == 0:
            print(f"Processed {oi + 1}/{len(origins_grid)} origins…")

    if not pts:
        print("No hits within range, abort.")
        return

    pts = np.asarray(pts, dtype=np.float32)
    tids = np.asarray(tids, dtype=np.uint32)
    print(f"Final points: {len(pts)}")

    if las_loaded and args.out.lower().endswith(".las"):
        print("Writing LAS…")
        write_las(pts, tids, args.out)
    else:
        if not las_loaded:
            print("laspy missing, switching to PLY.")
        out_path = args.out if args.out.lower().endswith(".ply") else args.out.rsplit('.', 1)[0] + '.ply'
        print("Writing PLY…")
        save_ply(pts, tids, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
