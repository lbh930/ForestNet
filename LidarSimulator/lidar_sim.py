import math
import numpy as np
import trimesh
import math
import numpy as np

from utils import generate_rays

try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    embree_loaded = True
except ImportError:
    from trimesh.ray.ray_triangle import RayMeshIntersector
    embree_loaded = False
    
if embree_loaded:
    print("Embree loaded! Using hardware-accelerated ray tracing.")
else:
    print("Falling back to pure Python ray tracing.")


def save_ply(points, ply_path):
    """
    Saves a list of 3D points to a PLY file (ASCII).
    Only x, y, z coordinates are saved.
    """
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    # 1) Load the mesh from an OBJ file
    #    'force="mesh"' tries to ensure we get a single mesh object
    obj_path = "test_tree.obj"
    mesh = trimesh.load(obj_path, force='mesh')

    if mesh.is_empty:
        print(f"Failed to load a valid mesh from {obj_path}.")
        return

    print(f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces from {obj_path}.")
    
    # Print Mesh Model Boundings
    print("Mesh bounding box:", mesh.bounds)

    # 2) Create a RayMeshIntersector from trimesh
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    # 3) Define scanning parameters
    camera_position      = np.array([0.0, 0.0, 40.0], dtype=np.float32)
    camera_forward       = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Example direction
    horizontal_fov_deg   = 77.2 #70.4 for DJI Zenmuse L2
    vertical_fov_deg     = 77.2 #77.2 for DJI Zenmuse L2
    horizontal_steps     = 1024
    vertical_steps       = 1024
    max_range            = 200.0

    # 4) Generate all rays
    print("Generating rays...")
    origins, directions = generate_rays(
        camera_pos       = camera_position,
        camera_dir       = camera_forward,
        horizontal_fov_deg = horizontal_fov_deg,
        vertical_fov_deg   = vertical_fov_deg,
        horizontal_samples = horizontal_steps,
        vertical_samples   = vertical_steps
    )
    
    #print (origins)
    #print (directions)

    total_rays = len(origins)
    print(f"Generated {total_rays} rays in total.")

    # 5) Perform ray intersection in batch
    print("Running batch intersection...")
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    print(f"Total intersections: {len(locations)}")

    # 6) Filter out hits that exceed max_range
    #    We need to compute distances, because trimesh doesn't directly return 't'
    hit_points = []
    hits_count = 0

    for i, loc in enumerate(locations):
        ray_id = index_ray[i]
        dist   = np.linalg.norm(loc - origins[ray_id])
        if dist <= max_range:
            hit_points.append(loc)
            hits_count += 1
            if hits_count % 2000 == 0:
                print(f"Processed {hits_count} hits so far...")

    print(f"Out of {total_rays} rays, {len(hit_points)} are within range {max_range}.")

    # 7) Save to PLY
    output_ply = "trimesh_scanned_points.ply"
    save_ply(hit_points, output_ply)
    print(f"Saved {len(hit_points)} points to {output_ply}.")


if __name__ == "__main__":
    main()
