import numpy as np
import math
import sys

import pyembree

def load_obj_to_arrays(obj_file_path):
    """
    Loads an OBJ file and returns two NumPy arrays:
      vertices: shape (N, 3)
      faces:    shape (M, 3), dtype = np.uint32
    This parser only handles lines of 'v' and 'f' (triangulated).
    """
    vertices_list = []
    faces_list = []
    
    with open(obj_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                # 'v x y z'
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices_list.append([x, y, z])
            elif parts[0] == 'f':
                # 'f i1 i2 i3' (or i1/.. i2/.. i3/..)
                idx = []
                for v_str in parts[1:]:
                    # take only the vertex index before '/'
                    val = v_str.split('/')[0]
                    idx.append(int(val) - 1)  # OBJ is 1-based
                if len(idx) == 3:
                    faces_list.append(idx)
                # If there's more than 3 indices, it should be triangulated first.
    
    vertices = np.array(vertices_list, dtype=np.float32)
    faces = np.array(faces_list, dtype=np.uint32)
    return vertices, faces


def rotation_matrix_from_ypr(yaw_deg, pitch_deg, roll_deg=0.0):
    """
    Generates a basic rotation matrix from yaw, pitch, and roll angles in degrees.
    Convention here is Z-Y-X (yaw around Z, pitch around Y, roll around X).
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # Rotation around Z (Yaw)
    Rz = np.array([
        [ math.cos(yaw), -math.sin(yaw), 0],
        [ math.sin(yaw),  math.cos(yaw), 0],
        [           0,             0,    1]
    ], dtype=np.float32)
    
    # Rotation around Y (Pitch)
    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [              0,  1,             0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=np.float32)
    
    # Rotation around X (Roll)
    Rx = np.array([
        [1,            0,             0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ], dtype=np.float32)
    
    return Rz @ Ry @ Rx


def generate_rays(camera_pos, yaw_deg, pitch_deg,
                  horizontal_fov_deg, vertical_fov_deg,
                  horizontal_samples, vertical_samples):
    """
    Generate a batch of ray origins and directions for a LiDAR-like scanner.
    
    Returns:
      origins:   shape (N, 3)
      directions: shape (N, 3)
    where N = horizontal_samples * vertical_samples.
    """
    # Base orientation
    base_rotation = rotation_matrix_from_ypr(yaw_deg, pitch_deg, 0.0)
    
    # Sweep horizontally from -H/2 to +H/2, and vertically from -V/2 to +V/2
    h_step = horizontal_fov_deg / float(horizontal_samples - 1)
    v_step = vertical_fov_deg   / float(vertical_samples   - 1)
    
    h_angles = [ -horizontal_fov_deg * 0.5 + i * h_step for i in range(horizontal_samples) ]
    v_angles = [ -vertical_fov_deg   * 0.5 + j * v_step for j in range(vertical_samples)   ]
    
    # We'll store all rays in a list, then convert to NumPy
    ray_origins = []
    ray_dirs = []
    
    for ha in h_angles:
        for va in v_angles:
            ha_rad = math.radians(ha)
            va_rad = math.radians(va)
            
            # Local direction in sensor coords:
            # We'll define a "forward" vector as -Z if pitch=0, etc.
            # This is the same logic as your naive code.
            dir_local_x =  math.cos(va_rad) * math.sin(ha_rad)
            dir_local_y =  math.cos(va_rad) * math.cos(ha_rad)
            dir_local_z = -math.sin(va_rad)
            
            direction_local = np.array([dir_local_x, dir_local_y, dir_local_z], dtype=np.float32)
            norm = np.linalg.norm(direction_local)
            if norm > 1e-8:
                direction_local /= norm
            
            # Transform to world space
            direction_world = base_rotation.dot(direction_local)
            
            ray_origins.append(camera_pos.astype(np.float32))
            ray_dirs.append(direction_world)
    
    ray_origins = np.array(ray_origins, dtype=np.float32)
    ray_dirs = np.array(ray_dirs, dtype=np.float32)
    return ray_origins, ray_dirs


def save_ply(points, ply_path):
    """
    Saves the given list of 3D points to a PLY file (ASCII format).
    Only x, y, z properties are saved.
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
    # 1) Load OBJ
    obj_path = "test_tree.obj"  # You can change this to your path
    vertices, faces = load_obj_to_arrays(obj_path)
    print(f"Loaded {len(vertices)} vertices, {len(faces)} faces from {obj_path}.")

    # 2) Create Embree Scene
    scene = rtc.EmbreeScene()
    # Add geometry (triangle mesh)
    # The add_triangles expects float32 arrays:
    geom_id = scene.add_triangles(vertices, faces)
    scene.commit()
    print("Embree BVH built and committed.")
    
    # 3) Define scanning parameters
    camera_position = np.array([0.0, 0.0, 30.0], dtype=np.float32)  # UAV at Z=30
    yaw_deg = 0.0
    pitch_deg = 30.0
    horizontal_fov_deg = 90.0
    vertical_fov_deg = 20.0
    horizontal_steps = 200
    vertical_steps = 50
    max_range = 200.0
    
    # 4) Generate all rays (origins + directions)
    origins, directions = generate_rays(
        camera_pos=camera_position,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        horizontal_samples=horizontal_steps,
        vertical_samples=vertical_steps
    )
    total_rays = len(origins)
    print(f"Generated {total_rays} rays.")
    
    # 5) Run intersection with Embree (batch)
    # scene.run expects shape (N,3) float32 arrays for origins & directions
    hits = scene.run(origins, directions)
    # hits is a dict with:
    # {
    #   't': distances,
    #   'geomID': geometry IDs,
    #   'primID': triangle IDs,
    #   'instID': instance IDs
    # }
    # If a ray fails to intersect anything, 'geomID' is -1, and t might be inf
    
    distances = hits['t']
    geomIDs = hits['geomID']
    
    # 6) For each ray that hits a face, compute the intersection point
    #    and filter out those beyond max_range
    hit_points = []
    for i in range(total_rays):
        if geomIDs[i] != -1:  # means it hit something
            t_val = distances[i]
            if t_val < max_range and not math.isinf(t_val):
                # intersection point = origin + t_val * direction
                px = origins[i][0] + t_val * directions[i][0]
                py = origins[i][1] + t_val * directions[i][1]
                pz = origins[i][2] + t_val * directions[i][2]
                hit_points.append((px, py, pz))
    
    print(f"Out of {total_rays} rays, {len(hit_points)} intersected within range {max_range}.")
    
    # 7) Save the intersection points as a PLY
    output_ply = "embree_scanned_points.ply"
    save_ply(hit_points, output_ply)
    print(f"Saved {len(hit_points)} points to {output_ply}.")


if __name__ == "__main__":
    main()
