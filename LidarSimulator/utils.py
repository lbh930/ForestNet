
import numpy as np
import math
def rotation_matrix_from_ypr(yaw_deg, pitch_deg, roll_deg=0.0):
    """
    Builds a basic rotation matrix from yaw, pitch, and roll angles in degrees.
    Convention: Z-Y-X rotation (yaw around Z, pitch around Y, roll around X).
    This function is not used anymore if we rely on a forward direction vector.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    Rz = np.array([
        [ math.cos(yaw), -math.sin(yaw), 0],
        [ math.sin(yaw),  math.cos(yaw), 0],
        [           0,             0,    1]
    ], dtype=np.float32)

    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [              0,  1,             0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=np.float32)

    Rx = np.array([
        [1,            0,             0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx

def rotation_negz_to_dir(forward_vec):
    """
    Build a rotation matrix that sends local -Z axis to the given forward_vec in world space.
    -Z in local coordinates is considered 'forward' in sensor's frame.
    forward_vec is the desired forward direction in the world space.
    """
    # Normalize the target direction
    forward_norm = forward_vec / np.linalg.norm(forward_vec)

    # Local 'forward' is (0, 0, -1)
    local_negz = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # If the vectors are already close, return identity
    dot_val = np.dot(local_negz, forward_norm)
    if abs(dot_val + 1.0) < 1e-6:
        # local_negz == -forward_norm -> 180 deg rotation
        # We'll handle it in the axis-angle approach below
        pass
    elif np.allclose(local_negz, forward_norm, atol=1e-6):
        # They are almost the same direction
        return np.eye(3, dtype=np.float32)

    # Calculate rotation axis using cross product
    axis = np.cross(local_negz, forward_norm)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-8:
        # local_negz and forward_norm are nearly collinear but opposite
        # This implies 180-degree rotation around any perpendicular axis
        # e.g. rotate around X or Y
        # Choose X by default if Z is collinear
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis /= axis_len

    # Calculate rotation angle
    # dot_val = cos(theta)
    # local_negz dot forward_norm = cos(theta), but local_negz = [0,0,-1]
    # If forward_norm also points in -Z, angle ~ 0
    # If forward_norm points in +Z, angle ~ pi
    angle = math.acos(np.clip(np.dot(local_negz, forward_norm), -1.0, 1.0))

    # Build axis-angle rotation matrix (Rodrigues' rotation formula)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis

    R = np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c  ]
    ], dtype=np.float32)

    return R

def R_y(angle_rad: float) -> np.ndarray:
    """
    Rotation matrix around local Y axis by angle_rad.
    Positive angle means rotating to the left when looking forward.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ], dtype=np.float32)

def R_x(angle_rad: float) -> np.ndarray:
    """
    Rotation matrix around local X axis by angle_rad.
    Positive angle means rotating up or down depending on sign.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ], dtype=np.float32)

def rotation_negz_to_dir(forward_vec: np.ndarray) -> np.ndarray:
    """
    Build a rotation matrix that sends local -Z axis to the given forward_vec in world space.
    local_negz = [0, 0, -1].
    """
    forward_norm = forward_vec / np.linalg.norm(forward_vec)
    local_negz   = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    dot_val      = np.dot(local_negz, forward_norm)

    if np.allclose(local_negz, forward_norm, atol=1e-6):
        return np.eye(3, dtype=np.float32)

    axis = np.cross(local_negz, forward_norm)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis /= axis_len

    angle = math.acos(np.clip(dot_val, -1.0, 1.0))
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis

    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*y*x + s*z, t*y*y + c,   t*y*z - s*x],
        [t*z*x - s*y, t*z*y + s*x, t*z*z + c  ]
    ], dtype=np.float32)

    return R

def generate_rays(camera_pos: np.ndarray,
                  camera_dir: np.ndarray,
                  horizontal_fov_deg: float,
                  vertical_fov_deg: float,
                  horizontal_samples: int,
                  vertical_samples: int):
    """
    Generates a batch of ray origins and directions for a LIDAR-like scanner,
    based on a forward vector (camera_dir) in world space.

    Local coordinate system:
      - local forward = -Z
      - local horizontal sweep = rotation around Y
      - local vertical sweep = rotation around X

    Steps:
      1) For each horizontal angle ha in [-hFOV/2, +hFOV/2],
         rotate around Y.
      2) For each vertical angle va in [-vFOV/2, +vFOV/2],
         rotate around X.
      3) The base local direction before rotation is (0,0,-1).
      4) Then we apply a final rotation (base_rotation) which sends -Z to camera_dir in world space.

    Args:
        camera_pos: (3,) The sensor position in world space.
        camera_dir: (3,) The desired forward direction in world space.
        horizontal_fov_deg: total horizontal FOV in degrees.
        vertical_fov_deg:   total vertical FOV in degrees.
        horizontal_samples: number of rays horizontally.
        vertical_samples:   number of rays vertically.

    Returns:
        origins:    shape (N, 3) array of ray origins.
        directions: shape (N, 3) array of normalized ray directions.
    """
    # rotation that sends local -Z to camera_dir in world coords
    base_rotation = rotation_negz_to_dir(camera_dir)

    h_step = horizontal_fov_deg / float(horizontal_samples - 1)
    v_step = vertical_fov_deg   / float(vertical_samples   - 1)

    h_angles = [ -horizontal_fov_deg * 0.5 + i * h_step for i in range(horizontal_samples) ]
    v_angles = [ -vertical_fov_deg   * 0.5 + j * v_step for j in range(vertical_samples)   ]

    forward_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    ray_origins = []
    ray_dirs = []

    total_count = horizontal_samples * vertical_samples
    count = 0

    for ha in h_angles:
        for va in v_angles:
            # convert angles to radians
            ha_rad = math.radians(ha)
            va_rad = math.radians(va)

            # step1: rotate around Y by ha_rad
            # step2: rotate around X by va_rad
            # order matters: typically "horizontal first, then vertical" or vice versa
            # Here we do: R_y(ha) * R_x(va) * forward_local
            # so "horizontal sweep" is outer, "vertical sweep" is inner
            dir_local = R_y(ha_rad).dot(R_x(va_rad)).dot(forward_local)

            # apply final rotation to align sensor frame to world frame
            direction_world = base_rotation.dot(dir_local)
            direction_world /= np.linalg.norm(direction_world)

            ray_origins.append(camera_pos.astype(np.float32))
            ray_dirs.append(direction_world.astype(np.float32))

            count += 1
            if count % 2000 == 0:
                print(f"Ray generation progress: {count}/{total_count}")

    ray_origins = np.array(ray_origins, dtype=np.float32)
    ray_dirs    = np.array(ray_dirs, dtype=np.float32)

    return ray_origins, ray_dirs

