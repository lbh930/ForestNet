import numpy as np


def build_voxel_layout(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    minimum_normalized_height_meters: float,
    maximum_normalized_height_meters: float,
    voxel_size_x_meters: float,
    voxel_size_y_meters: float,
    voxel_size_z_meters: float,
) -> dict:
    x_origin = float(np.min(x_coordinates))
    y_origin = float(np.min(y_coordinates))
    z_origin = float(minimum_normalized_height_meters)

    number_of_voxels_x = int(np.floor((np.max(x_coordinates) - x_origin) / voxel_size_x_meters)) + 1
    number_of_voxels_y = int(np.floor((np.max(y_coordinates) - y_origin) / voxel_size_y_meters)) + 1
    number_of_voxels_z = int(
        np.floor(
            (maximum_normalized_height_meters - minimum_normalized_height_meters)
            / voxel_size_z_meters
        )
    ) + 1

    total_number_of_voxels = number_of_voxels_x * number_of_voxels_y * number_of_voxels_z
    if total_number_of_voxels > 200_000_000:
        raise ValueError(
            "Voxel grid is too large for this minimal implementation. "
            "Use larger voxel_size or smaller AOI."
        )

    return {
        "x_origin": x_origin,
        "y_origin": y_origin,
        "z_origin": z_origin,
        "voxel_size_x_meters": voxel_size_x_meters,
        "voxel_size_y_meters": voxel_size_y_meters,
        "voxel_size_z_meters": voxel_size_z_meters,
        "number_of_voxels_x": number_of_voxels_x,
        "number_of_voxels_y": number_of_voxels_y,
        "number_of_voxels_z": number_of_voxels_z,
    }


def build_voxel_point_count_grid(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    normalized_height: np.ndarray,
    voxel_layout: dict,
) -> np.ndarray:
    x_origin = voxel_layout["x_origin"]
    y_origin = voxel_layout["y_origin"]
    z_origin = voxel_layout["z_origin"]
    voxel_size_x_meters = voxel_layout["voxel_size_x_meters"]
    voxel_size_y_meters = voxel_layout["voxel_size_y_meters"]
    voxel_size_z_meters = voxel_layout["voxel_size_z_meters"]
    number_of_voxels_x = voxel_layout["number_of_voxels_x"]
    number_of_voxels_y = voxel_layout["number_of_voxels_y"]
    number_of_voxels_z = voxel_layout["number_of_voxels_z"]

    x_voxel_index = np.floor((x_coordinates - x_origin) / voxel_size_x_meters).astype(np.int64)
    y_voxel_index = np.floor((y_coordinates - y_origin) / voxel_size_y_meters).astype(np.int64)
    z_voxel_index = np.floor((normalized_height - z_origin) / voxel_size_z_meters).astype(np.int64)

    point_mask = (
        (x_voxel_index >= 0)
        & (x_voxel_index < number_of_voxels_x)
        & (y_voxel_index >= 0)
        & (y_voxel_index < number_of_voxels_y)
        & (z_voxel_index >= 0)
        & (z_voxel_index < number_of_voxels_z)
    )

    x_voxel_index = x_voxel_index[point_mask]
    y_voxel_index = y_voxel_index[point_mask]
    z_voxel_index = z_voxel_index[point_mask]

    linear_voxel_index = z_voxel_index + number_of_voxels_z * (
        y_voxel_index + number_of_voxels_y * x_voxel_index
    )
    voxel_count_flat = np.bincount(
        linear_voxel_index,
        minlength=number_of_voxels_x * number_of_voxels_y * number_of_voxels_z,
    ).astype(np.uint32)

    return voxel_count_flat.reshape(
        number_of_voxels_x,
        number_of_voxels_y,
        number_of_voxels_z,
    )


def compute_observed_density_points_per_cubic_meter(
    voxel_point_count_grid: np.ndarray,
    voxel_size_x_meters: float,
    voxel_size_y_meters: float,
    voxel_size_z_meters: float,
) -> np.ndarray:
    voxel_volume_cubic_meters = (
        voxel_size_x_meters * voxel_size_y_meters * voxel_size_z_meters
    )
    return voxel_point_count_grid.astype(np.float64) / voxel_volume_cubic_meters
