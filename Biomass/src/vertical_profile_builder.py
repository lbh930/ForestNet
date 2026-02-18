import numpy as np


def aggregate_vertical_profile(
    density_grid: np.ndarray,
    z_origin: float,
    voxel_size_z_meters: float,
    profile_height_bin_size_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    profile_at_voxel_resolution = np.sum(density_grid, axis=(0, 1))

    ratio_float = profile_height_bin_size_meters / voxel_size_z_meters
    ratio_int = int(round(ratio_float))

    if ratio_int <= 0 or not np.isclose(ratio_int * voxel_size_z_meters, profile_height_bin_size_meters):
        raise ValueError(
            "profile_height_bin_size_meters must be a positive integer multiple "
            "of voxel_size_z_meters in this minimal implementation."
        )

    if ratio_int == 1:
        profile_values = profile_at_voxel_resolution
    else:
        padding_length = (-profile_at_voxel_resolution.size) % ratio_int
        if padding_length > 0:
            profile_at_voxel_resolution = np.pad(
                profile_at_voxel_resolution,
                (0, padding_length),
                mode="constant",
            )
        profile_values = profile_at_voxel_resolution.reshape(-1, ratio_int).sum(axis=1)

    height_bin_centers = z_origin + (np.arange(profile_values.size) + 0.5) * profile_height_bin_size_meters
    return height_bin_centers, profile_values


def compute_shape_normalized_profile(profile_values: np.ndarray) -> np.ndarray:
    total_value = float(np.sum(profile_values))
    if total_value <= 0:
        return np.zeros_like(profile_values)
    return profile_values / total_value
