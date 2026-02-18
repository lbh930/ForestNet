import numpy as np

from src.beer_lambert_occlusion_corrector import compute_beer_lambert_corrected_density
from src.vertical_profile_builder import aggregate_vertical_profile, compute_shape_normalized_profile
from src.voxel_density_builder import (
    build_voxel_point_count_grid,
    compute_observed_density_points_per_cubic_meter,
)


def run_subsampling_stability_check(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    normalized_height: np.ndarray,
    voxel_layout: dict,
    profile_height_bin_size_meters: float,
    occupancy_point_count_threshold: int,
    occlusion_config: dict,
    full_corrected_profile_values: np.ndarray,
    subsampling_keep_ratio: float,
    subsampling_random_seed: int,
) -> dict:
    random_number_generator = np.random.default_rng(subsampling_random_seed)
    point_mask = random_number_generator.random(normalized_height.shape[0]) < subsampling_keep_ratio

    if not np.any(point_mask):
        return {
            "subsampling_point_count": 0,
            "profile_pearson_correlation": None,
            "subsampled_corrected_profile_values": np.zeros_like(full_corrected_profile_values),
            "subsampled_shape_normalized_profile_values": np.zeros_like(full_corrected_profile_values),
        }

    voxel_point_count_grid = build_voxel_point_count_grid(
        x_coordinates[point_mask],
        y_coordinates[point_mask],
        normalized_height[point_mask],
        voxel_layout,
    )

    observed_density_points_per_cubic_meter = compute_observed_density_points_per_cubic_meter(
        voxel_point_count_grid,
        voxel_layout["voxel_size_x_meters"],
        voxel_layout["voxel_size_y_meters"],
        voxel_layout["voxel_size_z_meters"],
    )

    corrected_density_points_per_cubic_meter, _, _ = compute_beer_lambert_corrected_density(
        observed_density_points_per_cubic_meter,
        voxel_point_count_grid,
        occupancy_point_count_threshold,
        occlusion_config["vertical_processing_direction"],
        occlusion_config["extinction_coefficient_alpha"],
        occlusion_config["minimum_transmission_value"],
        occlusion_config["maximum_density_gain_factor"],
    )

    _, subsampled_corrected_profile_values = aggregate_vertical_profile(
        corrected_density_points_per_cubic_meter,
        voxel_layout["z_origin"],
        voxel_layout["voxel_size_z_meters"],
        profile_height_bin_size_meters,
    )

    full_shape_normalized_profile = compute_shape_normalized_profile(full_corrected_profile_values)
    subsampled_shape_normalized_profile = compute_shape_normalized_profile(subsampled_corrected_profile_values)

    if np.std(full_shape_normalized_profile) == 0 or np.std(subsampled_shape_normalized_profile) == 0:
        profile_pearson_correlation = None
    else:
        profile_pearson_correlation = float(
            np.corrcoef(full_shape_normalized_profile, subsampled_shape_normalized_profile)[0, 1]
        )

    return {
        "subsampling_point_count": int(np.sum(point_mask)),
        "profile_pearson_correlation": profile_pearson_correlation,
        "subsampled_corrected_profile_values": subsampled_corrected_profile_values,
        "subsampled_shape_normalized_profile_values": subsampled_shape_normalized_profile,
    }
