import numpy as np


def compute_beer_lambert_corrected_density(
    observed_density_points_per_cubic_meter: np.ndarray,
    voxel_point_count_grid: np.ndarray,
    occupancy_point_count_threshold: int,
    vertical_processing_direction: str,
    extinction_coefficient_alpha: float,
    minimum_transmission_value: float,
    maximum_density_gain_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if vertical_processing_direction != "top_to_bottom":
        raise ValueError(
            "This minimal implementation only supports vertical_processing_direction=top_to_bottom"
        )

    occupancy_grid = voxel_point_count_grid >= occupancy_point_count_threshold

    occupancy_grid_reversed = occupancy_grid[:, :, ::-1].astype(np.int32)
    cumulative_occupancy_reversed = np.cumsum(occupancy_grid_reversed, axis=2)
    cumulative_occupancy_above_reversed = cumulative_occupancy_reversed - occupancy_grid_reversed
    cumulative_occupancy_above = cumulative_occupancy_above_reversed[:, :, ::-1]

    transmission_grid = np.exp(-extinction_coefficient_alpha * cumulative_occupancy_above)
    correction_gain_grid = 1.0 / np.maximum(transmission_grid, minimum_transmission_value)
    correction_gain_grid = np.minimum(correction_gain_grid, maximum_density_gain_factor)
    corrected_density_points_per_cubic_meter = (
        observed_density_points_per_cubic_meter * correction_gain_grid
    )

    return (
        corrected_density_points_per_cubic_meter,
        correction_gain_grid,
        transmission_grid,
    )
