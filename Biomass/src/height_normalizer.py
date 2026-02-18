import numpy as np


def normalize_heights(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    ground_elevation_for_points: np.ndarray,
    minimum_normalized_height_meters: float,
    maximum_normalized_height_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_height = z_coordinates - ground_elevation_for_points
    point_mask = (
        (normalized_height >= minimum_normalized_height_meters)
        & (normalized_height <= maximum_normalized_height_meters)
    )

    return (
        x_coordinates[point_mask],
        y_coordinates[point_mask],
        normalized_height[point_mask],
    )
