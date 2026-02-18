from pathlib import Path

import numpy as np


def load_point_cloud_from_file(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_config = config["input_data"]
    point_cloud_file_path = Path(input_config["point_cloud_file_path"])
    point_cloud_file_format = input_config["point_cloud_file_format"].lower()

    if point_cloud_file_format in {"las", "laz"}:
        import laspy

        las = laspy.read(point_cloud_file_path)
        x_coordinates = np.asarray(las.x, dtype=np.float64)
        y_coordinates = np.asarray(las.y, dtype=np.float64)
        z_coordinates = np.asarray(las.z, dtype=np.float64)
        return x_coordinates, y_coordinates, z_coordinates

    if point_cloud_file_format == "csv":
        csv_array = np.genfromtxt(
            point_cloud_file_path,
            delimiter=",",
            names=True,
            dtype=np.float64,
            encoding="utf-8",
        )
        x_column_name = input_config["csv_x_column_name"]
        y_column_name = input_config["csv_y_column_name"]
        z_column_name = input_config["csv_z_column_name"]
        x_coordinates = np.asarray(csv_array[x_column_name], dtype=np.float64)
        y_coordinates = np.asarray(csv_array[y_column_name], dtype=np.float64)
        z_coordinates = np.asarray(csv_array[z_column_name], dtype=np.float64)
        return x_coordinates, y_coordinates, z_coordinates

    raise ValueError(f"Unsupported point_cloud_file_format: {point_cloud_file_format}")


def filter_points_by_area_of_interest(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    area_of_interest_bounds_xy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if area_of_interest_bounds_xy is None:
        return x_coordinates, y_coordinates, z_coordinates

    x_min, x_max, y_min, y_max = area_of_interest_bounds_xy
    point_mask = (
        (x_coordinates >= x_min)
        & (x_coordinates <= x_max)
        & (y_coordinates >= y_min)
        & (y_coordinates <= y_max)
    )
    return x_coordinates[point_mask], y_coordinates[point_mask], z_coordinates[point_mask]
