from collections import deque

import numpy as np


def compute_ground_elevation_grid_with_percentile(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    ground_grid_cell_size_meters: float,
    ground_elevation_percentile_value: float,
    minimum_points_required_per_ground_cell: int,
) -> tuple[np.ndarray, dict]:
    x_origin = float(np.min(x_coordinates))
    y_origin = float(np.min(y_coordinates))

    x_cell_index = np.floor((x_coordinates - x_origin) / ground_grid_cell_size_meters).astype(np.int64)
    y_cell_index = np.floor((y_coordinates - y_origin) / ground_grid_cell_size_meters).astype(np.int64)

    number_of_cells_x = int(np.max(x_cell_index)) + 1
    number_of_cells_y = int(np.max(y_cell_index)) + 1

    linear_cell_index = x_cell_index + number_of_cells_x * y_cell_index
    sorting_index = np.argsort(linear_cell_index)

    sorted_linear_cell_index = linear_cell_index[sorting_index]
    sorted_z_coordinates = z_coordinates[sorting_index]

    unique_linear_cell_index, first_occurrence_index, counts_per_cell = np.unique(
        sorted_linear_cell_index,
        return_index=True,
        return_counts=True,
    )

    ground_grid_flat = np.full(number_of_cells_x * number_of_cells_y, np.nan, dtype=np.float64)

    for cell_linear_index, start_index, point_count in zip(
        unique_linear_cell_index,
        first_occurrence_index,
        counts_per_cell,
        strict=False,
    ):
        if point_count < minimum_points_required_per_ground_cell:
            continue
        end_index = start_index + point_count
        cell_points_z = sorted_z_coordinates[start_index:end_index]
        ground_grid_flat[cell_linear_index] = np.percentile(
            cell_points_z,
            ground_elevation_percentile_value,
        )

    ground_grid = ground_grid_flat.reshape(number_of_cells_y, number_of_cells_x)

    ground_grid_metadata = {
        "x_origin": x_origin,
        "y_origin": y_origin,
        "ground_grid_cell_size_meters": ground_grid_cell_size_meters,
        "number_of_cells_x": number_of_cells_x,
        "number_of_cells_y": number_of_cells_y,
    }

    return ground_grid, ground_grid_metadata


def fill_empty_ground_grid_cells(
    ground_grid: np.ndarray,
    empty_ground_cell_fill_method_name: str,
) -> np.ndarray:
    valid_cell_mask = ~np.isnan(ground_grid)
    if not np.any(valid_cell_mask):
        raise ValueError("Ground grid has no valid cells. Check input data and ground parameters.")

    if empty_ground_cell_fill_method_name != "nearest_neighbor":
        raise ValueError(
            "Unsupported empty_ground_cell_fill_method_name: "
            f"{empty_ground_cell_fill_method_name}"
        )

    if np.all(valid_cell_mask):
        return ground_grid

    return fill_empty_ground_grid_cells_with_breadth_first_search(ground_grid)


def fill_empty_ground_grid_cells_with_breadth_first_search(
    ground_grid: np.ndarray,
) -> np.ndarray:
    filled_ground_grid = ground_grid.copy()
    visited_cell_mask = ~np.isnan(filled_ground_grid)

    queue = deque((int(row), int(column)) for row, column in np.argwhere(visited_cell_mask))
    neighbor_offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    number_of_rows, number_of_columns = filled_ground_grid.shape

    while queue:
        row, column = queue.popleft()
        cell_value = filled_ground_grid[row, column]
        for row_offset, column_offset in neighbor_offsets:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if neighbor_row < 0 or neighbor_row >= number_of_rows:
                continue
            if neighbor_column < 0 or neighbor_column >= number_of_columns:
                continue
            if visited_cell_mask[neighbor_row, neighbor_column]:
                continue
            filled_ground_grid[neighbor_row, neighbor_column] = cell_value
            visited_cell_mask[neighbor_row, neighbor_column] = True
            queue.append((neighbor_row, neighbor_column))

    return filled_ground_grid


def sample_ground_elevation_for_points(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    ground_grid: np.ndarray,
    ground_grid_metadata: dict,
) -> np.ndarray:
    x_origin = ground_grid_metadata["x_origin"]
    y_origin = ground_grid_metadata["y_origin"]
    ground_grid_cell_size_meters = ground_grid_metadata["ground_grid_cell_size_meters"]
    number_of_cells_x = ground_grid_metadata["number_of_cells_x"]
    number_of_cells_y = ground_grid_metadata["number_of_cells_y"]

    x_cell_index = np.floor((x_coordinates - x_origin) / ground_grid_cell_size_meters).astype(np.int64)
    y_cell_index = np.floor((y_coordinates - y_origin) / ground_grid_cell_size_meters).astype(np.int64)

    x_cell_index = np.clip(x_cell_index, 0, number_of_cells_x - 1)
    y_cell_index = np.clip(y_cell_index, 0, number_of_cells_y - 1)

    return ground_grid[y_cell_index, x_cell_index]
