from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import laspy
import numpy as np
import yaml
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from src.ground_surface_estimator import fill_empty_ground_grid_cells
from src.progress_logging import log_progress


@dataclass(frozen=True)
class GroundGridModel:
    ground_grid: np.ndarray
    x_origin: float
    y_origin: float
    cell_size_meters: float
    number_of_cells_x: int
    number_of_cells_y: int


@dataclass(frozen=True)
class StemEstimate:
    x_coordinate_meters: float
    y_coordinate_meters: float
    dbh_centimeters: float
    fit_rmse_meters: float
    fit_inlier_ratio: float
    slice_point_count: int
    assignment_radius_meters: float


@dataclass(frozen=True)
class StemFitSnapshot:
    tree_top_x_meters: float
    tree_top_y_meters: float
    seed_center_x_meters: float
    seed_center_y_meters: float
    fitted_center_x_meters: float
    fitted_center_y_meters: float
    fitted_radius_meters: float
    fit_rmse_meters: float
    fit_inlier_ratio: float
    fit_x_coordinates: np.ndarray
    fit_y_coordinates: np.ndarray


@dataclass
class DBHEstimatorParameters:
    chunk_size: int = 2_000_000
    max_chunks: int | None = None
    ground_grid_cell_size_meters: float = 0.5
    ground_valid_ratio_threshold: float = 0.02
    ground_smoothing_median_filter_size: int = 3
    ground_classification_value: int = 2
    chm_cell_size_meters: float = 0.40
    chm_gaussian_sigma: float = 1.0
    chm_local_max_window_size: int = 13
    chm_min_peak_height_meters: float = 4.0
    chm_min_peak_separation_meters: float = 1.5
    breast_height_meters: float = 1.3
    breast_height_window_meters: float = 0.35
    slice_downsample_xy_size_meters: float = 0.020
    minimum_slice_points_per_tree: int = 45
    top_to_slice_search_radius_meters: float = 4.0
    maximum_top_to_stem_offset_meters: float = 3.8
    local_stem_density_cell_size_meters: float = 0.08
    local_stem_min_cell_points: int = 3
    dbh_fit_radius_meters: float = 0.75
    minimum_points_for_dbh_fit: int = 8
    deduplication_min_stem_distance_meters: float = 1.0
    circle_ransac_iterations: int = 180
    circle_ransac_inlier_threshold_meters: float = 0.035
    circle_min_inlier_ratio: float = 0.30
    circle_max_rmse_meters: float = 0.07
    minimum_dbh_centimeters: float = 5.0
    maximum_dbh_centimeters: float = 150.0
    minimum_height_for_assignment_meters: float = 0.5
    maximum_tree_height_meters: float = 80.0
    height_assignment_base_radius_meters: float = 1.8
    height_assignment_dbh_multiplier: float = 5.5
    height_assignment_min_radius_meters: float = 1.4
    height_assignment_max_radius_meters: float = 4.8
    height_assignment_xy_downsample_meters: float = 0.18
    height_hist_bin_size_meters: float = 0.10
    height_percentile: float = 99.5
    minimum_points_for_height: int = 60
    random_seed: int = 42
    enable_diagnostic_plots: bool = True
    diagnostic_overview_max_points: int = 120_000
    diagnostic_fit_examples_count: int = 18
    diagnostic_points_per_fit_example: int = 320


def _load_estimator_parameters(configuration: dict) -> DBHEstimatorParameters:
    section = configuration.get("dbh_estimator", {})
    defaults = DBHEstimatorParameters()
    parameter_values = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    for key, value in section.items():
        if key in parameter_values:
            parameter_values[key] = value
    return DBHEstimatorParameters(**parameter_values)


def _sample_ground_elevation_for_points(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    ground_model: GroundGridModel,
) -> np.ndarray:
    x_cell_index = np.floor(
        (x_coordinates - ground_model.x_origin) / ground_model.cell_size_meters
    ).astype(np.int64)
    y_cell_index = np.floor(
        (y_coordinates - ground_model.y_origin) / ground_model.cell_size_meters
    ).astype(np.int64)
    x_cell_index = np.clip(x_cell_index, 0, ground_model.number_of_cells_x - 1)
    y_cell_index = np.clip(y_cell_index, 0, ground_model.number_of_cells_y - 1)
    return ground_model.ground_grid[y_cell_index, x_cell_index]


def _build_ground_grid_model(
    point_cloud_file_path: Path,
    parameters: DBHEstimatorParameters,
    enable_progress_prints: bool,
) -> GroundGridModel:
    with laspy.open(point_cloud_file_path) as point_reader:
        x_minimum = float(point_reader.header.mins[0])
        y_minimum = float(point_reader.header.mins[1])
        x_maximum = float(point_reader.header.maxs[0])
        y_maximum = float(point_reader.header.maxs[1])

        grid_cell_size = float(parameters.ground_grid_cell_size_meters)
        number_of_cells_x = int(np.floor((x_maximum - x_minimum) / grid_cell_size)) + 1
        number_of_cells_y = int(np.floor((y_maximum - y_minimum) / grid_cell_size)) + 1
        number_of_cells_flat = number_of_cells_x * number_of_cells_y

        min_ground_z_by_class = np.full(number_of_cells_flat, np.inf, dtype=np.float64)
        count_ground_by_class = np.zeros(number_of_cells_flat, dtype=np.int64)
        min_ground_z_by_all = np.full(number_of_cells_flat, np.inf, dtype=np.float64)
        count_ground_by_all = np.zeros(number_of_cells_flat, dtype=np.int64)

        processed_chunk_count = 0
        for chunk in point_reader.chunk_iterator(int(parameters.chunk_size)):
            processed_chunk_count += 1
            if parameters.max_chunks is not None and processed_chunk_count > parameters.max_chunks:
                break

            x_coordinates = np.asarray(chunk.x, dtype=np.float64)
            y_coordinates = np.asarray(chunk.y, dtype=np.float64)
            z_coordinates = np.asarray(chunk.z, dtype=np.float64)

            x_cell_index = np.floor((x_coordinates - x_minimum) / grid_cell_size).astype(np.int64)
            y_cell_index = np.floor((y_coordinates - y_minimum) / grid_cell_size).astype(np.int64)
            x_cell_index = np.clip(x_cell_index, 0, number_of_cells_x - 1)
            y_cell_index = np.clip(y_cell_index, 0, number_of_cells_y - 1)
            linear_cell_index = x_cell_index + number_of_cells_x * y_cell_index

            np.minimum.at(min_ground_z_by_all, linear_cell_index, z_coordinates)
            count_ground_by_all += np.bincount(
                linear_cell_index,
                minlength=number_of_cells_flat,
            )

            if hasattr(chunk, "classification"):
                classification_values = np.asarray(chunk.classification, dtype=np.uint8)
                ground_mask = classification_values == int(parameters.ground_classification_value)
                if np.any(ground_mask):
                    ground_linear_index = linear_cell_index[ground_mask]
                    ground_z = z_coordinates[ground_mask]
                    np.minimum.at(min_ground_z_by_class, ground_linear_index, ground_z)
                    count_ground_by_class += np.bincount(
                        ground_linear_index,
                        minlength=number_of_cells_flat,
                    )

        class_valid_mask = count_ground_by_class > 0
        all_valid_mask = count_ground_by_all > 0
        class_valid_ratio = float(np.mean(class_valid_mask))

        use_classified_ground = class_valid_ratio >= float(parameters.ground_valid_ratio_threshold)
        if use_classified_ground:
            selected_valid_mask = class_valid_mask
            selected_ground = min_ground_z_by_class
            log_progress(
                enable_progress_prints,
                f"Ground grid source: LAS classification={parameters.ground_classification_value}, "
                f"valid_ratio={class_valid_ratio:.4f}",
            )
        else:
            selected_valid_mask = all_valid_mask
            selected_ground = min_ground_z_by_all
            log_progress(
                enable_progress_prints,
                "Ground grid source: fallback all-point cell-minimum "
                f"(class_valid_ratio={class_valid_ratio:.4f})",
            )

        if not np.any(selected_valid_mask):
            raise ValueError("Ground grid estimation failed: no valid cells were found.")

        ground_grid_flat = np.full(number_of_cells_flat, np.nan, dtype=np.float64)
        ground_grid_flat[selected_valid_mask] = selected_ground[selected_valid_mask]
        ground_grid = ground_grid_flat.reshape(number_of_cells_y, number_of_cells_x)
        ground_grid = fill_empty_ground_grid_cells(ground_grid, "nearest_neighbor")

        if int(parameters.ground_smoothing_median_filter_size) > 1:
            ground_grid = ndimage.median_filter(
                ground_grid,
                size=int(parameters.ground_smoothing_median_filter_size),
                mode="nearest",
            )

        return GroundGridModel(
            ground_grid=ground_grid,
            x_origin=x_minimum,
            y_origin=y_minimum,
            cell_size_meters=grid_cell_size,
            number_of_cells_x=number_of_cells_x,
            number_of_cells_y=number_of_cells_y,
        )


def _downsample_points_on_xy_grid(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    cell_size_meters: float,
    keep_highest_z_per_cell: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_size_meters <= 0.0 or x_coordinates.size == 0:
        return x_coordinates, y_coordinates, z_coordinates

    x_origin = float(np.min(x_coordinates))
    y_origin = float(np.min(y_coordinates))
    number_of_cells_x = (
        int(np.floor((np.max(x_coordinates) - x_origin) / cell_size_meters)) + 1
    )

    x_cell_index = np.floor((x_coordinates - x_origin) / cell_size_meters).astype(np.int64)
    y_cell_index = np.floor((y_coordinates - y_origin) / cell_size_meters).astype(np.int64)
    linear_cell_index = x_cell_index + number_of_cells_x * y_cell_index

    if keep_highest_z_per_cell:
        sorting_index = np.lexsort((-z_coordinates, linear_cell_index))
        sorted_linear_cell_index = linear_cell_index[sorting_index]
        _, first_index = np.unique(sorted_linear_cell_index, return_index=True)
        selected_index = sorting_index[first_index]
    else:
        _, selected_index = np.unique(linear_cell_index, return_index=True)

    return (
        x_coordinates[selected_index],
        y_coordinates[selected_index],
        z_coordinates[selected_index],
    )


def _collect_breast_height_slice_points(
    point_cloud_file_path: Path,
    ground_model: GroundGridModel,
    parameters: DBHEstimatorParameters,
    enable_progress_prints: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_slice_height = (
        float(parameters.breast_height_meters) - float(parameters.breast_height_window_meters)
    )
    maximum_slice_height = (
        float(parameters.breast_height_meters) + float(parameters.breast_height_window_meters)
    )

    x_slices: list[np.ndarray] = []
    y_slices: list[np.ndarray] = []
    z_slices: list[np.ndarray] = []

    with laspy.open(point_cloud_file_path) as point_reader:
        processed_chunk_count = 0
        for chunk in point_reader.chunk_iterator(int(parameters.chunk_size)):
            processed_chunk_count += 1
            if parameters.max_chunks is not None and processed_chunk_count > parameters.max_chunks:
                break

            x_coordinates = np.asarray(chunk.x, dtype=np.float64)
            y_coordinates = np.asarray(chunk.y, dtype=np.float64)
            z_coordinates = np.asarray(chunk.z, dtype=np.float64)
            ground_elevation = _sample_ground_elevation_for_points(
                x_coordinates,
                y_coordinates,
                ground_model,
            )
            normalized_height = z_coordinates - ground_elevation
            slice_mask = (normalized_height >= minimum_slice_height) & (
                normalized_height <= maximum_slice_height
            )
            if not np.any(slice_mask):
                continue

            x_slices.append(x_coordinates[slice_mask])
            y_slices.append(y_coordinates[slice_mask])
            z_slices.append(normalized_height[slice_mask])

    if not x_slices:
        raise ValueError("No points found inside the DBH breast-height slice.")

    x_slice = np.concatenate(x_slices)
    y_slice = np.concatenate(y_slices)
    z_slice = np.concatenate(z_slices)
    log_progress(
        enable_progress_prints,
        f"Raw breast-height slice points: {x_slice.size}",
    )

    x_slice, y_slice, z_slice = _downsample_points_on_xy_grid(
        x_slice,
        y_slice,
        z_slice,
        float(parameters.slice_downsample_xy_size_meters),
        keep_highest_z_per_cell=False,
    )
    log_progress(
        enable_progress_prints,
        f"Downsampled breast-height slice points: {x_slice.size}",
    )
    return x_slice, y_slice, z_slice


def _circle_from_three_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float, float] | None:
    determinant = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(determinant) < 1e-10:
        return None

    a_term = x1**2 + y1**2
    b_term = x2**2 + y2**2
    c_term = x3**2 + y3**2
    center_x = (a_term * (y2 - y3) + b_term * (y3 - y1) + c_term * (y1 - y2)) / (
        2.0 * determinant
    )
    center_y = (a_term * (x3 - x2) + b_term * (x1 - x3) + c_term * (x2 - x1)) / (
        2.0 * determinant
    )
    radius = float(np.hypot(x1 - center_x, y1 - center_y))
    if not np.isfinite(radius):
        return None
    return center_x, center_y, radius


def _fit_circle_with_ransac(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    parameters: DBHEstimatorParameters,
    random_number_generator: np.random.Generator,
) -> tuple[float, float, float, float, float] | None:
    point_count = x_coordinates.size
    if point_count < 3:
        return None

    minimum_radius = float(parameters.minimum_dbh_centimeters) / 200.0
    maximum_radius = float(parameters.maximum_dbh_centimeters) / 200.0
    inlier_threshold = float(parameters.circle_ransac_inlier_threshold_meters)

    best_inlier_mask = None
    best_center_x = 0.0
    best_center_y = 0.0
    best_radius = 0.0
    best_inlier_count = -1

    for _ in range(int(parameters.circle_ransac_iterations)):
        sampled_index = random_number_generator.choice(point_count, size=3, replace=False)
        model = _circle_from_three_points(
            float(x_coordinates[sampled_index[0]]),
            float(y_coordinates[sampled_index[0]]),
            float(x_coordinates[sampled_index[1]]),
            float(y_coordinates[sampled_index[1]]),
            float(x_coordinates[sampled_index[2]]),
            float(y_coordinates[sampled_index[2]]),
        )
        if model is None:
            continue
        center_x, center_y, radius = model
        if radius < minimum_radius or radius > maximum_radius:
            continue

        distances = np.hypot(x_coordinates - center_x, y_coordinates - center_y)
        residual = np.abs(distances - radius)
        inlier_mask = residual <= inlier_threshold
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_center_x = float(center_x)
            best_center_y = float(center_y)
            best_radius = float(radius)

    if best_inlier_mask is None or best_inlier_count < 3:
        return None

    inlier_ratio = float(best_inlier_count / point_count)
    if inlier_ratio < float(parameters.circle_min_inlier_ratio):
        return None

    inlier_x = x_coordinates[best_inlier_mask]
    inlier_y = y_coordinates[best_inlier_mask]

    def objective(circle_parameters: np.ndarray) -> np.ndarray:
        center_x, center_y, radius = circle_parameters
        return np.hypot(inlier_x - center_x, inlier_y - center_y) - radius

    optimized = least_squares(
        objective,
        x0=np.array([best_center_x, best_center_y, best_radius], dtype=np.float64),
        bounds=(
            np.array([-np.inf, -np.inf, minimum_radius], dtype=np.float64),
            np.array([np.inf, np.inf, maximum_radius], dtype=np.float64),
        ),
        loss="soft_l1",
        f_scale=inlier_threshold,
        max_nfev=250,
    )
    if not optimized.success:
        return None

    refined_center_x = float(optimized.x[0])
    refined_center_y = float(optimized.x[1])
    refined_radius = float(optimized.x[2])
    if refined_radius >= 0.97 * maximum_radius:
        return None

    residual_all = np.hypot(x_coordinates - refined_center_x, y_coordinates - refined_center_y) - refined_radius
    fit_rmse = float(np.sqrt(np.mean(np.square(residual_all))))
    if fit_rmse > float(parameters.circle_max_rmse_meters):
        return None

    return refined_center_x, refined_center_y, refined_radius, fit_rmse, inlier_ratio


def _fit_dbh_with_fallback(
    fit_x_coordinates: np.ndarray,
    fit_y_coordinates: np.ndarray,
    seed_center_x: float,
    seed_center_y: float,
    parameters: DBHEstimatorParameters,
    random_number_generator: np.random.Generator,
) -> tuple[float, float, float, float, float] | None:
    fitted_circle = _fit_circle_with_ransac(
        fit_x_coordinates,
        fit_y_coordinates,
        parameters,
        random_number_generator,
    )
    if fitted_circle is not None:
        return fitted_circle

    radial_distance = np.hypot(
        fit_x_coordinates - seed_center_x,
        fit_y_coordinates - seed_center_y,
    )
    radius = float(np.percentile(radial_distance, 65.0))
    minimum_radius = float(parameters.minimum_dbh_centimeters) / 200.0
    maximum_radius = float(parameters.maximum_dbh_centimeters) / 200.0
    if radius < minimum_radius or radius > maximum_radius:
        return None
    fit_rmse = float(np.sqrt(np.mean(np.square(radial_distance - radius))))
    inlier_ratio = float(np.mean(np.abs(radial_distance - radius) <= 0.08))
    if fit_rmse > 0.25 or inlier_ratio < 0.10:
        return None
    return seed_center_x, seed_center_y, radius, fit_rmse, inlier_ratio


def _find_local_stem_seed(
    local_x_coordinates: np.ndarray,
    local_y_coordinates: np.ndarray,
    parameters: DBHEstimatorParameters,
) -> tuple[float, float] | None:
    if local_x_coordinates.size == 0:
        return None

    cell_size = float(parameters.local_stem_density_cell_size_meters)
    x_origin = float(np.min(local_x_coordinates))
    y_origin = float(np.min(local_y_coordinates))
    number_of_cells_x = int(np.floor((np.max(local_x_coordinates) - x_origin) / cell_size)) + 1

    x_cell_index = np.floor((local_x_coordinates - x_origin) / cell_size).astype(np.int64)
    y_cell_index = np.floor((local_y_coordinates - y_origin) / cell_size).astype(np.int64)
    linear_cell_index = x_cell_index + number_of_cells_x * y_cell_index

    cell_point_count = np.bincount(linear_cell_index)
    densest_cell_index = int(np.argmax(cell_point_count))
    densest_cell_count = int(cell_point_count[densest_cell_index])
    if densest_cell_count < int(parameters.local_stem_min_cell_points):
        return None

    densest_cell_mask = linear_cell_index == densest_cell_index
    seed_center_x = float(np.mean(local_x_coordinates[densest_cell_mask]))
    seed_center_y = float(np.mean(local_y_coordinates[densest_cell_mask]))
    return seed_center_x, seed_center_y


def _compute_stem_quality_score(stem: StemEstimate) -> float:
    return (
        stem.fit_inlier_ratio
        * math.sqrt(max(stem.slice_point_count, 1))
        / (1.0 + 6.0 * stem.fit_rmse_meters)
    )


def _deduplicate_stem_estimate_indices(
    stem_estimates: list[StemEstimate],
    minimum_distance_meters: float,
) -> list[int]:
    if not stem_estimates:
        return []

    ordering = np.argsort(
        np.asarray(
            [-_compute_stem_quality_score(stem) for stem in stem_estimates],
            dtype=np.float64,
        )
    )
    selected_indices: list[int] = []
    for index in ordering:
        candidate = stem_estimates[int(index)]
        if not selected_indices:
            selected_indices.append(int(index))
            continue
        selected_x = np.asarray(
            [stem_estimates[selected_index].x_coordinate_meters for selected_index in selected_indices],
            dtype=np.float64,
        )
        selected_y = np.asarray(
            [stem_estimates[selected_index].y_coordinate_meters for selected_index in selected_indices],
            dtype=np.float64,
        )
        distances = np.hypot(
            selected_x - candidate.x_coordinate_meters,
            selected_y - candidate.y_coordinate_meters,
        )
        if np.min(distances) < minimum_distance_meters:
            continue
        selected_indices.append(int(index))

    selected_indices.sort(
        key=lambda selected_index: (
            stem_estimates[selected_index].x_coordinate_meters,
            stem_estimates[selected_index].y_coordinate_meters,
        )
    )
    return selected_indices


def _build_tree_top_candidates(
    point_cloud_file_path: Path,
    ground_model: GroundGridModel,
    parameters: DBHEstimatorParameters,
    enable_progress_prints: bool,
) -> np.ndarray:
    chm_cell_size = float(parameters.chm_cell_size_meters)
    with laspy.open(point_cloud_file_path) as point_reader:
        x_minimum = float(point_reader.header.mins[0])
        y_minimum = float(point_reader.header.mins[1])
        x_maximum = float(point_reader.header.maxs[0])
        y_maximum = float(point_reader.header.maxs[1])

        number_of_cells_x = int(np.floor((x_maximum - x_minimum) / chm_cell_size)) + 1
        number_of_cells_y = int(np.floor((y_maximum - y_minimum) / chm_cell_size)) + 1
        chm_flat = np.full(number_of_cells_x * number_of_cells_y, -np.inf, dtype=np.float32)

        processed_chunk_count = 0
        for chunk in point_reader.chunk_iterator(int(parameters.chunk_size)):
            processed_chunk_count += 1
            if parameters.max_chunks is not None and processed_chunk_count > parameters.max_chunks:
                break

            x_coordinates = np.asarray(chunk.x, dtype=np.float64)
            y_coordinates = np.asarray(chunk.y, dtype=np.float64)
            z_coordinates = np.asarray(chunk.z, dtype=np.float64)
            ground_elevation = _sample_ground_elevation_for_points(
                x_coordinates,
                y_coordinates,
                ground_model,
            )
            normalized_height = z_coordinates - ground_elevation
            valid_height_mask = (
                (normalized_height >= float(parameters.minimum_height_for_assignment_meters))
                & (normalized_height <= float(parameters.maximum_tree_height_meters))
            )
            if not np.any(valid_height_mask):
                continue

            x_coordinates = x_coordinates[valid_height_mask]
            y_coordinates = y_coordinates[valid_height_mask]
            normalized_height = normalized_height[valid_height_mask]

            x_cell_index = np.floor((x_coordinates - x_minimum) / chm_cell_size).astype(np.int64)
            y_cell_index = np.floor((y_coordinates - y_minimum) / chm_cell_size).astype(np.int64)
            x_cell_index = np.clip(x_cell_index, 0, number_of_cells_x - 1)
            y_cell_index = np.clip(y_cell_index, 0, number_of_cells_y - 1)
            linear_cell_index = x_cell_index + number_of_cells_x * y_cell_index

            sorting_index = np.argsort(linear_cell_index)
            sorted_linear_cell_index = linear_cell_index[sorting_index]
            sorted_height = normalized_height[sorting_index]
            unique_linear_index, first_occurrence_index = np.unique(
                sorted_linear_cell_index,
                return_index=True,
            )
            maximum_height_per_cell = np.maximum.reduceat(
                sorted_height,
                first_occurrence_index,
            )
            np.maximum.at(chm_flat, unique_linear_index, maximum_height_per_cell)

    chm_grid = chm_flat.reshape(number_of_cells_y, number_of_cells_x)
    chm_grid[chm_grid < 0.0] = 0.0
    smoothed_chm = ndimage.gaussian_filter(
        chm_grid,
        sigma=float(parameters.chm_gaussian_sigma),
    )

    local_max_window = int(parameters.chm_local_max_window_size)
    if local_max_window < 3:
        local_max_window = 3
    if local_max_window % 2 == 0:
        local_max_window += 1
    local_maximum = ndimage.maximum_filter(
        smoothed_chm,
        size=local_max_window,
        mode="nearest",
    )
    candidate_peak_mask = (smoothed_chm == local_maximum) & (
        smoothed_chm >= float(parameters.chm_min_peak_height_meters)
    )
    peak_row_index, peak_column_index = np.nonzero(candidate_peak_mask)
    if peak_row_index.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    candidate_peak_height = smoothed_chm[peak_row_index, peak_column_index]
    candidate_peak_xy = np.column_stack(
        [
            x_minimum + (peak_column_index + 0.5) * chm_cell_size,
            y_minimum + (peak_row_index + 0.5) * chm_cell_size,
        ]
    ).astype(np.float64)

    separation_distance = float(parameters.chm_min_peak_separation_meters)
    sorting_index = np.argsort(-candidate_peak_height)
    selected_index: list[int] = []
    selected_xy: list[np.ndarray] = []
    for candidate_index in sorting_index:
        candidate_point = candidate_peak_xy[candidate_index]
        if selected_xy:
            distances = np.hypot(
                np.asarray([point[0] for point in selected_xy]) - candidate_point[0],
                np.asarray([point[1] for point in selected_xy]) - candidate_point[1],
            )
            if np.any(distances < separation_distance):
                continue
        selected_index.append(int(candidate_index))
        selected_xy.append(candidate_point)

    tree_top_candidates = candidate_peak_xy[np.asarray(selected_index, dtype=np.int64)]
    log_progress(
        enable_progress_prints,
        f"Detected tree-top candidates from CHM: {tree_top_candidates.shape[0]}",
    )
    return tree_top_candidates


def _fit_stems_from_tree_tops(
    x_slice: np.ndarray,
    y_slice: np.ndarray,
    tree_top_candidates_xy: np.ndarray,
    parameters: DBHEstimatorParameters,
    enable_progress_prints: bool,
) -> tuple[list[StemEstimate], list[StemEstimate], list[StemFitSnapshot]]:
    if tree_top_candidates_xy.size == 0:
        return [], [], []

    slice_tree = cKDTree(np.column_stack([x_slice, y_slice]))
    random_number_generator = np.random.default_rng(int(parameters.random_seed))
    stem_candidates: list[StemEstimate] = []
    fit_snapshots: list[StemFitSnapshot] = []

    for tree_top_xy in tree_top_candidates_xy:
        neighborhood_index = slice_tree.query_ball_point(
            tree_top_xy,
            r=float(parameters.top_to_slice_search_radius_meters),
        )
        if len(neighborhood_index) < int(parameters.minimum_points_for_dbh_fit):
            continue

        neighborhood_index_array = np.asarray(neighborhood_index, dtype=np.int64)
        local_x_coordinates = x_slice[neighborhood_index_array]
        local_y_coordinates = y_slice[neighborhood_index_array]

        seed_center = _find_local_stem_seed(
            local_x_coordinates,
            local_y_coordinates,
            parameters,
        )
        if seed_center is None:
            continue
        seed_center_x, seed_center_y = seed_center
        top_offset = float(
            np.hypot(seed_center_x - float(tree_top_xy[0]), seed_center_y - float(tree_top_xy[1]))
        )
        if top_offset > float(parameters.maximum_top_to_stem_offset_meters):
            continue

        local_distance = np.hypot(
            local_x_coordinates - seed_center_x,
            local_y_coordinates - seed_center_y,
        )
        fit_radius = float(parameters.dbh_fit_radius_meters)
        fit_mask = local_distance <= fit_radius
        if int(np.sum(fit_mask)) < int(parameters.minimum_points_for_dbh_fit):
            fit_mask = local_distance <= (fit_radius * 1.35)
        if int(np.sum(fit_mask)) < int(parameters.minimum_points_for_dbh_fit):
            continue

        fit_x_coordinates = local_x_coordinates[fit_mask]
        fit_y_coordinates = local_y_coordinates[fit_mask]
        fitted_result = _fit_dbh_with_fallback(
            fit_x_coordinates,
            fit_y_coordinates,
            seed_center_x,
            seed_center_y,
            parameters,
            random_number_generator,
        )
        if fitted_result is None:
            continue

        center_x, center_y, radius, fit_rmse, inlier_ratio = fitted_result
        dbh_centimeters = 200.0 * radius
        assignment_radius = float(
            np.clip(
                parameters.height_assignment_base_radius_meters
                + parameters.height_assignment_dbh_multiplier * radius * 2.0,
                parameters.height_assignment_min_radius_meters,
                parameters.height_assignment_max_radius_meters,
            )
        )
        stem_candidates.append(
            StemEstimate(
                x_coordinate_meters=center_x,
                y_coordinate_meters=center_y,
                dbh_centimeters=dbh_centimeters,
                fit_rmse_meters=fit_rmse,
                fit_inlier_ratio=inlier_ratio,
                slice_point_count=int(fit_x_coordinates.size),
                assignment_radius_meters=assignment_radius,
            )
        )
        example_point_limit = int(parameters.diagnostic_points_per_fit_example)
        if fit_x_coordinates.size > example_point_limit:
            sample_index = random_number_generator.choice(
                fit_x_coordinates.size,
                size=example_point_limit,
                replace=False,
            )
            snapshot_x_coordinates = fit_x_coordinates[sample_index]
            snapshot_y_coordinates = fit_y_coordinates[sample_index]
        else:
            snapshot_x_coordinates = fit_x_coordinates
            snapshot_y_coordinates = fit_y_coordinates
        fit_snapshots.append(
            StemFitSnapshot(
                tree_top_x_meters=float(tree_top_xy[0]),
                tree_top_y_meters=float(tree_top_xy[1]),
                seed_center_x_meters=seed_center_x,
                seed_center_y_meters=seed_center_y,
                fitted_center_x_meters=center_x,
                fitted_center_y_meters=center_y,
                fitted_radius_meters=radius,
                fit_rmse_meters=fit_rmse,
                fit_inlier_ratio=inlier_ratio,
                fit_x_coordinates=snapshot_x_coordinates,
                fit_y_coordinates=snapshot_y_coordinates,
            )
        )

    log_progress(enable_progress_prints, f"Raw stem candidates: {len(stem_candidates)}")
    selected_candidate_indices = _deduplicate_stem_estimate_indices(
        stem_candidates,
        float(parameters.deduplication_min_stem_distance_meters),
    )
    deduplicated_stems = [stem_candidates[index] for index in selected_candidate_indices]
    deduplicated_snapshots = [fit_snapshots[index] for index in selected_candidate_indices]
    log_progress(enable_progress_prints, f"Accepted stem count: {len(deduplicated_stems)}")
    return deduplicated_stems, stem_candidates, deduplicated_snapshots


def _estimate_tree_heights(
    point_cloud_file_path: Path,
    stems: list[StemEstimate],
    ground_model: GroundGridModel,
    parameters: DBHEstimatorParameters,
    enable_progress_prints: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not stems:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)

    stem_centers = np.array(
        [[stem.x_coordinate_meters, stem.y_coordinate_meters] for stem in stems],
        dtype=np.float64,
    )
    assignment_radius_by_stem = np.array(
        [stem.assignment_radius_meters for stem in stems],
        dtype=np.float64,
    )
    stem_tree = cKDTree(stem_centers)
    maximum_query_radius = float(np.max(assignment_radius_by_stem))

    height_bin_size = float(parameters.height_hist_bin_size_meters)
    number_of_bins = int(np.ceil(float(parameters.maximum_tree_height_meters) / height_bin_size)) + 1
    height_histogram = np.zeros((len(stems), number_of_bins), dtype=np.int64)
    assigned_point_count = np.zeros(len(stems), dtype=np.int64)

    processed_chunk_count = 0
    with laspy.open(point_cloud_file_path) as point_reader:
        for chunk in point_reader.chunk_iterator(int(parameters.chunk_size)):
            processed_chunk_count += 1
            if parameters.max_chunks is not None and processed_chunk_count > parameters.max_chunks:
                break

            x_coordinates = np.asarray(chunk.x, dtype=np.float64)
            y_coordinates = np.asarray(chunk.y, dtype=np.float64)
            z_coordinates = np.asarray(chunk.z, dtype=np.float64)
            ground_elevation = _sample_ground_elevation_for_points(
                x_coordinates,
                y_coordinates,
                ground_model,
            )
            normalized_height = z_coordinates - ground_elevation

            height_mask = (
                (normalized_height >= float(parameters.minimum_height_for_assignment_meters))
                & (normalized_height <= float(parameters.maximum_tree_height_meters))
            )
            if not np.any(height_mask):
                continue

            x_coordinates = x_coordinates[height_mask]
            y_coordinates = y_coordinates[height_mask]
            normalized_height = normalized_height[height_mask]

            x_coordinates, y_coordinates, normalized_height = _downsample_points_on_xy_grid(
                x_coordinates,
                y_coordinates,
                normalized_height,
                float(parameters.height_assignment_xy_downsample_meters),
                keep_highest_z_per_cell=True,
            )
            if normalized_height.size == 0:
                continue

            query_points = np.column_stack([x_coordinates, y_coordinates])
            nearest_distance, nearest_stem_index = stem_tree.query(
                query_points,
                k=1,
                distance_upper_bound=maximum_query_radius,
            )

            valid_query_mask = np.isfinite(nearest_distance) & (
                nearest_stem_index < len(stems)
            )
            if not np.any(valid_query_mask):
                continue

            nearest_distance = nearest_distance[valid_query_mask]
            nearest_stem_index = nearest_stem_index[valid_query_mask].astype(np.int64)
            normalized_height = normalized_height[valid_query_mask]

            distance_mask = nearest_distance <= assignment_radius_by_stem[nearest_stem_index]
            if not np.any(distance_mask):
                continue

            nearest_stem_index = nearest_stem_index[distance_mask]
            normalized_height = normalized_height[distance_mask]
            assigned_point_count += np.bincount(nearest_stem_index, minlength=len(stems))

            height_bin_index = np.floor(normalized_height / height_bin_size).astype(np.int64)
            height_bin_index = np.clip(height_bin_index, 0, number_of_bins - 1)
            linear_histogram_index = nearest_stem_index * number_of_bins + height_bin_index
            height_histogram += np.bincount(
                linear_histogram_index,
                minlength=len(stems) * number_of_bins,
            ).reshape(len(stems), number_of_bins)

    estimated_height = np.full(len(stems), np.nan, dtype=np.float64)
    for stem_index in range(len(stems)):
        total_count = int(np.sum(height_histogram[stem_index]))
        if total_count < int(parameters.minimum_points_for_height):
            continue
        target_rank = int(
            math.ceil(
                float(parameters.height_percentile) / 100.0 * total_count
            )
        )
        target_rank = max(1, target_rank)
        cumulative_count = np.cumsum(height_histogram[stem_index])
        percentile_bin_index = int(
            np.searchsorted(cumulative_count, target_rank, side="left")
        )
        estimated_height[stem_index] = (percentile_bin_index + 0.5) * height_bin_size

    valid_height_ratio = float(np.mean(np.isfinite(estimated_height)))
    log_progress(
        enable_progress_prints,
        f"Height estimation finished: valid_ratio={valid_height_ratio:.4f}",
    )
    return estimated_height, assigned_point_count


def _select_snapshot_indices_for_examples(
    fit_snapshots: list[StemFitSnapshot],
    maximum_examples: int,
) -> np.ndarray:
    if not fit_snapshots:
        return np.zeros(0, dtype=np.int64)
    if len(fit_snapshots) <= maximum_examples:
        return np.arange(len(fit_snapshots), dtype=np.int64)

    rmse_values = np.asarray([snapshot.fit_rmse_meters for snapshot in fit_snapshots], dtype=np.float64)
    sorting_index = np.argsort(rmse_values)
    half_count = maximum_examples // 2
    worst_index = sorting_index[-half_count:]
    best_index = sorting_index[: maximum_examples - half_count]
    selected = np.unique(np.concatenate([worst_index, best_index]))
    if selected.size > maximum_examples:
        selected = selected[:maximum_examples]
    return selected.astype(np.int64)


def _save_diagnostic_plots(
    diagnostics_directory_path: Path,
    x_slice: np.ndarray,
    y_slice: np.ndarray,
    tree_top_candidates_xy: np.ndarray,
    raw_stem_candidates: list[StemEstimate],
    accepted_stems: list[StemEstimate],
    fit_snapshots: list[StemFitSnapshot],
    tree_records: list[dict],
    parameters: DBHEstimatorParameters,
) -> None:
    if not bool(parameters.enable_diagnostic_plots):
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    diagnostics_directory_path.mkdir(parents=True, exist_ok=True)

    random_number_generator = np.random.default_rng(int(parameters.random_seed))
    overview_max_points = int(parameters.diagnostic_overview_max_points)
    if x_slice.size > overview_max_points:
        sampled_index = random_number_generator.choice(
            x_slice.size,
            size=overview_max_points,
            replace=False,
        )
        x_overview = x_slice[sampled_index]
        y_overview = y_slice[sampled_index]
    else:
        x_overview = x_slice
        y_overview = y_slice

    figure, axis = plt.subplots(figsize=(10.5, 9.0))
    axis.scatter(
        x_overview,
        y_overview,
        s=1.1,
        color="#8d99ae",
        alpha=0.16,
        linewidths=0.0,
        label=f"Breast-height slice points (sampled={x_overview.size})",
    )
    if tree_top_candidates_xy.size > 0:
        axis.scatter(
            tree_top_candidates_xy[:, 0],
            tree_top_candidates_xy[:, 1],
            s=30.0,
            marker="^",
            color="#f4a261",
            edgecolors="#6c584c",
            linewidths=0.4,
            alpha=0.88,
            label=f"CHM tree-top candidates ({tree_top_candidates_xy.shape[0]})",
        )

    if raw_stem_candidates:
        raw_x = np.asarray([stem.x_coordinate_meters for stem in raw_stem_candidates], dtype=np.float64)
        raw_y = np.asarray([stem.y_coordinate_meters for stem in raw_stem_candidates], dtype=np.float64)
        axis.scatter(
            raw_x,
            raw_y,
            s=20.0,
            color="#457b9d",
            alpha=0.72,
            label=f"Raw stem candidates ({len(raw_stem_candidates)})",
        )

    if accepted_stems:
        accepted_x = np.asarray([stem.x_coordinate_meters for stem in accepted_stems], dtype=np.float64)
        accepted_y = np.asarray([stem.y_coordinate_meters for stem in accepted_stems], dtype=np.float64)
        accepted_radius = np.asarray(
            [stem.dbh_centimeters / 200.0 for stem in accepted_stems],
            dtype=np.float64,
        )
        axis.scatter(
            accepted_x,
            accepted_y,
            s=45.0,
            color="#e63946",
            edgecolors="#660708",
            linewidths=0.45,
            alpha=0.95,
            label=f"Accepted stems ({len(accepted_stems)})",
            zorder=3,
        )
        for center_x, center_y, radius in zip(
            accepted_x,
            accepted_y,
            accepted_radius,
        ):
            axis.add_patch(
                Circle(
                    (float(center_x), float(center_y)),
                    float(radius),
                    fill=False,
                    color="#e63946",
                    linewidth=0.8,
                    alpha=0.45,
                )
            )

    axis.set_title("Stem Detection Overview (Human-check)")
    axis.set_xlabel("X (meters)")
    axis.set_ylabel("Y (meters)")
    axis.grid(alpha=0.22)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(diagnostics_directory_path / "stem_detection_overview.png", dpi=190)
    plt.close(figure)

    if tree_records:
        dbh_centimeters = np.asarray([record["dbh_centimeters"] for record in tree_records], dtype=np.float64)
        height_meters = np.asarray(
            [record["height_meters"] if record["height_meters"] is not None else np.nan for record in tree_records],
            dtype=np.float64,
        )
        fit_rmse_meters = np.asarray([record["fit_rmse_meters"] for record in tree_records], dtype=np.float64)
        fit_inlier_ratio = np.asarray([record["fit_inlier_ratio"] for record in tree_records], dtype=np.float64)

        quality_figure, quality_axes = plt.subplots(2, 2, figsize=(12.0, 9.5))
        quality_axes = quality_axes.ravel()

        quality_axes[0].hist(dbh_centimeters, bins=28, color="#457b9d", alpha=0.9)
        quality_axes[0].set_title("DBH Distribution")
        quality_axes[0].set_xlabel("DBH (cm)")
        quality_axes[0].set_ylabel("Tree Count")
        quality_axes[0].grid(alpha=0.2)

        valid_height_mask = np.isfinite(height_meters)
        quality_axes[1].hist(height_meters[valid_height_mask], bins=28, color="#2a9d8f", alpha=0.9)
        quality_axes[1].set_title("Height Distribution")
        quality_axes[1].set_xlabel("Height (m)")
        quality_axes[1].set_ylabel("Tree Count")
        quality_axes[1].grid(alpha=0.2)

        scatter = quality_axes[2].scatter(
            dbh_centimeters,
            fit_rmse_meters,
            c=fit_inlier_ratio,
            cmap="viridis",
            s=38.0,
            alpha=0.86,
            edgecolors="none",
        )
        quality_axes[2].set_title("DBH vs Fit RMSE (color=inlier ratio)")
        quality_axes[2].set_xlabel("DBH (cm)")
        quality_axes[2].set_ylabel("Fit RMSE (m)")
        quality_axes[2].grid(alpha=0.2)
        quality_figure.colorbar(scatter, ax=quality_axes[2], label="Inlier Ratio")

        quality_axes[3].scatter(
            dbh_centimeters,
            height_meters,
            s=35.0,
            color="#e63946",
            alpha=0.8,
            edgecolors="none",
        )
        quality_axes[3].set_title("DBH vs Height")
        quality_axes[3].set_xlabel("DBH (cm)")
        quality_axes[3].set_ylabel("Height (m)")
        quality_axes[3].grid(alpha=0.2)

        quality_figure.tight_layout()
        quality_figure.savefig(diagnostics_directory_path / "dbh_quality_summary.png", dpi=200)
        plt.close(quality_figure)

    if fit_snapshots:
        selected_snapshot_indices = _select_snapshot_indices_for_examples(
            fit_snapshots,
            int(parameters.diagnostic_fit_examples_count),
        )
        selected_snapshots = [fit_snapshots[index] for index in selected_snapshot_indices]
        panel_count = len(selected_snapshots)
        panel_columns = 3
        panel_rows = int(np.ceil(panel_count / panel_columns))
        examples_figure, example_axes = plt.subplots(
            panel_rows,
            panel_columns,
            figsize=(4.9 * panel_columns, 4.2 * panel_rows),
        )
        if isinstance(example_axes, np.ndarray):
            axis_array = example_axes.ravel()
        else:
            axis_array = np.array([example_axes], dtype=object)

        for panel_index, snapshot in enumerate(selected_snapshots):
            axis = axis_array[panel_index]
            axis.scatter(
                snapshot.fit_x_coordinates,
                snapshot.fit_y_coordinates,
                s=7.0,
                color="#8d99ae",
                alpha=0.72,
                linewidths=0.0,
            )
            axis.scatter(
                [snapshot.tree_top_x_meters],
                [snapshot.tree_top_y_meters],
                s=55.0,
                marker="^",
                color="#f4a261",
                edgecolors="#6c584c",
                linewidths=0.45,
                label="Tree top",
            )
            axis.scatter(
                [snapshot.seed_center_x_meters],
                [snapshot.seed_center_y_meters],
                s=54.0,
                marker="+",
                color="#1d3557",
                linewidths=1.4,
                label="Local stem seed",
            )
            axis.scatter(
                [snapshot.fitted_center_x_meters],
                [snapshot.fitted_center_y_meters],
                s=46.0,
                color="#e63946",
                edgecolors="#660708",
                linewidths=0.35,
                label="Fitted center",
            )
            axis.add_patch(
                Circle(
                    (snapshot.fitted_center_x_meters, snapshot.fitted_center_y_meters),
                    snapshot.fitted_radius_meters,
                    fill=False,
                    color="#e63946",
                    linewidth=1.3,
                    alpha=0.9,
                )
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.16)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                "DBH={:.1f}cm | RMSE={:.3f}m | Inlier={:.2f}".format(
                    snapshot.fitted_radius_meters * 200.0,
                    snapshot.fit_rmse_meters,
                    snapshot.fit_inlier_ratio,
                )
            )

        for panel_index in range(panel_count, axis_array.size):
            axis_array[panel_index].axis("off")

        handles, labels = axis_array[0].get_legend_handles_labels()
        if handles:
            examples_figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
        examples_figure.suptitle(
            "Circle Fitting Examples (best + worst by RMSE)",
            fontsize=14,
        )
        examples_figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        examples_figure.savefig(diagnostics_directory_path / "dbh_fit_examples.png", dpi=210)
        plt.close(examples_figure)


def _create_run_directory(root_directory: Path) -> Path:
    root_directory.mkdir(parents=True, exist_ok=True)
    run_indices = []
    for path in root_directory.iterdir():
        if not path.is_dir() or not path.name.startswith("run_"):
            continue
        suffix = path.name.removeprefix("run_")
        if suffix.isdigit():
            run_indices.append(int(suffix))
    next_index = (max(run_indices) + 1) if run_indices else 1
    run_directory = root_directory / f"run_{next_index:03d}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def run_dbh_estimation(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    input_config = configuration.get("input_data", {})
    output_config = configuration.get("dbh_estimator_output", {})
    runtime_progress_logging = configuration.get("runtime_progress_logging", {})
    enable_progress_prints = runtime_progress_logging.get("enable_progress_prints", True)
    parameters = _load_estimator_parameters(configuration)

    point_cloud_file_path = Path(
        input_config.get("point_cloud_file_path", "data/L1W.laz")
    )
    if not point_cloud_file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {point_cloud_file_path}")

    output_root_directory = Path(output_config.get("output_root_directory_path", "output"))
    output_json_filename = output_config.get(
        "output_json_filename",
        "forest_tree_dbh_height.json",
    )
    run_directory = _create_run_directory(output_root_directory)

    log_progress(enable_progress_prints, f"Input point cloud: {point_cloud_file_path}")
    segmentation_mode_used = "chm_peaks"
    log_progress(enable_progress_prints, f"Segmentation mode: {segmentation_mode_used}")

    ground_model = _build_ground_grid_model(
        point_cloud_file_path,
        parameters,
        enable_progress_prints,
    )

    x_slice, y_slice, _ = _collect_breast_height_slice_points(
        point_cloud_file_path,
        ground_model,
        parameters,
        enable_progress_prints,
    )

    tree_top_candidates_xy = _build_tree_top_candidates(
        point_cloud_file_path,
        ground_model,
        parameters,
        enable_progress_prints,
    )

    stems, raw_stem_candidates, fit_snapshots = _fit_stems_from_tree_tops(
        x_slice,
        y_slice,
        tree_top_candidates_xy,
        parameters,
        enable_progress_prints,
    )

    estimated_height, assigned_point_count = _estimate_tree_heights(
        point_cloud_file_path,
        stems,
        ground_model,
        parameters,
        enable_progress_prints,
    )

    tree_records = []
    for stem_index, stem in enumerate(stems):
        tree_records.append(
            {
                "tree_id": stem_index + 1,
                "x_meters": round(stem.x_coordinate_meters, 4),
                "y_meters": round(stem.y_coordinate_meters, 4),
                "dbh_centimeters": round(stem.dbh_centimeters, 3),
                "height_meters": None
                if not np.isfinite(estimated_height[stem_index])
                else round(float(estimated_height[stem_index]), 3),
                "slice_point_count": int(stem.slice_point_count),
                "height_point_count": int(assigned_point_count[stem_index]),
                "fit_inlier_ratio": round(float(stem.fit_inlier_ratio), 4),
                "fit_rmse_meters": round(float(stem.fit_rmse_meters), 4),
            }
        )

    _save_diagnostic_plots(
        run_directory / "diagnostics",
        x_slice,
        y_slice,
        tree_top_candidates_xy,
        raw_stem_candidates,
        stems,
        fit_snapshots,
        tree_records,
        parameters,
    )

    output_payload = {
        "input_point_cloud_file_path": str(point_cloud_file_path),
        "tree_count": len(tree_records),
        "segmentation_mode_used": segmentation_mode_used,
        "parameters": {
            key: getattr(parameters, key)
            for key in vars(parameters)
        },
        "trees": tree_records,
    }

    output_json_path = run_directory / output_json_filename
    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, indent=2)

    print(f"Output run directory: {run_directory}")
    print(f"Output json: {output_json_path}")
    print(f"Estimated tree count: {len(tree_records)}")
    return output_json_path
