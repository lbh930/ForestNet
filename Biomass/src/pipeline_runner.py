import json
from pathlib import Path

import numpy as np

from src.beer_lambert_occlusion_corrector import compute_beer_lambert_corrected_density
from src.configuration_loader import load_configuration
from src.ground_surface_estimator import (
    compute_ground_elevation_grid_with_percentile,
    fill_empty_ground_grid_cells,
    sample_ground_elevation_for_points,
)
from src.height_normalizer import normalize_heights
from src.point_cloud_reader import filter_points_by_area_of_interest, load_point_cloud_from_file
from src.progress_logging import log_progress
from src.quality_control_checks import run_subsampling_stability_check
from src.vertical_profile_builder import aggregate_vertical_profile, compute_shape_normalized_profile
from src.visualization_outputs import (
    save_correction_gain_distribution_histogram_png,
    save_ground_elevation_grid_png,
    save_normalized_height_histogram_png,
    save_observed_vs_corrected_vertical_profile_png,
    save_peak_density_height_surface_png,
    save_subsampling_profile_comparison_png,
    save_vertical_profile_csv,
    save_xy_density_map_png,
)
from src.voxel_density_builder import (
    build_voxel_layout,
    build_voxel_point_count_grid,
    compute_observed_density_points_per_cubic_meter,
)


def format_axis_range(values: np.ndarray) -> str:
    return f"{float(np.min(values)):.3f} to {float(np.max(values)):.3f}"


def run_pipeline(config_path: Path) -> None:
    configuration = load_configuration(config_path)

    input_data_config = configuration["input_data"]
    ground_surface_estimation_config = configuration["ground_surface_estimation"]
    height_normalization_config = configuration["height_normalization"]
    voxel_density_model_config = configuration["voxel_density_model"]
    occlusion_correction_config = configuration["occlusion_correction_beer_lambert"]
    vertical_profile_output_config = configuration["vertical_profile_output"]
    quality_control_config = configuration["quality_control"]
    output_files_config = configuration["output_files"]
    runtime_progress_logging_config = configuration.get("runtime_progress_logging", {})
    visualization_outputs_config = configuration.get("visualization_outputs", {})

    enable_progress_prints = runtime_progress_logging_config.get("enable_progress_prints", True)

    if ground_surface_estimation_config["ground_estimation_method_name"] != "grid_percentile":
        raise ValueError(
            "This minimal implementation only supports "
            "ground_estimation_method_name=grid_percentile"
        )

    log_progress(enable_progress_prints, "Loading point cloud")
    x_coordinates, y_coordinates, z_coordinates = load_point_cloud_from_file(configuration)
    log_progress(
        enable_progress_prints,
        "Loaded points: "
        f"count={x_coordinates.size}, "
        f"x_range=({format_axis_range(x_coordinates)}), "
        f"y_range=({format_axis_range(y_coordinates)}), "
        f"z_range=({format_axis_range(z_coordinates)})",
    )

    x_coordinates, y_coordinates, z_coordinates = filter_points_by_area_of_interest(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        input_data_config["area_of_interest_bounds_xy"],
    )

    if x_coordinates.size == 0:
        raise ValueError("No points remain after input loading and AOI filtering.")

    log_progress(
        enable_progress_prints,
        f"After AOI filter: count={x_coordinates.size}",
    )

    log_progress(enable_progress_prints, "Estimating ground surface")
    ground_grid, ground_grid_metadata = compute_ground_elevation_grid_with_percentile(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        ground_surface_estimation_config["ground_grid_cell_size_meters"],
        ground_surface_estimation_config["ground_elevation_percentile_value"],
        ground_surface_estimation_config["minimum_points_required_per_ground_cell"],
    )
    ground_valid_cell_ratio = float(np.mean(~np.isnan(ground_grid)))
    ground_valid_values = ground_grid[~np.isnan(ground_grid)]
    log_progress(
        enable_progress_prints,
        "Ground grid built: "
        f"shape={ground_grid.shape}, "
        f"valid_cell_ratio={ground_valid_cell_ratio:.4f}, "
        f"elevation_min={float(np.min(ground_valid_values)):.3f}, "
        f"elevation_p50={float(np.median(ground_valid_values)):.3f}, "
        f"elevation_max={float(np.max(ground_valid_values)):.3f}",
    )

    ground_grid = fill_empty_ground_grid_cells(
        ground_grid,
        ground_surface_estimation_config["empty_ground_cell_fill_method_name"],
    )

    ground_elevation_for_points = sample_ground_elevation_for_points(
        x_coordinates,
        y_coordinates,
        ground_grid,
        ground_grid_metadata,
    )

    log_progress(enable_progress_prints, "Normalizing heights")
    x_coordinates, y_coordinates, normalized_height = normalize_heights(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        ground_elevation_for_points,
        height_normalization_config["minimum_normalized_height_meters"],
        height_normalization_config["maximum_normalized_height_meters"],
    )

    if x_coordinates.size == 0:
        raise ValueError("No points remain after height normalization and height filtering.")

    normalized_height_percentiles = np.percentile(normalized_height, [5, 50, 95])
    log_progress(
        enable_progress_prints,
        "Height normalization done: "
        f"count={x_coordinates.size}, "
        f"p5={normalized_height_percentiles[0]:.3f}, "
        f"p50={normalized_height_percentiles[1]:.3f}, "
        f"p95={normalized_height_percentiles[2]:.3f}",
    )

    log_progress(enable_progress_prints, "Building voxel grid")
    voxel_layout = build_voxel_layout(
        x_coordinates,
        y_coordinates,
        height_normalization_config["minimum_normalized_height_meters"],
        height_normalization_config["maximum_normalized_height_meters"],
        voxel_density_model_config["voxel_size_x_meters"],
        voxel_density_model_config["voxel_size_y_meters"],
        voxel_density_model_config["voxel_size_z_meters"],
    )

    voxel_point_count_grid = build_voxel_point_count_grid(
        x_coordinates,
        y_coordinates,
        normalized_height,
        voxel_layout,
    )
    non_empty_voxel_count = int(np.count_nonzero(voxel_point_count_grid))
    non_empty_voxel_ratio = float(non_empty_voxel_count / voxel_point_count_grid.size)
    log_progress(
        enable_progress_prints,
        "Voxel grid done: "
        f"shape=({voxel_layout['number_of_voxels_x']},"
        f"{voxel_layout['number_of_voxels_y']},"
        f"{voxel_layout['number_of_voxels_z']}), "
        f"non_empty_voxel_count={non_empty_voxel_count}, "
        f"non_empty_voxel_ratio={non_empty_voxel_ratio:.6f}",
    )

    observed_density_points_per_cubic_meter = compute_observed_density_points_per_cubic_meter(
        voxel_point_count_grid,
        voxel_layout["voxel_size_x_meters"],
        voxel_layout["voxel_size_y_meters"],
        voxel_layout["voxel_size_z_meters"],
    )

    if occlusion_correction_config["enable_occlusion_correction"]:
        log_progress(enable_progress_prints, "Applying Beer-Lambert occlusion correction")
        (
            corrected_density_points_per_cubic_meter,
            correction_gain_grid,
            transmission_grid,
        ) = compute_beer_lambert_corrected_density(
            observed_density_points_per_cubic_meter,
            voxel_point_count_grid,
            voxel_density_model_config["occupancy_point_count_threshold"],
            occlusion_correction_config["vertical_processing_direction"],
            occlusion_correction_config["extinction_coefficient_alpha"],
            occlusion_correction_config["minimum_transmission_value"],
            occlusion_correction_config["maximum_density_gain_factor"],
        )
    else:
        corrected_density_points_per_cubic_meter = observed_density_points_per_cubic_meter.copy()
        correction_gain_grid = np.ones_like(observed_density_points_per_cubic_meter)
        transmission_grid = np.ones_like(observed_density_points_per_cubic_meter)

    occupied_voxel_mask = (
        voxel_point_count_grid >= voxel_density_model_config["occupancy_point_count_threshold"]
    )
    if np.any(occupied_voxel_mask):
        correction_gain_values = correction_gain_grid[occupied_voxel_mask]
    else:
        correction_gain_values = correction_gain_grid.ravel()

    correction_gain_percentiles = np.percentile(correction_gain_values, [50, 95])
    max_gain_value = occlusion_correction_config["maximum_density_gain_factor"]
    max_gain_fraction = float(np.mean(np.isclose(correction_gain_values, max_gain_value)))
    transmission_values = transmission_grid[occupied_voxel_mask] if np.any(occupied_voxel_mask) else transmission_grid.ravel()
    transmission_percentiles = np.percentile(transmission_values, [5, 50, 95])
    log_progress(
        enable_progress_prints,
        "Occlusion correction done: "
        f"gain_p50={correction_gain_percentiles[0]:.3f}, "
        f"gain_p95={correction_gain_percentiles[1]:.3f}, "
        f"gain_max={float(np.max(correction_gain_values)):.3f}, "
        f"max_gain_fraction={max_gain_fraction:.4f}, "
        f"transmission_p05={transmission_percentiles[0]:.4f}, "
        f"transmission_p50={transmission_percentiles[1]:.4f}, "
        f"transmission_p95={transmission_percentiles[2]:.4f}",
    )

    log_progress(enable_progress_prints, "Building vertical profiles")
    height_bin_centers, observed_profile_values = aggregate_vertical_profile(
        observed_density_points_per_cubic_meter,
        voxel_layout["z_origin"],
        voxel_layout["voxel_size_z_meters"],
        vertical_profile_output_config["profile_height_bin_size_meters"],
    )
    _, corrected_profile_values = aggregate_vertical_profile(
        corrected_density_points_per_cubic_meter,
        voxel_layout["z_origin"],
        voxel_layout["voxel_size_z_meters"],
        vertical_profile_output_config["profile_height_bin_size_meters"],
    )

    observed_peak_height = float(height_bin_centers[int(np.argmax(observed_profile_values))])
    corrected_peak_height = float(height_bin_centers[int(np.argmax(corrected_profile_values))])
    observed_profile_sum = float(np.sum(observed_profile_values))
    corrected_profile_sum = float(np.sum(corrected_profile_values))
    observed_fraction_below_2m = float(np.sum(observed_profile_values[height_bin_centers < 2.0]) / observed_profile_sum)
    corrected_fraction_below_2m = float(np.sum(corrected_profile_values[height_bin_centers < 2.0]) / corrected_profile_sum)
    log_progress(
        enable_progress_prints,
        "Profiles done: "
        f"observed_peak_height={observed_peak_height:.3f}, "
        f"corrected_peak_height={corrected_peak_height:.3f}, "
        f"observed_sum={observed_profile_sum:.1f}, "
        f"corrected_sum={corrected_profile_sum:.1f}, "
        f"observed_fraction_below_2m={observed_fraction_below_2m:.4f}, "
        f"corrected_fraction_below_2m={corrected_fraction_below_2m:.4f}",
    )

    if vertical_profile_output_config["output_shape_normalized_profile"]:
        observed_shape_normalized_profile_values = compute_shape_normalized_profile(observed_profile_values)
        corrected_shape_normalized_profile_values = compute_shape_normalized_profile(corrected_profile_values)
    else:
        observed_shape_normalized_profile_values = None
        corrected_shape_normalized_profile_values = None

    output_directory_path = Path(output_files_config["output_directory_path"])
    output_directory_path.mkdir(parents=True, exist_ok=True)

    if output_files_config["save_ground_elevation_grid"]:
        np.save(output_directory_path / "ground_elevation_grid.npy", ground_grid)
        with (output_directory_path / "ground_elevation_grid_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(ground_grid_metadata, file, indent=2)

    if output_files_config["save_height_normalized_point_cloud"]:
        normalized_point_cloud = np.column_stack([x_coordinates, y_coordinates, normalized_height])
        np.save(output_directory_path / "height_normalized_point_cloud.npy", normalized_point_cloud)

    if output_files_config["save_observed_voxel_density_grid"]:
        np.save(
            output_directory_path / "observed_voxel_density_points_per_cubic_meter.npy",
            observed_density_points_per_cubic_meter,
        )

    if output_files_config["save_corrected_voxel_density_grid"]:
        np.save(
            output_directory_path / "corrected_voxel_density_points_per_cubic_meter.npy",
            corrected_density_points_per_cubic_meter,
        )

    if output_files_config["save_observed_vertical_profile_csv"]:
        save_vertical_profile_csv(
            output_directory_path / "observed_vertical_profile.csv",
            height_bin_centers,
            observed_profile_values,
            observed_shape_normalized_profile_values,
        )

    if output_files_config["save_corrected_vertical_profile_csv"]:
        save_vertical_profile_csv(
            output_directory_path / "corrected_vertical_profile.csv",
            height_bin_centers,
            corrected_profile_values,
            corrected_shape_normalized_profile_values,
        )

    quality_control_report = {
        "input_point_count": int(z_coordinates.size),
        "normalized_point_count": int(normalized_height.size),
    }

    subsampling_stability_check_result = None
    if quality_control_config["enable_subsampling_stability_check"]:
        log_progress(enable_progress_prints, "Running subsampling stability check")
        subsampling_stability_check_result = run_subsampling_stability_check(
            x_coordinates,
            y_coordinates,
            normalized_height,
            voxel_layout,
            vertical_profile_output_config["profile_height_bin_size_meters"],
            voxel_density_model_config["occupancy_point_count_threshold"],
            occlusion_correction_config,
            corrected_profile_values,
            quality_control_config["subsampling_keep_ratio"],
            quality_control_config["subsampling_random_seed"],
        )
        quality_control_report["subsampling_stability_check"] = {
            "subsampling_point_count": subsampling_stability_check_result["subsampling_point_count"],
            "profile_pearson_correlation": subsampling_stability_check_result[
                "profile_pearson_correlation"
            ],
        }
        log_progress(
            enable_progress_prints,
            "Subsampling check done: "
            f"correlation={quality_control_report['subsampling_stability_check']['profile_pearson_correlation']}",
        )

    with (output_directory_path / "quality_control_report.json").open("w", encoding="utf-8") as file:
        json.dump(quality_control_report, file, indent=2)

    if visualization_outputs_config.get("enable_visualization_outputs", False):
        log_progress(enable_progress_prints, "Saving visualization images")
        figure_dpi_value = int(visualization_outputs_config.get("figure_dpi_value", 150))
        figure_colormap_name = visualization_outputs_config.get("figure_colormap_name", "viridis")

        if visualization_outputs_config.get("save_ground_elevation_grid_png", False):
            save_ground_elevation_grid_png(
                output_directory_path / "ground_elevation_grid.png",
                ground_grid,
                figure_dpi_value,
                figure_colormap_name,
            )

        if visualization_outputs_config.get("save_normalized_height_histogram_png", False):
            save_normalized_height_histogram_png(
                output_directory_path / "normalized_height_histogram.png",
                normalized_height,
                figure_dpi_value,
            )

        if visualization_outputs_config.get("save_observed_vs_corrected_vertical_profile_png", False):
            save_observed_vs_corrected_vertical_profile_png(
                output_directory_path / "observed_vs_corrected_vertical_profile.png",
                height_bin_centers,
                observed_profile_values,
                corrected_profile_values,
                figure_dpi_value,
            )

        observed_xy_density_map = np.sum(observed_density_points_per_cubic_meter, axis=2)
        corrected_xy_density_map = np.sum(corrected_density_points_per_cubic_meter, axis=2)

        if visualization_outputs_config.get("save_observed_xy_density_map_png", False):
            save_xy_density_map_png(
                output_directory_path / "observed_xy_density_map.png",
                observed_xy_density_map,
                "Observed XY Integrated Density Map",
                figure_dpi_value,
                figure_colormap_name,
            )

        if visualization_outputs_config.get("save_corrected_xy_density_map_png", False):
            save_xy_density_map_png(
                output_directory_path / "corrected_xy_density_map.png",
                corrected_xy_density_map,
                "Corrected XY Integrated Density Map",
                figure_dpi_value,
                figure_colormap_name,
            )

        if visualization_outputs_config.get("save_correction_gain_distribution_histogram_png", False):
            save_correction_gain_distribution_histogram_png(
                output_directory_path / "correction_gain_distribution_histogram.png",
                correction_gain_values,
                figure_dpi_value,
            )

        if (
            visualization_outputs_config.get("save_subsampling_profile_comparison_png", False)
            and subsampling_stability_check_result is not None
        ):
            full_shape_normalized_profile_values = compute_shape_normalized_profile(
                corrected_profile_values
            )
            save_subsampling_profile_comparison_png(
                output_directory_path / "subsampling_profile_comparison.png",
                height_bin_centers,
                full_shape_normalized_profile_values,
                subsampling_stability_check_result[
                    "subsampled_shape_normalized_profile_values"
                ],
                figure_dpi_value,
            )

        maximum_surface_grid_resolution = int(
            visualization_outputs_config.get(
                "maximum_surface_grid_resolution_for_peak_density_height_map",
                180,
            )
        )

        if visualization_outputs_config.get(
            "save_observed_peak_density_height_surface_png",
            False,
        ):
            save_peak_density_height_surface_png(
                output_directory_path / "observed_peak_density_height_surface.png",
                observed_density_points_per_cubic_meter,
                voxel_layout["z_origin"],
                voxel_layout["voxel_size_z_meters"],
                figure_dpi_value,
                figure_colormap_name,
                maximum_surface_grid_resolution,
                "Observed Peak-Density Height Surface (2.5D)",
            )

        if visualization_outputs_config.get(
            "save_corrected_peak_density_height_surface_png",
            False,
        ):
            save_peak_density_height_surface_png(
                output_directory_path / "corrected_peak_density_height_surface.png",
                corrected_density_points_per_cubic_meter,
                voxel_layout["z_origin"],
                voxel_layout["voxel_size_z_meters"],
                figure_dpi_value,
                figure_colormap_name,
                maximum_surface_grid_resolution,
                "Corrected Peak-Density Height Surface (2.5D)",
            )

    print(f"Output directory: {output_directory_path}")
    print(f"Input point count: {quality_control_report['input_point_count']}")
    print(f"Normalized point count: {quality_control_report['normalized_point_count']}")

    if "subsampling_stability_check" in quality_control_report:
        print("Subsampling stability check:")
        print(json.dumps(quality_control_report["subsampling_stability_check"], indent=2))
