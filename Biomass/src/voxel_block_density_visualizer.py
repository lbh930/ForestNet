from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from src.configuration_loader import load_configuration


def fast_subsample_density_grid(
    density_grid: np.ndarray,
    voxel_block_subsample_step_x: int,
    voxel_block_subsample_step_y: int,
    voxel_block_subsample_step_z: int,
) -> np.ndarray:
    return density_grid[
        ::voxel_block_subsample_step_x,
        ::voxel_block_subsample_step_y,
        ::voxel_block_subsample_step_z,
    ]


def select_voxels_for_fast_rendering(
    subsampled_density_grid: np.ndarray,
    minimum_density_percentile_to_render: float,
    maximum_number_of_voxel_blocks_to_render: int,
    density_threshold_override: float | None = None,
) -> tuple[np.ndarray, dict]:
    positive_voxel_mask = subsampled_density_grid > 0
    positive_voxel_count = int(np.count_nonzero(positive_voxel_mask))

    if positive_voxel_count == 0:
        empty_mask = np.zeros_like(subsampled_density_grid, dtype=bool)
        return empty_mask, {
            "positive_voxel_count_after_subsample": 0,
            "density_threshold": None,
            "selected_voxel_count_before_topk": 0,
            "selected_voxel_count_final": 0,
        }

    positive_density_values = subsampled_density_grid[positive_voxel_mask]
    if density_threshold_override is None:
        density_threshold = float(
            np.percentile(positive_density_values, minimum_density_percentile_to_render)
        )
        density_threshold_source = "self_percentile"
    else:
        density_threshold = float(density_threshold_override)
        density_threshold_source = "shared_reference"

    selected_voxel_mask = positive_voxel_mask & (subsampled_density_grid >= density_threshold)
    selected_voxel_count_before_topk = int(np.count_nonzero(selected_voxel_mask))

    if (
        maximum_number_of_voxel_blocks_to_render > 0
        and selected_voxel_count_before_topk > maximum_number_of_voxel_blocks_to_render
    ):
        flat_selected_indices = np.flatnonzero(selected_voxel_mask.ravel())
        flat_selected_density_values = subsampled_density_grid.ravel()[flat_selected_indices]

        topk_local_indices = np.argpartition(
            flat_selected_density_values,
            -maximum_number_of_voxel_blocks_to_render,
        )[-maximum_number_of_voxel_blocks_to_render:]

        topk_flat_indices = flat_selected_indices[topk_local_indices]
        selected_voxel_mask = np.zeros_like(subsampled_density_grid, dtype=bool)
        selected_voxel_mask.ravel()[topk_flat_indices] = True

    selected_voxel_count_final = int(np.count_nonzero(selected_voxel_mask))

    return selected_voxel_mask, {
        "positive_voxel_count_after_subsample": positive_voxel_count,
        "density_threshold": density_threshold,
        "density_threshold_source": density_threshold_source,
        "selected_voxel_count_before_topk": selected_voxel_count_before_topk,
        "selected_voxel_count_final": selected_voxel_count_final,
    }


def compute_density_threshold_from_subsampled_grid(
    subsampled_density_grid: np.ndarray,
    minimum_density_percentile_to_render: float,
) -> float:
    positive_density_values = subsampled_density_grid[subsampled_density_grid > 0]
    if positive_density_values.size == 0:
        return 0.0
    return float(np.percentile(positive_density_values, minimum_density_percentile_to_render))


def render_voxel_block_density_png(
    output_file_path: Path,
    subsampled_density_grid: np.ndarray,
    selected_voxel_mask: np.ndarray,
    density_color_scaling_method_name: str,
    voxel_block_colormap_name: str,
    voxel_block_transparency_alpha_value: float,
    view_elevation_degrees: float,
    view_azimuth_degrees: float,
    figure_dpi_value: int,
    enforce_equal_axis_aspect_ratio_1_to_1_to_1: bool,
    title: str,
) -> None:
    if density_color_scaling_method_name == "log1p":
        scaled_density_grid = np.log1p(subsampled_density_grid)
    elif density_color_scaling_method_name == "linear":
        scaled_density_grid = subsampled_density_grid
    else:
        raise ValueError(
            "Unsupported density_color_scaling_method_name: "
            f"{density_color_scaling_method_name}"
        )

    selected_scaled_density_values = scaled_density_grid[selected_voxel_mask]

    if selected_scaled_density_values.size == 0:
        normalized_min = 0.0
        normalized_max = 1.0
    else:
        normalized_min = float(np.min(selected_scaled_density_values))
        normalized_max = float(np.max(selected_scaled_density_values))
        if np.isclose(normalized_min, normalized_max):
            normalized_max = normalized_min + 1.0

    color_normalization = Normalize(vmin=normalized_min, vmax=normalized_max)
    colormap = plt.get_cmap(voxel_block_colormap_name)

    face_color_grid = np.zeros(selected_voxel_mask.shape + (4,), dtype=np.float32)
    if selected_scaled_density_values.size > 0:
        selected_face_colors = colormap(color_normalization(selected_scaled_density_values))
        selected_face_colors[:, 3] = voxel_block_transparency_alpha_value
        face_color_grid[selected_voxel_mask] = selected_face_colors

    figure = plt.figure(figsize=(10.0, 8.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.voxels(
        selected_voxel_mask,
        facecolors=face_color_grid,
        edgecolor=None,
    )
    axis.set_title(title)
    axis.set_xlabel("Voxel X Index (subsampled)")
    axis.set_ylabel("Voxel Y Index (subsampled)")
    axis.set_zlabel("Voxel Z Index (subsampled)")
    axis.set_xlim(0, selected_voxel_mask.shape[0])
    axis.set_ylim(0, selected_voxel_mask.shape[1])
    axis.set_zlim(0, selected_voxel_mask.shape[2])
    if enforce_equal_axis_aspect_ratio_1_to_1_to_1:
        axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=view_elevation_degrees, azim=view_azimuth_degrees)

    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_normalization, cmap=colormap),
        ax=axis,
        shrink=0.62,
        pad=0.08,
    )
    if density_color_scaling_method_name == "log1p":
        color_bar.set_label("log1p(Density)")
    else:
        color_bar.set_label("Density")

    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def print_voxel_visualization_progress(message: str) -> None:
    print(f"[VOXEL_BLOCK_VIS] {message}")


def generate_single_voxel_block_density_visualization(
    density_grid_file_path: Path,
    output_file_path: Path,
    title: str,
    visualization_config: dict,
    shared_density_threshold_override: float | None = None,
) -> None:
    if not density_grid_file_path.exists():
        raise FileNotFoundError(
            f"Density grid file not found: {density_grid_file_path}. "
            "Run run_pipeline.py first to generate voxel density grids."
        )

    print_voxel_visualization_progress(f"Loading density grid: {density_grid_file_path}")
    density_grid = np.load(density_grid_file_path)
    print_voxel_visualization_progress(
        f"Loaded grid shape={density_grid.shape}, total_voxels={density_grid.size}"
    )

    subsampled_density_grid = fast_subsample_density_grid(
        density_grid,
        int(visualization_config["voxel_block_subsample_step_x"]),
        int(visualization_config["voxel_block_subsample_step_y"]),
        int(visualization_config["voxel_block_subsample_step_z"]),
    )

    print_voxel_visualization_progress(
        "After fast subsample: "
        f"shape={subsampled_density_grid.shape}, "
        f"total_voxels={subsampled_density_grid.size}"
    )

    selected_voxel_mask, selection_stats = select_voxels_for_fast_rendering(
        subsampled_density_grid,
        float(visualization_config["minimum_density_percentile_to_render"]),
        int(visualization_config["maximum_number_of_voxel_blocks_to_render"]),
        density_threshold_override=shared_density_threshold_override,
    )

    if selection_stats["selected_voxel_count_final"] == 0:
        print_voxel_visualization_progress("No voxels selected for rendering. Skip this output.")
        return

    selected_density_values = subsampled_density_grid[selected_voxel_mask]
    density_percentiles = np.percentile(selected_density_values, [50, 95])

    print_voxel_visualization_progress(
        "Selection stats: "
        f"positive_voxels={selection_stats['positive_voxel_count_after_subsample']}, "
        f"percentile_threshold={selection_stats['density_threshold']:.4f}, "
        f"threshold_source={selection_stats['density_threshold_source']}, "
        f"selected_before_topk={selection_stats['selected_voxel_count_before_topk']}, "
        f"selected_final={selection_stats['selected_voxel_count_final']}, "
        f"selected_density_p50={density_percentiles[0]:.4f}, "
        f"selected_density_p95={density_percentiles[1]:.4f}, "
        f"selected_density_max={float(np.max(selected_density_values)):.4f}"
    )

    print_voxel_visualization_progress(f"Rendering voxel blocks -> {output_file_path}")
    render_voxel_block_density_png(
        output_file_path,
        subsampled_density_grid,
        selected_voxel_mask,
        visualization_config["density_color_scaling_method_name"],
        visualization_config["voxel_block_colormap_name"],
        float(visualization_config["voxel_block_transparency_alpha_value"]),
        float(visualization_config["view_elevation_degrees"]),
        float(visualization_config["view_azimuth_degrees"]),
        int(visualization_config["figure_dpi_value"]),
        bool(visualization_config["enforce_equal_axis_aspect_ratio_1_to_1_to_1"]),
        title,
    )


def generate_voxel_block_density_visualizations(
    config_path: Path,
    voxel_density_grid_source_mode_override: str | None = None,
) -> None:
    configuration = load_configuration(config_path)
    output_directory_path = Path(configuration["output_files"]["output_directory_path"])
    visualization_config = configuration["voxel_block_density_visualization"]

    output_directory_path.mkdir(parents=True, exist_ok=True)

    observed_density_grid_file_path = output_directory_path / "observed_voxel_density_points_per_cubic_meter.npy"
    corrected_density_grid_file_path = output_directory_path / "corrected_voxel_density_points_per_cubic_meter.npy"

    source_mode = visualization_config["voxel_density_grid_source_mode"].lower()
    if voxel_density_grid_source_mode_override is not None:
        source_mode = voxel_density_grid_source_mode_override.lower()

    if source_mode not in {"observed", "corrected", "both"}:
        raise ValueError(
            "voxel_density_grid_source_mode must be observed, corrected, or both"
        )

    print_voxel_visualization_progress(f"Render mode: {source_mode}")

    use_shared_density_threshold_reference = bool(
        visualization_config.get("use_shared_density_threshold_reference", False)
    )
    shared_density_threshold_reference_mode = visualization_config.get(
        "shared_density_threshold_reference_mode",
        "observed",
    ).lower()
    shared_density_threshold_override = None

    if use_shared_density_threshold_reference:
        if shared_density_threshold_reference_mode == "observed":
            reference_density_grid_file_path = observed_density_grid_file_path
        elif shared_density_threshold_reference_mode == "corrected":
            reference_density_grid_file_path = corrected_density_grid_file_path
        else:
            raise ValueError(
                "shared_density_threshold_reference_mode must be observed or corrected"
            )

        if not reference_density_grid_file_path.exists():
            raise FileNotFoundError(
                f"Reference density grid file not found: {reference_density_grid_file_path}"
            )

        reference_density_grid = np.load(reference_density_grid_file_path)
        reference_subsampled_density_grid = fast_subsample_density_grid(
            reference_density_grid,
            int(visualization_config["voxel_block_subsample_step_x"]),
            int(visualization_config["voxel_block_subsample_step_y"]),
            int(visualization_config["voxel_block_subsample_step_z"]),
        )
        shared_density_threshold_override = compute_density_threshold_from_subsampled_grid(
            reference_subsampled_density_grid,
            float(visualization_config["minimum_density_percentile_to_render"]),
        )
        print_voxel_visualization_progress(
            "Shared threshold enabled: "
            f"reference={shared_density_threshold_reference_mode}, "
            f"threshold={shared_density_threshold_override:.4f}"
        )

    if source_mode in {"observed", "both"}:
        generate_single_voxel_block_density_visualization(
            observed_density_grid_file_path,
            output_directory_path / "observed_density_voxel_blocks.png",
            "Observed Density Voxel Blocks (Minecraft-style)",
            visualization_config,
            shared_density_threshold_override=shared_density_threshold_override,
        )

    if source_mode in {"corrected", "both"}:
        generate_single_voxel_block_density_visualization(
            corrected_density_grid_file_path,
            output_directory_path / "corrected_density_voxel_blocks.png",
            "Corrected Density Voxel Blocks (Minecraft-style)",
            visualization_config,
            shared_density_threshold_override=shared_density_threshold_override,
        )
