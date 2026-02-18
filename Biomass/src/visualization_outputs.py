from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def save_vertical_profile_csv(
    output_file_path: Path,
    height_bin_centers: np.ndarray,
    profile_values: np.ndarray,
    shape_normalized_profile_values: np.ndarray | None,
) -> None:
    if shape_normalized_profile_values is None:
        output_array = np.column_stack([height_bin_centers, profile_values])
        header_line = "height_bin_center_meters,profile_value"
    else:
        output_array = np.column_stack(
            [height_bin_centers, profile_values, shape_normalized_profile_values]
        )
        header_line = "height_bin_center_meters,profile_value,shape_normalized_profile_value"

    np.savetxt(
        output_file_path,
        output_array,
        delimiter=",",
        header=header_line,
        comments="",
    )


def save_ground_elevation_grid_png(
    output_file_path: Path,
    ground_elevation_grid: np.ndarray,
    figure_dpi_value: int,
    figure_colormap_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    image = axis.imshow(
        ground_elevation_grid,
        origin="lower",
        cmap=figure_colormap_name,
        aspect="auto",
    )
    axis.set_title("Ground Elevation Grid")
    axis.set_xlabel("Grid X Index")
    axis.set_ylabel("Grid Y Index")
    figure.colorbar(image, ax=axis, label="Elevation (m)")
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def save_normalized_height_histogram_png(
    output_file_path: Path,
    normalized_height: np.ndarray,
    figure_dpi_value: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.hist(normalized_height, bins=80, color="#2a9d8f", alpha=0.9)
    axis.set_title("Normalized Height Distribution")
    axis.set_xlabel("Normalized Height (m)")
    axis.set_ylabel("Point Count")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def save_observed_vs_corrected_vertical_profile_png(
    output_file_path: Path,
    height_bin_centers: np.ndarray,
    observed_profile_values: np.ndarray,
    corrected_profile_values: np.ndarray,
    figure_dpi_value: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.plot(height_bin_centers, observed_profile_values, label="Observed", linewidth=2.0)
    axis.plot(height_bin_centers, corrected_profile_values, label="Corrected", linewidth=2.0)
    axis.set_title("Observed vs Corrected Vertical Profile")
    axis.set_xlabel("Height (m)")
    axis.set_ylabel("Profile Value")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def save_xy_density_map_png(
    output_file_path: Path,
    xy_density_map: np.ndarray,
    title: str,
    figure_dpi_value: int,
    figure_colormap_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    image = axis.imshow(
        xy_density_map.T,
        origin="lower",
        cmap=figure_colormap_name,
        aspect="auto",
    )
    axis.set_title(title)
    axis.set_xlabel("Voxel X Index")
    axis.set_ylabel("Voxel Y Index")
    figure.colorbar(image, ax=axis, label="Integrated Density")
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def save_correction_gain_distribution_histogram_png(
    output_file_path: Path,
    correction_gain_values: np.ndarray,
    figure_dpi_value: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.hist(correction_gain_values, bins=80, color="#e76f51", alpha=0.9)
    axis.set_title("Correction Gain Distribution")
    axis.set_xlabel("Correction Gain")
    axis.set_ylabel("Voxel Count")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def save_subsampling_profile_comparison_png(
    output_file_path: Path,
    height_bin_centers: np.ndarray,
    full_shape_normalized_profile_values: np.ndarray,
    subsampled_shape_normalized_profile_values: np.ndarray,
    figure_dpi_value: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.plot(
        height_bin_centers,
        full_shape_normalized_profile_values,
        label="Full Shape-Normalized",
        linewidth=2.0,
    )
    axis.plot(
        height_bin_centers,
        subsampled_shape_normalized_profile_values,
        label="Subsampled Shape-Normalized",
        linewidth=2.0,
    )
    axis.set_title("Subsampling Stability Profile Comparison")
    axis.set_xlabel("Height (m)")
    axis.set_ylabel("Normalized Profile")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)


def save_peak_density_height_surface_png(
    output_file_path: Path,
    density_grid: np.ndarray,
    z_origin: float,
    voxel_size_z_meters: float,
    figure_dpi_value: int,
    figure_colormap_name: str,
    maximum_surface_grid_resolution: int,
    title: str,
) -> None:
    peak_density_values = np.max(density_grid, axis=2)
    peak_height_indices = np.argmax(density_grid, axis=2)
    peak_height_values = z_origin + (peak_height_indices + 0.5) * voxel_size_z_meters

    peak_height_values = peak_height_values.astype(np.float64)
    peak_height_values[peak_density_values <= 0] = np.nan

    number_of_voxels_x, number_of_voxels_y = peak_density_values.shape
    stride = max(1, int(np.ceil(max(number_of_voxels_x, number_of_voxels_y) / maximum_surface_grid_resolution)))

    peak_height_values = peak_height_values[::stride, ::stride]
    peak_density_values = peak_density_values[::stride, ::stride]

    x_coordinates = np.arange(peak_height_values.shape[0])
    y_coordinates = np.arange(peak_height_values.shape[1])
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")

    valid_density_values = peak_density_values[peak_density_values > 0]
    if valid_density_values.size == 0:
        valid_density_values = np.array([0.0, 1.0], dtype=np.float64)

    color_normalization = Normalize(
        vmin=float(np.min(valid_density_values)),
        vmax=float(np.max(valid_density_values)),
    )
    color_map = cm.get_cmap(figure_colormap_name)
    face_colors = color_map(color_normalization(np.nan_to_num(peak_density_values, nan=0.0)))

    figure = plt.figure(figsize=(8.0, 6.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x_grid,
        y_grid,
        np.nan_to_num(peak_height_values, nan=0.0),
        facecolors=face_colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    axis.set_title(title)
    axis.set_xlabel("Voxel X Index (downsampled)")
    axis.set_ylabel("Voxel Y Index (downsampled)")
    axis.set_zlabel("Peak Density Height (m)")
    axis.view_init(elev=35, azim=-125)

    color_bar = figure.colorbar(
        cm.ScalarMappable(norm=color_normalization, cmap=color_map),
        ax=axis,
        shrink=0.6,
        pad=0.1,
    )
    color_bar.set_label("Peak Density Value")
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=figure_dpi_value)
    plt.close(figure)
