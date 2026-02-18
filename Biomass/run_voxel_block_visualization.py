#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.voxel_block_density_visualizer import generate_voxel_block_density_visualizations


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Generate Minecraft-style voxel block density visualization"
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML file",
    )
    argument_parser.add_argument(
        "--voxel-density-grid-source-mode",
        choices=["observed", "corrected", "both"],
        default=None,
        help="Optional override for voxel_density_grid_source_mode in config",
    )
    return argument_parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    generate_voxel_block_density_visualizations(
        arguments.config,
        voxel_density_grid_source_mode_override=arguments.voxel_density_grid_source_mode,
    )


if __name__ == "__main__":
    main()
