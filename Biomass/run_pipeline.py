#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.pipeline_runner import run_pipeline


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Minimal UAV LiDAR biomass proxy pipeline"
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML file",
    )
    return argument_parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_pipeline(arguments.config)


if __name__ == "__main__":
    main()
