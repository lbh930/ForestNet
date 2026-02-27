#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.dbh_estimator import run_dbh_estimation


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Estimate per-tree DBH and height from a forest point cloud"
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_dbh_estimator.yaml"),
        help="Path to DBH estimator config YAML file",
    )
    return argument_parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_dbh_estimation(arguments.config)


if __name__ == "__main__":
    main()
