from pathlib import Path

import yaml


def load_configuration(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
