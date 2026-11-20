
from pathlib import Path
import yaml

# Get the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config():
    """Load project settings from params.yaml."""

    config_path = PROJECT_ROOT / "params.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config
