"""Central config loader — all modules read hyperparameters/paths through this."""
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the repo root."""
    return REPO_ROOT / relative_path
