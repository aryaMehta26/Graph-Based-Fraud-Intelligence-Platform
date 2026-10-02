"""Load the single source of truth for experiment checkpoints."""
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "models" / "registry.yaml"

def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, Dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configs/models/registry.yaml") from exc
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    required = {"family", "parameters_b", "model_id", "revision", "architecture"}
    for key, value in data.items():
        missing = required - set(value)
        if missing: raise ValueError(f"Registry entry {key} missing: {sorted(missing)}")
    return data

def get_model(name: str, path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    registry = load_registry(path)
    if name not in registry: raise KeyError(f"Unknown model {name}; choose from {sorted(registry)}")
    return dict(registry[name], name=name)
