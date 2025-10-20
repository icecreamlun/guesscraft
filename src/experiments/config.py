from __future__ import annotations

from typing import Any, Dict, Optional


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


