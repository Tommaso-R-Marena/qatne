"""Utilities for data persistence and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class QATNEEncoder(json.JSONEncoder):
    """Custom JSON encoder for QATNE objects (e.g., numpy arrays)."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        return super().default(obj)


def save_results(data: dict[str, Any], filepath: str | Path) -> None:
    """Save results to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=QATNEEncoder, indent=2)


def load_results(filepath: str | Path) -> dict[str, Any]:
    """Load results from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def export_to_json(data: dict[str, Any]) -> str:
    """Export results to a JSON string."""
    return json.dumps(data, cls=QATNEEncoder, indent=2)
