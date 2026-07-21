"""Loads the versioned golden dataset JSON into typed GoldenCase objects."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import GoldenCase

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden_dataset"


def load_golden_dataset(version: str = "v1") -> list[GoldenCase]:
    path = GOLDEN_DIR / f"golden_dataset_{version}.json"
    if not path.exists():
        available = sorted(p.stem for p in GOLDEN_DIR.glob("golden_dataset_*.json"))
        raise FileNotFoundError(
            f"No golden dataset found for version '{version}' at {path}. "
            f"Available: {available}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenCase(**case) for case in raw["cases"]]
