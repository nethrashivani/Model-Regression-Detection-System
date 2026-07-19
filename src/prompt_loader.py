"""
Loads versioned prompt YAML files from /prompts into PromptConfig objects.

This is the only place that should ever touch the filesystem for prompts —
everything downstream (classifier, eval runner) works with PromptConfig objects,
never with raw file paths.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.models import PromptConfig

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_config(version: str) -> PromptConfig:
    """
    Load a prompt version by name (e.g. "v1") from /prompts/v1.yaml.

    Raises FileNotFoundError with a clear message if the version doesn't exist,
    and lets Pydantic's ValidationError surface directly if the YAML is malformed —
    a bad prompt file should fail loudly, not silently produce a broken config.
    """
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"No prompt file found for version '{version}' at {path}. "
            f"Available versions: {available}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PromptConfig(**raw)


def list_available_versions() -> list[str]:
    """Return all prompt version names currently in /prompts."""
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
