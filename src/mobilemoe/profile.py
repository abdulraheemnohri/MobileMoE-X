"""Validated model profile helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_profile(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    validate_profile(data)
    return data


def validate_profile(profile: dict[str, Any]) -> None:
    required = (
        "schema_version", "name", "source_model", "runtime_repository",
        "runtime", "model_family", "architecture", "parameters", "layers",
        "context_length", "moe",
    )
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"profile is missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != 1:
        raise ValueError("unsupported profile schema_version")
    for key in ("name", "source_model", "runtime_repository", "runtime", "model_family"):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("parameters", "layers", "context_length"):
        if not isinstance(profile[key], int) or profile[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(profile["moe"], bool):
        raise ValueError("moe must be a boolean")
    if profile["model_family"] == "LFM2" and profile["moe"]:
        raise ValueError("LFM2 profile must not be marked as a Mixture-of-Experts model")
    if "convolution_layers" in profile and "attention_layers" in profile:
        if profile["convolution_layers"] + profile["attention_layers"] != profile["layers"]:
            raise ValueError("convolution_layers + attention_layers must equal layers")
