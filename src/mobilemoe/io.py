"""Safe artifact persistence for quantized NumPy tensors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .quantization import QuantizedTensor


def save_artifact(artifact: QuantizedTensor, directory: str | Path) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "packed.npy", artifact.packed, allow_pickle=False)
    np.save(target / "scales.npy", artifact.scales, allow_pickle=False)
    manifest = artifact.manifest()
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_artifact(directory: str | Path) -> QuantizedTensor:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported artifact manifest version")
    packed = np.load(source / "packed.npy", allow_pickle=False)
    scales = np.load(source / "scales.npy", allow_pickle=False)
    return QuantizedTensor(
        packed=packed,
        scales=scales,
        original_shape=tuple(manifest["original_shape"]),
        original_size=int(manifest["original_size"]),
        bits=int(manifest["bits"]),
        group_size=int(manifest["group_size"]),
        scheme=str(manifest["scheme"]),
        version=int(manifest["version"]),
    )
