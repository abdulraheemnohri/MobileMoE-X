"""Portable weight-only INT8/INT4 MiniMoE checkpoint export and import.

This is a model-specific archive format. It is not GGUF, TFLite, LiteRT-LM,
or a native Android runtime format.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import torch

from .config import MiniMoEConfig
from .model import MiniMoEForCausalLM
from .quantization import QuantizedTensor, dequantize_tensor, quantize_tensor


FORMAT = "mobilemoe-mini-quantized-v1"
SUPPORTED_MODEL_BITS = (4, 8)
_ARRAY_NAME = re.compile(r"t[0-9]{6}_(?:packed|scales|data)\\.npy$")


def _array_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _read_array(archive: zipfile.ZipFile, name: str, names: set[str]) -> np.ndarray:
    if not isinstance(name, str) or not _ARRAY_NAME.fullmatch(name) or name not in names:
        raise ValueError(f"invalid or missing tensor array entry: {name!r}")
    try:
        return np.load(BytesIO(archive.read(name)), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise ValueError(f"invalid NumPy array payload: {name}") from exc


def _quantize_int8(array: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    quantized = np.empty(flat.size, dtype=np.int8)
    scales: list[float] = []
    for start in range(0, flat.size, group_size):
        chunk = flat[start : start + group_size]
        maximum = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        scale = maximum / 127.0 if maximum else 1.0
        scales.append(scale)
        quantized[start : start + chunk.size] = np.clip(
            np.rint(chunk / scale), -127, 127
        ).astype(np.int8)
    return quantized.reshape(array.shape), np.asarray(scales, dtype=np.float32)


def _dequantize_int8(
    quantized: np.ndarray, scales: np.ndarray, shape: tuple[int, ...], group_size: int
) -> np.ndarray:
    flat = np.asarray(quantized)
    if flat.dtype != np.int8 or flat.size != math.prod(shape):
        raise ValueError("INT8 tensor payload shape or dtype is invalid")
    flat = flat.reshape(-1)
    expected = math.ceil(flat.size / group_size)
    if scales.dtype != np.float32 or scales.size != expected:
        raise ValueError("INT8 scale count or dtype is invalid")
    if not np.all(np.isfinite(scales)) or np.any(scales < 0):
        raise ValueError("INT8 scales must be finite and non-negative")
    output = np.empty(flat.size, dtype=np.float32)
    for group, start in enumerate(range(0, flat.size, group_size)):
        end = min(start + group_size, flat.size)
        output[start:end] = flat[start:end].astype(np.float32) * scales[group]
    return output.reshape(shape)


def export_quantized_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, object]:
    """Export a normal MiniMoE training checkpoint to a portable INT4/INT8 archive."""
    if bits not in SUPPORTED_MODEL_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_MODEL_BITS}")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    source = Path(checkpoint_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == target.resolve():
        raise ValueError("output must not overwrite the source checkpoint")

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "mobilemoe-mini-v1":
        raise ValueError("unsupported source checkpoint; expected mobilemoe-mini-v1")
    config = MiniMoEConfig(**payload["config"]).validate()
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")

    # Validate key names and shapes before writing a deployable artifact.
    reference_model = MiniMoEForCausalLM(config)
    reference_model.load_state_dict(state, strict=True)

    entries: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for index, (key, tensor) in enumerate(sorted(state.items())):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {key!r} is not a tensor")
        prefix = f"t{index:06d}"
        if tensor.is_floating_point():
            source_array = tensor.detach().cpu().float().numpy()
            if not np.all(np.isfinite(source_array)):
                raise ValueError(f"state tensor {key!r} contains NaN or infinity")
            if bits == 4:
                artifact = quantize_tensor(source_array, bits=4, group_size=group_size)
                packed_name, scales_name = prefix + "_packed.npy", prefix + "_scales.npy"
                arrays[packed_name] = artifact.packed
                arrays[scales_name] = artifact.scales
                entries[key] = {
                    "kind": "quantized",
                    "bits": 4,
                    "shape": list(source_array.shape),
                    "source_dtype": str(tensor.dtype),
                    "packed_file": packed_name,
                    "scales_file": scales_name,
                }
            else:
                quantized, scales = _quantize_int8(source_array, group_size)
                data_name, scales_name = prefix + "_data.npy", prefix + "_scales.npy"
                arrays[data_name] = quantized
                arrays[scales_name] = scales
                entries[key] = {
                    "kind": "quantized",
                    "bits": 8,
                    "shape": list(source_array.shape),
                    "source_dtype": str(tensor.dtype),
                    "data_file": data_name,
                    "scales_file": scales_name,
                }
        else:
            data_name = prefix + "_data.npy"
            arrays[data_name] = tensor.detach().cpu().numpy()
            entries[key] = {
                "kind": "raw",
                "shape": list(tensor.shape),
                "source_dtype": str(tensor.dtype),
                "data_file": data_name,
            }

    manifest = {
        "format": FORMAT,
        "version": 1,
        "config": asdict(config),
        "quantization": {
            "scheme": "symmetric_groupwise",
            "bits": bits,
            "group_size": group_size,
            "scope": "all floating-point model state tensors",
        },
        "tensor_count": len(entries),
        "tensors": entries,
        "training": payload.get("training", {}),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            )
            for name, array in arrays.items():
                archive.writestr(name, _array_bytes(array))
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "checkpoint": str(target),
        "bits": bits,
        "group_size": group_size,
        "tensor_count": len(entries),
        "source_bytes": source.stat().st_size,
        "artifact_bytes": target.stat().st_size,
    }


def load_quantized_checkpoint(
    checkpoint_path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[MiniMoEForCausalLM, dict[str, object]]:
    """Load and dequantize a MiniMoE INT4/INT8 archive into the PyTorch reference model."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source, "r") as archive:
        names_list = archive.namelist()
        if len(names_list) != len(set(names_list)):
            raise ValueError("archive contains duplicate entry names")
        names = set(names_list)
        if "manifest.json" not in names:
            raise ValueError("quantized checkpoint is missing manifest.json")
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid quantized checkpoint manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ValueError("unsupported quantized checkpoint format")
        if manifest.get("version") != 1:
            raise ValueError("unsupported quantized checkpoint version")
        quantization = manifest.get("quantization", {})
        bits = quantization.get("bits")
        group_size = quantization.get("group_size")
        if bits not in SUPPORTED_MODEL_BITS or not isinstance(group_size, int) or group_size < 1:
            raise ValueError("invalid quantization metadata")
        config_data = manifest.get("config")
        entries = manifest.get("tensors")
        if not isinstance(config_data, dict) or not isinstance(entries, dict):
            raise ValueError("manifest is missing config or tensor metadata")
        config = MiniMoEConfig(**config_data).validate()
        model = MiniMoEForCausalLM(config)
        expected_keys = set(model.state_dict())
        if set(entries) != expected_keys:
            raise ValueError("quantized checkpoint tensor names do not match model configuration")
        state: dict[str, torch.Tensor] = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                raise ValueError(f"invalid metadata for tensor {key!r}")
            shape_value = entry.get("shape")
            if (
                not isinstance(shape_value, list)
                or any(not isinstance(dim, int) or dim < 0 for dim in shape_value)
            ):
                raise ValueError(f"invalid shape for tensor {key!r}")
            shape = tuple(shape_value)
            if entry.get("kind") == "raw":
                array = _read_array(archive, entry.get("data_file"), names)
                if tuple(array.shape) != shape or array.dtype.hasobject:
                    raise ValueError(f"raw tensor {key!r} has invalid shape or dtype")
            elif entry.get("kind") == "quantized":
                if entry.get("bits") != bits:
                    raise ValueError(f"mixed bit widths are not supported ({key})")
                scales = _read_array(archive, entry.get("scales_file"), names)
                if bits == 4:
                    packed = _read_array(archive, entry.get("packed_file"), names)
                    artifact = QuantizedTensor(
                        packed=packed,
                        scales=scales,
                        original_shape=shape,
                        original_size=math.prod(shape),
                        bits=4,
                        group_size=group_size,
                    )
                    array = dequantize_tensor(artifact)
                else:
                    quantized = _read_array(archive, entry.get("data_file"), names)
                    array = _dequantize_int8(quantized, scales, shape, group_size)
            else:
                raise ValueError(f"unknown tensor storage kind for {key!r}")
            state[key] = torch.from_numpy(np.ascontiguousarray(array))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a MiniMoE checkpoint to model-specific INT4 or INT8 weights"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=SUPPORTED_MODEL_BITS, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(export_quantized_checkpoint(
        args.checkpoint, args.output, bits=args.bits, group_size=args.group_size
    ), indent=2))


if __name__ == "__main__":
    main()
