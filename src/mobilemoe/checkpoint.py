"""Metadata-only inspection of local Hugging Face safetensors checkpoints.

Only safetensors headers are read; tensor payloads are never loaded into RAM.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


MAX_HEADER_BYTES = 100 * 1024 * 1024


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"{path.name}: file is shorter than safetensors header prefix")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > MAX_HEADER_BYTES:
            raise ValueError(f"{path.name}: invalid or excessive header size {header_size}")
        if 8 + header_size > file_size:
            raise ValueError(f"{path.name}: header extends beyond end of file")
        raw = stream.read(header_size)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: malformed safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: header must be a JSON object")

    data_size = file_size - 8 - header_size
    tensors: dict[str, Any] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}: invalid tensor entry {name!r}")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        dtype = item.get("dtype")
        if (
            not isinstance(shape, list)
            or any(not isinstance(dim, int) or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"{path.name}: invalid shape/offset/dtype for {name!r}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(f"{path.name}: out-of-range data offsets for {name!r}")
        intervals.append((start, end, name))
        tensors[name] = {"shape": shape, "dtype": dtype, "data_bytes": end - start}

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise ValueError(
                f"{path.name}: overlapping tensor data ranges "
                f"{previous[2]!r} and {current[2]!r}"
            )
    return tensors


def _classify_tensor(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("router", "gate")):
        return "routing_or_gate_candidate"
    if "expert" in lowered or ".experts." in lowered:
        return "expert_candidate"
    if any(token in lowered for token in ("norm", "layernorm", "rmsnorm")):
        return "normalization_candidate"
    if any(token in lowered for token in ("embed", "lm_head", "word_embeddings")):
        return "embedding_or_output_candidate"
    if any(token in lowered for token in ("attention", "attn", "query", "key", "value")):
        return "attention_candidate"
    return "unclassified"


def inspect_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Inspect config, shard index and tensor headers without reading tensor payloads."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("config.json not found; expected a local Hugging Face checkpoint")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")

    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("safetensors index is missing a valid weight_map")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
            raise ValueError("safetensors weight_map entries must map names to filenames")
        shard_names = sorted(set(weight_map.values()))
    else:
        single = root / "model.safetensors"
        shard_paths = [single] if single.is_file() else sorted(root.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError("no .safetensors files found in checkpoint directory")
        shard_names = [p.name for p in shard_paths]

    tensors: dict[str, dict[str, Any]] = {}
    shards: list[dict[str, Any]] = []
    for shard_name in shard_names:
        # Prevent an index from escaping the selected checkpoint directory.
        shard_path = (root / shard_name).resolve()
        if root.resolve() not in shard_path.parents or not shard_path.is_file():
            raise FileNotFoundError(f"missing or unsafe shard path: {shard_name}")
        shard_tensors = _read_safetensors_header(shard_path)
        shards.append({"file": shard_name, "size_bytes": shard_path.stat().st_size,
                       "tensor_count": len(shard_tensors)})
        for name, details in shard_tensors.items():
            if name in tensors:
                raise ValueError(f"duplicate tensor name across shards: {name}")
            tensors[name] = {
                **details,
                "shard": shard_name,
                "category": _classify_tensor(name),
            }

    if weight_map is not None:
        if set(weight_map) != set(tensors):
            missing = sorted(set(weight_map) - set(tensors))
            unexpected = sorted(set(tensors) - set(weight_map))
            raise ValueError(f"index/header mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
        for name, shard_name in weight_map.items():
            if tensors[name]["shard"] != shard_name:
                raise ValueError(f"index points tensor {name!r} to the wrong shard")

    category_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = {}
    for tensor in tensors.values():
        category_counts[tensor["category"]] = category_counts.get(tensor["category"], 0) + 1
        dtype_counts[tensor["dtype"]] = dtype_counts.get(tensor["dtype"], 0) + 1

    architecture_keys = (
        "model_type", "architectures", "hidden_size", "num_hidden_layers",
        "num_local_experts", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "intermediate_size", "vocab_size",
    )
    return {
        "format": "huggingface_safetensors_metadata",
        "checkpoint_path": str(root.resolve()),
        "config_summary": {key: config[key] for key in architecture_keys if key in config},
        "tensor_count": len(tensors),
        "shards": shards,
        "dtype_counts": dtype_counts,
        "category_counts": category_counts,
        "tensors": tensors,
        "payloads_loaded": False,
        "weight_map_validated": weight_map is not None,
    }
