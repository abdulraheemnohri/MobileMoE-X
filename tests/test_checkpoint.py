import json
import struct

import pytest

from mobilemoe.checkpoint import inspect_checkpoint


def _write_safetensors(path, tensors):
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_inspect_checkpoint_headers_only(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "olmoe",
        "architectures": ["OlmoeForCausalLM"],
        "num_hidden_layers": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
    }))
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.layers.0.mlp.experts.0.w1.weight": {
            "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]
        },
        "model.layers.0.mlp.router.weight": {
            "dtype": "F32", "shape": [64, 3], "data_offsets": [12, 780]
        },
        "__metadata__": {"format": "pt"},
    })
    result = inspect_checkpoint(tmp_path)
    assert result["payloads_loaded"] is False
    assert result["config_summary"]["model_type"] == "olmoe"
    assert result["tensor_count"] == 2
    assert result["tensors"]["model.layers.0.mlp.experts.0.w1.weight"]["category"] == "expert_candidate"
    assert result["tensors"]["model.layers.0.mlp.router.weight"]["category"] == "routing_or_gate_candidate"


def test_missing_checkpoint_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(tmp_path)


def test_index_mismatch_fails(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    _write_safetensors(tmp_path / "part.safetensors", {
        "tensor.a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    })
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"tensor.b": "part.safetensors"}
    }))
    with pytest.raises(ValueError, match="index/header mismatch"):
        inspect_checkpoint(tmp_path)
