import json
from pathlib import Path

import pytest

from mobilemoe.profile import load_profile, validate_profile


ROOT = Path(__file__).resolve().parents[1]


def test_lfm2_700m_profile_is_consistent():
    profile = load_profile(ROOT / "profiles" / "lfm2_700m.json")
    assert profile["source_model"] == "LiquidAI/LFM2-700M"
    assert profile["runtime_repository"] == "litert-community/LFM2-700M"
    assert profile["moe"] is False
    assert profile["convolution_layers"] == 10
    assert profile["attention_layers"] == 6


def test_lfm2_cannot_be_marked_moe():
    profile = {
        "schema_version": 1, "name": "LFM2", "source_model": "LiquidAI/LFM2-700M",
        "runtime_repository": "litert-community/LFM2-700M", "runtime": "LiteRT-LM",
        "model_family": "LFM2", "architecture": "hybrid", "parameters": 742489344,
        "layers": 16, "context_length": 32768, "moe": True,
    }
    with pytest.raises(ValueError, match="must not be marked"):
        validate_profile(profile)
