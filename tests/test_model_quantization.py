import torch
import pytest

from mobilemoe.config import MiniMoEConfig
from mobilemoe.model import MiniMoEForCausalLM
from mobilemoe.model_quantization import (
    export_quantized_checkpoint,
    load_quantized_checkpoint,
)


def _checkpoint(path):
    torch.manual_seed(23)
    config = MiniMoEConfig(
        context_length=16, d_model=16, n_layers=1, n_heads=2,
        n_experts=2, experts_per_token=1, d_ff=32,
    )
    model = MiniMoEForCausalLM(config).eval()
    torch.save({
        "format": "mobilemoe-mini-v1",
        "config": config.__dict__,
        "model_state_dict": model.state_dict(),
        "training": {"steps": 3},
    }, path)
    return model, config


@pytest.mark.parametrize(("bits", "max_mean_error"), [(8, 0.08), (4, 0.8)])
def test_model_checkpoint_export_import_forward_parity(tmp_path, bits, max_mean_error):
    source = tmp_path / "source.pt"
    original, _ = _checkpoint(source)
    artifact = tmp_path / f"minimoe-int{bits}.mmq"

    report = export_quantized_checkpoint(source, artifact, bits=bits, group_size=32)
    restored, manifest = load_quantized_checkpoint(artifact)

    tokens = torch.tensor([[72, 101, 108, 108, 111, 32, 77, 111]])
    with torch.no_grad():
        expected, _ = original(tokens)
        actual, _ = restored(tokens)
    mean_error = (expected - actual).abs().mean().item()

    assert artifact.is_file()
    assert report["bits"] == bits
    assert manifest["quantization"]["bits"] == bits
    assert manifest["tensor_count"] == len(original.state_dict())
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert mean_error < max_mean_error


def test_model_export_rejects_unsupported_bits(tmp_path):
    source = tmp_path / "source.pt"
    _checkpoint(source)
    with pytest.raises(ValueError, match="bits"):
        export_quantized_checkpoint(source, tmp_path / "bad.mmq", bits=2)


def test_quantized_loader_rejects_wrong_format(tmp_path):
    source = tmp_path / "wrong.mmq"
    source.write_bytes(b"not a zip archive")
    with pytest.raises(Exception):
        load_quantized_checkpoint(source)
