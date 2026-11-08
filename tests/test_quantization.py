import numpy as np
import pytest

from mobilemoe.quantization import dequantize_tensor, quantize_tensor, storage_report


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_round_trip_shape_and_finite(bits):
    rng = np.random.default_rng(7)
    source = rng.normal(size=(7, 19)).astype(np.float32)
    artifact = quantize_tensor(source, bits=bits, group_size=16)
    restored = dequantize_tensor(artifact)
    assert restored.shape == source.shape
    assert np.all(np.isfinite(restored))
    assert artifact.packed.dtype == np.uint8
    assert artifact.scales.dtype == np.float32


def test_int4_error_is_bounded_by_half_step_for_non_clipped_values():
    source = np.linspace(-1, 1, 129, dtype=np.float32)
    artifact = quantize_tensor(source, bits=4, group_size=129)
    restored = dequantize_tensor(artifact)
    assert np.max(np.abs(source - restored)) <= float(artifact.scales[0]) * 0.51


def test_zero_tensor_and_empty_tensor():
    zeros = np.zeros((2, 3), dtype=np.float32)
    assert np.array_equal(dequantize_tensor(quantize_tensor(zeros)), zeros)
    empty = np.array([], dtype=np.float32)
    assert dequantize_tensor(quantize_tensor(empty)).shape == (0,)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        quantize_tensor(np.ones(4), bits=3)
    with pytest.raises(ValueError):
        quantize_tensor(np.ones(4), group_size=0)
    with pytest.raises(ValueError):
        quantize_tensor(np.array([np.nan], dtype=np.float32))


def test_storage_report_includes_scale_overhead():
    artifact = quantize_tensor(np.ones(257, dtype=np.float32), bits=4, group_size=64)
    report = storage_report(artifact)
    assert report["payload_bytes"] == report["packed_bytes"] + report["scale_bytes"]
    assert report["effective_bits_per_value"] > 4
