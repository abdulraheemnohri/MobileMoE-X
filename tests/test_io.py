import numpy as np

from mobilemoe.io import load_artifact, save_artifact
from mobilemoe.quantization import dequantize_tensor, quantize_tensor


def test_artifact_persistence(tmp_path):
    source = np.arange(33, dtype=np.float32).reshape(3, 11)
    artifact = quantize_tensor(source, bits=2, group_size=8)
    save_artifact(artifact, tmp_path / "artifact")
    loaded = load_artifact(tmp_path / "artifact")
    assert loaded.bits == 2
    assert np.array_equal(dequantize_tensor(loaded), dequantize_tensor(artifact))
