"""Reference grouped quantization and bit packing.

The implementation is deliberately framework-independent. It is not a model runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


SUPPORTED_BITS = (1, 2, 4)


@dataclass(frozen=True)
class QuantizedTensor:
    """Packed tensor and metadata required for deterministic dequantization."""

    packed: np.ndarray
    scales: np.ndarray
    original_shape: tuple[int, ...]
    original_size: int
    bits: int
    group_size: int
    scheme: str = "symmetric_groupwise"
    version: int = 1

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "packed": None,
            "scales": self.scales.astype(np.float32).tolist(),
            "original_shape": list(self.original_shape),
            "original_size": self.original_size,
            "packed_bytes": int(self.packed.size),
            "scale_count": int(self.scales.size),
            "effective_bits_per_value": (
                (self.packed.size * 8 + self.scales.size * 32) / self.original_size
                if self.original_size else 0.0
            ),
        }


def _limits(bits: int) -> tuple[int, int]:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    if bits == 1:
        return -1, 1
    # Symmetric ranges intentionally avoid the extra negative two's-complement value.
    qmax = (1 << (bits - 1)) - 1
    return -qmax, qmax


def _pack(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack signed quantized integers in little-endian bit order."""
    qmin, qmax = _limits(bits)
    flat = np.asarray(values, dtype=np.int16).reshape(-1)
    if np.any(flat < qmin) or np.any(flat > qmax):
        raise ValueError("quantized values exceed representable symmetric range")
    mask = (1 << bits) - 1
    encoded = np.bitwise_and(flat, mask).astype(np.uint8)
    out = np.zeros(math.ceil(encoded.size * bits / 8), dtype=np.uint8)
    bit_pos = 0
    for value in encoded:
        byte_index, offset = divmod(bit_pos, 8)
        out[byte_index] |= np.uint8((int(value) << offset) & 0xFF)
        spill = offset + bits - 8
        if spill > 0:
            out[byte_index + 1] |= np.uint8(int(value) >> (bits - spill))
        bit_pos += bits
    return out


def _unpack(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bit width: {bits}")
    data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    needed = math.ceil(count * bits / 8)
    if data.size != needed:
        raise ValueError(f"packed length mismatch: expected {needed}, got {data.size}")
    mask = (1 << bits) - 1
    result = np.empty(count, dtype=np.int16)
    bit_pos = 0
    sign_bit = 1 << (bits - 1)
    for i in range(count):
        byte_index, offset = divmod(bit_pos, 8)
        value = int(data[byte_index]) >> offset
        if offset + bits > 8:
            value |= int(data[byte_index + 1]) << (8 - offset)
        value &= mask
        if bits > 1 and value & sign_bit:
            value -= 1 << bits
        result[i] = value
        bit_pos += bits
    return result


def quantize_tensor(
    tensor: np.ndarray, *, bits: int = 4, group_size: int = 128
) -> QuantizedTensor:
    """Quantize a finite numeric tensor in flattened contiguous groups."""
    qmin, qmax = _limits(bits)
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    array = np.asarray(tensor)
    if array.dtype.kind not in "fiu":
        raise TypeError("tensor must have a real numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("tensor contains NaN or infinity")
    flat = array.astype(np.float32, copy=False).reshape(-1)
    quantized = np.empty(flat.size, dtype=np.int16)
    scales = []
    for start in range(0, flat.size, group_size):
        chunk = flat[start : start + group_size]
        max_abs = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        scale = max_abs / qmax if max_abs else 1.0
        scales.append(scale)
        quantized[start : start + chunk.size] = np.clip(
            np.rint(chunk / scale), qmin, qmax
        ).astype(np.int16)
    return QuantizedTensor(
        packed=_pack(quantized, bits),
        scales=np.asarray(scales, dtype=np.float32),
        original_shape=tuple(array.shape),
        original_size=int(flat.size),
        bits=bits,
        group_size=group_size,
    )


def dequantize_tensor(artifact: QuantizedTensor) -> np.ndarray:
    """Reconstruct an approximate float32 tensor from a QuantizedTensor."""
    if artifact.original_size < 0 or artifact.group_size < 1:
        raise ValueError("invalid tensor metadata")
    expected_groups = math.ceil(artifact.original_size / artifact.group_size)
    if artifact.scales.size != expected_groups:
        raise ValueError("scale count does not match tensor metadata")
    if len(artifact.original_shape) and math.prod(artifact.original_shape) != artifact.original_size:
        raise ValueError("shape does not match original_size")
    if np.any(~np.isfinite(artifact.scales)) or np.any(artifact.scales <= 0):
        raise ValueError("scales must be finite and positive")
    q = _unpack(artifact.packed, artifact.original_size, artifact.bits)
    output = np.empty(artifact.original_size, dtype=np.float32)
    for group, start in enumerate(range(0, artifact.original_size, artifact.group_size)):
        end = min(start + artifact.group_size, artifact.original_size)
        output[start:end] = q[start:end].astype(np.float32) * artifact.scales[group]
    return output.reshape(artifact.original_shape)


def storage_report(artifact: QuantizedTensor) -> dict[str, float | int]:
    """Return physical payload and scale overhead, not a theoretical-only estimate."""
    packed_bytes = int(artifact.packed.nbytes)
    scale_bytes = int(artifact.scales.nbytes)
    return {
        "values": artifact.original_size,
        "packed_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "payload_bytes": packed_bytes + scale_bytes,
        "effective_bits_per_value": (
            (packed_bytes + scale_bytes) * 8 / artifact.original_size
            if artifact.original_size else 0.0
        ),
    }
