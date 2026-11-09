# Quantization format notes

These formats are generic research utilities for standalone numeric tensors. They are **not** the serialization format of `litert-community/LFM2-700M` and must not be used to rewrite the ready-made LiteRT-LM artifact.

## Implemented reference formats

- INT4: signed symmetric grouped quantization.
- INT2: signed symmetric grouped quantization; experimental and quality-sensitive.
- INT1: binary sign quantization; highly lossy and research-only.
- FP16/BF16: preserve the upstream model artifact as supplied; this package does not convert its runtime representation.

## Artifact layout

- `packed.npy`: packed unsigned bytes, least-significant bits first.
- `scales.npy`: float32 scale per contiguous group.
- `manifest.json`: shape, element count, group size, bit width and version.

The generic packer is not a standardized interchange format. A LiteRT-LM converter would need to support the model's operators, tensor layouts, quantization metadata and runtime kernels explicitly.

## Safe use

Do not apply INT1/INT2 to the LFM2 checkpoint based only on file-size expectations. First implement a supported model export path, then compare model quality and device performance against the original artifact. Router-specific guidance is irrelevant because LFM2 has no MoE router.


## MiniMoE model checkpoints (separate format)

The MiniMoE-specific exporter is distinct from the generic standalone NumPy
tensor packer above. It accepts a `mobilemoe-mini-v1` PyTorch training
checkpoint and writes a `.mmq` ZIP archive containing a JSON manifest plus
safe NumPy arrays (loaded with `allow_pickle=False`).

```bash
mobilemoe quantize-model artifacts/minimoe.pt artifacts/minimoe-int8.mmq --bits 8
mobilemoe quantize-model artifacts/minimoe.pt artifacts/minimoe-int4.mmq --bits 4
```

All floating-point state tensors are quantized independently in contiguous
groups using symmetric scales. INT8 stores signed byte values; INT4 stores
packed signed nibbles. Non-floating state entries, if present, are retained
without quantization. The PyTorch importer dequantizes the archive back into
the reference model's floating-point tensors:

```python
from mobilemoe.model_quantization import load_quantized_checkpoint

model, manifest = load_quantized_checkpoint("artifacts/minimoe-int4.mmq")
```

This provides a portable research artifact and an import/parity test path; it
does **not** provide low-bit kernels. Loading currently expands weights to
floating point in PyTorch, so runtime RAM and compute are not INT4/INT8. The
`.mmq` format is not GGUF, TFLite, LiteRT-LM, or a verified Android format.
