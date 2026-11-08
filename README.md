# MobileMoE-X

**Research toolkit for memory-aware Mixture-of-Experts inference on mobile devices.**

MobileMoE-X explores how to store and serve an existing MoE checkpoint on constrained devices while preserving its original expert/router semantics. The initial implementation provides a tested, framework-independent reference for grouped INT4, INT2 and binary INT1 tensor quantization. It does **not** claim to convert OLMoE into a mobile runtime or provide a production Android inference engine yet.

## Project principles

- Preserve source checkpoint weights; never overwrite the teacher checkpoint.
- Keep router, expert IDs, routing scores and top-k behavior unchanged.
- Quantize selected tensor groups into separate artifacts with explicit metadata.
- Treat INT4 as the first practical candidate; INT2 is experimental; INT1 is highly lossy research-only.
- Account for scales, padding and metadata when reporting effective storage.
- Never report simulated performance as measured device performance.

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Quantize a NumPy tensor saved as `.npy`:

```bash
mobilemoe quantize input.npy output_dir --bits 4 --group-size 128
mobilemoe inspect output_dir
mobilemoe inspect-checkpoint ./local-hf-checkpoint --output inventory.json
```

`inspect-checkpoint` reads only `config.json`, an optional safetensors index, and shard headers. It does not load tensor payloads into memory. It validates index-to-shard mappings and reports raw configuration fields and heuristic tensor-name categories. Categories are hints, not proof of tensor roles.

The output directory contains packed bytes, per-group scales and a JSON manifest. Original tensors are not modified.

## Formats

| Format | Representation | Status |
|---|---|---|
| FP16/BF16 | Source/reference storage | Preserve source artifact; no lossy conversion performed here |
| INT4 | Signed symmetric grouped quantization | Reference implementation |
| INT2 | Signed symmetric grouped quantization | Experimental |
| INT1 | Binary sign plus group scale | Research-only, very lossy |

INT1 represents values using only -1/+1 and cannot represent zero. INT2/INT1 quality depends heavily on calibration and model architecture. This repository does not claim that these files can be loaded by LiteRT-LM, TFLite, llama.cpp, GGUF or any other runtime.

## Teacher checkpoint

The intended research teacher is OLMoE-1B-7B. The actual checkpoint configuration and tensor names must be read from the selected checkpoint at conversion time; do not hard-code architecture assumptions. This toolkit currently accepts individual NumPy tensors, not a full OLMoE checkpoint.

## Safety and correctness boundaries

- No teacher retraining, distillation, pruning, expert merging or router modification.
- No automatic download of multi-gigabyte model weights.
- No fabricated Android compatibility or benchmark numbers.
- A successful quantization round-trip test is not proof of language-model quality.
- Use only checkpoints and datasets whose licenses permit your intended use.

## Development

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/FORMATS.md](docs/FORMATS.md), [docs/ANDROID_PLAN.md](docs/ANDROID_PLAN.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

See [LICENSE](LICENSE).