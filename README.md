# MobileMoE-X — LFM2 Mobile Runtime Track

A lightweight, local-first Android deployment project centered on **[litert-community/LFM2-700M](https://huggingface.co/litert-community/LFM2-700M)**.

> **Architecture correction:** LFM2-700M is not a Mixture-of-Experts model. It is a hybrid Liquid AI model with short-convolution and grouped-query-attention blocks. This repository now uses the ready-made LiteRT-LM community model instead of treating LFM2 as an MoE checkpoint.

## Selected model

- Runtime model: `litert-community/LFM2-700M`
- Base model: `LiquidAI/LFM2-700M`
- Parameters: 742,489,344 (~742.5M)
- Architecture: 16 blocks — 10 short-convolution + 6 grouped-query-attention
- Context length: 32,768 tokens
- Source precision: BF16
- License: LFM Open License v1.0 (review license terms before redistribution)

The model profile is stored at `profiles/lfm2_700m.json`. Model facts are based on the upstream [LiquidAI model card](https://huggingface.co/LiquidAI/LFM2-700M) and [LiteRT community repository](https://huggingface.co/litert-community/LFM2-700M).

## Current project status

| Component | Status |
|---|---|
| LFM2-700M model profile and validation | Implemented |
| LiteRT-LM community model selection | Configured |
| Generic INT4 / INT2 / INT1 tensor-format research tools | Implemented; not a LiteRT model converter |
| Safetensors metadata-only checkpoint inspector | Implemented |
| Android app and native LiteRT-LM integration | Not implemented |
| Pixel 6 performance and thermal benchmarks | Not measured |

## Quick start — Python utilities

Python 3.10+ and NumPy are required.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
mobilemoe show-profile
```

Quantize a standalone NumPy tensor for format experiments:

```bash
mobilemoe quantize tensor.npy output_dir --bits 4 --group-size 128
mobilemoe inspect output_dir
```

This generic tensor utility does **not** read or rewrite the LiteRT-LM model file.

## Run the selected model

Use the LiteRT-LM installation and execution instructions published with the [community model repository](https://huggingface.co/litert-community/LFM2-700M). The repository's exact artifact filename, CLI arguments and runtime compatibility must be taken from that model revision; MobileMoE-X does not invent or wrap undocumented commands.

## Design principles

- Prefer the already-converted LiteRT-LM model over rebuilding the architecture.
- Preserve the upstream model and tokenizer files; never overwrite the source.
- No fake MoE router, expert paging or expert cache: LFM2 is not an MoE model.
- No model retraining, distillation, pruning or weight modification by default.
- Keep inference local and offline after the model files are present.
- Never claim Android support, speed, memory use or battery performance without device measurements.
- Do not package large model weights inside the APK; use a user-controlled model import/download flow.

## Quantization research formats

The generic tensor tools include symmetric grouped INT4, experimental INT2 and research-only binary INT1. These are **not** LiteRT-LM artifact formats and cannot be applied to the community model unless a compatible conversion/export path is implemented and quality-tested. Do not convert or replace the ready-made community artifact with these formats.

## Documentation

- [Architecture and roadmap](docs/ARCHITECTURE.md)
- [Android integration plan](docs/ANDROID_PLAN.md)
- [Quantization format notes](docs/FORMATS.md)
- [Benchmark protocol](docs/BENCHMARKS.md)

## License

The repository's own license is in [LICENSE](LICENSE). The selected model has its own LFM Open License v1.0; those are separate licenses.