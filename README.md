# MobileMoE-X — MiniMoE

**A small, trainable, local-first Mixture-of-Experts language-model baseline.**

MobileMoE-X now includes a simple MiniMoE model built from scratch. It does not require a large teacher model, cloud APIs, or a downloaded multi-billion-parameter checkpoint.

## MiniMoE v1 configuration

| Setting | Default |
|---|---:|
| Vocabulary | 256 UTF-8 byte tokens |
| Context | 256 tokens |
| Transformer layers | 4 |
| Hidden size | 128 |
| Attention heads | 4 |
| Experts per MoE layer | 4 |
| Experts selected per token | Top-2 |
| Expert FFN size | 256 |
| Router | Learned linear router |
| Training | Local PyTorch |
| Inference | PyTorch CPU reference |

This configuration is intentionally small. Actual parameter count, memory use, speed and output quality must be measured from the trained artifact on the target device. Byte-level tokenization is simple and multilingual-safe at the byte representation level, but it is less efficient than a learned subword tokenizer.

## Install (Windows 10 / Linux)

Python 3.10+:

```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Linux / Android Debian shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

## Train locally

Prepare a UTF-8 `.txt` file with your own permitted training text. The file must contain at least 257 bytes with the default context.

```bash
mobilemoe-train data.txt --steps 500 --batch-size 8 --output artifacts/minimoe.pt
```

For a quick smoke test, use a small number of steps; useful language quality requires much more clean data and training.

## Generate text

```bash
mobilemoe-generate artifacts/minimoe.pt --prompt "Hello" --tokens 80
```

## Python API

```python
from mobilemoe.config import MiniMoEConfig
from mobilemoe.model import MiniMoEForCausalLM

model = MiniMoEForCausalLM(MiniMoEConfig())
```

## Quantization research tools

Standalone NumPy tensors can still be quantized to grouped INT4, experimental INT2 and research-only binary INT1:

```bash
mobilemoe quantize tensor.npy artifacts/tensor-int4 --bits 4 --group-size 128
mobilemoe inspect artifacts/tensor-int4
```

These are not yet model checkpoint exporters and cannot be loaded by the MiniMoE PyTorch model. The model checkpoint remains ordinary PyTorch weights.

## Design boundaries

- MiniMoE is trained from scratch; no OLMoE/LFM2 teacher is used by default.
- Top-2 routing is learned and executed in the model; each selected expert contributes using normalized router weights.
- This is a reference implementation, not yet an optimized Android runtime.
- No measured Pixel 6 performance or memory claims are published until benchmarked.
- Keep training data and model checkpoints local; verify dataset/model licensing.
- INT1 is highly lossy and INT2 is experimental.

## Roadmap

1. MiniMoE training and CPU reference inference.
2. Checkpoint validation, export format and reproducible evaluation.
3. Weight-only INT8/INT4 model export and dequantized parity tests.
4. ARM64 native inference kernels and Android integration.
5. Pixel 6 memory, latency, thermal and quality benchmarks.

## License

See [LICENSE](LICENSE).
