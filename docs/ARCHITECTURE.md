# MiniMoE-X architecture

## v1 model

MiniMoE-X v1 is a small decoder-only causal language model trained from scratch.

- Byte vocabulary: 256 IDs (each UTF-8 byte maps to one token).
- Token and learned position embeddings.
- Four pre-normalized decoder blocks by default.
- Causal multi-head self-attention.
- Each block has a learned linear router and four independent two-layer GELU expert MLPs.
- Top-2 experts are selected per token; their selected probabilities are renormalized and used to combine expert outputs.
- A load-balancing auxiliary loss is added to next-byte cross entropy.
- Token embedding and language-model output projection weights are tied.

Default shape: context 256, width 128, 4 layers, 4 heads, 4 experts, top-2, expert width 256. Configuration can be reduced for tests or experimentation.

## Training and inference

Training reads a local UTF-8 text file as raw bytes, samples next-byte windows and optimizes with AdamW. Standard checkpoints contain the config, model state and minimal training metadata. Generation currently recomputes the full visible context on each step; KV-cache inference is not implemented.

## Model checkpoint quantization

The model-specific exporter writes a versioned `.mmq` archive for INT8 or INT4 weight-only storage. It quantizes floating-point state tensors in independent groups and records the model configuration and tensor manifest. The importer validates model keys and dequantizes weights into the ordinary PyTorch reference model.

This is an export/import and numerical-parity milestone, not low-bit execution. The current importer expands weights to floating point, so it does not provide INT4/INT8 runtime memory savings or low-bit kernels. The `.mmq` format is not GGUF, TFLite, LiteRT-LM, or a verified Android model format. The generic tensor packer remains a separate research utility; INT2 is experimental and INT1 is highly lossy.

## Important limitations

- Byte tokens are robust and easy to implement, but not compute-efficient for natural language.
- No pretrained knowledge is present; quality depends on data, training compute and evaluation.
- The initial sparse dispatch implementation is a correctness-oriented PyTorch reference and may be slower than dense MLPs for this tiny model.
- There is no production serving runtime, KV cache, ARM64 kernel, Android app integration or measured Pixel 6 benchmark yet.

## Next milestones

Implement KV-cache inference with full-context parity tests, then add a deterministic evaluation and benchmark harness. After that, investigate native ARM64/Android execution and test actual device memory, latency, thermal and output quality. Do not claim compatibility or performance until tested end-to-end.
