# MiniMoE-X architecture

## v1 model

MiniMoE-X v1 is a small decoder-only causal language model trained from scratch.

- Byte vocabulary: 256 IDs (each UTF-8 byte maps to one token).
- Token and learned position embeddings.
- Four pre-normalized decoder blocks by default.
- Causal multi-head self-attention.
- Each block has a learned linear router and four independent two-layer GELU expert MLPs.
- Top-2 experts are selected per token; their selected probabilities are renormalized and used to combine expert outputs.
- A load-balancing auxiliary loss is added to next-token cross entropy.
- Token embedding and language-model output projection weights are tied.

Default shape: context 256, width 128, 4 layers, 4 heads, 4 experts, top-2, expert width 256. Configuration can be reduced for tests or experimentation.

## Training and inference

Training reads a local UTF-8 text file as raw bytes, samples next-byte windows and optimizes with AdamW. Checkpoints contain the config, model state and minimal training metadata. Generation uses greedy decoding when temperature is zero, otherwise temperature sampling with optional top-k filtering.

## Important limitations

- Byte tokens are robust and easy to implement, but not compute-efficient for natural language.
- No pretrained knowledge is present; quality depends on data, training compute and evaluation.
- The initial sparse dispatch implementation is a correctness-oriented PyTorch reference and may be slower than dense MLPs for this tiny model.
- This is not a production serving runtime and does not implement KV caching yet.
- No Android APK or native runtime is included in this stage.

## Next milestones

Model checkpoint format validation, deterministic evaluation, KV-cache inference, INT8/INT4 model export, then native Android runtime. Do not claim compatibility until tested end-to-end.
