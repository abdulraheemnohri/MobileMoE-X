# Architecture and implementation status

## Current implemented slice

The Python reference package accepts a single real-valued NumPy tensor, quantizes it groupwise, packs signed values into a byte stream, stores scales and a versioned JSON manifest, and reconstructs an approximate float32 tensor. It is intended for correctness experiments and format prototyping.

## Intended full pipeline

1. Inspect checkpoint configuration, tensor names, shapes, dtypes and license.
2. Build a tensor inventory without modifying source files.
3. Select eligible expert weight tensors; exclude router, normalization and sensitive tensors by default.
4. Run calibration/evaluation on representative prompts and datasets.
5. Produce a separate quantized artifact set with checksums and provenance.
6. Validate numerical error and model-level quality.
7. Integrate a runtime that explicitly supports the artifact format.
8. Benchmark on real target hardware before advertising device support.

## Invariants

- Keep original checkpoint immutable and record its cryptographic hash.
- Preserve router and top-k semantics exactly.
- Do not prune, merge or reorder experts.
- Keep embeddings, normalization, router and attention in a conservative precision until independently validated.
- Never silently fall back to a different quantization format.

## Not implemented yet

Full OLMoE checkpoint import/export, model graph conversion, tokenizer packaging, expert paging, cache scheduler, native ARM kernels, JNI, Android application, LiteRT-LM adapter and on-device inference. These require dedicated integration work and hardware validation; they are not represented by mocks in this repository.
