# Architecture and implementation status

## Selected runtime model

The default model is `litert-community/LFM2-700M`, based on `LiquidAI/LFM2-700M`. LFM2 is a hybrid edge model, **not an MoE model**. The published 700M configuration has 742,489,344 parameters, 16 blocks (10 short-convolution and 6 grouped-query-attention), 32K context and BF16 source weights.

The checked-in profile `profiles/lfm2_700m.json` captures these facts and marks the runtime integration as pending.

## Runtime-first design

1. Obtain the exact community LiteRT-LM model revision and review its license and file list.
2. Use the model's documented LiteRT-LM runtime path; do not re-export it through a different graph converter by default.
3. Validate tokenizer/chat template and model loading against a known prompt.
4. Build a thin Android app around the supported runtime API.
5. Add cancellation, streaming, bounded context, lifecycle handling and model-file integrity checks.
6. Benchmark on real target devices before setting memory or speed claims.

## Existing utilities

- Grouped INT4/INT2/INT1 quantization for standalone NumPy tensors.
- Packed tensor serialization with group scales and manifest.
- Safetensors metadata-only checkpoint inventory.
- Profile validation and CLI display.

The generic quantizer is not a LiteRT-LM converter. The safetensors inspector is for checkpoint inventory and does not load tensor payloads.

## Explicit non-goals for this model

- No synthetic MoE router, experts or expert paging.
- No pruning, expert merging, distillation or teacher retraining.
- No modification of the published LFM2 checkpoint.
- No claim that INT1/INT2/INT4 custom artifacts can be consumed by LiteRT-LM.

## Remaining implementation

- Confirm the exact community artifact format, revision, runtime API and license metadata.
- Add an Android/Kotlin app and native runtime binding against the supported LiteRT-LM API.
- Implement model download/import, checksum validation, storage checks and offline model registry.
- Add chat UI with real token streaming and cancellation.
- Add runtime logs, crash-safe state, memory telemetry and settings.
- Run integration tests and benchmark on Pixel 6 and other supported devices.
