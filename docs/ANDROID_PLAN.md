# Android runtime plan

Target profiles (4/6/8/12/16 GB RAM) are design targets, not compatibility guarantees. Device-specific testing is mandatory.

## Runtime requirements

- Android ARM64 build with a scalar reference kernel and optimized NEON kernels.
- Explicitly versioned native artifact reader matching the Python packer.
- JNI boundary with bounds checking and cancellation.
- Expert files stored separately; bounded LRU residency and optional prefetch.
- Memory governor budgets OS reserve, app/runtime, KV cache, temporary unpack buffers, expert cache and safety margin.
- Preserve exact expert IDs, router scores and top-k selection.
- No model weights bundled in the APK; user-controlled import/download with disk-space checks.
- Offline inference after all required assets are available.

## Integration gate

Do not claim LiteRT-LM/TFLite/llama.cpp compatibility until the selected runtime supports this MoE graph and custom quantized tensor format. A custom runtime or custom operators may be required. Benchmark first on a physical Pixel 6 and additional representative devices.

## UI scope for a future Android client

Dashboard, model library/import, runtime profile, memory budget, precision policy, chat, download manager, diagnostics, benchmark history, logs and privacy controls. UI must expose real backend state; it must not simulate inference or performance metrics.
