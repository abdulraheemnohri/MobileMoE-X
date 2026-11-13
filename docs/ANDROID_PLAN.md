# Android plan for MiniMoE-X

The MiniMoE PyTorch implementation is a desktop/host reference, not a direct Android deployment package.

Target-device work should proceed in stages:
1. Freeze the model config and checkpoint schema.
2. Export a tiny deterministic checkpoint and compare logits against the PyTorch reference.
3. Select an Android runtime only after verifying its support for the graph's top-k routing and sparse expert dispatch.
4. Implement ARM64 kernels and bounded memory behavior.
5. Add a Kotlin UI only after native inference can load the model and generate tokens on-device.

For a 4–16 GB phone, evaluate smaller configs (e.g. width 64/96, 2 layers, 2 experts) before the default. Measure process RSS, load time, tokens/second, battery and thermal behavior on each device. Do not infer mobile suitability from parameter count alone.
