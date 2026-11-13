# Benchmark protocol

No benchmark results are published by this initial implementation.

For each device and model artifact, record:
- exact device, SoC, Android build and available RAM;
- checkpoint revision, tokenizer revision, artifact manifest and SHA-256 hashes;
- prompt dataset/license, prompt lengths and generation settings;
- cold load time, warm load time, time-to-first-token and decode tokens/second;
- peak process RSS, peak native heap, storage I/O and expert-cache hit rate;
- output quality/perplexity or task accuracy against the unmodified reference;
- battery drain and thermal state over a repeatable run.

Run at least three repetitions after one warm-up; publish median and range. Separate synthetic microbenchmarks from end-to-end model results. Never infer a model's mobile feasibility from parameter bit-width alone.
