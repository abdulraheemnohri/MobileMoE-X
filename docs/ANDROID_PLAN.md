# Android integration plan — LFM2-700M

## Target

Android ARM64, starting with a real-device validation on Google Pixel 6. Device compatibility is not guaranteed merely because the model is described as edge-oriented.

## Runtime

Use the published `litert-community/LFM2-700M` LiteRT-LM artifact and its documented runtime API. First inspect the exact repository revision and available artifact files. Do not assume a filename, shell command, JNI API or Android AAR coordinate that is not documented by the model/runtime project.

## App modules

- **Model Library:** model ID, revision, file size, checksum, license and installed state.
- **Chat:** prompt, generated tokens, stop/cancel, conversation reset.
- **Runtime Settings:** context limit, generation parameters, thread/accelerator options only when supported.
- **Diagnostics:** runtime version, model load errors, device memory, thermal state and logs.
- **Storage:** import/download progress, free-space guard, checksum verification and safe deletion.
- **Privacy:** local conversation storage, clear history and no network use during inference.

## Resource policy

- Never bundle the model weights in the APK.
- Check available storage before download/import.
- Keep context limits configurable; 32K is the model's published context ceiling, not a promise that a phone can run 32K comfortably.
- Account for app/runtime memory, model buffers, KV cache, temporary allocations and Android OS reserve.
- Stop generation safely when the app is backgrounded or the user cancels.

## Release gate

Before calling the Android build functional, verify on physical hardware:
1. model loads from the actual community artifact;
2. tokenizer and chat template work;
3. prompt and generation produce non-empty output;
4. streaming and cancellation work;
5. offline inference works after assets are installed;
6. peak memory, latency, thermal behavior and battery impact are recorded.

No fake responses or synthetic benchmark metrics.
