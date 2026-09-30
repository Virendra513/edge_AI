# Deploying Llama-3.2-1B-Instruct (Q2_K GGUF) on Android

This folder is the staging area for packaging the Q2_K GGUF in
`../llama-3.2-1B-Instruct-gguf/` so it can run on an Android phone.

## What we have today

Source directory: `../llama-3.2-1B-Instruct-gguf/`

| File                                 | Size     | Notes                                 |
| ------------------------------------ | -------- | ------------------------------------- |
| `Llama-3.2-1B-Instruct.F16.gguf`     | ~2.4 GB  | Full precision. **Too large for phone.** |
| `Llama-3.2-1B-Instruct.Q2_K.gguf`    | ~554 MB  | **Chosen variant.** Smallest, lowest quality, fits on phones with ≤ 4 GB RAM. |
| `Llama-3.2-1B-Instruct.Q3_K_L.gguf`  | ~698 MB  | Aggressive quant                      |
| `Llama-3.2-1B-Instruct.Q3_K_M.gguf`  | ~659 MB  |                                       |
| `Llama-3.2-1B-Instruct.Q3_K_S.gguf`  | ~612 MB  |                                       |
| `Llama-3.2-1B-Instruct.Q4_K_M.gguf`  | ~770 MB  | **Recommended balance size↔quality**  |
| `Llama-3.2-1B-Instruct.Q4_K_S.gguf`  | ~740 MB  |                                       |
| `Llama-3.2-1B-Instruct.Q5_K_M.gguf`  | ~869 MB  | Higher quality, larger                |
| `Llama-3.2-1B-Instruct.Q5_K_S.gguf`  | ~851 MB  |                                       |
| `Llama-3.2-1B-Instruct.Q6_K.gguf`    | ~974 MB  | Near-lossless                         |
| `Llama-3.2-1B-Instruct.Q8_0.gguf`    | ~1.26 GB | Almost full quality                   |

## Why Q2_K

Q2_K is the smallest practical variant (~554 MB on disk) and is the
only one that fits comfortably on phones with ≤ 4 GB of RAM after
accounting for the OS, the llama.cpp KV cache, and runtime heap. It
trades some coherence for size; for a chat app where responses are
typically short this is acceptable. If you have headroom and want
better quality, swap to **Q4_K_M** by editing `COPY_VARIANT` in
`android/app/build.gradle.kts` (or re-running
`scripts/copy_model.sh Q4_K_M`).

## Goal

Run the 1B model on-device with:

- Reasonable latency (>5 tok/s on modern phones)
- Offline / no network required
- A clean chat interface (system prompt + multi-turn)
- The GGUF lives in app-private storage; no copy outside the device

## Folder layout

```
deploy/
├── README.md                   ← this file
├── android/                    ← Android Studio project (Kotlin + llama.cpp JNI)
│   ├── app/
│   │   ├── build.gradle.kts
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── assets/models/model.gguf   ← populated by copyModel
│   │       ├── cpp/                       ← JNI bridge (CMake + .cpp)
│   │       ├── java/com/example/llamachat/
│   │       │   ├── inference/             ← LlamaEngine, settings, sampler
│   │       │   └── ui/                    ← Activity, ViewModel, Adapter
│   │       └── res/                       ← layouts, themes, strings
│   ├── build.gradle.kts
│   ├── settings.gradle.kts
│   └── gradle.properties
├── scripts/
│   ├── build_llama_android.sh   ← one-time clone + NDK build of llama.cpp
│   └── copy_model.sh            ← stage the chosen GGUF into assets/
├── docs/
│   ├── architecture.md
│   └── benchmark.md
└── llama.cpp/                   ← populated by build_llama_android.sh (gitignored)
```

## Quick start

```bash
# 1. Clone llama.cpp sources next to the project.
scripts/build_llama_android.sh           # needs ANDROID_NDK_HOME set

# 2. Stage the Q2_K GGUF into assets/models/.
scripts/copy_model.sh                    # defaults to Q2_K

# 3. Open android/ in Android Studio, build, run on a device.
```

Gradle runs the copy task automatically during `assemble`; you can
also invoke it explicitly:

```bash
cd android && ./gradlew copyModel            # default Q2_K
cd android && ./gradlew copyModel -PCOPY_VARIANT=Q4_K_M
```

## Switching to a different quant

Either:

```bash
scripts/copy_model.sh Q4_K_M
```

or pass the variant into Gradle:

```bash
./gradlew copyModel -PCOPY_VARIANT=Q4_K_M
```

Both replace `app/src/main/assets/models/model.gguf` with the new
file. The next `assembleDebug` rebuilds the APK with the new asset.

## Runtime config

Default `InferenceSettings`:

- `n_ctx = 2048` (1B model is small; bump to 4096 on ≥ 6 GB devices)
- `n_threads = min(cores, 4)` (auto-resolved at load time)
- `n_gpu_layers = 0` (CPU only; flip to `-1` once you build with `GGML_VULKAN=ON`)
- `max_new_tokens = 512`
- `temperature = 0.7`, `top_p = 0.9`, `top_k = 40`

All of these are editable at runtime via the Settings dialog (toolbar
gear icon); changes take effect on the next turn. Changing `n_ctx` or
`n_gpu_layers` triggers a model reload.

## Out of scope

- Server / cloud fallback — fully on-device.
- Fine-tuning on-device — too heavy for a phone.
- iOS — same GGUF works with llama.cpp on iOS, but this project is
  Android-only.

## References

- llama.cpp: https://github.com/ggerganov/llama.cpp
- llama.cpp Android examples:
  `examples/llama.android/` in the llama.cpp repo
- Llama 3.2 model card: https://github.com/meta-llama/llama-models
- Android NDK: https://developer.android.com/ndk
