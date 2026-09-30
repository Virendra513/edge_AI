# Benchmarking Q2_K on Android

This page is a template for the numbers you should collect before
shipping. Drop your measurements into the table after each test run.

## How to measure

1. Pick a fixed prompt — 256 input tokens, 256 generated tokens is
   the most useful baseline because it forces a `prefill + decode`
   mix that's representative of a real turn.
2. Warm the model for ~20 tokens first; the first token after load
   is always slower because of memory bandwidth.
3. Use `LlamaChat`'s status row (it shows tok/s after each turn) or
   pull `logcat -s LlamaJNI` for the per-call breakdown.
4. Run on a fresh device with ≥ 60% battery. Phones throttle above
   ~40 °C; sustained generation will drift downward.

## Sample prompt (used for all rows below)

```
You are a helpful assistant. The user will ask a question. Answer concisely.
User: List the first five prime numbers, in order, separated by commas. Do not include any extra text.
Assistant:
```

## Target table

Fill in `tok/s` (decode), `prefill ms` (first token latency), and
peak RSS (from `dumpsys meminfo com.example.llamachat`).

| Device | SoC | n_threads | n_ctx | GPU? | tok/s | prefill ms | RSS MB |
| ------ | --- | --------- | ----- | ---- | ----- | ---------- | ------ |
| Pixel 8 | Tensor G3 | 4 | 2048 | off | _ | _ | _ |
| Pixel 6a | Tensor G1 | 4 | 2048 | off | _ | _ | _ |
| Galaxy A54 | Exynos 1380 | 4 | 2048 | off | _ | _ | _ |
| Redmi Note 12 | Snapdragon 685 | 4 | 2048 | off | _ | _ | _ |

## What "good" looks like

- **Decode ≥ 8 tok/s** on a 2023-era mid-range phone. The 1B model
  at Q2_K is tiny; if you're below 5 tok/s, threads are likely
  pinned to 1 (check `InferenceSettings.effectiveThreads`).
- **Prefill < 1500 ms** for 256 input tokens. Anything slower means
  the device is reading from internal storage rather than memory-
  mapping the GGUF — make sure `filesDir/models/model.gguf` exists.
- **RSS < 1.5 GB**. Q2_K is ~554 MB on disk; resident set adds
  context KV cache (≈ 64 MB at n_ctx=2048) + native heap overhead.
  If you see > 2 GB, ctx is probably over-sized.

## Tuning knobs

Try in this order; each step costs you a re-load.

1. **`n_threads = min(cores, 4)`** — 4 is usually the sweet spot
   on big.LITTLE SoCs; going higher contends for the little cores
   and hurts more than helps.
2. **`n_gpu_layers`** — start at 0; bump by 10 and watch tok/s. On
   most Adreno/Mali parts the Q2_K 1B fully offloads (`n_gpu_layers
   = -1`) and you get a 1.5×–2× speedup. Skip Vulkan entirely if
   you didn't compile with `GGML_VULKAN=ON`.
3. **`n_ctx`** — drop to 1024 if your turn lengths are short; KV
   cache memory scales linearly.
4. **Temperature / top-k** — no perf impact; tune for quality only.
5. **`mmap`** — enabled by default; if you ever call
   `llama_model_load_from_file` with the env var
   `LLAMA_NO_MMAP=1`, decode will tank.
