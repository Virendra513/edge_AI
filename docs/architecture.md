# Architecture

```
+------------------------------------------------------------+
|                        MainActivity                        |
|  RecyclerView (chat) · Toolbar · SettingsDialog            |
+-----------------▲----------------------+-------------------+
                  | Flow<String> tokens  | user text
                  | StateFlow<Status>    |
+-----------------+----------------------v-------------------+
|                     ChatViewModel                          |
|  holds InferenceSettings, conversation, generation Job     |
+-----------------▲----------------------+-------------------+
                  | coroutines           | coroutines
                  |                      |
+-----------------+----------------------v-------------------+
|                      LlamaEngine (Kotlin)                  |
|  ensureModel() · load() · generate() · stop() · unload()   |
|  Chat template formatter (Llama-3.2 special tokens)       |
+-----------------▲----------------------+-------------------+
                  | JNI                   | callback
                  |                       |
+-----------------+----------------------v-------------------+
|                       native_jni.cpp                       |
|  Java_com_example_llamachat_inference_LlamaEngine_*        |
|  Tokenizer · Sampler chain · llama_decode loop · EOG stop  |
+-----------------▲----------------------+-------------------+
                  | C++                   |
                  |                       |
+-----------------+----------------------v-------------------+
|                         llama.cpp                         |
|  static libs: libllama.a, libggml.a, libggml-base.a       |
|  bundled into libllamachat.so via CMakeLists.txt          |
+------------------------------------------------------------+
```

## Why this layout

- **Single `.so` (`libllamachat.so`)** keeps the APK small and avoids
  shipping both `libllama.so` and `libggml.so` separately. We link the
  static archives into one shared object.
- **One model + one context in the process.** For a 1B chat model
  there is no benefit to pooling. Reload happens when ctx size or
  GPU-layer count change (those are llama.cpp invariants).
- **Streaming via callback.** The native loop calls
  `callback.onToken(token, cumulative)` for every generated token.
  Kotlin emits on a `callbackFlow` running on `Dispatchers.IO`, so
  the UI never blocks.
- **Asset staging.** Bundling the GGUF under `assets/` works but
  reading from there is slow. `LlamaEngine.ensureModel` copies it
  to `filesDir/models/model.gguf` once, then llama.cpp `mmap`s it
  from a real file (much faster cold start on subsequent launches).

## Threading rules

| Layer | Thread |
| ----- | ------ |
| UI rendering | Main |
| Adapter updates | Main (`submitList`) |
| `engine.load()` / `ensureModel()` | `Dispatchers.IO` |
| `engine.generate()` | `Dispatchers.IO` (callback is invoked on same thread) |
| `engine.stop()` | Any thread (atomic flag) |

## Chat template

`LlamaEngine.buildPrompt` mirrors the Llama-3.2-Instruct template:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user1}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{assistant1}<|eot_id|>
...
<|start_header_id|>user<|end_header_id|>

{userN}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

Generation stops when the model emits the EOG token `<|eot_id|>`,
handled by `llama_vocab_is_eog` in the native bridge.
