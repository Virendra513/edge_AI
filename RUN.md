# RUN.md — How to build and deploy

This walks you through two paths:

1. **Local sanity check on this Linux box** — confirm the native
   library compiles and the Kotlin code wires up correctly before
   touching a phone.
2. **Build, install, and run on a real Android device** — the path
   you'll actually ship through.

Both paths assume the model variant is **Q2_K** (~554 MB), which is
the default everywhere in this project. If you want Q4_K_M or
another, substitute the name wherever you see `Q2_K`.

---

## 0. Prerequisites

| Tool               | Why                                                       | How to get it                                                                                            |
| ------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| JDK 17             | Android Gradle Plugin 8.5 requires JDK 17                  | `sudo apt install openjdk-17-jdk` (Ubuntu/Debian) or download from [Adoptium](https://adoptium.net/)      |
| Android SDK        | `adb`, `sdkmanager`, platform 34                          | Install [Android Studio](https://developer.android.com/studio) — SDK comes bundled                       |
| Android NDK r25+   | Compiles llama.cpp C++ via `externalNativeBuild`          | In Studio: **Settings → Languages & Frameworks → Android SDK → SDK Tools → NDK (Side by Side)**, install r26 |
| CMake ≥ 3.22       | Drives the JNI build                                      | `sudo apt install cmake` (Ubuntu/Debian) — NDK ships its own, but the host copy is needed by gradle       |
| Ninja              | Faster CMake builds                                       | `sudo apt install ninja-build`                                                                           |
| Git                | Clones llama.cpp                                          | Already installed on most systems                                                                        |
| `adb`              | Installs the APK on a device                              | Bundled with the Android SDK platform-tools                                                              |
| A physical device  | Q2_K on an x86_64 emulator is *painful*; ARM is much faster | Any phone running Android 7.0+ (API 24) — Pixel, Samsung, OnePlus, Redmi, etc.                           |

### Environment variables

```bash
export ANDROID_HOME="$HOME/Android/Sdk"                       # Android Studio default
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"     # the version you installed
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
```

Drop these into `~/.bashrc` so they survive new shells.

### Sanity check

```bash
java -version           # 17+
cmake --version         # 3.22+
adb version             # 1.0.41+
ls "$ANDROID_NDK_HOME"  # should print source.properties, etc.
```

---

## 1. One-time setup

### 1.1 Clone llama.cpp sources

The Gradle build expects llama.cpp at `deploy/llama.cpp/`. The script
fetches it for you:

```bash
cd /home/virendra/work/LFT/modle321B/deploy
bash scripts/build_llama_android.sh
```

That script:

1. `git clone --depth=1 https://github.com/ggerganov/llama.cpp` into
   `deploy/llama.cpp/`.
2. Configures it with the Android NDK toolchain for `arm64-v8a` (the
   default ABI in `app/build.gradle.kts`).
3. Builds the static archives (`libllama.a`, `libggml.a`,
   `libggml-base.a`) into `deploy/.build/llama-android/arm64-v8a/`.

You only run this once per machine (or whenever you want to update
llama.cpp). Subsequent Gradle builds reuse the source tree.

To build for additional ABIs (e.g. emulator testing on x86_64):

```bash
TARGETS="arm64-v8a x86_64" bash scripts/build_llama_android.sh
```

Then edit `app/build.gradle.kts` and add `x86_64` to `abiFilters`.

### 1.2 Stage the Q2_K GGUF

```bash
bash scripts/copy_model.sh         # defaults to Q2_K
# or
bash scripts/copy_model.sh Q4_K_M  # explicit
```

This copies `Llama-3.2-1B-Instruct.Q2_K.gguf` (≈ 554 MB) into
`android/app/src/main/assets/models/model.gguf`. At runtime,
`LlamaEngine.ensureModel` will copy this asset out to
`filesDir/models/model.gguf` on first launch so llama.cpp can `mmap`
it.

### 1.3 Bootstrap the Gradle wrapper

The wrapper jar is not checked in (it's a binary). Generate it once:

```bash
cd android
gradle wrapper --gradle-version 8.7          # uses your system gradle once
cd ..
```

If you don't have a system Gradle, just open `android/` in Android
Studio — it will offer to download the wrapper for you.

---

## 2. Local sanity check (x86_64 Linux)

You can't truly "run" the APK on x86_64 (Android is ARM-native), but
you *can* confirm that:

- The JNI module compiles.
- The Kotlin code is well-formed.
- The APK packages successfully.

```bash
cd android
./gradlew :app:assembleDebug --no-daemon
```

On success you'll find:

```
app/build/outputs/apk/debug/app-debug.apk
```

The first build will take 5–10 minutes because CMake rebuilds
llama.cpp from source. Subsequent builds are <30 seconds (Gradle
caches the `externalNativeBuild` outputs).

If the JNI build fails with `llama.cpp sources not found at…`, run
`scripts/build_llama_android.sh` first.

---

## 3. Build, install, and run on an Android device

### 3.1 Plug in your phone and confirm `adb` sees it

```bash
adb devices
```

Expected output:

```
List of devices attached
XXXXXXXX        device
```

If you see `unauthorized`, accept the "Allow USB debugging" prompt on
the phone, then re-run. On Ubuntu you may also need
[`android-tools-adb` udev rules](https://developer.android.com/studio/run/device).

### 3.2 Build the APK

```bash
cd android
./gradlew :app:assembleDebug --no-daemon
```

This produces `app/build/outputs/apk/debug/app-debug.apk`.

For a release build (smaller, no debug overlay):

```bash
./gradlew :app:assembleRelease
# output: app/build/outputs/apk/release/app-release.apk
# (signed with the debug key for local installs)
```

### 3.3 Install

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

Use `-r` to reinstall over an existing copy without uninstalling
first — useful while iterating.

### 3.4 Launch

```bash
adb shell am start -n com.example.llamachat/.ui.MainActivity
```

### 3.5 Watch the logs

```bash
adb logcat -s LlamaJNI:V AndroidRuntime:E System.err:W
```

Useful messages:

- `llama backend initialised` — native lib loaded.
- `Model loaded: /data/user/0/.../files/models/model.gguf, n_ctx=2048, …` — model ready.
- `Generated N tokens (stopped=…)` — one log line per turn.

### 3.6 Tail the loading progress

The first launch copies the GGUF out of the APK into app-private
storage. That's a 554 MB write, so it can take 10–30 seconds. Watch
the progress:

```bash
adb logcat -s LlamaJNI:V
```

The status row in the app shows "Preparing model (NN%)".

### 3.7 Stop / uninstall

```bash
adb shell am force-stop com.example.llamachat       # kill the running app
adb uninstall com.example.llamachat                 # remove app + cached model
```

---

## 4. Day-to-day workflow

```bash
# After editing Kotlin / native code:
cd android
./gradlew :app:installDebug --no-daemon            # build + install in one shot
adb shell am start -n com.example.llamachat/.ui.MainActivity
adb logcat -s LlamaJNI:V
```

For pure native changes (anything under `app/src/main/cpp/`), force
a rebuild by deleting the cached outputs:

```bash
rm -rf android/app/.cxx android/app/build/intermediates/cxx
./gradlew :app:installDebug --no-daemon
```

---

## 5. Switching to a different quant at runtime

Edit `app/build.gradle.kts` and bump `COPY_VARIANT`, or rerun:

```bash
bash scripts/copy_model.sh Q4_K_M
cd android && ./gradlew :app:installDebug
```

| Quant     | Size    | When to use                                  |
| --------- | ------- | -------------------------------------------- |
| Q2_K      | ~554 MB | **Default.** Low-RAM phones (≤ 4 GB).        |
| Q3_K_M    | ~659 MB | Low-RAM phones that need a quality bump.     |
| Q4_K_M    | ~770 MB | **Best balance.** ≥ 6 GB RAM phones.         |
| Q5_K_M    | ~869 MB | High quality, still works on mid-range.      |

---

## 6. Troubleshooting

| Symptom                                                            | Likely cause                                              | Fix                                                                                                                       |
| ------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `ANDROID_NDK_HOME is not set`                                      | NDK env var missing                                       | `export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"` (or whatever version you installed)                            |
| `llama.cpp sources not found at …/llama.cpp/CMakeLists.txt`        | Skipped the clone step                                    | `bash scripts/build_llama_android.sh`                                                                                     |
| `Failed to load model: …` Snackbar at startup                       | GGUF not in `assets/models/model.gguf`                    | `bash scripts/copy_model.sh` then rebuild                                                                                 |
| `JNI DETECTED ERROR IN APPLICATION: …` in logcat                   | Mismatch between Kotlin native declaration and JNI symbol | Make sure the package in `LlamaEngine.kt` matches `native_jni.cpp`'s `Java_com_example_llamachat_inference_LlamaEngine_*` |
| `error: undefined reference to common_batch_add`                   | llama.cpp version mismatch                                | llama.cpp ≥ 1.6 ships that helper; older versions don't. Run `bash scripts/build_llama_android.sh` to refresh the source.   |
| App stuck on "Loading model…" for >60s                              | Slow flash storage / very low-end device                  | Watch `adb logcat -s LlamaJNI:V`; if you see no progress at all, the asset copy is hung — uninstall and reinstall.         |
| First token takes >5s, subsequent tokens fine                       | Normal — first token is `prefill`, not `decode`            | Nothing to fix; expected behaviour                                                                                        |
| Tokens come out as garbage / repetition                            | Sampler settings are too aggressive                       | Settings dialog → temperature ≈ 0.7, top_p ≈ 0.9, top_k ≈ 40                                                              |
| `INSTALL_FAILED_NO_MATCHING_ABIS`                                  | APK is x86_64-only and phone is ARM                       | Rebuild with `arm64-v8a` in `abiFilters`                                                                                  |
| `dlopen failed: "libc++_shared.so"`                                | App is missing the NDK runtime                            | Set `-DANDROID_STL=c++_shared` in `app/build.gradle.kts` (already the default — only happens if you edited it)             |

---

## 7. Cleaning up

```bash
# Remove Gradle / CMake build cache:
cd android
./gradlew clean

# Drop the cached model in app-private storage (forces re-copy on next launch):
adb shell run-as com.example.llamachat rm -rf files/models

# Drop the entire app + caches:
adb uninstall com.example.llamachat
```

To start completely fresh (forces everything to rebuild):

```bash
rm -rf android/.cxx android/app/build android/build deploy/.build
cd android && ./gradlew clean
```

---

## 8. References

- llama.cpp README: https://github.com/ggerganov/llama.cpp
- llama.cpp Android example: `examples/llama.android/` in that repo
- Android NDK guide: https://developer.android.com/ndk/guides
- Gradle Android plugin: https://developer.android.com/build
