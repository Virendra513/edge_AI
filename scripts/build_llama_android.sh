#!/usr/bin/env bash
# Build the llama.cpp static library for Android via NDK.
#
# Why this exists:
#   Gradle's externalNativeBuild compiles our JNI bridge by adding
#   llama.cpp via add_subdirectory. That requires the llama.cpp source
#   tree to be present at android/../llama.cpp. Run this script to
#   clone it once per machine; Gradle will then take over.
#
# Prerequisites:
#   - ANDROID_NDK_HOME set (NDK r25+ recommended)
#   - cmake >= 3.22 on PATH
#   - ninja on PATH (recommended)
#
# Outputs:
#   android/app/build/intermediates/cxx/.../obj/<abi>/libllama.a
#   android/app/build/intermediates/cxx/.../obj/<abi>/libggml.a
#   android/app/build/intermediates/cxx/.../obj/<abi>/libggml-base.a
#   android/app/build/intermediates/cxx/.../obj/<abi>/libllamachat.so
#
# The .so is what gets packaged into the APK.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${ROOT}/llama.cpp"
JOBS="$(nproc)"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "ANDROID_NDK_HOME is not set. Install the NDK and re-run." >&2
    echo "Example: export ANDROID_NDK_HOME=\$HOME/Android/Sdk/ndk/26.1.10909125" >&2
    exit 1
fi

if [[ ! -d "${ANDROID_NDK_HOME}" ]]; then
    echo "ANDROID_NDK_HOME points to a missing directory: ${ANDROID_NDK_HOME}" >&2
    exit 1
fi

# 1. Clone llama.cpp if needed.
if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
    echo "Cloning llama.cpp into ${LLAMA_CPP_DIR}…"
    git clone --depth=1 https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_DIR}"
else
    echo "Updating llama.cpp in ${LLAMA_CPP_DIR}…"
    (cd "${LLAMA_CPP_DIR}" && git pull --ff-only || true)
fi

# 2. Build each ABI. arm64-v8a is the default; add others if needed.
ANDROID_API="${ANDROID_API:-24}"
TARGETS_DEFAULT=(arm64-v8a)
TARGETS="${TARGETS:-${TARGETS_DEFAULT[*]}}"

# Toolchain helper that ships with the NDK.
TOOLCHAIN="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake"

# Build out-of-tree so we don't litter the source tree.
BUILD_ROOT="${ROOT}/.build/llama-android"
mkdir -p "${BUILD_ROOT}"

for ABI in ${TARGETS}; do
    echo
    echo "=== Building llama.cpp for ${ABI} ==="
    BUILD_DIR="${BUILD_ROOT}/${ABI}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    cmake -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_PLATFORM="android-${ANDROID_API}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_OPENMP=OFF \
        "${LLAMA_CPP_DIR}"

    cmake --build . --config Release -j "${JOBS}"

    echo "Built: ${BUILD_DIR}/bin/ (static libs in ${BUILD_DIR}/lib/)"
done

echo
echo "Done. Gradle's externalNativeBuild will pick these up via"
echo "  add_subdirectory(${LLAMA_CPP_DIR}) in CMakeLists.txt."