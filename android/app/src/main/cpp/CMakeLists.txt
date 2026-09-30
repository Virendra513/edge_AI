# Builds libllamachat.so — JNI wrapper around llama.cpp.
#
# Project structure:
#
# deploy/
# ├── llama.cpp/
# └── android/
#     └── app/
#         └── src/
#             └── main/
#                 └── cpp/
#                     ├── CMakeLists.txt
#                     └── native_jni.cpp
#
# This CMake file:
#   1. Finds llama.cpp
#   2. Builds llama.cpp for Android
#   3. Builds our JNI wrapper
#   4. Links llama.cpp into libllamachat.so
#

cmake_minimum_required(VERSION 3.22.1)

project(llamachat LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# C++ configuration
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)


# ---------------------------------------------------------------------------
# Locate llama.cpp
#
# Current file:
#
# deploy/android/app/src/main/cpp/CMakeLists.txt
#
# llama.cpp:
#
# deploy/llama.cpp/
#
# Therefore we go:
#
# cpp -> main -> src -> app -> android -> deploy -> llama.cpp
# ---------------------------------------------------------------------------

set(LLAMA_CPP_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../llama.cpp"
)

# Print path during Gradle/CMake configuration
message(STATUS "========================================")
message(STATUS "llama.cpp configuration")
message(STATUS "LLAMA_CPP_DIR = ${LLAMA_CPP_DIR}")
message(STATUS "========================================")


# ---------------------------------------------------------------------------
# Verify llama.cpp exists
# ---------------------------------------------------------------------------

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")

    message(FATAL_ERROR
        "\n"
        "llama.cpp sources not found.\n"
        "\n"
        "Expected location:\n"
        "  ${LLAMA_CPP_DIR}\n"
        "\n"
        "Expected file:\n"
        "  ${LLAMA_CPP_DIR}/CMakeLists.txt\n"
        "\n"
        "Check that llama.cpp exists at:\n"
        "  /home/virendra/work/LFT/modle321B/deploy/llama.cpp\n"
    )

endif()

message(STATUS "Found llama.cpp successfully!")


# ---------------------------------------------------------------------------
# llama.cpp build configuration
#
# We are building an Android application.
#
# We DON'T need:
#   - llama CLI
#   - llama-server
#   - benchmarks
#   - examples
#   - tests
#   - Web UI
#   - quantization executable
#
# We only need the inference libraries.
# ---------------------------------------------------------------------------

set(LLAMA_BUILD_APP
    OFF
    CACHE BOOL
    "Build llama unified application"
    FORCE
)

set(LLAMA_BUILD_UI
    OFF
    CACHE BOOL
    "Build llama Web UI"
    FORCE
)

set(LLAMA_USE_PREBUILT_UI
    OFF
    CACHE BOOL
    "Use prebuilt llama Web UI"
    FORCE
)

set(LLAMA_BUILD_TOOLS
    OFF
    CACHE BOOL
    "Build llama tools"
    FORCE
)

set(LLAMA_BUILD_EXAMPLES
    OFF
    CACHE BOOL
    "Build llama examples"
    FORCE
)

set(LLAMA_BUILD_TESTS
    OFF
    CACHE BOOL
    "Build llama tests"
    FORCE
)

set(LLAMA_BUILD_SERVER
    OFF
    CACHE BOOL
    "Build llama server"
    FORCE
)

set(LLAMA_CURL
    OFF
    CACHE BOOL
    "Enable CURL"
    FORCE
)


# ---------------------------------------------------------------------------
# GGML configuration
#
# CPU inference for Android ARM64.
# ---------------------------------------------------------------------------

set(GGML_OPENMP
    OFF
    CACHE BOOL
    "Enable OpenMP"
    FORCE
)

set(GGML_VULKAN
    OFF
    CACHE BOOL
    "Enable Vulkan"
    FORCE
)

set(GGML_METAL
    OFF
    CACHE BOOL
    "Enable Metal"
    FORCE
)

set(GGML_OPENCL
    OFF
    CACHE BOOL
    "Enable OpenCL"
    FORCE
)

set(GGML_NATIVE
    OFF
    CACHE BOOL
    "Enable native CPU optimizations"
    FORCE
)


# ---------------------------------------------------------------------------
# Add llama.cpp
#
# EXCLUDE_FROM_ALL prevents unrelated llama.cpp targets from automatically
# becoming part of the Android application build.
# ---------------------------------------------------------------------------

add_subdirectory(
    "${LLAMA_CPP_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp_build"
    EXCLUDE_FROM_ALL
)


# ---------------------------------------------------------------------------
# JNI wrapper
#
# native_jni.cpp should contain the JNI functions that communicate with
# Kotlin/Java.
# ---------------------------------------------------------------------------

add_library(
    llamachat
    SHARED

    native_jni.cpp
)


# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------

target_include_directories(
    llamachat
    PRIVATE

    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}"
)


# ---------------------------------------------------------------------------
# Android system libraries
# ---------------------------------------------------------------------------

find_library(
    ANDROID_LOG_LIB
    log
)

find_library(
    ANDROID_LIB
    android
)


# ---------------------------------------------------------------------------
# Link llama.cpp
#
# llama.cpp provides the main inference library.
#
# Depending on the llama.cpp version, ggml targets are exposed separately.
# ---------------------------------------------------------------------------

target_link_libraries(
    llamachat
    PRIVATE

    llama

    ggml
    ggml-base

    ${ANDROID_LOG_LIB}
    ${ANDROID_LIB}
)


# ---------------------------------------------------------------------------
# Android compiler definitions
# ---------------------------------------------------------------------------

target_compile_definitions(
    llamachat
    PRIVATE

    LLAMA_ANDROID
)


# ---------------------------------------------------------------------------
# Android linker options
#
# Remove unused sections to reduce final .so size.
# ---------------------------------------------------------------------------

target_link_options(
    llamachat
    PRIVATE

    -Wl,--gc-sections
)


# ---------------------------------------------------------------------------
# Release optimization
# ---------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Release")

    target_compile_options(
        llamachat
        PRIVATE

        -O3
        -ffunction-sections
        -fdata-sections
    )

    target_link_options(
        llamachat
        PRIVATE

        -Wl,--gc-sections
        -s
    )

endif()


# ---------------------------------------------------------------------------
# Output information
# ---------------------------------------------------------------------------

message(STATUS "========================================")
message(STATUS "llamachat Android JNI configuration")
message(STATUS "ABI: ${ANDROID_ABI}")
message(STATUS "Android API: ${ANDROID_PLATFORM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "llama.cpp: ${LLAMA_CPP_DIR}")
message(STATUS "========================================")