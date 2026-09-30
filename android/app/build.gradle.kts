// App module: LlamaChat
// Loads the Q2_K GGUF (~554 MB) from app-private storage and exposes a
// Kotlin API around the native llama.cpp JNI bridge.

import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.Exec

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.llamachat"
    compileSdk = 34
    ndkVersion = "26.1.10909125"
    

    buildFeatures {
        // Generates ActivityMainBinding / MessageItemBinding / etc.
        viewBinding = true
    }

    defaultConfig {
        applicationId = "com.example.llamachat"
        minSdk = 24                 // Android 7.0 — covers ~98% of devices.
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        // Q2_K is the smallest variant and fits comfortably even on
        // older phones, but Android devices come in several ABIs.
        // arm64-v8a is the dominant one today (most phones since 2017).
        // armeabi-v7a covers older 32-bit phones.
        // x86_64 covers emulators and Chromebooks (Android subsystem).
        // Shipping all three keeps the APK visible on every device that
        // meets minSdk=24; splits let Play serve the right one.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        // Produce one APK per ABI in addition to the universal APK so
        // Play Store can serve the smallest matching download.
        splits {
            abi {
                isEnable = true
                reset()
                include("arm64-v8a")
                isUniversalApk = true
            }
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fvisibility=hidden", "-fexceptions")
                // llama.cpp is CPU-heavy — let the build optimise for the
                // target ABI. For shipping you can switch to -O2/-Os.
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_PLATFORM=android-24",
                    "-DGGML_OPENMP=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DLLAMA_CURL=OFF",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Sign with the debug key so `assembleRelease` runs locally.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // The Q2_K GGUF is ~554 MB — large for an APK asset. Compress it
    // so the resulting AAB stays under Play's 150 MB cap, and disable
    // PNG crunching (we ship no PNGs beyond the launcher).
    androidResources {
        noCompress += listOf("gguf")
    }

    packaging {
        resources {
            excludes += listOf("META-INF/AL2.0", "META-INF/LGPL2.1")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.4")
    implementation("androidx.activity:activity-ktx:1.9.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}

// Copy the chosen GGUF into assets/models/ whenever `copyModel` runs.
// Run `./gradlew copyModel` after editing COPY_VARIANT below.
tasks.register<Copy>("copyModel") {
    val variant = providers.gradleProperty("COPY_VARIANT").orElse("Q2_K")
    val srcRoot = file("../llama-3.2-1B-Instruct-gguf/")
    val src = file("$srcRoot/Llama-3.2-1B-Instruct.${variant.get()}.gguf")
    val dstDir = file("src/main/assets/models")
    from(src)
    into(dstDir)
    rename("(.+)\\.gguf", "model.gguf")
    doFirst {
        if (!src.exists()) {
            throw GradleException(
                "GGUF not found: ${src.path}\n" +
                "Run `scripts/copy_model.sh` or check COPY_VARIANT (current: ${variant.get()})"
            )
        }
    }
}

// Make `assemble` depend on the copy so a fresh checkout always has a
// model staged.
afterEvaluate {
    tasks.named("preBuild").configure { dependsOn("copyModel") }
}
