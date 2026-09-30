package com.example.llamachat.inference

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.channels.awaitClose

/**
 * Thin Kotlin wrapper around the native llama.cpp JNI surface.
 *
 * Lifecycle:
 *   1. [init]   — call once before any other method.
 *   2. [ensureModel] — copies the bundled GGUF from `assets/models/`
 *      into the app's private files dir on first launch.
 *   3. [load]   — load it into llama.cpp. Heavy: blocking.
 *   4. [generate] — streams tokens as a [Flow]. Cooperative cancel
 *      via [stop].
 *   5. [unload] — frees context + model. Safe to call repeatedly.
 *
 * All long-running calls are dispatched off the main thread by the
 * [Flow.flowOn] inside [generate].
 */
class LlamaEngine(private val appContext: Context) {

    @Volatile private var loaded = false
    @Volatile private var currentCtx = 2048
    @Volatile private var currentThreads = 1

    /** Path the model was last loaded from. Useful for diagnostics. */
    @Volatile var loadedPath: String? = null
        private set

    fun init() {
        // Idempotent on the C side; safe to call multiple times.
        nativeInit()
    }

    /**
     * Copy the GGUF bundled in `assets/models/model.gguf` to
     * `filesDir/models/model.gguf` so llama.cpp can `mmap` it. Skips
     * the copy if the file already exists with matching length.
     *
     * @return absolute path to the staged file.
     */
    suspend fun ensureModel(
        assetName: String = "models/model.gguf",
        onProgress: (copied: Long, total: Long) -> Unit = { _, _ -> }
    ): String = withContext(Dispatchers.IO) {
        val modelsDir = File(appContext.filesDir, "models").apply { mkdirs() }
        val target = File(modelsDir, "model.gguf")

        // Quick size check before opening the asset stream — avoids
        // reading the whole file twice on the happy path.
        val expectedSize = try {
            appContext.assets.openFd(assetName).use { it.length }
        } catch (t: Throwable) {
            throw IllegalStateException("Missing asset: $assetName", t)
        }

        if (target.exists() && target.length() == expectedSize) {
            onProgress(expectedSize, expectedSize)
            return@withContext target.absolutePath
        }

        target.delete()
        appContext.assets.open(assetName).use { input ->
            FileOutputStream(target).use { output ->
                val buf = ByteArray(1 shl 16) // 64 KiB
                var copied = 0L
                while (true) {
                    val n = input.read(buf)
                    if (n <= 0) break
                    output.write(buf, 0, n)
                    copied += n
                    onProgress(copied, expectedSize)
                }
            }
        }
        target.absolutePath
    }

    /** Blocking model load. Do NOT call on the main thread. */
    suspend fun load(
        modelPath: String,
        settings: InferenceSettings = InferenceSettings.DEFAULT,
    ): Boolean = withContext(Dispatchers.IO) {
        val threads = InferenceSettings.effectiveThreads(settings.nThreads)
        val ok = nativeLoad(
            modelPath,
            settings.nCtx,
            threads,
            settings.nGpuLayers,
        )
        loaded = ok
        if (ok) {
            loadedPath = modelPath
            currentCtx = settings.nCtx
            currentThreads = threads
        }
        ok
    }

    fun unload() {
        if (loaded) {
            nativeUnload()
            loaded = false
            loadedPath = null
        }
    }

    fun stop() {
        nativeStop()
    }

    /**
     * Build the prompt string for a conversation. Mirrors the
     * Llama-3.2-Instruct chat template. We do this in Kotlin instead
     * of letting llama.cpp parse JSON because it's easier to debug
     * and we don't need the model's tokenizer until we're already
     * past load().
     */
    fun buildPrompt(messages: List<Message>, addAssistantHeader: Boolean = true): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")
        for (m in messages) {
            val role = when (m.role) {
                Message.Role.SYSTEM -> "system"
                Message.Role.USER -> "user"
                Message.Role.ASSISTANT -> "assistant"
            }
            sb.append("<|start_header_id|>").append(role).append("<|end_header_id|>\n\n")
            sb.append(m.content.trim()).append("<|eot_id|>")
        }
        if (addAssistantHeader) {
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        }
        return sb.toString()
    }

    /**
     * Stream tokens from the model. Cooperative cancellation: call
     * [stop] from any thread to break out of the loop.
     *
     * Tokens are emitted as small `String`s, NOT character-by-character.
     * Llama.cpp's piece decoder can return multi-byte UTF-8 fragments
     * for a single token; callers should append them directly.
     */
    
    
    fun generate(
        prompt: String,
        sampler: SamplerParams = SamplerParams.DEFAULT,
        maxTokens: Int = 256,
    ): Flow<String> = callbackFlow {
        val cb = object : TokenCallback {
            override fun onToken(token: String, cumulative: Int) {
                trySend(token)
            }
        }
        // nativeGenerate is synchronous — by the time it returns every
        // token has already been delivered via trySend.
        nativeGenerate(prompt, sampler, maxTokens, cb)
        close()
        awaitClose { }
    }.flowOn(Dispatchers.IO)

    /** Internal helper for tests/diagnostics: tokenize a string. */
    fun isReady(): Boolean = loaded

    /** Last configured context length — exposed for the UI status row. */
    fun currentCtxSize(): Int = currentCtx

    /** Last configured thread count. */
    fun currentThreadCount(): Int = currentThreads

    // ------------------------------------------------------------------
    // Native surface — symbols are defined in cpp/native_jni.cpp.
    // ------------------------------------------------------------------
    private external fun nativeInit()
    private external fun nativeLoad(
        path: String,
        nCtx: Int,
        nThreads: Int,
        nGpuLayers: Int,
    ): Boolean
    private external fun nativeUnload()
    private external fun nativeStop()
    private external fun nativeApplyChatTemplate(
        messagesJson: String,
        addAssistant: Boolean,
    ): String
    private external fun nativeGenerate(
        prompt: String,
        sampler: SamplerParams,
        maxTokens: Int,
        callback: TokenCallback,
    ): Int

    companion object {
        @Volatile private var libraryLoaded = false

        /**
         * The JNI library is named `libllamachat.so` (see CMakeLists).
         * Loading it twice (e.g. across Activity restarts) is safe but
         * noisy in logcat.
         */
        fun loadLibraryOnce() {
            if (!libraryLoaded) {
                System.loadLibrary("llamachat")
                libraryLoaded = true
            }
        }
    }
}

/**
 * Producer-scope helper kept here for callers that prefer a suspending
 * variant of [LlamaEngine.generate]. Not currently used by the UI but
 * handy for tests.
 */
suspend fun ProducerScope<String>.drain(generate: () -> Int): Int {
    // Reserved for future API expansion; intentionally minimal.
    return generate()
}
