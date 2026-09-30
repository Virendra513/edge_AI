package com.example.llamachat.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.llamachat.inference.InferenceSettings
import com.example.llamachat.inference.LlamaEngine
import com.example.llamachat.inference.Message
import com.example.llamachat.inference.SamplerParams
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Owns the chat state and brokers calls to [LlamaEngine].
 *
 * The ViewModel survives Activity configuration changes.
 *
 * Responsibilities:
 *  - Initialize llama.cpp
 *  - Load the GGUF model
 *  - Maintain chat messages
 *  - Start/stop generation
 *  - Stream generated tokens into the UI
 *  - Maintain inference status
 *  - Apply inference settings
 */
class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = LlamaEngine(app)

    // ---------------------------------------------------------------------
    // Chat messages
    // ---------------------------------------------------------------------

    private val _messages =
        MutableStateFlow<List<Message>>(emptyList())

    val messages: StateFlow<List<Message>> =
        _messages.asStateFlow()

    // ---------------------------------------------------------------------
    // Status
    // ---------------------------------------------------------------------

    private val _status: MutableStateFlow<Status> =
        MutableStateFlow<Status>(Status.Idle)

    val status: StateFlow<Status> =
        _status.asStateFlow()

    /** True once the GGUF has been successfully loaded into llama.cpp. */
    fun isModelReady(): Boolean = engine.isReady()

    // ---------------------------------------------------------------------
    // Settings
    // ---------------------------------------------------------------------

    val settings: MutableStateFlow<InferenceSettings> =
        MutableStateFlow(InferenceSettings.DEFAULT)

    // ---------------------------------------------------------------------
    // Generation job
    // ---------------------------------------------------------------------

    private var generationJob: Job? = null

    // ---------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------

    init {
        LlamaEngine.loadLibraryOnce()
        engine.init()
    }

    // ---------------------------------------------------------------------
    // Model loading
    // ---------------------------------------------------------------------

    /**
     * Called once when the Activity starts.
     *
     * Copies the GGUF model from assets if necessary and loads it
     * through llama.cpp. UI consumers should observe [status] for
     * progress and error reporting — every state change goes through
     * [_status] so updates are delivered on the main thread.
     */
    fun loadModel() {
        viewModelScope.launch {

            try {

                _status.value = Status.Loading(0)

                val path = engine.ensureModel { copied, total ->

                    val pct =
                        if (total > 0) {
                            ((copied.toDouble() / total) * 100.0)
                                .toInt()
                                .coerceIn(0, 100)
                        } else {
                            0
                        }

                    _status.value = Status.Loading(pct)
                }

                val ok = engine.load(
                    path,
                    settings.value
                )

                if (!ok) {

                    val error =
                        IllegalStateException(
                            "llama_load returned false"
                        )

                    _status.value =
                        Status.Error(error.message ?: "Model loading failed")

                    return@launch
                }

                _status.value = Status.Idle

            } catch (t: Throwable) {

                _status.value =
                    Status.Error(
                        t.message ?: "Model loading failed"
                    )
            }
        }
    }

    // ---------------------------------------------------------------------
    // Send user message
    // ---------------------------------------------------------------------

    fun sendUserMessage(text: String) {

        if (text.isBlank()) {
            return
        }

        // Do not allow another generation while one is running.
        if (generationJob?.isActive == true) {
            return
        }

        // Do not allow generation before the model is loaded.
        // Without this guard the JNI side logs "model not loaded"
        // and the user sees nothing.
        if (!engine.isReady()) {

            _status.value =
                Status.Error(
                    "Model is still loading. " +
                        "Please wait until it finishes."
                )

            return
        }

        val next =
            _messages.value +
                Message(
                    Message.Role.USER,
                    text
                )

        _messages.value = next

        startAssistantTurn(next)
    }

    // ---------------------------------------------------------------------
    // Stop generation
    // ---------------------------------------------------------------------

    fun stopGeneration() {

        // Tell llama.cpp to stop its native generation loop.
        engine.stop()

        // Cancel Kotlin coroutine.
        generationJob?.cancel()
        generationJob = null

        // Immediately update UI.
        _status.value = Status.Idle
    }

    // ---------------------------------------------------------------------
    // Clear conversation
    // ---------------------------------------------------------------------

    fun clear() {

        stopGeneration()

        _messages.value = emptyList()

        _status.value = Status.Idle
    }

    // ---------------------------------------------------------------------
    // Apply inference settings
    // ---------------------------------------------------------------------

    fun applySettings(
        newSettings: InferenceSettings
    ) {

        /*
         * llama.cpp does not resize an existing context.
         *
         * Therefore, if nCtx or GPU layer configuration changes,
         * reload the model.
         */
        val oldSettings = settings.value

        val reload =
            newSettings.nCtx != oldSettings.nCtx ||
            newSettings.nGpuLayers != oldSettings.nGpuLayers

        settings.value = newSettings

        if (reload && engine.isReady()) {

            viewModelScope.launch {

                try {

                    val path =
                        engine.loadedPath
                            ?: return@launch

                    _status.value = Status.Loading(0)

                    val ok =
                        engine.load(
                            path,
                            newSettings
                        )

                    if (ok) {
                        _status.value = Status.Idle
                    } else {
                        _status.value =
                            Status.Error(
                                "Failed to reload model"
                            )
                    }

                } catch (t: Throwable) {

                    _status.value =
                        Status.Error(
                            t.message
                                ?: "Failed to reload model"
                        )
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Start assistant generation
    // ---------------------------------------------------------------------

    private fun startAssistantTurn(
        history: List<Message>
    ) {

        // Build the Llama-3.2 prompt.
        val prompt =
            engine.buildPrompt(
                history,
                addAssistantHeader = true
            )

        // Create sampler configuration.
        val sampler =
            SamplerParams(
                temperature =
                    settings.value.sampler.temperature,

                topP =
                    settings.value.sampler.topP,

                topK =
                    settings.value.sampler.topK,

                repeatPenalty =
                    settings.value.sampler.repeatPenalty,

                nPrev =
                    settings.value.sampler.nPrev
            )

        /*
         * Add an empty assistant message immediately.
         *
         * The generated tokens will be appended to this message
         * as they arrive from llama.cpp.
         */
        val assistantIndex =
            history.size

        _messages.value =
            history +
                Message(
                    Message.Role.ASSISTANT,
                    ""
                )

        // -----------------------------------------------------------------
        // Start coroutine
        // -----------------------------------------------------------------

        generationJob =
            viewModelScope.launch {

                val start =
                    System.nanoTime()

                var tokens = 0

                /*
                 * Explicit type is useful here and avoids Kotlin
                 * inference problems around MutableStateFlow<Status>.
                 */
                _status.value =
                    Status.Generating(
                        tokens = 0,
                        max = settings.value.maxNewTokens
                    )

                try {

                    /*
                     * engine.generate() returns a Flow<String>.
                     *
                     * collect() waits until generation is completely
                     * finished.
                     */
                    engine.generate(
                        prompt,
                        sampler,
                        settings.value.maxNewTokens
                    ).collect { piece ->

                        tokens++

                        // -------------------------------------------------
                        // Update assistant message
                        // -------------------------------------------------

                        val current =
                            _messages.value

                        if (assistantIndex < current.size) {

                            val updated =
                                current.toMutableList()

                            updated[assistantIndex] =
                                current[assistantIndex].copy(
                                    content =
                                        current[assistantIndex].content +
                                            piece
                                )

                            _messages.value =
                                updated
                        }

                        // -------------------------------------------------
                        // Update generation status
                        // -------------------------------------------------

                        _status.value =
                            Status.Generating(
                                tokens = tokens,
                                max =
                                    settings.value.maxNewTokens
                            )
                    }

                    // -----------------------------------------------------
                    // Generation completed
                    // -----------------------------------------------------

                    val elapsed =
                        (System.nanoTime() - start) /
                            1e9

                    val tokPerSec =
                        if (elapsed > 0.05) {
                            tokens / elapsed
                        } else {
                            0.0
                        }

                    _status.value =
                        Status.Done(
                            tokens = tokens,
                            tokPerSec = tokPerSec
                        )

                } catch (t: Throwable) {

                    /*
                     * Coroutine cancellation is expected when the user
                     * presses Stop. Don't display it as an error.
                     */
                    if (t is kotlinx.coroutines.CancellationException) {

                        _status.value =
                            Status.Idle

                    } else {

                        _status.value =
                            Status.Error(
                                t.message
                                    ?: "Inference error"
                            )
                    }
                } finally {

                    /*
                     * Clear the reference so the next send isn't gated
                     * by a stale Job. sendUserMessage() also checks
                     * isActive, but nulling here is the simpler
                     * correctness invariant: at most one Job is "live"
                     * at a time, and any completed one is gone.
                     */
                    generationJob = null
                }
            }
    }

    // ---------------------------------------------------------------------
    // Status definitions
    // ---------------------------------------------------------------------

    sealed class Status {

        /**
         * No model operation is currently running.
         */
        data object Idle : Status()

        /**
         * GGUF model is being copied/loaded.
         */
        data class Loading(
            val pct: Int
        ) : Status()

        /**
         * Model is generating tokens.
         */
        data class Generating(
            val tokens: Int,
            val max: Int
        ) : Status()

        /**
         * Generation completed successfully.
         */
        data class Done(
            val tokens: Int,
            val tokPerSec: Double
        ) : Status()

        /**
         * Model loading or generation failed.
         */
        data class Error(
            val message: String
        ) : Status()
    }
}