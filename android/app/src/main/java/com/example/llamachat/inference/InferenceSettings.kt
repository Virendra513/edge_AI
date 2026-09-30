package com.example.llamachat.inference

/**
 * Top-level inference knobs. Persisted in SharedPreferences by
 * ChatViewModel; passed into LlamaEngine.load().
 */
data class InferenceSettings(
    val nCtx: Int = 2048,
    val nThreads: Int = 0,            // 0 -> auto (min(cores, 4))
    val nGpuLayers: Int = 0,          // 0 -> CPU only
    val maxNewTokens: Int = 512,
    val sampler: SamplerParams = SamplerParams.DEFAULT,
    val systemPrompt: String = "You are a helpful assistant.",
) {
    companion object {
        val DEFAULT = InferenceSettings()

        fun effectiveThreads(requested: Int): Int {
            val cores = Runtime.getRuntime().availableProcessors()
            val target = if (requested <= 0) cores else requested
            return target.coerceIn(1, 4)
        }
    }
}
