package com.example.llamachat.inference

/**
 * Plain data holder for sampler parameters. Field names MUST match the
 * JNI bridge (native_jni.cpp readFloat/readInt).
 */
data class SamplerParams(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val repeatPenalty: Float = 1.1f,
    val nPrev: Int = 64,
) {
    companion object {
        val DEFAULT = SamplerParams()
    }
}
