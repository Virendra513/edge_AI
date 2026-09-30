package com.example.llamachat.inference

/**
 * One turn in the chat. Used both for UI rendering and for building
 * the prompt string in [LlamaEngine.buildPrompt].
 */
data class Message(
    val role: Role,
    val content: String,
) {
    enum class Role { SYSTEM, USER, ASSISTANT }
}
