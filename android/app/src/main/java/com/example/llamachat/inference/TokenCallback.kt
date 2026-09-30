package com.example.llamachat.inference

/**
 * Receives per-token callbacks from the native bridge. `cumulative` is
 * the 1-based count of tokens emitted so far in this generation, useful
 * for "12 / 256 tokens" status UI.
 */
interface TokenCallback {
    fun onToken(token: String, cumulative: Int)
}
