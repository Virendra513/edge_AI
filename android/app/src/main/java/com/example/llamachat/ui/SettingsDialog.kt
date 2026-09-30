package com.example.llamachat.ui

import android.app.AlertDialog
import android.content.Context
import android.view.LayoutInflater
import com.example.llamachat.R
import com.example.llamachat.databinding.DialogSettingsBinding
import com.example.llamachat.inference.InferenceSettings
import com.example.llamachat.inference.SamplerParams

/**
 * Modal settings editor. Returns the new [InferenceSettings] via the
 * `onResult` lambda (or `null` if the user cancelled).
 */
object SettingsDialog {

    fun show(
        ctx: Context,
        current: InferenceSettings,
        onResult: (InferenceSettings) -> Unit,
    ) {
        val b = DialogSettingsBinding.inflate(LayoutInflater.from(ctx))
        b.inTemperature.setText(current.sampler.temperature.toString())
        b.inTopP.setText(current.sampler.topP.toString())
        b.inTopK.setText(current.sampler.topK.toString())
        b.inMaxTokens.setText(current.maxNewTokens.toString())
        b.inThreads.setText(current.nThreads.toString())
        b.inCtx.setText(current.nCtx.toString())
        b.inGpuLayers.setText(current.nGpuLayers.toString())
        b.inSystem.setText(current.systemPrompt)

        AlertDialog.Builder(ctx)
            .setTitle(R.string.settings_title)
            .setView(b.root)
            .setPositiveButton(R.string.settings_save) { _, _ ->
                val updated = InferenceSettings(
                    nCtx         = b.inCtx.text.toString().toIntOrNull() ?: current.nCtx,
                    nThreads     = b.inThreads.text.toString().toIntOrNull() ?: current.nThreads,
                    nGpuLayers   = b.inGpuLayers.text.toString().toIntOrNull() ?: current.nGpuLayers,
                    maxNewTokens = b.inMaxTokens.text.toString().toIntOrNull() ?: current.maxNewTokens,
                    sampler      = SamplerParams(
                        temperature = b.inTemperature.text.toString().toFloatOrNull()
                            ?: current.sampler.temperature,
                        topP        = b.inTopP.text.toString().toFloatOrNull()
                            ?: current.sampler.topP,
                        topK        = b.inTopK.text.toString().toIntOrNull()
                            ?: current.sampler.topK,
                        repeatPenalty = current.sampler.repeatPenalty,
                        nPrev       = current.sampler.nPrev,
                    ),
                    systemPrompt = b.inSystem.text?.toString()?.ifBlank { current.systemPrompt }
                        ?: current.systemPrompt,
                )
                onResult(updated)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .setNeutralButton(R.string.settings_reset) { _, _ ->
                onResult(InferenceSettings.DEFAULT)
            }
            .show()
    }
}
