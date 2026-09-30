package com.example.llamachat.ui

import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.view.inputmethod.EditorInfo
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.llamachat.R
import com.example.llamachat.databinding.ActivityMainBinding
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.launch

/**
 * Single-screen chat Activity. Streams model state from
 * [ChatViewModel] into the [MessageAdapter] and status row.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val vm: ChatViewModel by viewModels()
    private val adapter = MessageAdapter()

    /** Last status seen by [renderStatus]. Used to fire a snackbar
     *  only on the Idle/Generating/Done → Error edge, not on every
     *  re-collect after a config change. */
    private var lastStatus: ChatViewModel.Status? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        binding.recycler.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        binding.recycler.adapter = adapter

        binding.input.movementMethod = ScrollingMovementMethod()

        // The sendButton's label flips between "Send" and "Stop" based
        // on the engine state (see renderStatus). We dispatch on the
        // current status here rather than swapping the click listener,
        // so the same listener stays installed across status changes.
        binding.sendButton.setOnClickListener { onSendOrStopClicked() }
        binding.input.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                onSendOrStopClicked()
                true
            } else false
        }
        // Wire up the TextInputLayout's end icon (send icon) so users
        // on higher Android versions get a visible tap target. Without
        // this, the IME action is the only way to send, which can be
        // unreliable on API 30+ with multi-line input.
        binding.inputLayout.setEndIconOnClickListener { onSendOrStopClicked() }

        binding.toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_clear -> { vm.clear(); true }
                R.id.action_settings -> {
                    SettingsDialog.show(this, vm.settings.value) { vm.applySettings(it) }
                    true
                }
                else -> false
            }
        }

        observeState()

        // Initial model load. All UI feedback flows through the
        // status StateFlow → observeState() below, so we don't need
        // to touch views from this call.
        binding.loadingOverlay.visibility = View.VISIBLE
        binding.loadingSubtitle.text = getString(R.string.status_copying_model, 0)
        vm.loadModel()
    }

    private fun onSendOrStopClicked() {
        // The button label is the source of truth: "Stop" while the
        // engine is generating, "Send" otherwise. Dispatching on the
        // status here (rather than rewriting the listener on each
        // status change) keeps the wiring in one place and avoids
        // stale-lambda surprises on configuration changes.
        if (vm.status.value is ChatViewModel.Status.Generating) {
            vm.stopGeneration()
        } else {
            sendCurrentInput()
        }
    }

    private fun sendCurrentInput() {
        val text = binding.input.text?.toString()?.trim().orEmpty()
        if (text.isEmpty()) return
        binding.input.setText("")
        vm.sendUserMessage(text)
    }

    private fun observeState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    vm.messages.collect { msgs ->
                        adapter.submitList(msgs) {
                            if (msgs.isNotEmpty()) {
                                binding.recycler.scrollToPosition(msgs.lastIndex)
                            }
                        }
                        binding.emptyState.visibility =
                            if (msgs.isEmpty()) View.VISIBLE else View.GONE
                    }
                }
                launch {
                    vm.status.collect { st -> renderStatus(st) }
                }
            }
        }
    }

    private fun renderStatus(st: ChatViewModel.Status) {
        val prev = lastStatus
        lastStatus = st
        if (st is ChatViewModel.Status.Error && prev !is ChatViewModel.Status.Error) {
            Snackbar.make(binding.root,
                getString(R.string.err_load_failed, st.message),
                Snackbar.LENGTH_LONG).show()
        }
        when (st) {
            ChatViewModel.Status.Idle -> {
                binding.loadingOverlay.visibility = View.GONE
                binding.statusText.text = getString(R.string.status_idle)
                binding.sendButton.text = getString(R.string.action_send)
                binding.sendButton.isEnabled = vm.isModelReady()
            }
            is ChatViewModel.Status.Loading -> {
                binding.loadingOverlay.visibility = View.VISIBLE
                binding.loadingSubtitle.text =
                    getString(R.string.status_copying_model, st.pct)
                binding.sendButton.isEnabled = false
            }
            is ChatViewModel.Status.Generating -> {
                binding.loadingOverlay.visibility = View.GONE
                binding.statusText.text = getString(R.string.status_tokens, st.tokens, st.max)
                binding.sendButton.text = getString(R.string.action_stop)
                binding.sendButton.isEnabled = true
            }
            is ChatViewModel.Status.Done -> {
                binding.loadingOverlay.visibility = View.GONE
                binding.statusText.text = getString(R.string.status_tokens_per_s, st.tokPerSec)
                binding.sendButton.text = getString(R.string.action_send)
                binding.sendButton.isEnabled = vm.isModelReady()
            }
            is ChatViewModel.Status.Error -> {
                binding.loadingOverlay.visibility = View.GONE
                binding.statusText.text = st.message
                binding.sendButton.text = getString(R.string.action_send)
                binding.sendButton.isEnabled = vm.isModelReady()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isFinishing) {
            // Don't unload on rotation — let the engine survive config
            // changes. Only free on full process teardown.
            // vm.engine is private; rely on AndroidViewModel to be GC'd.
        }
    }
}
