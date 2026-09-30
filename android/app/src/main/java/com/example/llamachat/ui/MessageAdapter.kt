package com.example.llamachat.ui

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.llamachat.R
import com.example.llamachat.databinding.MessageItemBinding
import com.example.llamachat.inference.Message

/**
 * Renders a chat history as right-aligned user bubbles and
 * left-aligned assistant bubbles. Plain ListAdapter with DiffUtil —
 * no Paging, no Compose — keeps the dependency footprint small.
 */
class MessageAdapter : ListAdapter<Message, MessageAdapter.VH>(DIFF) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = MessageItemBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return VH(binding)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        holder.bind(getItem(position))
    }

    class VH(private val b: MessageItemBinding) : RecyclerView.ViewHolder(b.root) {
        fun bind(msg: Message) {
            b.bubble.text = msg.content
            when (msg.role) {
                Message.Role.USER -> {
                    b.bubble.setBackgroundResource(R.drawable.bubble_user)
                    b.bubble.setTextColor(b.root.context.getColor(R.color.text_user))
                    b.root.gravity = android.view.Gravity.END
                    b.meta.text = b.root.context.getString(R.string.label_user)
                }
                Message.Role.ASSISTANT -> {
                    b.bubble.setBackgroundResource(R.drawable.bubble_assistant)
                    b.bubble.setTextColor(b.root.context.getColor(R.color.text_assistant))
                    b.root.gravity = android.view.Gravity.START
                    b.meta.text = b.root.context.getString(R.string.label_assistant)
                }
                Message.Role.SYSTEM -> {
                    // System prompts aren't shown in the chat UI.
                    b.root.visibility = android.view.View.GONE
                    return
                }
            }
            b.root.visibility = android.view.View.VISIBLE
        }
    }

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<Message>() {
            override fun areItemsTheSame(old: Message, new: Message) =
                old === new || (old.role == new.role && old.content === new.content)
            override fun areContentsTheSame(old: Message, new: Message) =
                old == new
        }
    }
}
