package com.aios.android.domain.model

data class Chat(
    val id: String = "",
    val title: String = "New Chat",
    val messages: List<Message> = emptyList(),
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val model: String = "gpt-4o",
    val isLocal: Boolean = false
)
