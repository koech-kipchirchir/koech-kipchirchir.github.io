package com.aios.android.domain.model

data class Message(
    val id: String = "",
    val chatId: String = "",
    val role: MessageRole = MessageRole.USER,
    val content: String = "",
    val contentParts: List<ContentPart> = emptyList(),
    val createdAt: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false,
    val isError: Boolean = false,
    val metadata: Map<String, String> = emptyMap()
)

enum class MessageRole { USER, ASSISTANT, SYSTEM }

sealed class ContentPart {
    data class Text(val text: String) : ContentPart()
    data class Image(val url: String, val mimeType: String = "image/jpeg") : ContentPart()
    data class File(val name: String, val url: String, val mimeType: String = "application/octet-stream") : ContentPart()
    data class Code(val language: String, val code: String) : ContentPart()
}
