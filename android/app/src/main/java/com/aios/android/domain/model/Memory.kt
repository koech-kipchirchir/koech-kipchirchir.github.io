package com.aios.android.domain.model

data class Memory(
    val id: String = "",
    val key: String = "",
    val value: String = "",
    val category: String = "general",
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val source: String = "manual"
)

data class Document(
    val id: String = "",
    val name: String = "",
    val path: String = "",
    val sizeBytes: Long = 0,
    val mimeType: String = "application/octet-stream",
    val createdAt: Long = System.currentTimeMillis(),
    val isLocal: Boolean = true,
    val content: String = ""
)

data class AiosModel(
    val id: String = "",
    val name: String = "",
    val provider: String = "",
    val sizeBytes: Long = 0,
    val isDownloaded: Boolean = false,
    val isBuiltin: Boolean = false,
    val capabilities: List<String> = emptyList()
)
