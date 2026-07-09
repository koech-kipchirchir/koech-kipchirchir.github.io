package com.example.aios.sse

sealed class AIEvent {

    data class Token(val text: String) : AIEvent()

    data class Tool(
        val name: String,
        val output: String
    ) : AIEvent()

    data class Debug(val info: String) : AIEvent()

    data class Reflection(val status: String) : AIEvent()
}