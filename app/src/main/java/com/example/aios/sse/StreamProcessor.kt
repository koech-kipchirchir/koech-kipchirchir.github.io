package com.example.aios.sse

import java.io.InputStream

class StreamProcessor {

    private val parser = SSEParser()

    suspend fun processStream(
        input: InputStream,
        emit: (AIEvent) -> Unit
    ) {
        val reader = input.bufferedReader()

        var currentEvent: String? = null

        reader.forEachLine { line: String ->

            val (type, value) = parser.parse(line)

            when (type) {

                "event" -> currentEvent = value

                "data" -> {

                    when (currentEvent) {

                        "token" -> emit(AIEvent.Token(value ?: ""))

                        "tool" -> {
                            val parts = value?.split("::") ?: listOf()
                            emit(
                                AIEvent.Tool(
                                    name = parts.getOrNull(0) ?: "unknown",
                                    output = parts.getOrNull(1) ?: ""
                                )
                            )
                        }

                        "debug" -> emit(AIEvent.Debug(value ?: ""))

                        "reflection" -> emit(AIEvent.Reflection(value ?: ""))
                    }
                }
            }
        }
    }
}