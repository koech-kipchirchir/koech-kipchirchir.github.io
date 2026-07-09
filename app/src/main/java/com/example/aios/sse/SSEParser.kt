package com.example.aios.sse

class SSEParser {

    fun parse(line: String): Pair<String?, String?> {
        val trimmed = line.trim()

        if (trimmed.isEmpty()) return null to null

        return when {
            trimmed.startsWith("event:") ->
                "event" to trimmed.removePrefix("event:").trim()

            trimmed.startsWith("data:") ->
                "data" to trimmed.removePrefix("data:").trim()

            else -> null to null
        }
    }
}