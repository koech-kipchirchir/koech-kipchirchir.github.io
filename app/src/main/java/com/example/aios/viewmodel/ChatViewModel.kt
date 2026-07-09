package com.example.aios.viewmodel

import android.app.Application
import android.content.Context
import android.net.Uri
import android.speech.tts.TextToSpeech
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.RecognitionListener
import android.os.Bundle
import android.content.Intent
import androidx.compose.runtime.*
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.*
import com.example.aios.AndroidTools
import com.example.aios.ToolExecutionManager
import com.example.aios.sse.StreamProcessor
import com.example.aios.sse.AIEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import com.google.gson.Gson
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale
import java.util.concurrent.TimeUnit

// ==========================================
// 1. STATE MODELS
// ==========================================
data class Message(
    val text: String,
    val isUser: Boolean,
    val type: String = "text", // "text", "thinking", "tool_call", "error", "code"
    val toolName: String? = null,
    val status: String? = null, // "pending", "success", "failed"
    val imageUris: List<Uri>? = null,
)

// ==========================================
// 2. ROOM DB ENTITIES & DAOS
// ==========================================
@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val text: String,
    val isUser: Boolean,
    val type: String = "text",
    val toolName: String? = null,
    val status: String? = null
)

@Dao
interface ChatDao {
    @Query("SELECT * FROM messages ORDER BY id ASC")
    fun getAllMessagesFlow(): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages ORDER BY id ASC")
    suspend fun getAllMessages(): List<MessageEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMessage(message: MessageEntity): Long

    @Query("DELETE FROM messages")
    suspend fun clearHistory()
}

@Entity(tableName = "memory")
data class MemoryEntity(
    @PrimaryKey val key: String,
    val value: String
)

@Dao
interface MemoryDao {
    @Query("SELECT * FROM memory ORDER BY `key` ASC")
    fun getAllMemoryFlow(): Flow<List<MemoryEntity>>

    @Query("SELECT * FROM memory ORDER BY `key` ASC")
    suspend fun getAllMemory(): List<MemoryEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMemory(memory: MemoryEntity)

    @Query("DELETE FROM memory WHERE `key` = :key")
    suspend fun deleteMemory(key: String)
}

@Database(entities = [MessageEntity::class, MemoryEntity::class], version = 3, exportSchema = false)
abstract class ChatDatabase : RoomDatabase() {
    abstract fun chatDao(): ChatDao
    abstract fun memoryDao(): MemoryDao

    companion object {
        @Volatile
        private var INSTANCE: ChatDatabase? = null

        fun getDatabase(context: Context): ChatDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    ChatDatabase::class.java,
                    "chat_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}

// ==========================================
// 3. VIEWMODEL
// ==========================================
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val context = application.applicationContext
    private val database = ChatDatabase.getDatabase(application)
    private val chatDao = database.chatDao()
    private val memoryDao = database.memoryDao()
    private val gson = Gson()
    private val streamProcessor = StreamProcessor()

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.MINUTES)
        .build()

    // Message list state
    private val _messagesState = MutableStateFlow<List<Message>>(emptyList())
    val messagesState: StateFlow<List<Message>> = _messagesState.asStateFlow()

    // Memory list state
    private val _memoryState = MutableStateFlow<List<MemoryEntity>>(emptyList())
    val memoryState: StateFlow<List<MemoryEntity>> = _memoryState.asStateFlow()

    private val _streamText = mutableStateOf("")
    val streamText: State<String> = _streamText

    var loading by mutableStateOf(false)
        private set

    // Text-to-Speech
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    // Speech-to-Text (STT) Voice Input
    var isListening by mutableStateOf(false)
        private set
    private var speechRecognizer: SpeechRecognizer? = null
    var voiceTranscript = mutableStateOf("")
        private set

    // SharedPreferences Settings configuration
    private val prefs = context.getSharedPreferences("aios_settings", Context.MODE_PRIVATE)

    var llmProvider by mutableStateOf(prefs.getString("provider", "GEMINI_DIRECT") ?: "GEMINI_DIRECT")
    var geminiApiKey by mutableStateOf(prefs.getString("gemini_api_key", "") ?: "")
    var geminiModel by mutableStateOf(prefs.getString("gemini_model", "gemini-1.5-flash") ?: "gemini-1.5-flash")
    var customServerUrl by mutableStateOf(prefs.getString("custom_server_url", "http://10.0.2.2:8000/") ?: "http://10.0.2.2:8000/")
    var voiceOutputEnabled by mutableStateOf(prefs.getBoolean("voice_output_enabled", false))
    var systemInstruction by mutableStateOf(prefs.getString("system_instruction", "You are AIOS, the world's most advanced AI operating system deeply integrated into the user's Android device. Unlike ChatGPT or any other LLM, you have REAL native access to the device hardware and software. You can control the flashlight, set alarms, make calls, send texts, open apps, read clipboard, search contacts, manage calendar events, browse the web, control volume and brightness, manage files, track location, and remember facts about the user permanently. You are autonomous — when the user asks you to do something, DO IT by calling the appropriate tool. Never say you can't do something if a tool exists for it. Be concise, friendly, and proactive. Always summarize tool results clearly.") ?: "You are AIOS, the world's most advanced AI operating system deeply integrated into the user's Android device. Unlike ChatGPT or any other LLM, you have REAL native access to the device hardware and software. You can control the flashlight, set alarms, make calls, send texts, open apps, read clipboard, search contacts, manage calendar events, browse the web, control volume and brightness, manage files, track location, and remember facts about the user permanently. You are autonomous — when the user asks you to do something, DO IT by calling the appropriate tool. Never say you can't do something if a tool exists for it. Be concise, friendly, and proactive. Always summarize tool results clearly.")

    init {
        // Collect Message updates
        viewModelScope.launch {
            chatDao.getAllMessagesFlow().collect { savedEntities ->
                _messagesState.value = savedEntities.map {
                    Message(it.text, it.isUser, it.type, it.toolName, it.status)
                }
            }
        }

        // Collect Memory updates
        viewModelScope.launch {
            memoryDao.getAllMemoryFlow().collect { list ->
                _memoryState.value = list
            }
        }

        // Init Text to Speech
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                ttsReady = true
            }
        }
    }

    // Save preferences
    fun updateSettings(provider: String, apiKey: String, model: String, serverUrl: String, voiceEnabled: Boolean, systemPrompt: String) {
        llmProvider = provider
        geminiApiKey = apiKey
        geminiModel = model
        customServerUrl = serverUrl
        voiceOutputEnabled = voiceEnabled
        systemInstruction = systemPrompt

        prefs.edit().apply {
            putString("provider", provider)
            putString("gemini_api_key", apiKey)
            putString("gemini_model", model)
            putString("custom_server_url", serverUrl)
            putBoolean("voice_output_enabled", voiceEnabled)
            putString("system_instruction", systemPrompt)
            apply()
        }
    }

    fun speak(text: String) {
        if (voiceOutputEnabled && ttsReady) {
            val cleanText = text.replace(Regex("[*#_`~>\\[\\]()]"), "")
            tts?.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, "AIOS_TTS_ID")
        }
    }

    // ==========================================
    // SPEECH TO TEXT DICTATION CONTROLS
    // ==========================================
    fun startListening() {
        viewModelScope.launch(Dispatchers.Main) {
            try {
                if (speechRecognizer == null) {
                    speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
                    speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                        override fun onReadyForSpeech(params: Bundle?) {
                            isListening = true
                            voiceTranscript.value = "Listening..."
                        }

                        override fun onBeginningOfSpeech() {
                            voiceTranscript.value = "Listening..."
                        }

                        override fun onRmsChanged(rmsdB: Float) {}
                        override fun onBufferReceived(buffer: ByteArray?) {}

                        override fun onEndOfSpeech() {
                            isListening = false
                        }

                        override fun onError(error: Int) {
                            isListening = false
                            val message = when (error) {
                                SpeechRecognizer.ERROR_AUDIO -> "Audio error"
                                SpeechRecognizer.ERROR_CLIENT -> "Client error"
                                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "No microphone permission"
                                SpeechRecognizer.ERROR_NETWORK -> "Network error"
                                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Timeout"
                                SpeechRecognizer.ERROR_NO_MATCH -> "No speech match"
                                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
                                SpeechRecognizer.ERROR_SERVER -> "Server error"
                                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Speech timeout"
                                else -> "Speech recognition failed"
                            }
                            voiceTranscript.value = "Error: $message"
                        }

                        override fun onResults(results: Bundle?) {
                            isListening = false
                            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            val text = matches?.firstOrNull() ?: ""
                            if (text.isNotBlank()) {
                                voiceTranscript.value = text
                            }
                        }

                        override fun onPartialResults(partialResults: Bundle?) {
                            val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            val text = matches?.firstOrNull() ?: ""
                            if (text.isNotBlank()) {
                                voiceTranscript.value = text
                            }
                        }

                        override fun onEvent(eventType: Int, params: Bundle?) {}
                    })
                }

                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                }

                speechRecognizer?.startListening(intent)
                isListening = true
                voiceTranscript.value = "Listening..."
            } catch (e: Exception) {
                isListening = false
                voiceTranscript.value = "Recognizer error: ${e.localizedMessage}"
            }
        }
    }

    fun stopListening() {
        viewModelScope.launch(Dispatchers.Main) {
            speechRecognizer?.stopListening()
            isListening = false
        }
    }

    // ==========================================
    // SEND MESSAGE ROUTER
    // ==========================================
    fun sendMessage(text: String, imageUris: List<Uri>? = null) {
        if (text.isBlank() && imageUris.isNullOrEmpty()) return
        loading = true

        viewModelScope.launch(Dispatchers.IO) {
            try {
                // 1. Insert user query with image refs
                val msgText = if (imageUris.isNullOrEmpty()) {
                    text
                } else {
                    val imgRefs = imageUris.joinToString("\n") { uri ->
                        "![Image]($uri)"
                    }
                    "$imgRefs\n\n$text"
                }

                chatDao.insertMessage(
                    MessageEntity(text = msgText, isUser = true, type = "text")
                )

                val newMsg = Message(
                    text = msgText,
                    isUser = true,
                    imageUris = imageUris,
                )
                val currentList = _messagesState.value.toMutableList()
                currentList.add(newMsg)
                _messagesState.value = currentList

                // 2. Select execution logic
                if (llmProvider == "GEMINI_DIRECT") {
                    executeGeminiLoop()
                } else {
                    executeBackendSseLoop(text)
                }
            } catch (e: Exception) {
                postError("Engine Error: ${e.localizedMessage}")
            } finally {
                withContext(Dispatchers.Main) {
                    loading = false
                }
            }
        }
    }

    // ==========================================
    // DIRECT GEMINI AGENT LOOP
    // ==========================================
    private suspend fun executeGeminiLoop() {
        val apiKey = geminiApiKey
        if (apiKey.isBlank()) {
            postError("API Key is missing. Add your Google Gemini API Key in settings.")
            return
        }

        withContext(Dispatchers.Main) {
            _streamText.value = ""
        }

        var shouldContinue = true
        var loopCount = 0
        val maxLoops = 5

        while (shouldContinue && loopCount < maxLoops) {
            loopCount++
            shouldContinue = false // default to stop unless a tool runs

            val history = chatDao.getAllMessages()
            val memories = memoryDao.getAllMemory()

            val contentsList = mutableListOf<Map<String, Any>>()

            for (msg in history) {
                val role = if (msg.isUser) "user" else "model"
                val parts = mutableListOf<Map<String, Any>>()

                if (msg.type == "tool_call") {
                    if (msg.status == "success" || msg.status == "failed") {
                        parts.add(mapOf("text" to "System: Tool '${msg.toolName}' result: ${msg.text}"))
                        contentsList.add(mapOf("role" to "user", "parts" to parts))
                    }
                } else if (msg.type == "text") {
                    parts.add(mapOf("text" to msg.text))
                    contentsList.add(mapOf("role" to role, "parts" to parts))
                }
            }

            // Injects user memory facts into instructions
            val memoryPrompt = if (memories.isNotEmpty()) {
                "\n\nHere are some personal facts you remember about the user:\n" +
                        memories.joinToString("\n") { "- ${it.key}: ${it.value}" }
            } else {
                ""
            }
            val finalSystemInstruction = systemInstruction + memoryPrompt

            // Construct payload
            val requestMap = mutableMapOf<String, Any>()
            requestMap["contents"] = contentsList
            requestMap["systemInstruction"] = mapOf(
                "parts" to listOf(mapOf("text" to finalSystemInstruction))
            )

            // Register system tools
            val toolDeclarations = listOf(
                mapOf(
                    "name" to "set_alarm",
                    "description" to "Sets an alarm on the user's Android device.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "hour" to mapOf("type" to "INTEGER", "description" to "The hour of the alarm (0-23)"),
                            "minute" to mapOf("type" to "INTEGER", "description" to "The minute of the alarm (0-59)"),
                            "message" to mapOf("type" to "STRING", "description" to "The label/message for the alarm")
                        ),
                        "required" to listOf("hour", "minute")
                    )
                ),
                mapOf(
                    "name" to "toggle_flashlight",
                    "description" to "Turns the device flashlight/torch on or off.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "enabled" to mapOf("type" to "BOOLEAN", "description" to "True to turn on, false to turn off")
                        ),
                        "required" to listOf("enabled")
                    )
                ),
                mapOf(
                    "name" to "get_device_status",
                    "description" to "Retrieves the current status of the device, including battery level, charging status, volume, and current time.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf<String, Any>()
                    )
                ),
                mapOf(
                    "name" to "send_sms",
                    "description" to "Drafts or sends an SMS message to a phone number.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "phoneNumber" to mapOf("type" to "STRING", "description" to "The recipient phone number"),
                            "message" to mapOf("type" to "STRING", "description" to "The message body")
                        ),
                        "required" to listOf("phoneNumber", "message")
                    )
                ),
                mapOf(
                    "name" to "make_call",
                    "description" to "Launches the phone dialer to call a specific phone number.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "phoneNumber" to mapOf("type" to "STRING", "description" to "The phone number to call")
                        ),
                        "required" to listOf("phoneNumber")
                    )
                ),
                mapOf(
                    "name" to "vibrate",
                    "description" to "Vibrates the device.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "durationMs" to mapOf("type" to "INTEGER", "description" to "The duration of the vibration in milliseconds")
                        ),
                        "required" to listOf("durationMs")
                    )
                ),
                mapOf(
                    "name" to "open_app",
                    "description" to "Searches installed apps and opens the first app matching the given name.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "appName" to mapOf("type" to "STRING", "description" to "The name of the application to open, e.g., youtube, maps, gmail")
                        ),
                        "required" to listOf("appName")
                    )
                ),
                mapOf(
                    "name" to "get_current_location",
                    "description" to "Retrieves the current coordinates (latitude and longitude) of the device.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf<String, Any>()
                    )
                ),
                mapOf(
                    "name" to "save_memory_fact",
                    "description" to "Saves a personal fact or preference about the user into your long-term memory bank.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "key" to mapOf("type" to "STRING", "description" to "A short unique identifier for the fact, e.g., user_name, pet_type, likes_coffee"),
                            "value" to mapOf("type" to "STRING", "description" to "The fact description or value, e.g., John, cat named Whiskers, true")
                        ),
                        "required" to listOf("key", "value")
                    )
                ),
                mapOf(
                    "name" to "forget_memory_fact",
                    "description" to "Removes a specific personal fact or preference from your long-term memory bank by key.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "key" to mapOf("type" to "STRING", "description" to "The key of the fact to forget")
                        ),
                        "required" to listOf("key")
                    )
                ),
                // ========== PHASE 3: CLIPBOARD ==========
                mapOf(
                    "name" to "read_clipboard",
                    "description" to "Reads the current text content from the device clipboard.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf<String, Any>()
                    )
                ),
                mapOf(
                    "name" to "write_clipboard",
                    "description" to "Copies the given text to the device clipboard.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "text" to mapOf("type" to "STRING", "description" to "The text to copy to clipboard")
                        ),
                        "required" to listOf("text")
                    )
                ),
                // ========== PHASE 3: CONTACTS ==========
                mapOf(
                    "name" to "search_contacts",
                    "description" to "Searches the user's device contacts by name and returns matching contact names and phone numbers.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "name" to mapOf("type" to "STRING", "description" to "The contact name or partial name to search for")
                        ),
                        "required" to listOf("name")
                    )
                ),
                // ========== PHASE 3: WEB ==========
                mapOf(
                    "name" to "web_search",
                    "description" to "Opens a web search in the device browser for the given query.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "query" to mapOf("type" to "STRING", "description" to "The search query to look up on the web")
                        ),
                        "required" to listOf("query")
                    )
                ),
                mapOf(
                    "name" to "open_url",
                    "description" to "Opens a specific URL in the device's default web browser.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "url" to mapOf("type" to "STRING", "description" to "The full URL to open, e.g. https://google.com")
                        ),
                        "required" to listOf("url")
                    )
                ),
                // ========== PHASE 3: CALENDAR ==========
                mapOf(
                    "name" to "get_calendar_events",
                    "description" to "Retrieves the user's upcoming calendar events for the next specified number of days.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "daysAhead" to mapOf("type" to "INTEGER", "description" to "Number of days ahead to check for events (default: 1 = today)")
                        )
                    )
                ),
                mapOf(
                    "name" to "create_calendar_event",
                    "description" to "Creates a new calendar event with a title, description, start/end time, and optional location.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "title" to mapOf("type" to "STRING", "description" to "The title of the event"),
                            "description" to mapOf("type" to "STRING", "description" to "A description for the event"),
                            "startTimeMillis" to mapOf("type" to "INTEGER", "description" to "Start time as Unix epoch milliseconds"),
                            "endTimeMillis" to mapOf("type" to "INTEGER", "description" to "End time as Unix epoch milliseconds"),
                            "location" to mapOf("type" to "STRING", "description" to "The location of the event")
                        ),
                        "required" to listOf("title", "startTimeMillis", "endTimeMillis")
                    )
                ),
                // ========== PHASE 3: SYSTEM CONTROLS ==========
                mapOf(
                    "name" to "set_volume",
                    "description" to "Sets the device volume level for a specific audio stream (media, ring, alarm, or notification).",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "volumeLevel" to mapOf("type" to "INTEGER", "description" to "The volume level to set (0 to max)"),
                            "streamType" to mapOf("type" to "STRING", "description" to "The audio stream: 'media', 'ring', 'alarm', or 'notification'. Defaults to 'media'.")
                        ),
                        "required" to listOf("volumeLevel")
                    )
                ),
                mapOf(
                    "name" to "set_brightness",
                    "description" to "Sets the screen brightness level (0-255).",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "level" to mapOf("type" to "INTEGER", "description" to "Brightness level from 0 (darkest) to 255 (brightest)")
                        ),
                        "required" to listOf("level")
                    )
                ),
                // ========== PHASE 3: FILE MANAGEMENT ==========
                mapOf(
                    "name" to "list_files",
                    "description" to "Lists files in a common device directory such as Downloads, DCIM, Documents, or Music.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "directory" to mapOf("type" to "STRING", "description" to "The directory name: 'Downloads', 'DCIM', 'Documents', 'Music', or 'Pictures'. Defaults to 'Downloads'.")
                        )
                    )
                ),
                mapOf(
                    "name" to "read_file",
                    "description" to "Reads the text content of a file on the device given its path.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "filePath" to mapOf("type" to "STRING", "description" to "The absolute file path to read")
                        ),
                        "required" to listOf("filePath")
                    )
                ),
                mapOf(
                    "name" to "create_file",
                    "description" to "Creates a new text file with the given content in a specified directory.",
                    "parameters" to mapOf(
                        "type" to "OBJECT",
                        "properties" to mapOf(
                            "fileName" to mapOf("type" to "STRING", "description" to "The name of the file to create, e.g. notes.txt"),
                            "content" to mapOf("type" to "STRING", "description" to "The text content to write to the file"),
                            "directory" to mapOf("type" to "STRING", "description" to "The directory: 'Downloads', 'Documents', etc. Defaults to 'Downloads'.")
                        ),
                        "required" to listOf("fileName", "content")
                    )
                )
            )
            requestMap["tools"] = listOf(mapOf("functionDeclarations" to toolDeclarations))

            val requestJson = gson.toJson(requestMap)
            val mediaType = "application/json; charset=utf-8".toMediaType()
            val requestBody = requestJson.toRequestBody(mediaType)

            val url = "https://generativelanguage.googleapis.com/v1beta/models/$geminiModel:streamGenerateContent?key=$apiKey"
            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            val response = okHttpClient.newCall(request).execute()
            if (!response.isSuccessful || response.body == null) {
                val errorMsg = response.body?.string() ?: ""
                postError("Gemini Error (${response.code}): $errorMsg")
                return
            }

            val reader = BufferedReader(InputStreamReader(response.body!!.byteStream()))
            var line: String?
            var activeTextResponse = ""
            var streamMsgId: Int? = null

            while (reader.readLine().also { line = it } != null) {
                val trimmed = line!!.trim()
                if (trimmed.isEmpty() || trimmed == "[" || trimmed == "]") continue

                var cleanLine = trimmed
                if (cleanLine.startsWith(",")) {
                    cleanLine = cleanLine.substring(1).trim()
                }

                try {
                    val chunk = gson.fromJson(cleanLine, Map::class.java)
                    val candidates = chunk?.get("candidates") as? List<*>
                    val candidate = candidates?.firstOrNull() as? Map<*, *>
                    val content = candidate?.get("content") as? Map<*, *>
                    val parts = content?.get("parts") as? List<*>

                    if (parts != null) {
                        for (partObj in parts) {
                            val part = partObj as? Map<*, *>

                            // 1. Text Streaming
                            val textPart = part?.get("text") as? String
                            if (!textPart.isNullOrEmpty()) {
                                activeTextResponse += textPart
                                withContext(Dispatchers.Main) {
                                    _streamText.value = activeTextResponse
                                }

                                val entity = MessageEntity(
                                    id = streamMsgId ?: 0,
                                    text = activeTextResponse,
                                    isUser = false,
                                    type = "text"
                                )
                                val newId = chatDao.insertMessage(entity)
                                if (streamMsgId == null) {
                                    streamMsgId = newId.toInt()
                                }
                            }

                            // 2. Function Calls
                            val functionCall = part?.get("functionCall") as? Map<*, *>
                            if (functionCall != null) {
                                val toolName = functionCall.get("name") as? String
                                val args = functionCall.get("args") as? Map<*, *>
                                val argsJson = gson.toJson(args)

                                if (toolName != null) {
                                    // Insert pending tool bubble
                                    val toolEntityId = chatDao.insertMessage(
                                        MessageEntity(
                                            text = "Running local tool '$toolName'...",
                                            isUser = false,
                                            type = "tool_call",
                                            toolName = toolName,
                                            status = "pending"
                                        )
                                    ).toInt()

                                    // Run tool on-device or handle local memory DB tool
                                    val result = if (toolName == "save_memory_fact" || toolName == "forget_memory_fact") {
                                        executeMemoryTool(toolName, args)
                                    } else {
                                        ToolExecutionManager.execute(context, toolName, argsJson)
                                    }

                                    // Save success result
                                    chatDao.insertMessage(
                                        MessageEntity(
                                            id = toolEntityId,
                                            text = result,
                                            isUser = false,
                                            type = "tool_call",
                                            toolName = toolName,
                                            status = "success"
                                        )
                                    )

                                    // Continue loop to report execution outcome back to model
                                    shouldContinue = true
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    // Ignore JSON framing tokens
                }
            }

            // Speak the response if voice enabled
            if (activeTextResponse.isNotBlank()) {
                withContext(Dispatchers.Main) {
                    speak(activeTextResponse)
                }
            }

            withContext(Dispatchers.Main) {
                _streamText.value = ""
            }
        }
    }

    private suspend fun executeMemoryTool(toolName: String, args: Map<*, *>?): String {
        return try {
            when (toolName) {
                "save_memory_fact" -> {
                    val key = (args?.get("key") as? String) ?: ""
                    val value = (args?.get("value") as? String) ?: ""
                    if (key.isNotBlank() && value.isNotBlank()) {
                        memoryDao.insertMemory(MemoryEntity(key, value))
                        "Successfully remembered: $key = $value."
                    } else {
                        "Error: Missing key or value."
                    }
                }
                "forget_memory_fact" -> {
                    val key = (args?.get("key") as? String) ?: ""
                    if (key.isNotBlank()) {
                        memoryDao.deleteMemory(key)
                        "Successfully forgot fact: $key."
                    } else {
                        "Error: Missing key."
                    }
                }
                else -> "Error: Unknown memory tool '$toolName'."
            }
        } catch (e: Exception) {
            "Memory error: ${e.localizedMessage}"
        }
    }

    fun deleteMemoryFact(key: String) {
        viewModelScope.launch(Dispatchers.IO) {
            memoryDao.deleteMemory(key)
        }
    }

    // ==========================================
    // FASTAPI SSE BACKEND LOOP (FALLBACK)
    // ==========================================
    private suspend fun executeBackendSseLoop(text: String) {
        val url = if (customServerUrl.endsWith("/")) "${customServerUrl}api/v1/chat/stream" else "$customServerUrl/api/v1/chat/stream"
        
        val requestJson = gson.toJson(mapOf("message" to text))
        val mediaType = "application/json; charset=utf-8".toMediaType()
        val requestBody = requestJson.toRequestBody(mediaType)

        val request = Request.Builder()
            .url(url)
            .post(requestBody)
            .build()

        val response = okHttpClient.newCall(request).execute()
        if (!response.isSuccessful || response.body == null) {
            postError("Backend offline or returned code ${response.code}")
            return
        }

        val inputStream = response.body!!.byteStream()

        withContext(Dispatchers.Main) {
            _streamText.value = ""
        }

        var streamMsgId: Int? = null
        var activeText = ""

        streamProcessor.processStream(inputStream) { event ->
            viewModelScope.launch(Dispatchers.IO) {
                when (event) {
                    is AIEvent.Token -> {
                        activeText += event.text
                        withContext(Dispatchers.Main) {
                            _streamText.value = activeText
                        }

                        val entity = MessageEntity(
                            id = streamMsgId ?: 0,
                            text = activeText,
                            isUser = false,
                            type = "text"
                        )
                        val newId = chatDao.insertMessage(entity)
                        if (streamMsgId == null) {
                            streamMsgId = newId.toInt()
                        }
                    }

                    is AIEvent.Tool -> {
                        val toolName = event.name
                        val params = event.output

                        val toolEntityId = chatDao.insertMessage(
                            MessageEntity(
                                text = "Running tool '$toolName'...",
                                isUser = false,
                                type = "tool_call",
                                toolName = toolName,
                                status = "pending"
                            )
                        ).toInt()

                        val result = ToolExecutionManager.execute(context, toolName, params)

                        chatDao.insertMessage(
                            MessageEntity(
                                id = toolEntityId,
                                text = result,
                                isUser = false,
                                type = "tool_call",
                                toolName = toolName,
                                status = "success"
                            )
                        )
                    }

                    is AIEvent.Reflection -> {
                        chatDao.insertMessage(
                            MessageEntity(
                                text = event.status,
                                isUser = false,
                                type = "thinking"
                            )
                        )
                    }

                    is AIEvent.Debug -> {
                        // Debug log callback
                    }
                }
            }
        }

        if (activeText.isNotBlank()) {
            withContext(Dispatchers.Main) {
                speak(activeText)
                _streamText.value = ""
            }
        }
    }

    private suspend fun postError(msg: String) {
        chatDao.insertMessage(
            MessageEntity(text = msg, isUser = false, type = "error")
        )
    }

    fun clearChatMemory() {
        viewModelScope.launch(Dispatchers.IO) {
            chatDao.clearHistory()
        }
    }

    override fun onCleared() {
        super.onCleared()
        tts?.stop()
        tts?.shutdown()
        viewModelScope.launch(Dispatchers.Main) {
            speechRecognizer?.destroy()
        }
    }
}