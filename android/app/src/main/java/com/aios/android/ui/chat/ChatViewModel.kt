package com.aios.android.ui.chat

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aios.android.data.repository.ChatRepository
import com.aios.android.domain.model.Chat
import com.aios.android.domain.model.Message
import com.aios.android.domain.model.MessageRole
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ChatUiState(
    val chats: List<Chat> = emptyList(),
    val currentChat: Chat? = null,
    val messages: List<Message> = emptyList(),
    val inputText: String = "",
    val isStreaming: Boolean = false,
    val isLoading: Boolean = false,
    val error: String? = null,
    val attachedImageUris: List<Uri> = emptyList()
)

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val chatRepository: ChatRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            chatRepository.getAllChats().collect { chats ->
                _uiState.value = _uiState.value.copy(chats = chats)
            }
        }
    }

    fun onInputChanged(text: String) { _uiState.value = _uiState.value.copy(inputText = text) }

    fun selectChat(chatId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            chatRepository.getMessages(chatId).collect { messages ->
                val chat = _uiState.value.chats.find { it.id == chatId }
                _uiState.value = _uiState.value.copy(
                    currentChat = chat,
                    messages = messages,
                    isLoading = false
                )
            }
        }
    }

    fun createNewChat() {
        viewModelScope.launch {
            val chat = chatRepository.createChat()
            _uiState.value = _uiState.value.copy(currentChat = chat, messages = emptyList(), inputText = "")
        }
    }

    fun sendMessage() {
        val text = _uiState.value.inputText.trim()
        if (text.isBlank()) return

        viewModelScope.launch {
            val chat = _uiState.value.currentChat
            val chatId = chat?.id ?: chatRepository.createChat().id

            // Send user message
            chatRepository.sendMessage(chatId, text)
            _uiState.value = _uiState.value.copy(inputText = "", isStreaming = true)

            // TODO: Stream response from API and append to messages
        }
    }

    fun attachImage(uri: Uri) {
        val uris = _uiState.value.attachedImageUris + uri
        _uiState.value = _uiState.value.copy(attachedImageUris = uris)
    }

    fun clearAttachments() {
        _uiState.value = _uiState.value.copy(attachedImageUris = emptyList())
    }

    fun deleteChat(chatId: String) {
        viewModelScope.launch { chatRepository.deleteChat(chatId) }
    }
}
