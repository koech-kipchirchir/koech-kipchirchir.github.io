package com.aios.android.ui.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aios.android.data.repository.ChatRepository
import com.aios.android.domain.model.Chat
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HistoryUiState(
    val chats: List<Chat> = emptyList(),
    val searchQuery: String = "",
    val isLoading: Boolean = false
)

@HiltViewModel
class HistoryViewModel @Inject constructor(
    private val chatRepository: ChatRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(HistoryUiState())
    val uiState: StateFlow<HistoryUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            chatRepository.getAllChats().collect { chats ->
                _uiState.value = _uiState.value.copy(chats = chats, isLoading = false)
            }
        }
    }

    fun onSearchQueryChanged(query: String) { _uiState.value = _uiState.value.copy(searchQuery = query) }
    fun deleteChat(chatId: String) { viewModelScope.launch { chatRepository.deleteChat(chatId) } }

    val filteredChats: StateFlow<List<Chat>> = _uiState.map { state ->
        if (state.searchQuery.isBlank()) state.chats
        else state.chats.filter { it.title.contains(state.searchQuery, ignoreCase = true) }
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
}
