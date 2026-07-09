package com.aios.android.ui.memory

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aios.android.data.repository.MemoryRepository
import com.aios.android.domain.model.Memory
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class MemoryUiState(
    val memories: List<Memory> = emptyList(),
    val searchQuery: String = "",
    val isLoading: Boolean = false,
    val editingMemory: Memory? = null,
    val editKey: String = "",
    val editValue: String = ""
)

@HiltViewModel
class MemoryViewModel @Inject constructor(
    private val memoryRepository: MemoryRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(MemoryUiState())
    val uiState: StateFlow<MemoryUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            memoryRepository.getAllMemories().collect { memories ->
                _uiState.value = _uiState.value.copy(memories = memories, isLoading = false)
            }
        }
    }

    fun onSearchQueryChanged(query: String) { _uiState.value = _uiState.value.copy(searchQuery = query) }

    fun createMemory() {
        viewModelScope.launch {
            val s = _uiState.value
            memoryRepository.createMemory(s.editKey, s.editValue)
            _uiState.value = s.copy(editingMemory = null, editKey = "", editValue = "")
        }
    }

    fun startEditing(memory: Memory) {
        _uiState.value = _uiState.value.copy(editingMemory = memory, editKey = memory.key, editValue = memory.value)
    }

    fun cancelEditing() {
        _uiState.value = _uiState.value.copy(editingMemory = null, editKey = "", editValue = "")
    }

    fun onEditKeyChanged(key: String) { _uiState.value = _uiState.value.copy(editKey = key) }
    fun onEditValueChanged(value: String) { _uiState.value = _uiState.value.copy(editValue = value) }

    fun deleteMemory(id: String) { viewModelScope.launch { memoryRepository.deleteMemory(id) } }
}
