package com.aios.android.ui.models

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aios.android.domain.model.AiosModel
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ModelsUiState(
    val models: List<AiosModel> = emptyList(),
    val isLoading: Boolean = false,
    val downloadingModelId: String? = null
)

@HiltViewModel
class ModelsViewModel @Inject constructor() : ViewModel() {
    private val _uiState = MutableStateFlow(ModelsUiState())
    val uiState: StateFlow<ModelsUiState> = _uiState.asStateFlow()

    private val availableModels = listOf(
        AiosModel(id = "gpt-4o", name = "GPT-4o", provider = "OpenAI", isBuiltin = true, isDownloaded = true, capabilities = listOf("chat", "vision")),
        AiosModel(id = "gpt-4o-mini", name = "GPT-4o Mini", provider = "OpenAI", isBuiltin = true, isDownloaded = true, capabilities = listOf("chat", "vision")),
        AiosModel(id = "claude-3-opus", name = "Claude 3 Opus", provider = "Anthropic", isBuiltin = true, isDownloaded = true, capabilities = listOf("chat")),
        AiosModel(id = "llama-3.1-8b", name = "Llama 3.1 8B", provider = "Meta", isBuiltin = false, isDownloaded = false, capabilities = listOf("chat"), sizeBytes = 4L * 1024 * 1024 * 1024),
        AiosModel(id = "llama-3.1-70b", name = "Llama 3.1 70B", provider = "Meta", isBuiltin = false, isDownloaded = false, capabilities = listOf("chat"), sizeBytes = 35L * 1024 * 1024 * 1024),
        AiosModel(id = "mistral-7b", name = "Mistral 7B", provider = "Mistral", isBuiltin = false, isDownloaded = false, capabilities = listOf("chat"), sizeBytes = 4L * 1024 * 1024 * 1024),
    )

    init {
        _uiState.value = _uiState.value.copy(models = availableModels)
    }

    fun downloadModel(modelId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(downloadingModelId = modelId)
            delay(3000)
            val updated = _uiState.value.models.map {
                if (it.id == modelId) it.copy(isDownloaded = true) else it
            }
            _uiState.value = _uiState.value.copy(models = updated, downloadingModelId = null)
        }
    }

    fun removeModel(modelId: String) {
        val updated = _uiState.value.models.map {
            if (it.id == modelId) it.copy(isDownloaded = false) else it
        }
        _uiState.value = _uiState.value.copy(models = updated)
    }
}
