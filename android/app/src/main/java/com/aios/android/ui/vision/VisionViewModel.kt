package com.aios.android.ui.vision

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class VisionUiState(
    val imageUri: Uri? = null,
    val isLoading: Boolean = false,
    val ocrText: String = "",
    val detections: List<String> = emptyList(),
    val caption: String = "",
    val error: String? = null
)

@HiltViewModel
class VisionViewModel @Inject constructor() : ViewModel() {
    private val _uiState = MutableStateFlow(VisionUiState())
    val uiState: StateFlow<VisionUiState> = _uiState.asStateFlow()

    fun onImageSelected(uri: Uri) {
        _uiState.value = _uiState.value.copy(imageUri = uri, ocrText = "", detections = emptyList(), caption = "", error = null)
    }

    fun runOcr() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            kotlinx.coroutines.delay(1500)
            _uiState.value = _uiState.value.copy(isLoading = false, ocrText = "Sample OCR text from the image.")
        }
    }

    fun runDetection() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            kotlinx.coroutines.delay(1500)
            _uiState.value = _uiState.value.copy(
                isLoading = false,
                detections = listOf("person (0.95)", "chair (0.82)", "laptop (0.78)")
            )
        }
    }

    fun runCaption() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            kotlinx.coroutines.delay(1500)
            _uiState.value = _uiState.value.copy(isLoading = false, caption = "A person sitting at a desk with a laptop.")
        }
    }

    fun clearImage() { _uiState.value = VisionUiState() }
}
