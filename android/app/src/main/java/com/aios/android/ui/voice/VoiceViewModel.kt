package com.aios.android.ui.voice

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

data class VoiceUiState(
    val isRecording: Boolean = false,
    val isProcessing: Boolean = false,
    val transcribedText: String = "",
    val error: String? = null,
    val audioLevel: Float = 0f
)

@HiltViewModel
class VoiceViewModel @Inject constructor() : ViewModel() {
    private val _uiState = MutableStateFlow(VoiceUiState())
    val uiState: StateFlow<VoiceUiState> = _uiState.asStateFlow()

    fun startRecording() { _uiState.value = _uiState.value.copy(isRecording = true, error = null) }

    fun stopRecording() {
        _uiState.value = _uiState.value.copy(isRecording = false, isProcessing = true)
        viewModelScope.launch {
            // TODO: Send audio to backend API for transcription
            kotlinx.coroutines.delay(1500)
            _uiState.value = _uiState.value.copy(
                isProcessing = false,
                transcribedText = "This is a placeholder transcription."
            )
        }
    }

    fun onAudioLevelChanged(level: Float) { _uiState.value = _uiState.value.copy(audioLevel = level) }
    fun clearTranscription() { _uiState.value = _uiState.value.copy(transcribedText = "") }
}
