package com.aios.android.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aios.android.util.PreferencesManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SettingsUiState(
    val apiUrl: String = "http://10.0.2.2:8000/v1",
    val apiKey: String = "",
    val model: String = "gpt-4o",
    val theme: String = "dark",
    val language: String = "en",
    val isDarkTheme: Boolean = true,
    val notificationsEnabled: Boolean = true,
    val offlineCacheEnabled: Boolean = true,
    val saved: Boolean = false
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val preferencesManager: PreferencesManager
) : ViewModel() {
    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState: StateFlow<SettingsUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            _uiState.value = SettingsUiState(
                apiUrl = preferencesManager.apiUrl.first(),
                apiKey = preferencesManager.apiKey.first(),
                model = preferencesManager.model.first(),
                theme = preferencesManager.theme.first(),
                isDarkTheme = preferencesManager.theme.first() == "dark"
            )
        }
    }

    fun onApiUrlChanged(url: String) { _uiState.value = _uiState.value.copy(apiUrl = url) }
    fun onApiKeyChanged(key: String) { _uiState.value = _uiState.value.copy(apiKey = key) }
    fun onModelChanged(model: String) { _uiState.value = _uiState.value.copy(model = model) }
    fun onThemeChanged(dark: Boolean) { _uiState.value = _uiState.value.copy(isDarkTheme = dark, theme = if (dark) "dark" else "light") }
    fun onLanguageChanged(lang: String) { _uiState.value = _uiState.value.copy(language = lang) }
    fun onNotificationsChanged(enabled: Boolean) { _uiState.value = _uiState.value.copy(notificationsEnabled = enabled) }
    fun onOfflineCacheChanged(enabled: Boolean) { _uiState.value = _uiState.value.copy(offlineCacheEnabled = enabled) }

    fun save() {
        viewModelScope.launch {
            val s = _uiState.value
            preferencesManager.setApiUrl(s.apiUrl)
            preferencesManager.setApiKey(s.apiKey)
            preferencesManager.setModel(s.model)
            preferencesManager.setTheme(s.theme)
            preferencesManager.setLanguage(s.language)
            _uiState.value = s.copy(saved = true)
        }
    }
}
