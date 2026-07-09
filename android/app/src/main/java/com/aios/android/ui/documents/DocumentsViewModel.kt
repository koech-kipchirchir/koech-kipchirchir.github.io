package com.aios.android.ui.documents

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aios.android.data.repository.DocumentRepository
import com.aios.android.domain.model.Document
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class DocumentsUiState(
    val documents: List<Document> = emptyList(),
    val isLoading: Boolean = false,
    val searchQuery: String = ""
)

@HiltViewModel
class DocumentsViewModel @Inject constructor(
    private val documentRepository: DocumentRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(DocumentsUiState())
    val uiState: StateFlow<DocumentsUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            documentRepository.getAllDocuments().collect { docs ->
                _uiState.value = _uiState.value.copy(documents = docs, isLoading = false)
            }
        }
    }

    fun onSearchQueryChanged(query: String) { _uiState.value = _uiState.value.copy(searchQuery = query) }

    fun importDocument(uri: Uri) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            documentRepository.importDocument(uri)
            _uiState.value = _uiState.value.copy(isLoading = false)
        }
    }

    fun deleteDocument(id: String) { viewModelScope.launch { documentRepository.deleteDocument(id) } }
}
