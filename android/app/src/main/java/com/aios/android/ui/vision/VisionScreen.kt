package com.aios.android.ui.vision

import android.net.Uri
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.aios.android.ui.components.ImagePicker

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VisionScreen(
    viewModel: VisionViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()

    Scaffold(
        topBar = { TopAppBar(title = { Text("Vision") }) }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(scrollState)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                ImagePicker(onImageSelected = viewModel::onImageSelected)
                Spacer(Modifier.width(12.dp))
                Text("Select an image", style = MaterialTheme.typography.bodyLarge)
            }
            Spacer(Modifier.height(12.dp))

            state.imageUri?.let { uri ->
                Card(modifier = Modifier.fillMaxWidth().heightIn(max = 300.dp)) {
                    Box {
                        AsyncImage(
                            model = ImageRequest.Builder(LocalContext.current)
                                .data(uri).crossfade(true).build(),
                            contentDescription = "Selected image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                        IconButton(
                            onClick = viewModel::clearImage,
                            modifier = Modifier.align(Alignment.TopEnd)
                        ) {
                            Icon(Icons.Default.Clear, "Remove")
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = viewModel::runOcr, enabled = !state.isLoading) { Text("OCR") }
                    OutlinedButton(onClick = viewModel::runDetection, enabled = !state.isLoading) { Text("Detect") }
                    OutlinedButton(onClick = viewModel::runCaption, enabled = !state.isLoading) { Text("Caption") }
                }
                Spacer(Modifier.height(12.dp))

                if (state.isLoading) {
                    CircularProgressIndicator()
                }

                if (state.ocrText.isNotBlank()) {
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(12.dp)) {
                            Text("OCR Text", style = MaterialTheme.typography.titleSmall)
                            Spacer(Modifier.height(4.dp))
                            Text(state.ocrText)
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }

                if (state.detections.isNotEmpty()) {
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(12.dp)) {
                            Text("Detections", style = MaterialTheme.typography.titleSmall)
                            state.detections.forEach { det ->
                                Text("• $det", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }

                if (state.caption.isNotBlank()) {
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(12.dp)) {
                            Text("Caption", style = MaterialTheme.typography.titleSmall)
                            Spacer(Modifier.height(4.dp))
                            Text(state.caption)
                        }
                    }
                }
            }

            state.error?.let { error ->
                Spacer(Modifier.height(8.dp))
                Text(error, color = MaterialTheme.colorScheme.error)
            }
        }
    }
}
