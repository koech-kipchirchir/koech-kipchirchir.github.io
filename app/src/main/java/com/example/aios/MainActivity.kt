package com.example.aios

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import com.example.aios.viewmodel.ChatViewModel
import com.example.aios.ui.ChatScreen
import com.example.aios.ui.theme.AIOSTheme

class MainActivity : ComponentActivity() {

    // Properly initialize the ViewModel using the standard delegate extension
    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            AIOSTheme {
                // Renders the connected ChatScreen user interface directly
                ChatScreen(viewModel)
            }
        }
    }
}