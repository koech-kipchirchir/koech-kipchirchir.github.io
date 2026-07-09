package com.example.aios.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.example.aios.viewmodel.ChatViewModel
import com.example.aios.viewmodel.Message
import com.example.aios.viewmodel.MemoryEntity
import kotlinx.coroutines.launch
import java.io.File

// =====================================
// DESIGN TOKENS
// =====================================
private val DarkBg = Color(0xFF080812)
private val CardBg = Color(0xFF12121E)
private val CardBg2 = Color(0xFF1A1A2E)
private val NeonPurple = Color(0xFF9D5CF6)
private val NeonCyan = Color(0xFF00D4FF)
private val NeonGreen = Color(0xFF00E5A0)
private val NeonRed = Color(0xFFFF4D6A)
private val TextPrimary = Color(0xFFF1F3F9)
private val TextSecondary = Color(0xFF8B8FA8)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val messages by viewModel.messagesState.collectAsState()
    val streamText by viewModel.streamText
    val listState = rememberLazyListState()
    var showSettings by remember { mutableStateOf(false) }

    // Auto-scroll logic when messages or streams update
    LaunchedEffect(messages.size, streamText) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.lastIndex)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBg)
    ) {
        // Subtle radial glow background effects
        Box(
            modifier = Modifier
                .size(400.dp)
                .align(Alignment.TopEnd)
                .background(
                    Brush.radialGradient(
                        colors = listOf(NeonPurple.copy(alpha = 0.12f), Color.Transparent)
                    )
                )
        )
        Box(
            modifier = Modifier
                .size(350.dp)
                .align(Alignment.BottomStart)
                .background(
                    Brush.radialGradient(
                        colors = listOf(NeonCyan.copy(alpha = 0.08f), Color.Transparent)
                    )
                )
        )

        Scaffold(
            containerColor = Color.Transparent,
            topBar = { AIOSTopBar(onSettingsClick = { showSettings = true }, onClearClick = { viewModel.clearChatMemory() }) }
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .imePadding()
            ) {
                // CHAT TIMELINE
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // Greeting card if empty
                    if (messages.isEmpty() && streamText.isBlank()) {
                        item {
                            WelcomeCard()
                        }
                    }

                    itemsIndexed(
                        items = messages,
                        key = { index, _ -> index }
                    ) { _, message ->
                        AnimatedVisibility(
                            visible = true,
                            enter = fadeIn(animationSpec = tween(300)) +
                                    slideInVertically(animationSpec = tween(300)) { it / 2 }
                        ) {
                            MessageRow(message)
                        }
                    }

                    // Live Streaming Bubble
                    if (streamText.isNotBlank()) {
                        item {
                            MessageRow(Message(text = streamText, isUser = false, type = "text"))
                        }
                    }

                    item { Spacer(modifier = Modifier.height(8.dp)) }
                }

                // VOICE STATE AND TEXT INPUT BAR
                InputArea(viewModel)
            }
        }

        // SETTINGS DIALOG PANEL
        if (showSettings) {
            SettingsPanel(
                viewModel = viewModel,
                onDismiss = { showSettings = false }
            )
        }
    }
}

// =====================================
// TOP BAR
// =====================================
@Composable
fun AIOSTopBar(onSettingsClick: () -> Unit, onClearClick: () -> Unit) {
    val infiniteTransition = rememberInfiniteTransition(label = "status_pulse")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(tween(1200), RepeatMode.Reverse),
        label = "alpha"
    )

    Surface(
        color = CardBg.copy(alpha = 0.9f),
        modifier = Modifier
            .fillMaxWidth()
            .border(
                0.5.dp,
                Color.White.copy(alpha = 0.07f),
                RoundedCornerShape(bottomStart = 28.dp, bottomEnd = 28.dp)
            ),
        shape = RoundedCornerShape(bottomStart = 28.dp, bottomEnd = 28.dp),
        tonalElevation = 6.dp
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            // Status row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(NeonGreen.copy(alpha = alpha))
                    )
                    Spacer(Modifier.width(6.dp))
                    Text(
                        "AIOS Kernel · Online",
                        color = NeonGreen.copy(alpha = alpha),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Text(
                    "Neural Agent v2.0",
                    color = TextSecondary,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace
                )
            }

            Spacer(Modifier.height(8.dp))

            // App title & controls
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "AIOS.2",
                        fontSize = 26.sp,
                        fontWeight = FontWeight.ExtraBold,
                        style = LocalTextStyle.current.copy(
                            brush = Brush.horizontalGradient(
                                colors = listOf(NeonPurple, NeonCyan)
                            )
                        )
                    )
                    Text(
                        "Your Personal AI Operating System",
                        color = TextSecondary,
                        fontSize = 11.sp
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    IconButton(
                        onClick = onClearClick,
                        colors = IconButtonDefaults.iconButtonColors(
                            containerColor = NeonRed.copy(alpha = 0.1f)
                        ),
                        modifier = Modifier.size(40.dp)
                    ) {
                        Icon(
                            Icons.Default.Delete,
                            "Clear History",
                            tint = NeonRed,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    IconButton(
                        onClick = onSettingsClick,
                        colors = IconButtonDefaults.iconButtonColors(
                            containerColor = NeonCyan.copy(alpha = 0.1f)
                        ),
                        modifier = Modifier.size(40.dp)
                    ) {
                        Icon(
                            Icons.Default.Settings,
                            "Settings",
                            tint = NeonCyan,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }
        }
    }
}

// =====================================
// WELCOME CARD
// =====================================
@Composable
fun WelcomeCard() {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBg2),
        shape = RoundedCornerShape(24.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, NeonPurple.copy(alpha = 0.2f), RoundedCornerShape(24.dp))
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(72.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.radialGradient(listOf(NeonPurple.copy(0.3f), NeonCyan.copy(0.1f)))
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Face,
                    "AIOS",
                    tint = NeonPurple,
                    modifier = Modifier.size(36.dp)
                )
            }
            Spacer(Modifier.height(16.dp))
            Text(
                "Hello, I'm AIOS",
                fontSize = 22.sp,
                fontWeight = FontWeight.ExtraBold,
                color = TextPrimary
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Your AI-powered Android operating system. I can control your device, remember personal details, open apps, check your location, and much more.",
                fontSize = 14.sp,
                color = TextSecondary,
                lineHeight = 20.sp,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(20.dp))

            // Capability chips
            val capabilities = listOf(
                "🔦 Flashlight",
                "⏰ Set Alarm",
                "📱 Open Apps",
                "📍 Location",
                "💬 Send SMS",
                "📞 Make Calls",
                "🧠 Long Memory",
                "🎙️ Voice Input"
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.Start
                ) {
                    val chunks = capabilities.chunked(4)
                    chunks.forEach { row ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 6.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            row.forEach { cap ->
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(20.dp))
                                        .background(NeonPurple.copy(alpha = 0.12f))
                                        .border(0.5.dp, NeonPurple.copy(alpha = 0.25f), RoundedCornerShape(20.dp))
                                        .padding(horizontal = 10.dp, vertical = 5.dp)
                                ) {
                                    Text(cap, color = TextPrimary, fontSize = 11.sp)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =====================================
// SIMPLE MARKDOWN RENDERER
// =====================================
@Composable
fun MarkdownText(text: String, modifier: Modifier = Modifier) {
    val annotated = buildAnnotatedString {
        var remaining = text
        while (remaining.isNotEmpty()) {
            when {
                // Bold (**text**)
                remaining.startsWith("**") -> {
                    val end = remaining.indexOf("**", 2)
                    if (end > 2) {
                        val boldText = remaining.substring(2, end)
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold, color = TextPrimary)) {
                            append(boldText)
                        }
                        remaining = remaining.substring(end + 2)
                    } else {
                        append(remaining[0])
                        remaining = remaining.substring(1)
                    }
                }
                // Inline code (`code`)
                remaining.startsWith("`") && remaining.length > 1 -> {
                    val end = remaining.indexOf("`", 1)
                    if (end > 1) {
                        val code = remaining.substring(1, end)
                        withStyle(SpanStyle(fontFamily = FontFamily.Monospace, color = NeonCyan, fontSize = 13.sp)) {
                            append(code)
                        }
                        remaining = remaining.substring(end + 1)
                    } else {
                        append(remaining[0])
                        remaining = remaining.substring(1)
                    }
                }
                // Italic (*text*)
                remaining.startsWith("*") && remaining.length > 1 && !remaining.startsWith("**") -> {
                    val end = remaining.indexOf("*", 1)
                    if (end > 1) {
                        val italicText = remaining.substring(1, end)
                        withStyle(SpanStyle(fontStyle = androidx.compose.ui.text.font.FontStyle.Italic, color = TextSecondary)) {
                            append(italicText)
                        }
                        remaining = remaining.substring(end + 1)
                    } else {
                        append(remaining[0])
                        remaining = remaining.substring(1)
                    }
                }
                // Headers
                remaining.startsWith("### ") -> {
                    val end = remaining.indexOf("\n")
                    val header = if (end > 4) remaining.substring(4, end) else remaining.substring(4)
                    remaining = if (end > 0) remaining.substring(end + 1) else ""
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontSize = 14.sp, color = NeonPurple)) {
                        append("$header\n")
                    }
                }
                remaining.startsWith("## ") -> {
                    val end = remaining.indexOf("\n")
                    val header = if (end > 3) remaining.substring(3, end) else remaining.substring(3)
                    remaining = if (end > 0) remaining.substring(end + 1) else ""
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontSize = 15.sp, color = NeonPurple)) {
                        append("$header\n")
                    }
                }
                remaining.startsWith("# ") -> {
                    val end = remaining.indexOf("\n")
                    val header = if (end > 2) remaining.substring(2, end) else remaining.substring(2)
                    remaining = if (end > 0) remaining.substring(end + 1) else ""
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontSize = 17.sp, color = NeonPurple)) {
                        append("$header\n")
                    }
                }
                // List items
                remaining.startsWith("- ") || remaining.startsWith("* ") -> {
                    val end = remaining.indexOf("\n")
                    val item = if (end > 2) remaining.substring(2, end) else remaining.substring(2)
                    remaining = if (end > 0) remaining.substring(end + 1) else ""
                    append("  • $item\n")
                }
                // Table row
                remaining.startsWith("|") -> {
                    val end = remaining.indexOf("\n")
                    val row = if (end > 0) remaining.substring(1, end) else remaining.substring(1)
                    remaining = if (end > 0) remaining.substring(end + 1) else ""
                    if (!row.contains("---")) {
                        val cols = row.split("|").filter { it.isNotBlank() }
                        if (cols.isNotEmpty()) {
                            withStyle(SpanStyle(fontFamily = FontFamily.Monospace, fontSize = 12.sp)) {
                                append(cols.joinToString(" │ ") + "\n")
                            }
                        }
                    }
                }
                // Code blocks
                remaining.startsWith("```") -> {
                    val end = remaining.indexOf("```", 3)
                    if (end > 3) {
                        val code = remaining.substring(remaining.indexOf("\n", 3) + 1, end)
                        remaining = remaining.substring(end + 3)
                        withStyle(SpanStyle(fontFamily = FontFamily.Monospace, color = NeonGreen, fontSize = 12.sp, background = Color(0x33000000))) {
                            append("\n$code\n")
                        }
                    } else {
                        remaining = remaining.substring(3)
                    }
                }
                else -> {
                    append(remaining[0])
                    remaining = remaining.substring(1)
                }
            }
        }
    }

    Text(
        text = annotated,
        modifier = modifier,
        fontSize = 15.sp,
        lineHeight = 22.sp
    )
}

// =====================================
// CODE BLOCK CARD
// =====================================
@Composable
fun CodeBlockCard(code: String, language: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xFF0D0E14)),
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, Color.White.copy(0.06f), RoundedCornerShape(14.dp))
    ) {
        Column {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF15161D))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(language, color = TextSecondary, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                Text("📋", color = TextSecondary, fontSize = 12.sp)
            }
            Text(
                text = code,
                color = NeonGreen.copy(0.9f),
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
                lineHeight = 18.sp,
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(12.dp)
            )
        }
    }
}

// =====================================
// MESSAGE ROW
// =====================================
@Composable
fun MessageRow(message: Message) {
    val isUser = message.isUser
    val alignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        contentAlignment = alignment
    ) {
        Column(
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
        ) {
            // Show images if present
            message.imageUris?.forEach { uri ->
                AsyncImage(
                    model = uri,
                    contentDescription = "Attached image",
                    modifier = Modifier
                        .padding(bottom = 6.dp)
                        .widthIn(max = 240.dp)
                        .heightIn(max = 240.dp)
                        .clip(RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.Crop
                )
            }

            when (message.type) {
                "tool_call" -> ToolExecutionCard(
                    toolName = message.toolName ?: "unknown",
                    output = message.text,
                    status = message.status ?: "success"
                )
                "thinking" -> ReflectionCard(thought = message.text)
                "error" -> ErrorCard(error = message.text)
                "code" -> CodeBlockCard(
                    code = message.text,
                    language = message.toolName ?: "code"
                )
                else -> {
                    // Text Bubble with markdown
                    val bubbleShape = RoundedCornerShape(
                        topStart = 20.dp, topEnd = 20.dp,
                        bottomStart = if (isUser) 20.dp else 6.dp,
                        bottomEnd = if (isUser) 6.dp else 20.dp
                    )
                    val bubbleBg = if (isUser) {
                        Brush.linearGradient(listOf(NeonPurple, NeonPurple.copy(0.8f)))
                    } else {
                        Brush.linearGradient(listOf(CardBg2, CardBg2))
                    }

                    Box(
                        modifier = Modifier
                            .widthIn(max = 310.dp)
                            .clip(bubbleShape)
                            .background(bubbleBg)
                            .border(
                                0.5.dp,
                                if (isUser) Color.White.copy(0.2f) else Color.White.copy(0.06f),
                                bubbleShape
                            )
                            .padding(horizontal = 14.dp, vertical = 10.dp)
                    ) {
                        MarkdownText(text = message.text)
                    }
                }
            }
        }
    }
}

// =====================================
// TOOL EXECUTION CARD
// =====================================
@Composable
fun ToolExecutionCard(toolName: String, output: String, status: String) {
    val isPending = status == "pending"
    val isFailed = status == "failed"
    val accentColor = when {
        isPending -> NeonCyan
        isFailed -> NeonRed
        else -> NeonGreen
    }

    val toolIcon = when (toolName) {
        "set_alarm" -> Icons.Default.Notifications
        "toggle_flashlight" -> Icons.Default.Star
        "get_device_status" -> Icons.Default.Info
        "send_sms" -> Icons.AutoMirrored.Filled.Send
        "make_call" -> Icons.Default.Phone
        "vibrate" -> Icons.Default.Refresh
        "open_app" -> Icons.Default.PlayArrow
        "get_current_location" -> Icons.Default.LocationOn
        "save_memory_fact" -> Icons.Default.Favorite
        "forget_memory_fact" -> Icons.Default.Delete
        // Phase 3 tools
        "read_clipboard" -> Icons.Default.ContentPaste
        "write_clipboard" -> Icons.Default.ContentCopy
        "search_contacts" -> Icons.Default.Person
        "web_search" -> Icons.Default.Search
        "open_url" -> Icons.Default.Language
        "get_calendar_events" -> Icons.Default.DateRange
        "create_calendar_event" -> Icons.Default.Event
        "set_volume" -> Icons.Default.VolumeUp
        "set_brightness" -> Icons.Default.LightMode
        "list_files" -> Icons.Default.Folder
        "read_file" -> Icons.Default.Description
        "create_file" -> Icons.Default.NoteAdd
        else -> Icons.Default.Build
    }

    Card(
        colors = CardDefaults.cardColors(containerColor = CardBg),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, accentColor.copy(alpha = 0.2f), RoundedCornerShape(18.dp))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Tool icon
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(accentColor.copy(alpha = 0.12f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = toolIcon,
                    contentDescription = "Tool",
                    tint = accentColor,
                    modifier = Modifier.size(22.dp)
                )
            }

            Spacer(Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = toolName.replace("_", " ").split(" ").joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } },
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    color = accentColor
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = output,
                    fontSize = 12.sp,
                    color = TextSecondary,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Spacer(Modifier.width(8.dp))

            if (isPending) {
                CircularProgressIndicator(
                    color = NeonCyan,
                    modifier = Modifier.size(18.dp),
                    strokeWidth = 2.dp
                )
            } else if (isFailed) {
                Icon(Icons.Default.Warning, "Failed", tint = NeonRed, modifier = Modifier.size(20.dp))
            } else {
                Icon(Icons.Default.Check, "Success", tint = NeonGreen, modifier = Modifier.size(20.dp))
            }
        }
    }
}

// =====================================
// REFLECTION CARD
// =====================================
@Composable
fun ReflectionCard(thought: String) {
    var expanded by remember { mutableStateOf(false) }
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBg.copy(alpha = 0.5f)),
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier
            .widthIn(max = 300.dp)
            .border(0.5.dp, Color.White.copy(alpha = 0.06f), RoundedCornerShape(14.dp))
            .clickable { expanded = !expanded }
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Build, "Thinking", tint = TextSecondary, modifier = Modifier.size(14.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Reflection", fontSize = 11.sp, color = TextSecondary, fontWeight = FontWeight.SemiBold)
                }
                Icon(
                    if (expanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                    "Expand",
                    tint = TextSecondary,
                    modifier = Modifier.size(16.dp)
                )
            }
            if (expanded) {
                Spacer(Modifier.height(6.dp))
                Text(
                    thought,
                    fontSize = 12.sp,
                    color = TextSecondary,
                    fontFamily = FontFamily.Monospace,
                    lineHeight = 16.sp
                )
            }
        }
    }
}

// =====================================
// ERROR CARD
// =====================================
@Composable
fun ErrorCard(error: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBg),
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, NeonRed.copy(alpha = 0.25f), RoundedCornerShape(14.dp))
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(Icons.Default.Warning, "Error", tint = NeonRed, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(10.dp))
            Text(error, color = NeonRed.copy(0.9f), fontSize = 13.sp)
        }
    }
}

// =====================================
// INPUT AREA WITH MIC BUTTON
// =====================================
@Composable
fun InputArea(viewModel: ChatViewModel) {
    val context = LocalContext.current
    var textState by remember { mutableStateOf("") }
    val voiceTranscript by viewModel.voiceTranscript
    val isListening = viewModel.isListening

    // Sync voice transcript into text field
    LaunchedEffect(voiceTranscript) {
        if (voiceTranscript.isNotBlank() && voiceTranscript != "Listening...") {
            textState = voiceTranscript
        }
    }

    // Animated waveform for loading state
    val infiniteTransition = rememberInfiniteTransition(label = "wave")
    val waveHeight1 by infiniteTransition.animateFloat(
        initialValue = 4f, targetValue = 22f,
        animationSpec = infiniteRepeatable(tween(600, easing = LinearEasing), RepeatMode.Reverse),
        label = "w1"
    )
    val waveHeight2 by infiniteTransition.animateFloat(
        initialValue = 18f, targetValue = 6f,
        animationSpec = infiniteRepeatable(tween(400, easing = LinearEasing), RepeatMode.Reverse),
        label = "w2"
    )
    val waveHeight3 by infiniteTransition.animateFloat(
        initialValue = 8f, targetValue = 26f,
        animationSpec = infiniteRepeatable(tween(500, easing = LinearEasing), RepeatMode.Reverse),
        label = "w3"
    )

    // Mic pulse animation when listening
    val micScale by infiniteTransition.animateFloat(
        initialValue = 1f, targetValue = 1.25f,
        animationSpec = infiniteRepeatable(tween(700, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "mic_pulse"
    )

    // Image picker launcher
    var selectedImageUris by remember { mutableStateOf<List<Uri>>(emptyList()) }
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        selectedImageUris = uris
    }

    // Runtime permission launcher for RECORD_AUDIO
    val micPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) viewModel.startListening()
    }

    Surface(
        color = CardBg,
        modifier = Modifier
            .fillMaxWidth()
            .border(
                0.5.dp,
                Color.White.copy(alpha = 0.06f),
                RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp)
            ),
        shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp),
        tonalElevation = 10.dp
    ) {
        Column(
            modifier = Modifier
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 14.dp)
        ) {
            // Loading waveform
            AnimatedVisibility(visible = viewModel.loading) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 10.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Spacer(Modifier.width(4.dp).height(waveHeight1.dp).background(NeonPurple, RoundedCornerShape(2.dp)))
                    Spacer(Modifier.width(5.dp))
                    Spacer(Modifier.width(4.dp).height(waveHeight2.dp).background(NeonCyan, RoundedCornerShape(2.dp)))
                    Spacer(Modifier.width(5.dp))
                    Spacer(Modifier.width(4.dp).height(waveHeight3.dp).background(NeonPurple, RoundedCornerShape(2.dp)))
                    Spacer(Modifier.width(5.dp))
                    Spacer(Modifier.width(4.dp).height(waveHeight1.dp).background(NeonCyan, RoundedCornerShape(2.dp)))
                    Spacer(Modifier.width(10.dp))
                    Text("AIOS thinking...", color = TextSecondary, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
                }
            }

            // Voice listening status bar
            AnimatedVisibility(visible = isListening) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(NeonGreen.copy(alpha = 0.1f))
                        .border(0.5.dp, NeonGreen.copy(alpha = 0.3f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(NeonGreen.copy(alpha = micScale))
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            if (voiceTranscript == "Listening...") "Listening..." else voiceTranscript.ifBlank { "Listening..." },
                            color = NeonGreen,
                            fontSize = 13.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    TextButton(onClick = { viewModel.stopListening() }) {
                        Text("Stop", color = NeonRed, fontSize = 12.sp)
                    }
                }
            }

            // Image preview row
            if (selectedImageUris.isNotEmpty()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                        .padding(bottom = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    selectedImageUris.forEach { uri ->
                        Box(modifier = Modifier.size(60.dp)) {
                            AsyncImage(
                                model = uri,
                                contentDescription = "Selected image",
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(RoundedCornerShape(8.dp)),
                                contentScale = ContentScale.Crop
                            )
                        }
                    }
                }
            }

            // Main input row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Text input field
                TextField(
                    value = textState,
                    onValueChange = { textState = it },
                    modifier = Modifier.weight(1f),
                    placeholder = {
                        Text(
                            "Ask AIOS anything...",
                            color = TextSecondary,
                            fontSize = 14.sp
                        )
                    },
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color.White.copy(alpha = 0.05f),
                        unfocusedContainerColor = Color.White.copy(alpha = 0.04f),
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        cursorColor = NeonPurple
                    ),
                    shape = RoundedCornerShape(20.dp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text),
                    maxLines = 5
                )

                Spacer(Modifier.width(8.dp))

                // Image Attachment Button
                IconButton(
                    onClick = { imagePickerLauncher.launch("image/*") },
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = Color.White.copy(0.06f)
                    ),
                    modifier = Modifier.size(48.dp)
                ) {
                    Icon(
                        Icons.Default.Image,
                        "Attach Image",
                        tint = TextSecondary,
                        modifier = Modifier.size(22.dp)
                    )
                }

                Spacer(Modifier.width(6.dp))

                // Mic Button
                val hasAudioPerm = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
                IconButton(
                    onClick = {
                        if (isListening) {
                            viewModel.stopListening()
                        } else {
                            if (hasAudioPerm) {
                                viewModel.startListening()
                            } else {
                                micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            }
                        }
                    },
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = if (isListening) NeonGreen.copy(0.2f) else Color.White.copy(0.06f)
                    ),
                    modifier = Modifier
                        .size(48.dp)
                        .then(if (isListening) Modifier.scale(micScale) else Modifier)
                ) {
                    Icon(
                        imageVector = if (isListening) Icons.Default.Close else Icons.Default.Mic,
                        contentDescription = "Voice Input",
                        tint = if (isListening) NeonGreen else TextSecondary,
                        modifier = Modifier.size(22.dp)
                    )
                }

                Spacer(Modifier.width(6.dp))

                // Send Button
                IconButton(
                    onClick = {
                        val trimmed = textState.trim()
                        if (trimmed.isNotEmpty() || selectedImageUris.isNotEmpty()) {
                            viewModel.sendMessage(trimmed, selectedImageUris)
                            textState = ""
                            selectedImageUris = emptyList()
                        }
                    },
                    enabled = !viewModel.loading && (textState.isNotBlank() || selectedImageUris.isNotEmpty()),
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = if (textState.isNotBlank() || selectedImageUris.isNotEmpty()) NeonPurple else Color.White.copy(0.04f),
                        disabledContainerColor = Color.White.copy(0.02f)
                    ),
                    modifier = Modifier.size(48.dp)
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        "Send",
                        tint = if (textState.isNotBlank() || selectedImageUris.isNotEmpty()) Color.White else TextSecondary,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
    }
}

// =====================================
// SETTINGS PANEL
// =====================================
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsPanel(
    viewModel: ChatViewModel,
    onDismiss: () -> Unit
) {
    val memoryList by viewModel.memoryState.collectAsState()

    var provider by remember { mutableStateOf(viewModel.llmProvider) }
    var apiKey by remember { mutableStateOf(viewModel.geminiApiKey) }
    var model by remember { mutableStateOf(viewModel.geminiModel) }
    var serverUrl by remember { mutableStateOf(viewModel.customServerUrl) }
    var voiceEnabled by remember { mutableStateOf(viewModel.voiceOutputEnabled) }
    var systemPrompt by remember { mutableStateOf(viewModel.systemInstruction) }
    var keyVisible by remember { mutableStateOf(false) }
    var showMemoryBank by remember { mutableStateOf(false) }

    Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = RoundedCornerShape(28.dp),
            color = CardBg,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp)
                .border(1.dp, Color.White.copy(alpha = 0.08f), RoundedCornerShape(28.dp)),
            tonalElevation = 20.dp
        ) {
            Column(
                modifier = Modifier.padding(22.dp).fillMaxWidth()
            ) {
                // Header
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(NeonPurple.copy(0.15f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Default.Settings, null, tint = NeonPurple, modifier = Modifier.size(20.dp))
                    }
                    Spacer(Modifier.width(12.dp))
                    Text(
                        "AIOS Configuration",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                }

                Spacer(Modifier.height(18.dp))

                LazyColumn(
                    modifier = Modifier
                        .weight(1f, fill = false)
                        .fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    // ── Provider Selector ──
                    item {
                        SettingLabel("LLM Engine Provider")
                        Spacer(Modifier.height(6.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            listOf("GEMINI_DIRECT" to "Gemini Direct", "BACKEND_SSE" to "SSE Backend").forEach { (id, label) ->
                                val selected = provider == id
                                SelectChip(label, selected, NeonPurple, Modifier.weight(1f)) { provider = id }
                            }
                        }
                    }

                    // ── Gemini-specific settings ──
                    if (provider == "GEMINI_DIRECT") {
                        item {
                            SettingLabel("Gemini API Key")
                            Spacer(Modifier.height(6.dp))
                            TextField(
                                value = apiKey,
                                onValueChange = { apiKey = it },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(14.dp),
                                colors = settingsTextFieldColors(),
                                placeholder = { Text("AIzaSy...", color = TextSecondary) },
                                visualTransformation = if (keyVisible) VisualTransformation.None else PasswordVisualTransformation(),
                                trailingIcon = {
                                    IconButton(onClick = { keyVisible = !keyVisible }) {
                                        Icon(
                                            if (keyVisible) Icons.Default.Info else Icons.Default.Lock,
                                            null,
                                            tint = TextSecondary
                                        )
                                    }
                                }
                            )
                        }

                        item {
                            SettingLabel("Gemini Model")
                            Spacer(Modifier.height(6.dp))
                            val models = listOf(
                                "gemini-2.5-flash-preview-05-20" to "2.5 Flash ⚡",
                                "gemini-2.5-pro-preview-06-05" to "2.5 Pro 🧠",
                                "gemini-2.0-flash" to "2.0 Flash",
                                "gemini-1.5-flash" to "1.5 Flash",
                                "gemini-1.5-pro" to "1.5 Pro"
                            )
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                models.chunked(2).forEach { row ->
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        row.forEach { (id, label) ->
                                            val selected = model == id
                                            SelectChip(label, selected, NeonCyan, Modifier.weight(1f)) { model = id }
                                        }
                                        // Fill empty cell if row has odd count
                                        if (row.size == 1) Spacer(Modifier.weight(1f))
                                    }
                                }
                            }
                        }
                    } else {
                        item {
                            SettingLabel("FastAPI Server Base URL")
                            Spacer(Modifier.height(6.dp))
                            TextField(
                                value = serverUrl,
                                onValueChange = { serverUrl = it },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(14.dp),
                                colors = settingsTextFieldColors(),
                                placeholder = { Text("http://10.0.2.2:8000/", color = TextSecondary) }
                            )
                        }
                    }

                    // ── Voice Output Toggle ──
                    item {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(14.dp))
                                .background(Color.White.copy(0.03f))
                                .padding(horizontal = 14.dp, vertical = 10.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text("Voice Output (TTS)", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                                Text("Read responses aloud", color = TextSecondary, fontSize = 11.sp)
                            }
                            Switch(
                                checked = voiceEnabled,
                                onCheckedChange = { voiceEnabled = it },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = NeonPurple,
                                    checkedTrackColor = NeonPurple.copy(0.4f)
                                )
                            )
                        }
                    }

                    // ── System Prompt ──
                    item {
                        SettingLabel("System Personality Instruction")
                        Spacer(Modifier.height(6.dp))
                        TextField(
                            value = systemPrompt,
                            onValueChange = { systemPrompt = it },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(90.dp),
                            shape = RoundedCornerShape(14.dp),
                            colors = settingsTextFieldColors(),
                            maxLines = 5
                        )
                    }

                    // ── Memory Bank Section ──
                    item {
                        // Header row - toggle section
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(14.dp))
                                .background(NeonCyan.copy(alpha = 0.06f))
                                .border(0.5.dp, NeonCyan.copy(0.2f), RoundedCornerShape(14.dp))
                                .clickable { showMemoryBank = !showMemoryBank }
                                .padding(horizontal = 14.dp, vertical = 12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Favorite, null, tint = NeonCyan, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(10.dp))
                                Column {
                                    Text(
                                        "Memory Bank",
                                        color = TextPrimary,
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                    Text(
                                        "${memoryList.size} stored fact${if (memoryList.size != 1) "s" else ""}",
                                        color = NeonCyan,
                                        fontSize = 11.sp
                                    )
                                }
                            }
                            Icon(
                                if (showMemoryBank) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                null,
                                tint = TextSecondary
                            )
                        }
                    }

                    // Memory Bank items
                    if (showMemoryBank) {
                        if (memoryList.isEmpty()) {
                            item {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Color.White.copy(0.03f))
                                        .padding(16.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        "No memories saved yet.\nTell AIOS to remember something!",
                                        color = TextSecondary,
                                        fontSize = 13.sp,
                                        lineHeight = 18.sp
                                    )
                                }
                            }
                        } else {
                            items(memoryList.size) { idx ->
                                val fact = memoryList[idx]
                                MemoryFactRow(fact = fact, onDelete = { viewModel.deleteMemoryFact(fact.key) })
                            }
                        }
                    }
                }

                Spacer(Modifier.height(18.dp))

                // Action buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = TextSecondary),
                        border = androidx.compose.foundation.BorderStroke(1.dp, Color.White.copy(0.1f)),
                        shape = RoundedCornerShape(14.dp)
                    ) { Text("Cancel") }

                    Button(
                        onClick = {
                            viewModel.updateSettings(provider, apiKey, model, serverUrl, voiceEnabled, systemPrompt)
                            onDismiss()
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = NeonPurple),
                        shape = RoundedCornerShape(14.dp)
                    ) { Text("Save Config", fontWeight = FontWeight.Bold) }
                }
            }
        }
    }
}

// =====================================
// MEMORY FACT ROW
// =====================================
@Composable
fun MemoryFactRow(fact: MemoryEntity, onDelete: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White.copy(0.03f))
            .border(0.5.dp, Color.White.copy(0.06f), RoundedCornerShape(12.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(NeonCyan.copy(0.1f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Favorite, null, tint = NeonCyan, modifier = Modifier.size(16.dp))
        }
        Spacer(Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(fact.key, color = NeonCyan, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            Text(fact.value, color = TextPrimary, fontSize = 13.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
        }
        IconButton(
            onClick = onDelete,
            modifier = Modifier.size(32.dp)
        ) {
            Icon(Icons.Default.Delete, "Forget", tint = NeonRed.copy(0.7f), modifier = Modifier.size(16.dp))
        }
    }
}

// =====================================
// HELPER COMPOSABLES
// =====================================
@Composable
private fun SettingLabel(text: String) {
    Text(
        text = text,
        color = TextSecondary,
        fontSize = 12.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = 0.5.sp
    )
}

@Composable
private fun SelectChip(
    label: String,
    selected: Boolean,
    accentColor: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(if (selected) accentColor.copy(0.15f) else Color.White.copy(0.03f))
            .border(
                1.dp,
                if (selected) accentColor.copy(0.5f) else Color.White.copy(0.06f),
                RoundedCornerShape(12.dp)
            )
            .clickable { onClick() }
            .padding(vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            label,
            color = if (selected) TextPrimary else TextSecondary,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            fontSize = 12.sp
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun settingsTextFieldColors() = TextFieldDefaults.colors(
    focusedContainerColor = Color.White.copy(0.05f),
    unfocusedContainerColor = Color.White.copy(0.04f),
    focusedIndicatorColor = Color.Transparent,
    unfocusedIndicatorColor = Color.Transparent,
    focusedTextColor = TextPrimary,
    unfocusedTextColor = TextPrimary,
    cursorColor = NeonPurple
)
