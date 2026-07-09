package com.aios.android.ui.components

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.sp
import com.aios.android.ui.theme.CodeBackground
import com.aios.android.ui.theme.DarkPrimary

@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    style: androidx.compose.ui.text.TextStyle = MaterialTheme.typography.bodyMedium
) {
    Text(
        text = parseMarkdown(text),
        modifier = modifier,
        style = style
    )
}

fun parseMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    val lines = text.split("\n")
    var inCodeBlock = false
    val codeBuilder = StringBuilder()

    for (line in lines) {
        when {
            line.startsWith("```") -> {
                if (inCodeBlock) {
                    withStyle(style = SpanStyle(
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.sp,
                        background = CodeBackground
                    )) { append(codeBuilder.toString().trimEnd()) }
                    append("\n")
                    codeBuilder.clear()
                    inCodeBlock = false
                } else {
                    inCodeBlock = true
                    codeBuilder.clear()
                }
            }
            inCodeBlock -> {
                if (codeBuilder.isNotEmpty()) codeBuilder.append("\n")
                codeBuilder.append(line)
            }
            line.startsWith("# ") -> {
                withStyle(style = SpanStyle(fontWeight = FontWeight.Bold, fontSize = 20.sp)) {
                    append(line.removePrefix("# ")); append("\n")
                }
            }
            line.startsWith("## ") -> {
                withStyle(style = SpanStyle(fontWeight = FontWeight.Bold, fontSize = 17.sp)) {
                    append(line.removePrefix("## ")); append("\n")
                }
            }
            line.startsWith("### ") -> {
                withStyle(style = SpanStyle(fontWeight = FontWeight.SemiBold, fontSize = 15.sp)) {
                    append(line.removePrefix("### ")); append("\n")
                }
            }
            line.startsWith("**") && line.endsWith("**") -> {
                withStyle(style = SpanStyle(fontWeight = FontWeight.Bold)) {
                    append(line.removeSurrounding("**")); append("\n")
                }
            }
            line.startsWith("*") && line.endsWith("*") && !line.startsWith("**") -> {
                withStyle(style = SpanStyle(fontStyle = androidx.compose.ui.text.font.FontStyle.Italic)) {
                    append(line.removeSurrounding("*")); append("\n")
                }
            }
            line.startsWith("> ") -> {
                withStyle(style = SpanStyle(color = DarkPrimary)) {
                    append(line.removePrefix("> ")); append("\n")
                }
            }
            line.startsWith("- ") || line.startsWith("* ") -> {
                append("  •  "); append(line.drop(2)); append("\n")
            }
            line.matches(Regex("\\d+\\..*")) -> {
                append("  ${line}"); append("\n")
            }
            line.startsWith("[") && line.contains("](") -> {
                val parts = line.split("](")
                val linkText = parts[0].removePrefix("[")
                val url = parts[1].removeSuffix(")")
                withStyle(style = SpanStyle(
                    color = DarkPrimary, textDecoration = TextDecoration.Underline
                )) {
                    append(linkText)
                }
                append("\n")
            }
            line.isBlank() -> append("\n")
            else -> { append(line); append("\n") }
        }
    }
    if (inCodeBlock && codeBuilder.isNotEmpty()) {
        withStyle(style = SpanStyle(
            fontFamily = FontFamily.Monospace, fontSize = 13.sp, background = CodeBackground
        )) { append(codeBuilder.toString()) }
    }
}
