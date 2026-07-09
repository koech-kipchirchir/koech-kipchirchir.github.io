package com.aios.android.util

import android.net.Uri
import android.provider.OpenableColumns
import android.content.Context
import java.io.File

object FileUtils {
    fun getFileName(context: Context, uri: Uri): String {
        var name = "file"
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex >= 0 && cursor.moveToFirst()) {
                name = cursor.getString(nameIndex)
            }
        }
        return name
    }

    fun getFileSize(context: Context, uri: Uri): Long {
        var size = 0L
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (sizeIndex >= 0 && cursor.moveToFirst()) {
                size = cursor.getLong(sizeIndex)
            }
        }
        return size
    }

    fun copyToCache(context: Context, uri: Uri): File? {
        return try {
            val inputStream = context.contentResolver.openInputStream(uri) ?: return null
            val name = getFileName(context, uri)
            val file = File(context.cacheDir, "uploads/$name")
            file.parentFile?.mkdirs()
            file.outputStream().use { out -> inputStream.copyTo(out) }
            inputStream.close()
            file
        } catch (e: Exception) {
            null
        }
    }
}
