package com.example.aios

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object ToolExecutionManager {
    private val gson = Gson()

    private fun parseParams(params: String): Map<String, Any> {
        if (params.isBlank()) return emptyMap()
        return try {
            val type = object : TypeToken<Map<String, Any>>() {}.type
            gson.fromJson(params, type) ?: emptyMap()
        } catch (e: Exception) {
            // Fallback for simple string formatted arguments, like: hour=7,minute=30
            val map = mutableMapOf<String, Any>()
            params.split(",").forEach { part ->
                val pair = part.split("=")
                if (pair.size == 2) {
                    val key = pair[0].trim()
                    val value = pair[1].trim()
                    map[key] = value
                }
            }
            map
        }
    }

    suspend fun execute(context: Context, command: String, params: String): String = withContext(Dispatchers.Main) {
        try {
            val map = parseParams(params)
            when (command) {
                "set_alarm" -> {
                    val hour = (map["hour"] as? Number)?.toInt()
                        ?: (map["hour"] as? String)?.toIntOrNull()
                        ?: 8
                    val minute = (map["minute"] as? Number)?.toInt()
                        ?: (map["minute"] as? String)?.toIntOrNull()
                        ?: 0
                    val message = (map["message"] as? String) ?: "AIOS Reminder"
                    AndroidTools.setSystemAlarm(context, hour, minute, message)
                    "Alarm set for $hour:${String.format("%02d", minute)}."
                }
                "toggle_flashlight" -> {
                    val enabled = (map["enabled"] as? Boolean)
                        ?: (map["enabled"] as? String)?.toBoolean()
                        ?: false
                    AndroidTools.toggleFlashlight(context, enabled)
                    "Flashlight turned ${if (enabled) "ON" else "OFF"}."
                }
                "get_device_status" -> {
                    val status = AndroidTools.getDeviceStatus(context)
                    gson.toJson(status)
                }
                "send_sms" -> {
                    val phoneNumber = (map["phoneNumber"] as? String) ?: ""
                    val message = (map["message"] as? String) ?: ""
                    if (phoneNumber.isNotBlank()) {
                        AndroidTools.sendSMS(context, phoneNumber, message)
                        "Drafted SMS to $phoneNumber."
                    } else {
                        "Error: Missing phone number."
                    }
                }
                "make_call" -> {
                    val phoneNumber = (map["phoneNumber"] as? String) ?: ""
                    if (phoneNumber.isNotBlank()) {
                        AndroidTools.makeCall(context, phoneNumber)
                        "Dialing $phoneNumber..."
                    } else {
                        "Error: Missing phone number."
                    }
                }
                "vibrate" -> {
                    val durationMs = (map["durationMs"] as? Number)?.toLong()
                        ?: (map["durationMs"] as? String)?.toLongOrNull()
                        ?: 500L
                    AndroidTools.vibrate(context, durationMs)
                    "Vibrated device for $durationMs ms."
                }
                "open_app" -> {
                    val appName = (map["appName"] as? String) ?: ""
                    if (appName.isNotBlank()) {
                        AndroidTools.openApp(context, appName)
                    } else {
                        "Error: Missing application name."
                    }
                }
                "get_current_location" -> {
                    val locationMap = AndroidTools.getCurrentLocation(context)
                    gson.toJson(locationMap)
                }
                
                // ========== PHASE 3: NEW ROUTES ==========
                
                "read_clipboard" -> {
                    AndroidTools.readClipboard(context)
                }
                "write_clipboard" -> {
                    val text = (map["text"] as? String) ?: ""
                    if (text.isNotBlank()) {
                        AndroidTools.writeClipboard(context, text)
                    } else {
                        "Error: Missing text content to copy."
                    }
                }
                "search_contacts" -> {
                    val name = (map["name"] as? String) ?: ""
                    if (name.isNotBlank()) {
                        AndroidTools.searchContacts(context, name)
                    } else {
                        "Error: Missing name to search contacts."
                    }
                }
                "web_search" -> {
                    val query = (map["query"] as? String) ?: ""
                    if (query.isNotBlank()) {
                        AndroidTools.webSearch(context, query)
                    } else {
                        "Error: Missing query string to search."
                    }
                }
                "open_url" -> {
                    val url = (map["url"] as? String) ?: ""
                    if (url.isNotBlank()) {
                        AndroidTools.openUrl(context, url)
                    } else {
                        "Error: Missing URL."
                    }
                }
                "get_calendar_events" -> {
                    val daysAhead = (map["daysAhead"] as? Number)?.toInt()
                        ?: (map["daysAhead"] as? String)?.toIntOrNull()
                        ?: 1
                    AndroidTools.getCalendarEvents(context, daysAhead)
                }
                "create_calendar_event" -> {
                    val title = (map["title"] as? String) ?: ""
                    val description = (map["description"] as? String) ?: ""
                    val startTimeMillis = (map["startTimeMillis"] as? Number)?.toLong()
                        ?: (map["startTimeMillis"] as? String)?.toLongOrNull()
                        ?: System.currentTimeMillis()
                    val endTimeMillis = (map["endTimeMillis"] as? Number)?.toLong()
                        ?: (map["endTimeMillis"] as? String)?.toLongOrNull()
                        ?: (startTimeMillis + 3600000L) // + 1 hour default
                    val location = (map["location"] as? String) ?: ""
                    
                    if (title.isNotBlank()) {
                        AndroidTools.createCalendarEvent(context, title, description, startTimeMillis, endTimeMillis, location)
                    } else {
                        "Error: Missing title to create calendar event."
                    }
                }
                "set_volume" -> {
                    val volumeLevel = (map["volumeLevel"] as? Number)?.toInt()
                        ?: (map["volumeLevel"] as? String)?.toIntOrNull()
                        ?: 5
                    val streamType = (map["streamType"] as? String) ?: "media"
                    AndroidTools.setVolume(context, volumeLevel, streamType)
                }
                "set_brightness" -> {
                    val level = (map["level"] as? Number)?.toInt()
                        ?: (map["level"] as? String)?.toIntOrNull()
                        ?: 128
                    AndroidTools.setBrightness(context, level)
                }
                "list_files" -> {
                    val directory = (map["directory"] as? String) ?: "Downloads"
                    AndroidTools.listFiles(context, directory)
                }
                "read_file" -> {
                    val filePath = (map["filePath"] as? String) ?: ""
                    if (filePath.isNotBlank()) {
                        AndroidTools.readTextFile(context, filePath)
                    } else {
                        "Error: Missing filePath to read."
                    }
                }
                "create_file" -> {
                    val fileName = (map["fileName"] as? String) ?: ""
                    val content = (map["content"] as? String) ?: ""
                    val directory = (map["directory"] as? String) ?: "Downloads"
                    if (fileName.isNotBlank()) {
                        AndroidTools.createTextFile(context, fileName, content, directory)
                    } else {
                        "Error: Missing fileName to create file."
                    }
                }
                
                else -> {
                    "Tool '$command' is not implemented on-device."
                }
            }
        } catch (e: Exception) {
            "Error executing $command: ${e.localizedMessage}"
        }
    }
}