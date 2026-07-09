package com.example.aios

import android.app.SearchManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.os.Environment
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.provider.ContactsContract
import android.provider.Settings
import android.widget.Toast
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object AndroidTools {

    /**
     * Sets an alarm with a custom message and time.
     */
    fun setSystemAlarm(context: Context, hour: Int, minute: Int, message: String = "AIOS Reminder") {
        try {
            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
                putExtra(AlarmClock.EXTRA_MESSAGE, message)
                putExtra(AlarmClock.EXTRA_SKIP_UI, false) // Shows validation UI
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            Toast.makeText(context, "AIOS: Setting alarm for $hour:$minute", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Failed to set alarm: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Toggles the device flashlight.
     */
    fun toggleFlashlight(context: Context, enabled: Boolean) {
        try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
            val cameraId = cameraManager.cameraIdList.firstOrNull()
            if (cameraId != null) {
                cameraManager.setTorchMode(cameraId, enabled)
                val statusText = if (enabled) "ON" else "OFF"
                Toast.makeText(context, "AIOS: Flashlight turned $statusText", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, "Camera flashlight not available", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(context, "Flashlight error: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Retrieves key device stats: battery percentage, charging state, volumes, and time.
     */
    fun getDeviceStatus(context: Context): Map<String, Any> {
        val status = mutableMapOf<String, Any>()
        try {
            val batteryStatusIntent = context.registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = batteryStatusIntent?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryStatusIntent?.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, -1) ?: -1
            val batteryPct = if (level >= 0 && scale > 0) (level * 100 / scale.toFloat()).toInt() else -1

            val statusCharge = batteryStatusIntent?.getIntExtra(android.os.BatteryManager.EXTRA_STATUS, -1) ?: -1
            val isCharging = statusCharge == android.os.BatteryManager.BATTERY_STATUS_CHARGING ||
                    statusCharge == android.os.BatteryManager.BATTERY_STATUS_FULL

            status["battery_level"] = "$batteryPct%"
            status["charging"] = isCharging

            // Volume Levels
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            status["media_volume"] = "$currentVolume/$maxVolume"

            // System Time
            val currentTime = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
            status["system_time"] = currentTime

        } catch (e: Exception) {
            status["error"] = e.localizedMessage ?: "Unknown error"
        }
        return status
    }

    /**
     * Drafts an SMS to a number.
     */
    fun sendSMS(context: Context, phoneNumber: String, message: String) {
        try {
            val intent = Intent(Intent.ACTION_SENDTO).apply {
                data = android.net.Uri.parse("smsto:$phoneNumber")
                putExtra("sms_body", message)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            Toast.makeText(context, "AIOS: Drafted SMS to $phoneNumber", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "SMS error: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Launches dialer with a phone number.
     */
    fun makeCall(context: Context, phoneNumber: String) {
        try {
            val intent = Intent(Intent.ACTION_DIAL).apply {
                data = android.net.Uri.parse("tel:$phoneNumber")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            Toast.makeText(context, "AIOS: Dialing $phoneNumber", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Dialer error: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Triggers short haptic vibration.
     */
    fun vibrate(context: Context, durationMs: Long) {
        try {
            val vibrator = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as android.os.VibratorManager
                vibratorManager.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                context.getSystemService(Context.VIBRATOR_SERVICE) as android.os.Vibrator
            }

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                vibrator.vibrate(android.os.VibrationEffect.createOneShot(durationMs, android.os.VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(durationMs)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Searches installed apps and opens the first app matching the given name.
     */
    fun openApp(context: Context, appName: String): String {
        try {
            val pm = context.packageManager
            val packages = pm.getInstalledApplications(android.content.pm.PackageManager.GET_META_DATA)
            for (app in packages) {
                val label = pm.getApplicationLabel(app).toString().lowercase()
                if (label.contains(appName.lowercase())) {
                    val launchIntent = pm.getLaunchIntentForPackage(app.packageName)
                    if (launchIntent != null) {
                        launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        context.startActivity(launchIntent)
                        return "Successfully opened $label."
                    }
                }
            }
            return "Application '$appName' not found on this device."
        } catch (e: Exception) {
            return "Failed to open app '$appName': ${e.localizedMessage}"
        }
    }

    /**
     * Queries coordinates (latitude, longitude) using native location providers.
     */
    fun getCurrentLocation(context: Context): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        try {
            val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as android.location.LocationManager
            val providers = locationManager.getProviders(true)
            var bestLocation: android.location.Location? = null

            for (provider in providers) {
                @Suppress("MissingPermission")
                val loc = locationManager.getLastKnownLocation(provider) ?: continue
                if (bestLocation == null || loc.accuracy < bestLocation.accuracy) {
                    bestLocation = loc
                }
            }

            if (bestLocation != null) {
                result["latitude"] = bestLocation.latitude
                result["longitude"] = bestLocation.longitude
                result["accuracy"] = "${bestLocation.accuracy}m"
                result["provider"] = bestLocation.provider ?: "unknown"
            } else {
                result["error"] = "Location unavailable. Please make sure location access is enabled on the device."
            }
        } catch (e: SecurityException) {
            result["error"] = "Location permission denied by user."
        } catch (e: Exception) {
            result["error"] = e.localizedMessage ?: "Unknown location query error."
        }
        return result
    }

    /**
     * Reads text content from clipboard.
     */
    fun readClipboard(context: Context): String {
        return try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            if (clipboard.hasPrimaryClip()) {
                val item = clipboard.primaryClip?.getItemAt(0)
                val text = item?.text?.toString()
                if (!text.isNullOrEmpty()) {
                    text
                } else {
                    "Clipboard is empty."
                }
            } else {
                "Clipboard is empty."
            }
        } catch (e: Exception) {
            "Error reading clipboard: ${e.localizedMessage}"
        }
    }

    /**
     * Copies text content to clipboard.
     */
    fun writeClipboard(context: Context, text: String): String {
        return try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("AIOS Copy", text)
            clipboard.setPrimaryClip(clip)
            "Successfully copied to clipboard."
        } catch (e: Exception) {
            "Error writing to clipboard: ${e.localizedMessage}"
        }
    }

    /**
     * Searches device contacts by display name.
     */
    fun searchContacts(context: Context, name: String): String {
        return try {
            val contentResolver: ContentResolver = context.contentResolver
            val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER
            )
            val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf("%$name%")
            val cursor = contentResolver.query(uri, projection, selection, selectionArgs, null)

            val results = mutableListOf<String>()
            cursor?.use {
                val nameIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                val numberIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                var count = 0
                while (it.moveToNext() && count < 10) {
                    val contactName = if (nameIndex >= 0) it.getString(nameIndex) else "Unknown"
                    val contactNumber = if (numberIndex >= 0) it.getString(numberIndex) else "Unknown"
                    results.add("$contactName: $contactNumber")
                    count++
                }
            }
            if (results.isEmpty()) {
                "No contacts found matching '$name'."
            } else {
                "Found contacts:\n" + results.joinToString("\n")
            }
        } catch (e: Exception) {
            "Error searching contacts: ${e.localizedMessage}"
        }
    }

    /**
     * Launches a web search in default browser.
     */
    fun webSearch(context: Context, query: String): String {
        try {
            val intent = Intent(Intent.ACTION_WEB_SEARCH).apply {
                putExtra(SearchManager.QUERY, query)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            return "Opened web search for '$query'."
        } catch (e: Exception) {
            return "Error launching web search: ${e.localizedMessage}"
        }
    }

    /**
     * Opens a specific URL in browser.
     */
    fun openUrl(context: Context, url: String): String {
        try {
            val formattedUrl = if (!url.startsWith("http://") && !url.startsWith("https://")) {
                "https://$url"
            } else {
                url
            }
            val intent = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(formattedUrl)).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            return "Opened browser at $formattedUrl."
        } catch (e: Exception) {
            return "Error opening URL: ${e.localizedMessage}"
        }
    }

    /**
     * Retrieves calendar events for next N days.
     */
    fun getCalendarEvents(context: Context, daysAhead: Int): String {
        return try {
            val contentResolver = context.contentResolver
            val uri = CalendarContract.Events.CONTENT_URI
            val projection = arrayOf(
                CalendarContract.Events.TITLE,
                CalendarContract.Events.DTSTART,
                CalendarContract.Events.DTEND,
                CalendarContract.Events.EVENT_LOCATION
            )
            
            val startMillis = System.currentTimeMillis()
            val endMillis = startMillis + (daysAhead * 24 * 60 * 60 * 1000L)
            
            val selection = "(${CalendarContract.Events.DTSTART} >= ?) AND (${CalendarContract.Events.DTSTART} <= ?)"
            val selectionArgs = arrayOf(startMillis.toString(), endMillis.toString())
            val sortOrder = "${CalendarContract.Events.DTSTART} ASC"
            
            val cursor = contentResolver.query(uri, projection, selection, selectionArgs, sortOrder)
            val events = mutableListOf<String>()
            
            cursor?.use {
                val titleIdx = it.getColumnIndex(CalendarContract.Events.TITLE)
                val startIdx = it.getColumnIndex(CalendarContract.Events.DTSTART)
                val endIdx = it.getColumnIndex(CalendarContract.Events.DTEND)
                val locIdx = it.getColumnIndex(CalendarContract.Events.EVENT_LOCATION)
                
                var count = 0
                val sdf = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())
                
                while (it.moveToNext() && count < 20) {
                    val title = if (titleIdx >= 0) it.getString(titleIdx) else "Untitled"
                    val start = if (startIdx >= 0) it.getLong(startIdx) else 0L
                    val end = if (endIdx >= 0) it.getLong(endIdx) else 0L
                    val location = if (locIdx >= 0) it.getString(locIdx) ?: "No Location" else "No Location"
                    
                    val timeStr = "${sdf.format(Date(start))} to ${sdf.format(Date(end))}"
                    events.add("- $title ($timeStr) @ $location")
                    count++
                }
            }
            
            if (events.isEmpty()) {
                "No calendar events found for the next $daysAhead day(s)."
            } else {
                "Upcoming events:\n" + events.joinToString("\n")
            }
        } catch (e: SecurityException) {
            "Error: Calendar permission not granted."
        } catch (e: Exception) {
            "Error reading calendar: ${e.localizedMessage}"
        }
    }

    /**
     * Creates draft calendar event.
     */
    fun createCalendarEvent(
        context: Context,
        title: String,
        description: String,
        startTimeMillis: Long,
        endTimeMillis: Long,
        location: String
    ): String {
        try {
            val intent = Intent(Intent.ACTION_INSERT).apply {
                data = CalendarContract.Events.CONTENT_URI
                putExtra(CalendarContract.Events.TITLE, title)
                putExtra(CalendarContract.Events.DESCRIPTION, description)
                putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startTimeMillis)
                putExtra(CalendarContract.EXTRA_EVENT_END_TIME, endTimeMillis)
                putExtra(CalendarContract.Events.EVENT_LOCATION, location)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
            return "Successfully opened calendar event editor for '$title'."
        } catch (e: Exception) {
            return "Error creating calendar event: ${e.localizedMessage}"
        }
    }

    /**
     * Configures volume of a specific audio stream.
     */
    fun setVolume(context: Context, volumeLevel: Int, streamType: String): String {
        try {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val stream = when (streamType.lowercase()) {
                "ring" -> AudioManager.STREAM_RING
                "alarm" -> AudioManager.STREAM_ALARM
                "notification" -> AudioManager.STREAM_NOTIFICATION
                else -> AudioManager.STREAM_MUSIC
            }
            val maxVolume = audioManager.getStreamMaxVolume(stream)
            val targetVolume = volumeLevel.coerceIn(0, maxVolume)
            val currentVolume = audioManager.getStreamVolume(stream)
            
            audioManager.setStreamVolume(stream, targetVolume, AudioManager.FLAG_SHOW_UI)
            return "Set $streamType volume from $currentVolume to $targetVolume (Max: $maxVolume)."
        } catch (e: Exception) {
            return "Error setting volume: ${e.localizedMessage}"
        }
    }

    /**
     * Adjusts system screen brightness.
     */
    fun setBrightness(context: Context, level: Int): String {
        try {
            if (Settings.System.canWrite(context)) {
                val target = level.coerceIn(0, 255)
                Settings.System.putInt(context.contentResolver, Settings.System.SCREEN_BRIGHTNESS, target)
                return "Set screen brightness to $target."
            } else {
                val intent = Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS).apply {
                    data = android.net.Uri.parse("package:${context.packageName}")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(intent)
                return "WRITE_SETTINGS permission required. Opened system settings panel."
            }
        } catch (e: Exception) {
            return "Error setting brightness: ${e.localizedMessage}"
        }
    }

    /**
     * Lists files in public directories.
     */
    fun listFiles(context: Context, directoryName: String): String {
        try {
            val dir = when (directoryName.lowercase()) {
                "dcim" -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
                "documents" -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
                "music" -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC)
                "pictures" -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                else -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            }
            
            if (dir.exists() && dir.isDirectory) {
                val files = dir.listFiles()
                if (files.isNullOrEmpty()) {
                    return "No files found in $directoryName."
                } else {
                    val list = files.take(25).joinToString("\n") { file ->
                        val sizeKB = file.length() / 1024
                        "- ${file.name} (${sizeKB} KB)"
                    }
                    return "Files in $directoryName:\n$list"
                }
            } else {
                return "Directory $directoryName is not accessible or does not exist."
            }
        } catch (e: Exception) {
            return "Error listing files: ${e.localizedMessage}"
        }
    }

    /**
     * Reads a text file's contents.
     */
    fun readTextFile(context: Context, filePath: String): String {
        try {
            val file = File(filePath)
            if (file.exists() && file.isFile) {
                val content = file.readText()
                return if (content.length > 5000) {
                    content.take(5000) + "\n... [truncated]"
                } else {
                    content
                }
            } else {
                return "File not found at: $filePath"
            }
        } catch (e: Exception) {
            return "Error reading file: ${e.localizedMessage}"
        }
    }

    /**
     * Creates a text file.
     */
    fun createTextFile(context: Context, fileName: String, content: String, directoryName: String): String {
        try {
            val dir = when (directoryName.lowercase()) {
                "documents" -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
                else -> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            }
            
            if (!dir.exists()) {
                dir.mkdirs()
            }
            
            val file = File(dir, fileName)
            file.writeText(content)
            return "File created successfully at: ${file.absolutePath}"
        } catch (e: Exception) {
            return "Error creating file: ${e.localizedMessage}"
        }
    }
}