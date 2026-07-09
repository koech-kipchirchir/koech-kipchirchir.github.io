package com.aios.android.util

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "aios_prefs")

class PreferencesManager(private val context: Context) {
    companion object {
        val API_URL = stringPreferencesKey("api_url")
        val API_KEY = stringPreferencesKey("api_key")
        val MODEL = stringPreferencesKey("model")
        val THEME = stringPreferencesKey("theme")
        val TOKEN = stringPreferencesKey("token")
        val LANGUAGE = stringPreferencesKey("language")
    }

    val apiUrl: Flow<String> = context.dataStore.data.map { it[API_URL] ?: "http://10.0.2.2:8000/v1" }
    val apiKey: Flow<String> = context.dataStore.data.map { it[API_KEY] ?: "" }
    val model: Flow<String> = context.dataStore.data.map { it[MODEL] ?: "gpt-4o" }
    val theme: Flow<String> = context.dataStore.data.map { it[THEME] ?: "dark" }
    val token: Flow<String> = context.dataStore.data.map { it[TOKEN] ?: "" }
    val language: Flow<String> = context.dataStore.data.map { it[LANGUAGE] ?: "en" }

    suspend fun setApiUrl(url: String) { context.dataStore.edit { it[API_URL] = url } }
    suspend fun setApiKey(key: String) { context.dataStore.edit { it[API_KEY] = key } }
    suspend fun setModel(model: String) { context.dataStore.edit { it[MODEL] = model } }
    suspend fun setTheme(theme: String) { context.dataStore.edit { it[THEME] = theme } }
    suspend fun setToken(token: String) { context.dataStore.edit { it[TOKEN] = token } }
    suspend fun setLanguage(lang: String) { context.dataStore.edit { it[LANGUAGE] = lang } }

    suspend fun getToken(): String = context.dataStore.data.first()[TOKEN] ?: ""
    suspend fun clearToken() { context.dataStore.edit { it.remove(TOKEN) } }
}
