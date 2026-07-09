package com.aios.android.domain.model

data class User(
    val id: String = "",
    val email: String = "",
    val displayName: String = "",
    val photoUrl: String = "",
    val token: String = "",
    val isAuthenticated: Boolean = false
)
