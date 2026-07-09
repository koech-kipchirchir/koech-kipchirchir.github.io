package com.aios.android.data.repository

import com.aios.android.data.remote.api.AuthApi
import com.aios.android.data.remote.dto.AuthRequest
import com.aios.android.data.remote.dto.AuthResponse
import com.aios.android.domain.model.User
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AuthRepository @Inject constructor(
    private val authApi: AuthApi
) {
    private var _currentUser: User? = null
    val currentUser: User? get() = _currentUser

    suspend fun login(email: String, password: String): Result<User> {
        return try {
            val response = authApi.login(AuthRequest(email, password))
            if (response.isSuccessful) {
                val body = response.body()!!
                val user = User(
                    id = body.user?.id ?: "",
                    email = body.user?.email ?: email,
                    displayName = body.user?.displayName ?: email,
                    photoUrl = body.user?.photoUrl ?: "",
                    token = body.token,
                    isAuthenticated = true
                )
                _currentUser = user
                Result.success(user)
            } else {
                Result.failure(Exception("Login failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun register(email: String, password: String): Result<User> {
        return try {
            val response = authApi.register(AuthRequest(email, password))
            if (response.isSuccessful) {
                val body = response.body()!!
                val user = User(
                    id = body.user?.id ?: "",
                    email = body.user?.email ?: email,
                    token = body.token,
                    isAuthenticated = true
                )
                _currentUser = user
                Result.success(user)
            } else {
                Result.failure(Exception("Registration failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun refreshToken(token: String): Result<String> {
        return try {
            val response = authApi.refreshToken("Bearer $token")
            if (response.isSuccessful) {
                Result.success(response.body()?.token ?: token)
            } else {
                Result.failure(Exception("Token refresh failed"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun logout() {
        _currentUser = null
    }
}
