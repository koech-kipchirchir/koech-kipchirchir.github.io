package com.aios.android.data.remote.api

import com.aios.android.data.remote.dto.*
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

interface ChatApi {
    @POST("chat/completions")
    @Streaming
    suspend fun streamChat(@Body request: ChatRequest): Response<ResponseBody>

    @POST("chat/completions")
    suspend fun chat(@Body request: ChatRequest): Response<ChatResponse>
}

interface AuthApi {
    @POST("auth/login")
    suspend fun login(@Body request: AuthRequest): Response<AuthResponse>

    @POST("auth/register")
    suspend fun register(@Body request: AuthRequest): Response<AuthResponse>

    @POST("auth/refresh")
    suspend fun refreshToken(@Header("Authorization") token: String): Response<AuthResponse>

    @GET("auth/me")
    suspend fun getProfile(@Header("Authorization") token: String): Response<UserDto>
}

interface VisionApi {
    @Multipart
    @POST("vision/analyze")
    suspend fun analyzeImage(
        @Header("Authorization") token: String,
        @Part image: okhttp3.MultipartBody.Part,
        @Part("features") features: okhttp3.RequestBody
    ): Response<ResponseBody>
}

interface VoiceApi {
    @Multipart
    @POST("voice/transcribe")
    suspend fun transcribe(
        @Header("Authorization") token: String,
        @Part audio: okhttp3.MultipartBody.Part
    ): Response<ResponseBody>

    @POST("voice/synthesize")
    suspend fun synthesize(
        @Header("Authorization") token: String,
        @Body text: Map<String, String>
    ): Response<ResponseBody>
}
