package com.aios.android.data.repository

import com.aios.android.data.local.dao.ChatDao
import com.aios.android.data.local.dao.MessageDao
import com.aios.android.data.local.entity.ChatEntity
import com.aios.android.data.local.entity.MessageEntity
import com.aios.android.data.remote.api.ChatApi
import com.aios.android.data.remote.dto.*
import com.aios.android.domain.model.Chat
import com.aios.android.domain.model.Message
import com.aios.android.domain.model.MessageRole
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ChatRepository @Inject constructor(
    private val chatDao: ChatDao,
    private val messageDao: MessageDao,
    private val chatApi: ChatApi
) {
    fun getAllChats(): Flow<List<Chat>> =
        chatDao.getAllChats().map { entities -> entities.map { it.toDomain() } }

    fun getMessages(chatId: String): Flow<List<Message>> =
        messageDao.getMessagesByChatId(chatId).map { entities -> entities.map { it.toDomain() } }

    suspend fun createChat(title: String = "New Chat"): Chat {
        val chat = ChatEntity(
            id = UUID.randomUUID().toString(),
            title = title,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
            model = "gpt-4o",
            isLocal = true
        )
        chatDao.insertChat(chat)
        return chat.toDomain()
    }

    suspend fun sendMessage(chatId: String, text: String): Message {
        val userMsg = MessageEntity(
            id = UUID.randomUUID().toString(),
            chatId = chatId,
            role = "user",
            content = text,
            createdAt = System.currentTimeMillis(),
            isStreaming = false,
            isError = false,
            metadataJson = "{}"
        )
        messageDao.insertMessage(userMsg)

        val chat = chatDao.getChatById(chatId)
        val messages = messageDao.getMessagesByChatId(chatId).let { /* use current value */ }

        return userMsg.toDomain()
    }

    suspend fun deleteChat(chatId: String) {
        chatDao.deleteChatById(chatId)
    }

    suspend fun deleteMessage(messageId: String) {
        messageDao.getMessageById(messageId)?.let { messageDao.deleteMessagesByChatId(it.chatId) }
    }
}

private fun ChatEntity.toDomain() = Chat(
    id = id, title = title, createdAt = createdAt,
    updatedAt = updatedAt, model = model, isLocal = isLocal
)

private fun MessageEntity.toDomain() = Message(
    id = id, chatId = chatId,
    role = if (role == "user") MessageRole.USER else MessageRole.ASSISTANT,
    content = content, createdAt = createdAt,
    isStreaming = isStreaming, isError = isError
)
