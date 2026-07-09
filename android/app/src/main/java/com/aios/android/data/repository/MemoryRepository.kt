package com.aios.android.data.repository

import com.aios.android.data.local.dao.MemoryDao
import com.aios.android.data.local.entity.MemoryEntity
import com.aios.android.domain.model.Memory
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class MemoryRepository @Inject constructor(
    private val memoryDao: MemoryDao
) {
    fun getAllMemories(): Flow<List<Memory>> =
        memoryDao.getAllMemories().map { entities -> entities.map { it.toDomain() } }

    fun getMemoriesByCategory(category: String): Flow<List<Memory>> =
        memoryDao.getMemoriesByCategory(category).map { entities -> entities.map { it.toDomain() } }

    fun searchMemories(query: String): Flow<List<Memory>> =
        memoryDao.searchMemories(query).map { entities -> entities.map { it.toDomain() } }

    suspend fun createMemory(key: String, value: String, category: String = "general"): Memory {
        val entity = MemoryEntity(
            id = UUID.randomUUID().toString(),
            key = key,
            value = value,
            category = category,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
            source = "manual"
        )
        memoryDao.insertMemory(entity)
        return entity.toDomain()
    }

    suspend fun updateMemory(id: String, key: String, value: String) {
        memoryDao.getMemoryByKey(key)?.let {
            memoryDao.updateMemory(it.copy(value = value, updatedAt = System.currentTimeMillis()))
        }
    }

    suspend fun deleteMemory(id: String) {
        memoryDao.deleteMemoryById(id)
    }
}

private fun MemoryEntity.toDomain() = Memory(
    id = id, key = key, value = value, category = category,
    createdAt = createdAt, updatedAt = updatedAt, source = source
)
