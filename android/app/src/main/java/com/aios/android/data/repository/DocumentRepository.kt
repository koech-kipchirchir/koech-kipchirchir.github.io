package com.aios.android.data.repository

import android.content.Context
import android.net.Uri
import com.aios.android.data.local.dao.DocumentDao
import com.aios.android.data.local.entity.DocumentEntity
import com.aios.android.domain.model.Document
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DocumentRepository @Inject constructor(
    private val documentDao: DocumentDao,
    private val context: Context
) {
    fun getAllDocuments(): Flow<List<Document>> =
        documentDao.getAllDocuments().map { entities -> entities.map { it.toDomain() } }

    fun searchDocuments(query: String): Flow<List<Document>> =
        documentDao.searchDocuments(query).map { entities -> entities.map { it.toDomain() } }

    suspend fun importDocument(uri: Uri): Result<Document> {
        return try {
            val name = uri.lastPathSegment ?: "document_${System.currentTimeMillis()}"
            val inputStream = context.contentResolver.openInputStream(uri)
            val bytes = inputStream?.readBytes() ?: return Result.failure(Exception("Cannot read file"))
            inputStream.close()

            val fileName = "doc_${UUID.randomUUID()}_${name}"
            val fileDir = java.io.File(context.filesDir, "documents")
            fileDir.mkdirs()
            val file = java.io.File(fileDir, fileName)
            file.writeBytes(bytes)

            val mimeType = context.contentResolver.getType(uri) ?: "application/octet-stream"
            val entity = DocumentEntity(
                id = UUID.randomUUID().toString(),
                name = name ?: fileName,
                path = file.absolutePath,
                sizeBytes = bytes.size.toLong(),
                mimeType = mimeType,
                createdAt = System.currentTimeMillis(),
                isLocal = true
            )
            documentDao.insertDocument(entity)
            Result.success(entity.toDomain())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun deleteDocument(id: String) {
        documentDao.deleteDocumentById(id)
    }
}

private fun DocumentEntity.toDomain() = Document(
    id = id, name = name, path = path, sizeBytes = sizeBytes,
    mimeType = mimeType, createdAt = createdAt, isLocal = isLocal
)
