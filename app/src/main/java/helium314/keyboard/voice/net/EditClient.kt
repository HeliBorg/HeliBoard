// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.net

import helium314.keyboard.voice.Backend
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class EditClient(private val backend: Backend, timeoutSeconds: Int, caCertPem: String = "") {

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    private val http = OkHttpClient.Builder()
        .callTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
        .connectTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
        .trustCustomCa(caCertPem)
        .build()

    @Serializable
    private data class Message(val role: String, val content: String)

    @Serializable
    private data class ChatRequest(val model: String, val messages: List<Message>, val temperature: Double = 0.2)

    @Serializable
    private data class Choice(val message: Message)

    @Serializable
    private data class ChatResponse(val choices: List<Choice> = emptyList())

    fun edit(text: String, instruction: String): String {
        val messages = listOf(
            Message(
                "system",
                "You edit text. Apply the user's instruction to the given text and return only the edited text, with no commentary or quotes."
            ),
            Message("user", "Text:\n$text\n\nInstruction:\n$instruction"),
        )
        val payload = json.encodeToString(ChatRequest(backend.model, messages))
        val request = Request.Builder()
            .url("${backend.baseUrl}/v1/chat/completions")
            .apply { if (backend.apiKey.isNotEmpty()) header("Authorization", "Bearer ${backend.apiKey}") }
            .post(payload.toRequestBody("application/json".toMediaType()))
            .build()
        http.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException("HTTP ${response.code}: $body")
            return json.decodeFromString<ChatResponse>(body).choices.firstOrNull()?.message?.content?.trim().orEmpty()
        }
    }
}
