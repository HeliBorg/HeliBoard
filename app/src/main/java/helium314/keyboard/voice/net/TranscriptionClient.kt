// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.net

import helium314.keyboard.voice.Backend
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class TranscriptionClient(timeoutSeconds: Int, caCertPem: String = "") {

    private val json = Json { ignoreUnknownKeys = true }

    private val http = OkHttpClient.Builder()
        .callTimeout(0, TimeUnit.MILLISECONDS)
        .connectTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
        .readTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
        .trustCustomCa(caCertPem)
        .build()

    @Serializable
    private data class TranscriptionResponse(val text: String = "")

    @Serializable
    private data class StreamEvent(val type: String = "", val delta: String = "", val text: String = "")

    fun transcribe(backend: Backend, wav: ByteArray, streaming: Boolean, onPartial: (String) -> Unit): String =
        if (streaming) stream(backend, wav, onPartial) else batch(backend, wav)

    private fun buildRequest(backend: Backend, wav: ByteArray, streaming: Boolean): Request {
        val form = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", "audio.wav", wav.toRequestBody("audio/wav".toMediaType()))
            .addFormDataPart("model", backend.model)
            .addFormDataPart("response_format", "json")
        if (streaming) form.addFormDataPart("stream", "true")
        return Request.Builder()
            .url("${backend.baseUrl}/v1/audio/transcriptions")
            .apply { if (backend.apiKey.isNotEmpty()) header("Authorization", "Bearer ${backend.apiKey}") }
            .post(form.build())
            .build()
    }

    private fun batch(backend: Backend, wav: ByteArray): String {
        http.newCall(buildRequest(backend, wav, false)).execute().use { response ->
            val payload = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException("HTTP ${response.code}: $payload")
            return json.decodeFromString<TranscriptionResponse>(payload).text.trim()
        }
    }

    private fun stream(backend: Backend, wav: ByteArray, onPartial: (String) -> Unit): String {
        http.newCall(buildRequest(backend, wav, true)).execute().use { response ->
            val source = response.body?.source() ?: throw IOException("empty response body")
            if (!response.isSuccessful) throw IOException("HTTP ${response.code}: ${source.readUtf8()}")
            val accumulated = StringBuilder()
            while (!source.exhausted()) {
                val line = source.readUtf8Line() ?: break
                if (!line.startsWith("data:")) continue
                val data = line.removePrefix("data:").trim()
                if (data.isEmpty() || data == "[DONE]") continue
                val event = runCatching { json.decodeFromString<StreamEvent>(data) }.getOrNull() ?: continue
                when {
                    event.delta.isNotEmpty() -> accumulated.append(event.delta)
                    event.type == "transcript.text.done" && event.text.isNotEmpty() -> {
                        accumulated.setLength(0)
                        accumulated.append(event.text)
                    }
                    event.text.isNotEmpty() -> {
                        if (accumulated.isNotEmpty()) accumulated.append(' ')
                        accumulated.append(event.text)
                    }
                    else -> continue
                }
                onPartial(accumulated.toString())
            }
            return accumulated.toString().trim()
        }
    }
}
