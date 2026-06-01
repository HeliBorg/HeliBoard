// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.net

import android.util.Base64
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.voice.Backend
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

sealed interface RealtimeEvent {
    data class Segment(val text: String, val final: Boolean) : RealtimeEvent
    data class ServerError(val message: String) : RealtimeEvent
    data object Other : RealtimeEvent
}

private val realtimeJson = Json { ignoreUnknownKeys = true }

fun parseRealtimeEvent(text: String): RealtimeEvent {
    val obj = runCatching { realtimeJson.parseToJsonElement(text).jsonObject }.getOrNull()
        ?: return RealtimeEvent.Other
    val type = obj["type"]?.jsonPrimitive?.contentOrNull ?: return RealtimeEvent.Other
    return when {
        type.endsWith("input_audio_transcription.delta") ->
            obj["delta"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotEmpty() }
                ?.let { RealtimeEvent.Segment(it, final = false) } ?: RealtimeEvent.Other
        type.endsWith("input_audio_transcription.completed") ->
            RealtimeEvent.Segment(obj["transcript"]?.jsonPrimitive?.contentOrNull.orEmpty(), final = true)
        type == "error" -> RealtimeEvent.ServerError(
            obj["error"]?.jsonObject?.get("message")?.jsonPrimitive?.contentOrNull
                ?: obj["message"]?.jsonPrimitive?.contentOrNull ?: "unknown error"
        )
        else -> RealtimeEvent.Other
    }
}

fun realtimeUrl(baseUrl: String, model: String): String {
    val ws = baseUrl.trimEnd('/')
        .replaceFirst("https://", "wss://")
        .replaceFirst("http://", "ws://")
    val m = model.replace(" ", "%20")
    return "$ws/v1/realtime?model=$m&transcription_model=$m&intent=transcription"
}

class RealtimeTranscriptionClient(timeoutSeconds: Int, caCertPem: String = "") {

    private val http = OkHttpClient.Builder()
        .connectTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .trustCustomCa(caCertPem)
        .build()

    interface Listener {
        fun onSegment(text: String, final: Boolean)
        fun onClosed(cause: Throwable?)
    }

    fun open(backend: Backend, listener: Listener): Session {
        val request = Request.Builder()
            .url(realtimeUrl(backend.baseUrl, backend.model))
            .apply { if (backend.apiKey.isNotEmpty()) header("Authorization", "Bearer ${backend.apiKey}") }
            .build()
        val session = Session()
        session.ws = http.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(
                    "{\"type\":\"session.update\",\"session\":{\"input_audio_transcription\":{\"model\":\"${backend.model}\"}}}"
                )
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                when (val event = parseRealtimeEvent(text)) {
                    is RealtimeEvent.Segment -> if (event.text.isNotEmpty()) listener.onSegment(event.text, event.final)
                    is RealtimeEvent.ServerError -> Log.w(TAG, "realtime server error: ${event.message}")
                    RealtimeEvent.Other -> {}
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                listener.onClosed(t)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                listener.onClosed(null)
            }
        })
        return session
    }

    class Session {
        internal var ws: WebSocket? = null

        fun appendAudio(pcm: ByteArray) {
            val b64 = Base64.encodeToString(pcm, Base64.NO_WRAP)
            ws?.send("{\"type\":\"input_audio_buffer.append\",\"audio\":\"$b64\"}")
        }

        fun commit() {
            ws?.send("{\"type\":\"input_audio_buffer.commit\"}")
        }

        fun close() {
            ws?.close(1000, null)
            ws = null
        }
    }

    companion object {
        private const val TAG = "VoxBoard"
    }
}
