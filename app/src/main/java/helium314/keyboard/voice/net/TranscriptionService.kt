// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.net

import helium314.keyboard.voice.VoiceConfig
import java.io.IOException

class TranscriptionService(private val config: VoiceConfig) {

    fun transcribe(wav: ByteArray, streaming: Boolean, onPartial: (String) -> Unit): String {
        val client = TranscriptionClient(config.requestTimeoutSeconds, config.caCertPem)
        var lastError: Throwable? = null
        for (backend in config.transcriptionBackends) {
            repeat(config.retries + 1) {
                try {
                    return client.transcribe(backend, wav, streaming, onPartial)
                } catch (e: Exception) {
                    lastError = e
                }
            }
        }
        throw lastError ?: IOException("no transcription backend configured")
    }
}
