// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream

class AudioRecorder(private val maxSeconds: Int, private val onChunk: ((ByteArray) -> Unit)? = null) {

    private val sampleRate = 16000
    private val channelBytes = 2

    @Volatile
    private var recording = false
    private var thread: Thread? = null
    private val pcm = ByteArrayOutputStream()

    @SuppressLint("MissingPermission")
    fun start() {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val bufSize = maxOf(minBuf, sampleRate * channelBytes)
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize
        )
        pcm.reset()
        recording = true
        record.startRecording()
        val maxBytes = if (maxSeconds > 0) sampleRate * channelBytes * maxSeconds else Int.MAX_VALUE
        val readSize = if (onChunk != null) sampleRate * channelBytes / 10 else bufSize
        thread = Thread {
            val buf = ByteArray(readSize)
            while (recording) {
                val n = record.read(buf, 0, buf.size)
                if (n > 0) {
                    pcm.write(buf, 0, n)
                    onChunk?.invoke(buf.copyOf(n))
                    if (pcm.size() >= maxBytes) recording = false
                }
            }
            record.stop()
            record.release()
        }.also { it.start() }
    }

    fun stop(): ByteArray {
        recording = false
        thread?.join()
        thread = null
        return WavEncoder.pcm16ToWav(pcm.toByteArray(), sampleRate)
    }
}
