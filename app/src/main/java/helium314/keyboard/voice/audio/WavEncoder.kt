// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.audio

import java.io.ByteArrayOutputStream

object WavEncoder {

    fun pcm16ToWav(data: ByteArray, sampleRate: Int, channels: Int = 1): ByteArray {
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val out = ByteArrayOutputStream(44 + data.size)
        fun ascii(s: String) = out.write(s.toByteArray(Charsets.US_ASCII))
        fun le32(v: Int) {
            out.write(v and 0xff); out.write((v shr 8) and 0xff)
            out.write((v shr 16) and 0xff); out.write((v shr 24) and 0xff)
        }
        fun le16(v: Int) { out.write(v and 0xff); out.write((v shr 8) and 0xff) }
        ascii("RIFF"); le32(36 + data.size); ascii("WAVE")
        ascii("fmt "); le32(16); le16(1); le16(channels)
        le32(sampleRate); le32(byteRate); le16(channels * bitsPerSample / 8); le16(bitsPerSample)
        ascii("data"); le32(data.size); out.write(data)
        return out.toByteArray()
    }
}
