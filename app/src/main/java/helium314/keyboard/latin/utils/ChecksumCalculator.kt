/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import java.security.NoSuchAlgorithmException

object ChecksumCalculator {
    @Throws(IOException::class)
    fun checksum(`in`: InputStream): String? { // This code from the Android documentation for MessageDigest. Nearly verbatim.
        val digester: MessageDigest = try {
            MessageDigest.getInstance("SHA-256")
        } catch (_: NoSuchAlgorithmException) {
            return null // Platform does not support SHA-256 : can't check, so return null
        }
        val bytes = ByteArray(8192)
        var byteCount: Int
        while (`in`.read(bytes).also { byteCount = it } > 0) {
            digester.update(bytes, 0, byteCount)
        }
        val digest = digester.digest()
        val s = StringBuilder()
        for (i in digest.indices) {
            s.append(String.format("%1$02x", digest[i]))
        }
        return s.toString()
    }

    fun checksum(file: File) = runCatching {
        file.inputStream().use { checksum(it) }
    }.getOrNull()
}
