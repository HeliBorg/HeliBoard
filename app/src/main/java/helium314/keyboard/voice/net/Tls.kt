// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice.net

import okhttp3.OkHttpClient
import okhttp3.tls.HandshakeCertificates
import okhttp3.tls.decodeCertificatePem

fun OkHttpClient.Builder.trustCustomCa(caCertPem: String): OkHttpClient.Builder {
    val pem = caCertPem.trim()
    if (pem.isEmpty()) return this
    val certificate = pem.decodeCertificatePem()
    val handshakeCertificates = HandshakeCertificates.Builder()
        .addPlatformTrustedCertificates()
        .addTrustedCertificate(certificate)
        .build()
    return sslSocketFactory(handshakeCertificates.sslSocketFactory(), handshakeCertificates.trustManager)
}
