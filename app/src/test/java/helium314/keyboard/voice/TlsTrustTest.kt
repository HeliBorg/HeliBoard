// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import helium314.keyboard.voice.net.TranscriptionClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.tls.HandshakeCertificates
import okhttp3.tls.HeldCertificate
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import javax.net.ssl.SSLHandshakeException
import kotlin.test.assertFailsWith

class TlsTrustTest {

    private lateinit var server: MockWebServer
    private lateinit var caPem: String

    @Before
    fun setUp() {
        val cert = HeldCertificate.Builder()
            .addSubjectAlternativeName("localhost")
            .addSubjectAlternativeName("127.0.0.1")
            .build()
        val serverCertificates = HandshakeCertificates.Builder()
            .heldCertificate(cert)
            .build()
        server = MockWebServer()
        server.useHttps(serverCertificates.sslSocketFactory(), false)
        server.start()
        caPem = cert.certificatePem()
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    private fun backend() = Backend("test", server.url("").toString().trimEnd('/'), "model")

    @Test
    fun trustsSelfSignedServerWhenCaProvided() {
        server.enqueue(MockResponse().setBody("""{"text":"secure"}"""))
        val result = TranscriptionClient(5, caPem).transcribe(backend(), ByteArray(8), false) {}
        assertEquals("secure", result)
    }

    @Test
    fun rejectsSelfSignedServerWithoutCa() {
        server.enqueue(MockResponse().setBody("""{"text":"secure"}"""))
        assertFailsWith<SSLHandshakeException> {
            TranscriptionClient(5, "").transcribe(backend(), ByteArray(8), false) {}
        }
    }
}
