// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import helium314.keyboard.voice.net.TranscriptionService
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class TranscriptionServiceTest {

    private lateinit var server: MockWebServer

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun fallsBackToSecondBackendWhenFirstFails() {
        server.enqueue(MockResponse().setResponseCode(500).setBody("boom"))
        server.enqueue(MockResponse().setBody("""{"text":"from fallback"}"""))
        val base = server.url("").toString().trimEnd('/')
        val config = VoiceConfig(
            transcriptionBackends = listOf(
                Backend("primary", base, "m1"),
                Backend("fallback", base, "m2"),
            ),
            editBackend = Backend("edit", base, "m"),
            recordMaxSeconds = 0,
            requestTimeoutSeconds = 5,
            retries = 0,
            streaming = false,
            caCertPem = "",
        )
        val result = TranscriptionService(config).transcribe(ByteArray(8), false) {}
        assertEquals("from fallback", result)
        assertEquals(2, server.requestCount)
    }

    @Test
    fun retriesBeforeMovingToNextBackend() {
        server.enqueue(MockResponse().setResponseCode(500))
        server.enqueue(MockResponse().setBody("""{"text":"ok"}"""))
        val base = server.url("").toString().trimEnd('/')
        val config = VoiceConfig(
            transcriptionBackends = listOf(Backend("only", base, "m1")),
            editBackend = Backend("edit", base, "m"),
            recordMaxSeconds = 0,
            requestTimeoutSeconds = 5,
            retries = 1,
            streaming = false,
            caCertPem = "",
        )
        val result = TranscriptionService(config).transcribe(ByteArray(8), false) {}
        assertEquals("ok", result)
        assertEquals(2, server.requestCount)
    }
}
