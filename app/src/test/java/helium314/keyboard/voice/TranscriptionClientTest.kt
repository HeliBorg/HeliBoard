// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import helium314.keyboard.voice.net.TranscriptionClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class TranscriptionClientTest {

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

    private fun backend() = Backend("test", server.url("").toString().trimEnd('/'), "model")

    @Test
    fun batchReturnsText() {
        server.enqueue(MockResponse().setBody("""{"text":"hello world"}"""))
        val result = TranscriptionClient(5).transcribe(backend(), ByteArray(8), false) {}
        assertEquals("hello world", result)
    }

    @Test
    fun streamingAppendsPerSegmentText() {
        val sse = buildString {
            append("data: {\"text\":\"This is the first sentence.\"}\n\n")
            append("data: {\"text\":\"Here comes the second sentence.\"}\n\n")
            append("data: {\"text\":\"And finally a third.\"}\n\n")
        }
        server.enqueue(MockResponse().setHeader("Content-Type", "text/event-stream").setBody(sse))
        val partials = mutableListOf<String>()
        val result = TranscriptionClient(5).transcribe(backend(), ByteArray(8), true) { partials.add(it) }
        assertEquals(3, partials.size)
        assertEquals("This is the first sentence. Here comes the second sentence. And finally a third.", result)
    }

    @Test
    fun streamingHandlesOpenAiDeltaThenDone() {
        val sse = buildString {
            append("data: {\"type\":\"transcript.text.delta\",\"delta\":\"Hel\"}\n\n")
            append("data: {\"type\":\"transcript.text.delta\",\"delta\":\"lo\"}\n\n")
            append("data: {\"type\":\"transcript.text.done\",\"text\":\"Hello\"}\n\n")
        }
        server.enqueue(MockResponse().setHeader("Content-Type", "text/event-stream").setBody(sse))
        val result = TranscriptionClient(5).transcribe(backend(), ByteArray(8), true) {}
        assertEquals("Hello", result)
    }
}
