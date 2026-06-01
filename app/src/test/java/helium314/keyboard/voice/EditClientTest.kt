// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import helium314.keyboard.voice.net.EditClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

class EditClientTest {

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
    fun returnsAssistantContent() {
        server.enqueue(
            MockResponse().setBody(
                """{"choices":[{"message":{"role":"assistant","content":"The meeting is scheduled for Tuesday."}}]}"""
            )
        )
        val backend = Backend("edit", server.url("").toString().trimEnd('/'), "gpt-4o-mini", "key")
        val result = EditClient(backend, 5).edit("the meeting is on tuesday", "make it formal")
        assertEquals("The meeting is scheduled for Tuesday.", result)
    }

    @Test
    fun sendsBearerAuthorization() {
        server.enqueue(MockResponse().setBody("""{"choices":[{"message":{"role":"assistant","content":"x"}}]}"""))
        val backend = Backend("edit", server.url("").toString().trimEnd('/'), "gpt-4o-mini", "secret-token")
        EditClient(backend, 5).edit("a", "b")
        val recorded = server.takeRequest()
        assertEquals("Bearer secret-token", recorded.getHeader("Authorization"))
    }
}
