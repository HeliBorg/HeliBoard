// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import helium314.keyboard.voice.net.RealtimeEvent
import helium314.keyboard.voice.net.parseRealtimeEvent
import helium314.keyboard.voice.net.realtimeUrl
import org.junit.Assert.assertEquals
import org.junit.Test

class RealtimeTranscriptionTest {

    @Test
    fun completedEventYieldsFinalSegment() {
        val event = parseRealtimeEvent(
            """{"type":"conversation.item.input_audio_transcription.completed","transcript":"Thank you."}"""
        )
        assertEquals(RealtimeEvent.Segment("Thank you.", final = true), event)
    }

    @Test
    fun deltaEventYieldsNonFinalSegment() {
        val event = parseRealtimeEvent(
            """{"type":"conversation.item.input_audio_transcription.delta","delta":"Than"}"""
        )
        assertEquals(RealtimeEvent.Segment("Than", final = false), event)
    }

    @Test
    fun errorEventYieldsServerError() {
        val event = parseRealtimeEvent(
            """{"type":"error","error":{"message":"bad request"}}"""
        )
        assertEquals(RealtimeEvent.ServerError("bad request"), event)
    }

    @Test
    fun unknownEventIsOther() {
        assertEquals(RealtimeEvent.Other, parseRealtimeEvent("""{"type":"session.created"}"""))
        assertEquals(RealtimeEvent.Other, parseRealtimeEvent("not json"))
    }

    @Test
    fun httpsBaseUrlBecomesWssRealtimeEndpoint() {
        assertEquals(
            "wss://10.44.0.4:8443/v1/realtime?model=deepdml/faster-whisper-large-v3-turbo-ct2" +
                "&transcription_model=deepdml/faster-whisper-large-v3-turbo-ct2&intent=transcription",
            realtimeUrl("https://10.44.0.4:8443/", "deepdml/faster-whisper-large-v3-turbo-ct2")
        )
    }

    @Test
    fun httpBaseUrlBecomesWs() {
        assertEquals(
            "ws://host:8001/v1/realtime?model=m&transcription_model=m&intent=transcription",
            realtimeUrl("http://host:8001", "m")
        )
    }
}
