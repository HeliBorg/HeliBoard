package helium314.keyboard.settings

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.latin.settings.Settings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.system.measureNanoTime

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class SettingsContainerTest {

    private lateinit var container: SettingsContainer

    @Before
    fun setup() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        container = SettingsContainer(context)
    }

    @Test
    fun testFilterFunctionality() {
        // Just verify it doesn't crash and returns some results for empty or common strings
        val res = container.filter("a")
        assertTrue(res.isNotEmpty())
    }

    @Test
    fun testFilterPerformance() {
        val searches = listOf("a", "b", "c", "theme", "color", "sound", "vib", "dictionary", "key", "layout")

        // Warmup
        for (i in 1..10) {
            for (s in searches) {
                container.filter(s)
            }
        }

        val time = measureNanoTime {
            for (i in 1..100) {
                for (s in searches) {
                    container.filter(s)
                }
            }
        }
        println("Filter performance: ${time / 1_000_000} ms")
    }

    @Test
    fun twoThumbGestureGatedAutospaceSettingIsRegistered() {
        assertEquals(Settings.PREF_COMBINING_AUTOSPACE_ONLY_AFTER_GESTURE,
            container[Settings.PREF_COMBINING_AUTOSPACE_ONLY_AFTER_GESTURE]?.key)
    }

    @Test
    fun twoThumbLowLevelBackspaceSettingIsHiddenFromSearchRegistry() {
        assertNull(container[Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD])
    }

    @Test
    fun twoThumbAdvancedTogglesAreRegistered() {
        assertEquals(Settings.PREF_COMBINING_BACKSPACE_DELETES_COMPOSING_TEXT,
            container[Settings.PREF_COMBINING_BACKSPACE_DELETES_COMPOSING_TEXT]?.key)
        assertEquals(Settings.PREF_MULTIPART_FULL_WORD_SUGGESTIONS,
            container[Settings.PREF_MULTIPART_FULL_WORD_SUGGESTIONS]?.key)
        assertEquals(Settings.PREF_GESTURE_DEBUG_ACCUMULATE_FRAGMENTS,
            container[Settings.PREF_GESTURE_DEBUG_ACCUMULATE_FRAGMENTS]?.key)
    }
}
