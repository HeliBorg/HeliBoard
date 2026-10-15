// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ToolbarContent.CLIPBOARD
import helium314.keyboard.latin.utils.ToolbarContent.EMOJI_CATEGORIES
import helium314.keyboard.latin.utils.ToolbarContent.SUGGESTIONS
import helium314.keyboard.latin.utils.ToolbarContent.TOOLBAR_KEYS
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Suggestions in the toolbar and "show suggestions" are one setting, and old settings carry over. */
@RunWith(RobolectricTestRunner::class)
class ToolbarContentsPrefsTest {
    private lateinit var prefs: SharedPreferences

    @Before fun setUp() {
        prefs = RuntimeEnvironment.getApplication().getSharedPreferences("toolbar_contents_test", Context.MODE_PRIVATE)
        prefs.edit(commit = true) { clear() }
    }

    private val showSuggestions get() = prefs.getBoolean(Settings.PREF_SHOW_SUGGESTIONS, true)

    @Test fun freshInstallShowsEverything() =
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), readToolbarContents(prefs))

    @Test fun choosingSuggestionsInTheToolbarSwitchesShowSuggestions() {
        writeToolbarContents(prefs, setOf(TOOLBAR_KEYS, CLIPBOARD))
        assertFalse(showSuggestions)
        writeToolbarContents(prefs, setOf(TOOLBAR_KEYS, SUGGESTIONS))
        assertTrue(showSuggestions)
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS), readToolbarContents(prefs))
    }

    @Test fun showSuggestionsSwitchesSuggestionsInTheToolbar() {
        writeToolbarContents(prefs, setOf(TOOLBAR_KEYS, SUGGESTIONS, EMOJI_CATEGORIES))
        prefs.edit { putBoolean(Settings.PREF_SHOW_SUGGESTIONS, false) }
        assertEquals(setOf(TOOLBAR_KEYS, EMOJI_CATEGORIES), readToolbarContents(prefs))
        prefs.edit { putBoolean(Settings.PREF_SHOW_SUGGESTIONS, true) }
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS, EMOJI_CATEGORIES), readToolbarContents(prefs))
    }

    @Test fun suggestionsAreNotStoredWithTheOtherContents() {
        prefs.edit {
            putString(Settings.PREF_TOOLBAR_CONTENTS, toolbarContentsToPref(listOf(SUGGESTIONS, CLIPBOARD)))
            putBoolean(Settings.PREF_SHOW_SUGGESTIONS, false)
        }
        assertEquals(setOf(CLIPBOARD), readToolbarContents(prefs))
        writeToolbarContents(prefs, setOf(SUGGESTIONS, CLIPBOARD))
        assertEquals(setOf(CLIPBOARD), parseToolbarContents(prefs.getString(Settings.PREF_TOOLBAR_CONTENTS, null)!!))
    }

    private fun migrate(mode: String?, hidingGlobal: Boolean?, showSuggestions: Boolean): Set<ToolbarContent> {
        prefs.edit(commit = true) {
            clear()
            mode?.let { putString(Settings.PREF_TOOLBAR_MODE, it) }
            hidingGlobal?.let { putBoolean(Settings.PREF_TOOLBAR_HIDING_GLOBAL, it) }
            putBoolean(Settings.PREF_SHOW_SUGGESTIONS, showSuggestions)
        }
        migrateLegacyToolbarSettings(prefs)
        assertFalse(prefs.contains(Settings.PREF_TOOLBAR_MODE), "old mode left behind")
        assertFalse(prefs.contains(Settings.PREF_TOOLBAR_HIDING_GLOBAL), "old switch left behind")
        return readToolbarContents(prefs)
    }

    @Test fun oldModesKeepWhatTheyShowed() {
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), migrate("EXPANDABLE", true, true))
        assertEquals(setOf(TOOLBAR_KEYS, CLIPBOARD, EMOJI_CATEGORIES), migrate("TOOLBAR_KEYS", true, true))
        assertEquals(setOf(SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), migrate("SUGGESTION_STRIP", true, true))
        assertEquals(emptySet(), migrate("HIDDEN", true, true))
        assertEquals(setOf(CLIPBOARD, EMOJI_CATEGORIES), migrate("HIDDEN", false, true))
    }

    @Test fun suggestionsNeededBothTheModeAndTheSwitch() {
        assertEquals(setOf(TOOLBAR_KEYS, CLIPBOARD, EMOJI_CATEGORIES), migrate("EXPANDABLE", true, false))
        assertFalse(showSuggestions)
        migrate("TOOLBAR_KEYS", true, true)
        assertFalse(showSuggestions, "a mode without suggestions must turn show suggestions off")
    }

    @Test fun onlyTheSwitchStoredMeansTheOldDefaultMode() =
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), migrate(null, false, true))

    @Test fun nothingToTakeOverChangesNothing() {
        prefs.edit(commit = true) { putBoolean(Settings.PREF_SHOW_SUGGESTIONS, true) }
        migrateLegacyToolbarSettings(prefs)
        assertFalse(prefs.contains(Settings.PREF_TOOLBAR_CONTENTS))
        assertTrue(showSuggestions)
    }

    @Test fun takingOverTwiceChangesNothing() {
        val first = migrate("TOOLBAR_KEYS", true, true)
        migrateLegacyToolbarSettings(prefs)
        assertEquals(first, readToolbarContents(prefs))
        assertFalse(showSuggestions)
    }
}
