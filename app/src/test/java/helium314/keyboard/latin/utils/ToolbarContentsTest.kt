// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import helium314.keyboard.latin.utils.ToolbarContent.CLIPBOARD
import helium314.keyboard.latin.utils.ToolbarContent.EMOJI_CATEGORIES
import helium314.keyboard.latin.utils.ToolbarContent.SUGGESTIONS
import helium314.keyboard.latin.utils.ToolbarContent.TOOLBAR_KEYS
import org.junit.Test
import kotlin.test.assertEquals

class ToolbarContentsTest {

    @Test fun oldDefaultKeepsEverything() =
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), toolbarContentsFromLegacy(null, true))

    @Test fun oldModesKeepTheirMainKeyboardStrip() {
        assertEquals(setOf(TOOLBAR_KEYS, CLIPBOARD, EMOJI_CATEGORIES), toolbarContentsFromLegacy("TOOLBAR_KEYS", true))
        assertEquals(setOf(SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), toolbarContentsFromLegacy("SUGGESTION_STRIP", true))
    }

    @Test fun oldHiddenFollowsTheExtraSwitch() {
        assertEquals(emptySet(), toolbarContentsFromLegacy("HIDDEN", true))
        assertEquals(setOf(CLIPBOARD, EMOJI_CATEGORIES), toolbarContentsFromLegacy("HIDDEN", false))
    }

    @Test fun unknownOldModeFallsBackToDefault() =
        assertEquals(setOf(TOOLBAR_KEYS, SUGGESTIONS, CLIPBOARD, EMOJI_CATEGORIES), toolbarContentsFromLegacy("NONSENSE", true))

    @Test fun mainKeyboardModeFollowsTheFirstTwoContents() {
        assertEquals(ToolbarMode.EXPANDABLE, toolbarModeOf(setOf(TOOLBAR_KEYS, SUGGESTIONS)))
        assertEquals(ToolbarMode.TOOLBAR_KEYS, toolbarModeOf(setOf(TOOLBAR_KEYS, EMOJI_CATEGORIES)))
        assertEquals(ToolbarMode.SUGGESTION_STRIP, toolbarModeOf(setOf(SUGGESTIONS, CLIPBOARD)))
        assertEquals(ToolbarMode.HIDDEN, toolbarModeOf(setOf(CLIPBOARD, EMOJI_CATEGORIES)))
        assertEquals(ToolbarMode.HIDDEN, toolbarModeOf(emptySet()))
    }

    @Test fun storedValueRoundTrips() {
        for (contents in listOf(emptySet(), setOf(EMOJI_CATEGORIES), setOf(TOOLBAR_KEYS, CLIPBOARD), ToolbarContent.entries.toSet()))
            assertEquals(contents, parseToolbarContents(toolbarContentsToPref(contents)))
    }

    @Test fun storedValueIgnoresUnknownEntries() =
        assertEquals(setOf(SUGGESTIONS), parseToolbarContents("SUGGESTIONS|SOMETHING_REMOVED|"))
}
