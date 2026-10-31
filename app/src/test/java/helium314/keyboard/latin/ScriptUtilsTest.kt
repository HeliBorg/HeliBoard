// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.utils.ScriptUtils.SCRIPT_CYRILLIC
import helium314.keyboard.latin.utils.ScriptUtils.SCRIPT_DEVANAGARI
import helium314.keyboard.latin.utils.ScriptUtils.SCRIPT_LATIN
import helium314.keyboard.latin.utils.ScriptUtils.needsWordSegmentation
import helium314.keyboard.latin.utils.ScriptUtils.script
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ScriptUtilsTest {
    @Test fun defaultScript() {
        assertEquals(SCRIPT_LATIN, "en".constructLocale().script())
        assertEquals(SCRIPT_DEVANAGARI, "hi".constructLocale().script())
        assertEquals(SCRIPT_LATIN, "hi_zz".constructLocale().script())
        assertEquals(SCRIPT_LATIN, "sr-Latn".constructLocale().script())
        assertEquals(SCRIPT_CYRILLIC, "mk".constructLocale().script())
        assertEquals(SCRIPT_CYRILLIC, "fr-Cyrl".constructLocale().script())
    }

    @Test fun needsWordSegmentationThai() {
        assertTrue(needsWordSegmentation("th".constructLocale()))
    }

    @Test fun needsWordSegmentationNonThai() {
        assertFalse(needsWordSegmentation("en".constructLocale()))
        assertFalse(needsWordSegmentation("ja".constructLocale()))
        assertFalse(needsWordSegmentation("zh".constructLocale()))
        assertFalse(needsWordSegmentation("lo".constructLocale()))
        assertFalse(needsWordSegmentation("km".constructLocale()))
        assertFalse(needsWordSegmentation("ko".constructLocale()))
        assertFalse(needsWordSegmentation("my".constructLocale()))
    }
}
