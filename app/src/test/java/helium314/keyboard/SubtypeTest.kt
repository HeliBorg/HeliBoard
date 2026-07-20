package helium314.keyboard

import helium314.keyboard.keyboard.KeyboardElement
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.keyboard_parser.LocaleKeyboardInfos
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsSubtype.Companion.toSettingsSubtype
import helium314.keyboard.latin.utils.LayoutType
import helium314.keyboard.latin.utils.POPUP_KEYS_LAYOUT
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional
import helium314.keyboard.latin.utils.getResourceSubtypes
import helium314.keyboard.latin.utils.locale
import helium314.keyboard.latin.utils.prefs
import java.io.File
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowLog
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
@Config(shadows = [
    ShadowInputMethodManager2::class
])
class SubtypeTest {
    private val latinIME = Robolectric.setupService(LatinIME::class.java)
    private val params = KeyboardParams()

    init {
        ShadowLog.setupLogging()
        ShadowLog.stream = System.out
        params.mId = KeyboardLayoutSet.getFakeKeyboardId(KeyboardElement.ALPHABET)
        params.mPopupKeyOrder.add(POPUP_KEYS_LAYOUT)
        LocaleKeyboardInfos.addLocaleKeyTextsToParams(latinIME, params, LocaleKeyboardInfos.POPUP_KEYS_NORMAL)
    }

    @Test fun emptyAdditionalSubtypesResultsInEmptyList() {
        // avoid issues where empty string results in additional subtype for undefined locale
        val prefs = latinIME.prefs()
        prefs.edit().putString(Settings.PREF_ADDITIONAL_SUBTYPES, "").apply()
        assertTrue(SubtypeSettings.getAdditionalSubtypes().isEmpty())
        val from = SubtypeSettings.getResourceSubtypesForLocale("es".constructLocale()).first()

        // no change, and "changed" subtype actually is resource subtype -> still expect empty list
        SubtypeUtilsAdditional.changeAdditionalSubtype(from.toSettingsSubtype(), from.toSettingsSubtype(), latinIME)
        assertEquals(emptyList(), SubtypeSettings.getAdditionalSubtypes().map { it.toSettingsSubtype() })
    }

    @Test fun subtypeStaysEnabledOnEdits() {
        val prefs = latinIME.prefs()
        prefs.edit().putString(Settings.PREF_ADDITIONAL_SUBTYPES, "").apply() // clear it for convenience

        // edit enabled resource subtype
        val from = SubtypeSettings.getResourceSubtypesForLocale("es".constructLocale()).first()
        SubtypeSettings.addEnabledSubtype(prefs, from)
        val to = from.toSettingsSubtype().withLayout(LayoutType.SYMBOLS, "symbols_arabic")
        SubtypeUtilsAdditional.changeAdditionalSubtype(from.toSettingsSubtype(), to, latinIME)
        assertEquals(to, SubtypeSettings.getEnabledSubtypes(false).single().toSettingsSubtype())

        // change the new subtype back to the original resource subtype (which may itself
        // carry a regional SYMBOLS layout, so revert to it directly instead of assuming none)
        val toNew = from.toSettingsSubtype()
        SubtypeUtilsAdditional.changeAdditionalSubtype(to, toNew, latinIME)
        assertEquals(emptyList(), SubtypeSettings.getAdditionalSubtypes().map { it.toSettingsSubtype() })
        assertEquals(from.toSettingsSubtype(), SubtypeSettings.getEnabledSubtypes(false).single().toSettingsSubtype())
    }

    @Test fun symbolsOverrideCanBeRemoved() {
        clearSubtypePrefs()
        val from = SubtypeSettings.getResourceSubtypesForLocale("es".constructLocale()).first()
        SubtypeSettings.addEnabledSubtype(latinIME.prefs(), from)
        assertEquals("symbols_es", from.toSettingsSubtype().layoutName(LayoutType.SYMBOLS))

        // removing the SYMBOLS override differs from the resource subtype, so the subtype
        // must stay enabled as an additional subtype instead of vanishing or colliding
        val noSymbols = from.toSettingsSubtype().withoutLayout(LayoutType.SYMBOLS)
        assertEquals(null, noSymbols.layoutName(LayoutType.SYMBOLS))
        assertTrue(!noSymbols.isSameAsDefault())
        SubtypeUtilsAdditional.changeAdditionalSubtype(from.toSettingsSubtype(), noSymbols, latinIME)
        assertEquals(listOf(noSymbols), SubtypeSettings.getAdditionalSubtypes().map { it.toSettingsSubtype() })
        assertEquals(listOf(noSymbols), SubtypeSettings.getEnabledSubtypes(false).map { it.toSettingsSubtype() })
        clearSubtypePrefs()
    }

    @Test fun stalePrefsHealAfterResourceSubtypeChange() {
        clearSubtypePrefs()
        val prefs = latinIME.prefs()
        val de = SubtypeSettings.getResourceSubtypesForLocale("de".constructLocale()).first()
        val canonical = de.toSettingsSubtype()
        assertEquals("symbols_de", canonical.layoutName(LayoutType.SYMBOLS))
        // second enabled subtype that sorts before de, so healing must really match, not just take the first
        val ca = SubtypeSettings.getResourceSubtypesForLocale("ca".constructLocale()).first().toSettingsSubtype()

        // simulate prefs written by a version whose de subtype had no SYMBOLS layout in method.xml
        val stale = canonical.withoutLayout(LayoutType.SYMBOLS)
        prefs.edit().putString(Settings.PREF_ENABLED_SUBTYPES, ca.toPref() + Separators.SETS + stale.toPref()).apply()
        prefs.edit().putString(Settings.PREF_SELECTED_SUBTYPE, stale.toPref()).apply()
        SubtypeSettings.reloadEnabledSubtypes(latinIME)

        // the enabled subtype resolves to the current resource subtype, and the pref is healed
        assertEquals(listOf(ca, canonical), SubtypeSettings.getEnabledSubtypes(false).map { it.toSettingsSubtype() })
        assertEquals(ca.toPref() + Separators.SETS + canonical.toPref(), prefs.getString(Settings.PREF_ENABLED_SUBTYPES, ""))

        // the selected subtype falls back to the locale+main layout match (not the first enabled) and heals its pref
        assertEquals(canonical, SubtypeSettings.getSelectedSubtype(prefs).toSettingsSubtype())
        assertEquals(canonical.toPref(), prefs.getString(Settings.PREF_SELECTED_SUBTYPE, ""))

        // edits through the subtype settings take effect on the healed prefs
        val edited = canonical.withLayout(LayoutType.SYMBOLS, "symbols_arabic")
        SubtypeUtilsAdditional.changeAdditionalSubtype(canonical, edited, latinIME)
        assertEquals(listOf(ca, edited), SubtypeSettings.getEnabledSubtypes(false).map { it.toSettingsSubtype() })
        assertEquals(edited, SubtypeSettings.getSelectedSubtype(prefs).toSettingsSubtype())
        clearSubtypePrefs()
    }

    @Test fun allResourceSubtypeLayoutsExist() {
        // a typo in a KeyboardLayoutSet extra value in method.xml would silently fall back to
        // the default layout in LayoutUtils.getContent, so check all referenced layouts exist
        val layoutsDir = File("src/main/assets/layouts")
        assertTrue(layoutsDir.isDirectory)
        var checked = 0
        var symbolsOverrides = 0
        getResourceSubtypes(latinIME.resources).forEach { subtype ->
            val layouts = LayoutType.getLayoutMap(subtype.toSettingsSubtype().getExtraValueOf(ExtraValue.KEYBOARD_LAYOUT_SET) ?: "")
            layouts.forEach { (type, name) ->
                // "+" layouts have no own file, they extend the base layout by locale extra keys
                val fileName = if (type == LayoutType.MAIN) name.removeSuffix("+") else name
                val exists = File(layoutsDir, type.name.lowercase()).listFiles()
                    ?.any { it.name.startsWith("$fileName.") } == true
                assertTrue(exists, "no layout file for $type layout $name of subtype ${subtype.locale()}")
                checked++
                if (type == LayoutType.SYMBOLS) symbolsOverrides++
            }
        }
        assertTrue(checked >= 40, "expected to check at least 40 layout references, but found $checked")
        // guard against a typo on the layout-TYPE side silently dropping an override
        // (22 regional + 6 arabic SYMBOLS overrides at the time of writing)
        assertTrue(symbolsOverrides >= 28, "expected at least 28 SYMBOLS overrides, but found $symbolsOverrides")
        assertEquals("symbols_de",
            SubtypeSettings.getResourceSubtypesForLocale("de".constructLocale()).first().toSettingsSubtype().layoutName(LayoutType.SYMBOLS))
    }

    private fun clearSubtypePrefs() {
        latinIME.prefs().edit()
            .putString(Settings.PREF_ADDITIONAL_SUBTYPES, "")
            .putString(Settings.PREF_ENABLED_SUBTYPES, "")
            .remove(Settings.PREF_SELECTED_SUBTYPE)
            .apply()
        SubtypeSettings.reloadEnabledSubtypes(latinIME)
    }
}
