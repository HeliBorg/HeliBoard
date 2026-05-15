// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard

import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodSubtype
import com.android.inputmethod.keyboard.ProximityInfo
import helium314.keyboard.keyboard.Key.KeyParams
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.internal.KeySpecParser.KeySpecParserError
import helium314.keyboard.keyboard.internal.KeyboardBuilder
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.TouchPositionCorrection
import helium314.keyboard.keyboard.internal.UniqueKeysCache
import helium314.keyboard.keyboard.internal.keyboard_parser.LayoutParser
import helium314.keyboard.keyboard.internal.keyboard_parser.POPUP_KEYS_NORMAL
import helium314.keyboard.keyboard.internal.keyboard_parser.addLocaleKeyTextsToParams
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.latin.RichInputMethodSubtype
import helium314.keyboard.latin.utils.LayoutUtilsCustom
import helium314.keyboard.latin.utils.POPUP_KEYS_LAYOUT
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import org.robolectric.shadows.ShadowLog
import java.io.File
import java.util.Locale
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
@Config(shadows = [
    ShadowInputMethodManager2::class,
    ShadowProximityInfo::class,
])
class ParserTest {
    private val latinIME = Robolectric.setupService(LatinIME::class.java)
    private val params = KeyboardParams()

    init {
        ShadowLog.setupLogging()
        ShadowLog.stream = System.out
        params.mId = KeyboardLayoutSet.getFakeKeyboardId(KeyboardId.ELEMENT_ALPHABET)
        params.mPopupKeyOrder.add(POPUP_KEYS_LAYOUT)
        addLocaleKeyTextsToParams(latinIME, params, POPUP_KEYS_NORMAL)
    }

    // todo: add tests for background type, also consider e.g. emoji key has functional bg by default

    @Test fun simpleParser() {
        val layoutStrings = listOf(
"""
a
b
c

d
e
f
""", // normal
"""
a
b
c

d
e
f
""", // spaces in the empty line
"""
a
b
c

d
e
f
""".replace("\n", "\r\n"), // windows file endings
"""
a
b
c


d
e
f

""", // too many newlines
"""
a
b x
c v

d
e
f
""", // spaces in the end
"""
a
b
c

d
e
f""", // no newline at the end
        )
        val wantedKeyLabels = listOf(listOf("a", "b", "c"), listOf("d", "e", "f"))
        layoutStrings.forEach { layout ->
            val keyLabels = LayoutParser.parseSimpleString(layout)
                .map { row -> row.map { it.toKeyParams(params).mLabel } }
            assertEquals(wantedKeyLabels, keyLabels)
        }
    }

    @Test fun simpleKey() {
        assertIsExpected("""[[{ "$": "auto_text_key" "label": "a" }]]""", Expected('a'.code, "a"))
        assertIsExpected("""[[{ "$": "text_key" "label": "a" }]]""", Expected('a'.code, "a"))
        assertIsExpected("""[[{ "label": "a" }]]""", Expected('a'.code, "a"))
    }

    @Test fun labelAndExplicitCode() {
        assertIsExpected("""[[{ "$": "text_key" "label": "a", "code": 98 }]]""", Expected('b'.code, "a"))
    }

    @Test fun labelAndImplicitCode() {
        assertIsExpected("""[[{ "$": "text_key" "label": "a|b" }]]""", Expected('b'.code, "a"))
    }

    @Test fun labelAndImplicitText() {
        assertIsExpected("""[[{ "$": "text_key" "label": "a|bb" }]]""", Expected(KeyCode.MULTIPLE_CODE_POINTS, "a", text = "bb"))
        // todo: should this actually work?
        assertIsExpected("""[[{ "$": "text_key" "label": "a|" }]]""", Expected(KeyCode.MULTIPLE_CODE_POINTS, "a", text = ""))
    }

    @Test fun labelAndImplicitAndExplicitCode() { // explicit code overrides implicit code
        assertIsExpected("""[[{ "code": 32, "label": "a|b" }]]""", Expected(' '.code, "a"))
        assertIsExpected("""[[{ "code": 32, "label": "a|!code/key_delete" }]]""", Expected(' '.code, "a"))
        assertIsExpected("""[[{ "code": 32, "label": "a|!code/-1" }]]""", Expected(' '.code, "a"))
        assertIsExpected("""[[{ "code": -1, "label": "a|!code/key_delete" }]]""", Expected(KeyCode.CTRL, "a"))
        // todo: should text be null? it's not used at all (it could be, but it really should not)
        assertIsExpected("""[[{ "code": 32, "label": "a|bb" }]]""", Expected(' '.code, "a", text = "bb"))
        assertIsExpected("""[[{ "code": 32, "label": "a|bb", "popup": { "main": { "code": 32, "label": "!icon/undo|!code/key_delete" } } }]]""", Expected(' '.code, "a", text = "bb", popups = listOf(null to ' '.code)))
        assertIsExpected("""[[{ "code": 32, "label": "a|bb", "popup": { "main": { "code": -1, "label": "!icon/undo|!code/key_delete" } } }]]""", Expected(' '.code, "a", text = "bb", popups = listOf(null to KeyCode.CTRL)))
        assertIsExpected("""[[{ "code": 32, "label": "a|bb", "popup": { "main": { "code": 32, "label": "a|!code/key_delete" } } }]]""", Expected(' '.code, "a", text = "bb", popups = listOf("a" to ' '.code)))
        assertIsExpected("""[[{ "code": 32, "label": "a|bb", "popup": { "main": { "code": -1, "label": "a|!code/key_delete" } } }]]""", Expected(' '.code, "a", text = "bb", popups = listOf("a" to KeyCode.CTRL)))
    }

    @Test fun keyWithIconAndExplicitCode() {
        assertIsExpected("""[[{ "label": "!icon/clipboard", "code": 55 }]]""", Expected(55, icon = "clipboard"))
        assertIsExpected("""[[{ "label": "!icon/clipboard", "code": -1 }]]""", Expected(KeyCode.CTRL, icon = "clipboard"))
    }

    @Test fun keyWithIconAndImplicitCode() {
        assertIsExpected("""[[{ "label": "!icon/clipboard_action_key|!code/key_clipboard" }]]""", Expected(KeyCode.CLIPBOARD, icon = "clipboard_action_key"))
        assertIsExpected("""[[{ "label": "!icon/clipboard_action_key|!code/key_clipboard", "popup": { "main": { "label": "!icon/undo|!code/key_delete" } } }]]""", Expected(KeyCode.CLIPBOARD, icon = "clipboard_action_key", popups = listOf(null to KeyCode.DELETE)))
    }

    @Test fun popupKeyWithIconAndExplicitCode() {
        assertIsExpected("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key", "code": 32 }
      ]
    } }]]""", Expected('a'.code, "a", popups = listOf(null to ' '.code)))
    }

    @Test fun popupKeyWithIconAndExplicitAndImplicitCode() {
        assertIsExpected("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|", "code": 32 }
      ]
    } }]]""", Expected('a'.code, "a", popups = listOf(null to ' '.code)))
        assertIsExpected("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|abc", "code": 32 }
      ]
    } }]]""", Expected('a'.code, "a", popups = listOf(null to ' '.code)))
    }

    @Test fun labelAndImplicitCodeForPopup() {
        assertIsExpected("""[[{ "$": "text_key" "label": "a|b", "popup": { "main": { "label": "b|a" } } }]]""", Expected('b'.code, "a", popups = listOf("b" to 'a'.code)))
        assertIsExpected("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|" }
      ]
    } }]]""", Expected('a'.code, "a",
            popups = listOf(null to KeyCode.MULTIPLE_CODE_POINTS))
        )
    }

    @Test fun `| works`() {
        assertIsExpected("""[[{ "label": "|", "popup": { "main": { "label": "|" } } }]]""", Expected('|'.code, "|", popups = listOf("|" to '|'.code)))
    }

    @Test fun currencyKey() {
        assertIsExpected("""[[{ "label": "$$$" }]]""", Expected('$'.code, "$", popups = listOf("£", "€", "¢", "¥", "₱").map { it to it.first().code }))
    }

    @Test fun currencyKeyWithOtherCurrencyCode() {
        assertIsExpected("""[[{ "label": "$$$", code: -805 }]]""", Expected('¥'.code, "$", popups = listOf("£", "€", "¢", "¥", "₱").map { it to it.first().code }))
    }

    @Test fun currencyPopup() {
        assertIsExpected("""[[{ "label": "p", "popup": { "main": { "label": "$$$" } } }]]""", Expected('p'.code, "p", null, null, listOf("$" to '$'.code)))
        assertIsExpected("""[[{ "label": "p", "popup": { "main": { "label": "a", "code": -804 } } }]]""", Expected('p'.code, "p", null, null, listOf("a" to '€'.code)))
        assertIsExpected("""[[{ "label": "p", "popup": { "main": { "label": "!icon/clipboard_action_key", "code": -804 } } }]]""", Expected('p'.code, "p", null, null, listOf(null to '€'.code)))
    }

    @Test fun weirdCurrencyKey() {
        assertIsExpected("""[[{ "code": -801, "label": "currency_slot_1", "popup": {
      "main": { "code": -802, "label": "currency_slot_2" },
      "relevant": [
        { "code": -806, "label": "currency_slot_6" },
        { "code": -803, "label": "currency_slot_3" },
        { "code": -804, "label": "currency_slot_4" },
        { "code": -805, "label": "currency_slot_5" },
        { "code": -804, "label": "$$$4" }
      ]
    } }]]""", Expected('$'.code, "$", popups = listOf("£" to '£'.code, "₱" to '₱'.code, "€" to '€'.code, "¢" to '¢'.code, "¥" to '¥'.code, "¥" to '€'.code)))
    }

    @Test fun caseSelector() {
        assertIsExpected("""[[{ "$": "case_selector",
      "lower": { "code":  105, "label": "i" },
      "upper": { "code":  304, "label": "İ" }
    }]]""", Expected(105, "i"))
    }

    @Test fun caseSelectorWithPopup() {
        assertIsExpected("""[[{ "$": "case_selector",
      "lower": { "code":   59, "label": ";", "popup": {
        "relevant": [
          { "code":   58, "label": ":" }
        ]
      } },
      "upper": { "code":   58, "label": ":", "popup": {
        "relevant": [
          { "code":   59, "label": ";" }
        ]
      } }
    }]]""", Expected(';'.code, ";", popups = listOf(":").map { it to it.first().code }))
    }

    @Test fun shiftSelector() {
        assertIsExpected("""[[{ "$": "shift_state_selector",
      "shiftedManual": { "code":   62, "label": ">", "popup": {
        "relevant": [
          { "code":   46, "label": "." }
        ]
      } },
      "default": { "code":   46, "label": ".", "popup": {
        "relevant": [
          { "code":   62, "label": ">" }
        ]
      } }
    }]]""", Expected('.'.code, ".", popups = listOf(">").map { it to it.first().code }))
    }

    @Test fun nestedSelectors() {
        assertIsExpected("""[[{ "$": "shift_state_selector",
      "shiftedManual": { "code":   34, "label": "\"", "popup": {
        "relevant": [
          { "code":   33, "label": "!" },
          { "code":   39, "label": "'"}
        ]
      } },
      "default": { "$": "variation_selector",
        "email":   { "code":   64, "label": "@" },
        "uri":     { "code":   47, "label": "/" },
        "default": { "code":   39, "label": "'", "popup": {
          "relevant": [
            { "code":   33, "label": "!" },
            { "code":   34, "label": "\"" }
          ]
        } }
      }
    }]]""", Expected('\''.code, "'", popups = listOf("!", "\"").map { it to it.first().code }))
    }

    @Test fun layoutDirectionSelector() {
        assertIsExpected("""[[{ "$": "layout_direction_selector",
      "ltr": { "code":   40, "label": "(", "popup": {
        "main": { "code":   60, "label": "<" },
        "relevant": [
          { "code":   91, "label": "[" },
          { "code":  123, "label": "{" }
        ]
      } },
      "rtl": { "code":   41, "label": "(", "popup": {
        "main": { "code":   62, "label": "<" },
        "relevant": [
          { "code":   93, "label": "[" },
          { "code":  125, "label": "{" }
        ]
      } }
    }]]""", Expected('('.code, "(", popups = listOf("<", "[", "{").map { it to it.first().code }))
    }

    @Test fun autoMultiTextKey() {
        assertIsExpected("""[[{ "label": "্র" }]]""", Expected(KeyCode.MULTIPLE_CODE_POINTS, "্র", text = "্র"))
    }

    @Test fun multiTextKey() { // pointless without codepoints!
        assertIsExpected("""[[{ "$": "multi_text_key", "codePoints": [2509, 2480], "label": "্র" }]]""", Expected(KeyCode.MULTIPLE_CODE_POINTS, "্র", text = "্র"))
        assertIsExpected("""[[{ "$": "multi_text_key", "codePoints": [2509, 2480], "label": "x" }]]""", Expected(KeyCode.MULTIPLE_CODE_POINTS, "x", text = "্র"))
    }

    @Test fun negativeCode() {
        assertIsExpected("""[[{ "code":   -7, "label": "delete" }]]""", Expected(-7, icon = "delete_key"))
    }

    @Test fun keyWithType() {
        assertIsExpected("""[[{ "code":   57, "label": "9", "type": "numeric" }]]""", Expected(57, "9"))
        assertIsExpected("""[[{ "code":   -7, "label": "delete", "type": "enter_editing" }]]""", Expected(-7, icon = "delete_key"))
        // -207 gets translated to -202 in Int.toKeyEventCode
        assertIsExpected("""[[{ "code": -207, "label": "view_phone2", "type": "system_gui" }]]""", Expected(-202, "?123"))
    }

    @Test fun spaceKey() {
        assertIsExpected("""[[{ "code":   32, "label": "space" }]]""", Expected(32, icon = "space_key"))
    }

    @Test fun invalidKeys() {
        assertFailsWith<KeySpecParserError> {
            LayoutParser.parseJsonString("""[[{ "label": "!icon/clipboard_action_key" }]]""")
                .map { row -> row.mapNotNull { it.compute(params)?.toKeyParams(params) } }
        }
    }

    @Test fun popupWithCodeAndLabel() {
        val key = LayoutParser.parseJsonString("""[[{ "label": "w", "popup": {
          "main": { "code":   55, "label": "!" }
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals("!", key.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals('7'.code, key.toKeyParams(params).mPopupKeys?.first()?.mCode)
    }

    @Test fun popupWithCodeAndIcon() {
        val key = LayoutParser.parseJsonString("""[[{ "label": "w", "popup": {
          "main": { "code":   55, "label": "!icon/clipboard_action_key" }
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("clipboard_action_key", key.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals('7'.code, key.toKeyParams(params).mPopupKeys?.first()?.mCode)
    }

    @Test fun popupToolbarKey() {
        val key = LayoutParser.parseJsonString("""[[{ "label": "x", "popup": {
          "main": { "label": "undo" }
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("undo", key.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals(KeyCode.UNDO, key.toKeyParams(params).mPopupKeys?.first()?.mCode)
    }

    @Test fun popupKeyWithIconAndImplicitText() {
        val key = LayoutParser.parseJsonString("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|aa" }
      ]
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("go_key", key.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals(KeyCode.MULTIPLE_CODE_POINTS, key.toKeyParams(params).mPopupKeys?.first()?.mCode)
        assertEquals("aa", key.toKeyParams(params).mPopupKeys?.first()?.mOutputText)

        val key2 = LayoutParser.parseJsonString("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|" }
      ]
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key2.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("go_key", key2.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals(KeyCode.MULTIPLE_CODE_POINTS, key2.toKeyParams(params).mPopupKeys?.first()?.mCode)
        assertEquals("", key2.toKeyParams(params).mPopupKeys?.first()?.mOutputText)
    }

    // output text is null here, maybe should be changed?
    @Test fun popupKeyWithIconAndCodeAndImplicitText() {
        val key = LayoutParser.parseJsonString("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|", "code": 55 }
      ]
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("go_key", key.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals(55, key.toKeyParams(params).mPopupKeys?.first()?.mCode)
        assertEquals(null, key.toKeyParams(params).mPopupKeys?.first()?.mOutputText)

        val key2 = LayoutParser.parseJsonString("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|a", "code": 55 }
      ]
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key2.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("go_key", key2.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals(55, key2.toKeyParams(params).mPopupKeys?.first()?.mCode)
        assertEquals(null, key2.toKeyParams(params).mPopupKeys?.first()?.mOutputText)

        val key3 = LayoutParser.parseJsonString("""[[{ "label": "a", "popup": { "relevant": [
       { "label": "!icon/go_key|aa", "code": 55 }
      ]
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals(null, key3.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals("go_key", key3.toKeyParams(params).mPopupKeys?.first()?.mIconName)
        assertEquals(55, key3.toKeyParams(params).mPopupKeys?.first()?.mCode)
        assertEquals(null, key3.toKeyParams(params).mPopupKeys?.first()?.mOutputText)
    }

    @Test fun invalidPopupKeys() {
        assertFailsWith<KeySpecParserError> {
            LayoutParser.parseJsonString("""[[{ "label": "a", "popup": {
          "main": { "label": "!icon/clipboard_action_key" }
    } }]]""").map { row -> row.mapNotNull { it.compute(params)?.toKeyParams(params) } }
        }
    }

    @Test fun popupSymbolAlpha() {
        val key = LayoutParser.parseJsonString("""[[{ "label": "c", "popup": {
          "main": { "code":   -10001, "label": "x" }
    } }]]""").flatMap { row -> row.mapNotNull { it.compute(params) } }.single()
        assertEquals("x", key.toKeyParams(params).mPopupKeys?.first()?.mLabel)
        assertEquals(-10001, key.toKeyParams(params).mPopupKeys?.first()?.mCode)
    }

    @Test fun canLoadKeyboard() {
        val editorInfo = EditorInfo()
        val subtype = SubtypeUtilsAdditional.createEmojiCapableAdditionalSubtype(Locale.ENGLISH, "qwerty", true)
        val (kb, keys) = buildKeyboard(editorInfo, subtype, KeyboardId.ELEMENT_ALPHABET)
        assertEquals(kb.sortedKeys.size, keys.sumOf { it.size })
    }

    @Test fun `dvorak has 4 rows`() {
        val editorInfo = EditorInfo()
        val subtype = SubtypeUtilsAdditional.createEmojiCapableAdditionalSubtype(Locale.ENGLISH, "dvorak", true)
        val (_, keys) = buildKeyboard(editorInfo, subtype, KeyboardId.ELEMENT_ALPHABET)
        assertEquals(keys.size, 4)
    }

    @Test fun `de_DE has extra keys`() {
        val editorInfo = EditorInfo()
        val subtype = SubtypeUtilsAdditional.createEmojiCapableAdditionalSubtype(Locale.GERMANY, "qwertz+", true)
        val (_, keys) = buildKeyboard(editorInfo, subtype, KeyboardId.ELEMENT_ALPHABET)
        assertEquals(11, keys[0].size)
        assertEquals(11, keys[1].size)
        assertEquals(10, keys[2].size)
        val (_, keys2) = buildKeyboard(editorInfo, subtype, KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED)
        assertEquals(11, keys2[0].size)
        assertEquals(11, keys2[1].size)
        assertEquals(10, keys2[2].size)
    }

    @Test fun `popup key count does not depend on shift for (for simple layout)`() {
        val editorInfo = EditorInfo()
        val subtype = SubtypeUtilsAdditional.createEmojiCapableAdditionalSubtype(Locale.ENGLISH, "qwerty", true)
        val (kb, keys) = buildKeyboard(editorInfo, subtype, KeyboardId.ELEMENT_ALPHABET)
        val (kb2, keys2) = buildKeyboard(editorInfo, subtype, KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED)
        assertEquals(kb.sortedKeys.size, kb2.sortedKeys.size)
        keys.forEachIndexed { i, kpList -> kpList.forEachIndexed { j, kp ->
            assertEquals(kp.mPopupKeys?.size, keys2[i][j].mPopupKeys?.size)
        } }
        kb.sortedKeys.forEachIndexed { index, key ->
            assertEquals(key.popupKeys?.size, kb2.sortedKeys[index].popupKeys?.size)
        }
    }

    @Test fun parseExistingLayouts() {
        val dir = File("src/main/assets/layouts")
        dir.walk().forEach {
            if (it.isDirectory) return@forEach
            val content = it.readText()
            val data = if (it.name.endsWith(".json"))
                LayoutParser.parseJsonString(content)
            else LayoutParser.parseSimpleString(content)
            data.flatten().forEach { key -> key.compute(params)?.toKeyParams(params) }
        }
    }

    @Test fun simpleWithLabelPopupHasCode() {
        val keys = LayoutParser.parseSimpleString("""
            a symbol
            b esc
            c undo

            d $$$
            e $$$1
            f blah
            tab timestamp
    """).flatMap { row -> row.mapNotNull { it.compute(params)?.toKeyParams(params) } }
        assertEquals("?123", keys[0].mPopupKeys?.first()?.mLabel)
        assertEquals(KeyCode.SYMBOL, keys[0].mPopupKeys?.first()?.mCode)
        assertEquals("ESC", keys[1].mPopupKeys?.first()?.mLabel)
        assertEquals(KeyCode.ESCAPE, keys[1].mPopupKeys?.first()?.mCode)
        assertEquals(null, keys[2].mPopupKeys?.first()?.mLabel)
        assertEquals("undo", keys[2].mPopupKeys?.first()?.mIconName)
        assertEquals(KeyCode.UNDO, keys[2].mPopupKeys?.first()?.mCode)
        assertEquals("$", keys[3].mPopupKeys?.first()?.mLabel)
        assertEquals('$'.code, keys[3].mPopupKeys?.first()?.mCode)
        assertEquals("£", keys[4].mPopupKeys?.first()?.mLabel)
        assertEquals('£'.code, keys[4].mPopupKeys?.first()?.mCode)
        assertEquals("blah", keys[5].mPopupKeys?.first()?.mLabel)
        assertEquals(KeyCode.MULTIPLE_CODE_POINTS, keys[5].mPopupKeys?.first()?.mCode)
        assertEquals("tab_key", keys[6].mIconName)
        assertEquals(KeyCode.TAB, keys[6].mCode)
        assertEquals("⌚", keys[6].mPopupKeys?.first()?.mLabel)
        assertEquals(KeyCode.TIMESTAMP, keys[6].mPopupKeys?.first()?.mCode)
    }

    private data class Expected(val code: Int, val label: String? = null, val icon: String? = null, val text: String? = null, val popups: List<Pair<String?, Int>>? = null)

    private fun assertIsExpected(json: String, expected: Expected) {
        assertAreExpected(json, listOf(expected))
    }

    private fun assertAreExpected(json: String, expected: List<Expected>) {
        val keys = LayoutParser.parseJsonString(json)
            .map { row -> row.mapNotNull { it.compute(params) } }.flatten()
        keys.forEachIndexed { index, keyData ->
            println("data: key ${keyData.label}: code ${keyData.code}, popups: ${keyData.popup.getPopupKeyLabels(params)}")
            val keyParams = keyData.toKeyParams(params)
            println("params: key ${keyParams.mLabel}: code ${keyParams.mCode}, popups: ${keyParams.mPopupKeys?.toList()}")
            assertEquals(expected[index].label, keyParams.mLabel)
            assertEquals(expected[index].icon, keyParams.mIconName)
            assertEquals(expected[index].code, keyParams.mCode)
            // todo (later): what's wrong with popup order?
            assertEquals(expected[index].popups?.sortedBy { it.first }, keyParams.mPopupKeys?.mapNotNull { it.mLabel to it.mCode }?.sortedBy { it.first })
            assertEquals(expected[index].text, keyParams.outputText)
            assertTrue(LayoutUtilsCustom.checkKeys(listOf(listOf(keyParams))))
        }
    }

    private fun buildKeyboard(editorInfo: EditorInfo, subtype: InputMethodSubtype, elementId: Int): Pair<Keyboard, List<List<KeyParams>>> {
        val layoutParams = KeyboardLayoutSet.Params()
        val editorInfoField = KeyboardLayoutSet.Params::class.java.getDeclaredField("mEditorInfo").apply { isAccessible = true }
        editorInfoField.set(layoutParams, editorInfo)
        val subtypeField = KeyboardLayoutSet.Params::class.java.getDeclaredField("mSubtype").apply { isAccessible = true }
        subtypeField.set(layoutParams, RichInputMethodSubtype.get(subtype))
        val widthField = KeyboardLayoutSet.Params::class.java.getDeclaredField("mKeyboardWidth").apply { isAccessible = true }
        widthField.setInt(layoutParams, 500)
        val heightField = KeyboardLayoutSet.Params::class.java.getDeclaredField("mKeyboardHeight").apply { isAccessible = true }
        heightField.setInt(layoutParams, 300)

        val keysInRowsField = KeyboardBuilder::class.java.getDeclaredField("keysInRows").apply { isAccessible = true }

        val id = KeyboardId(elementId, layoutParams)
        val builder = KeyboardBuilder(latinIME, KeyboardParams(UniqueKeysCache.NO_CACHE))
        builder.load(id)
        @Suppress("UNCHECKED_CAST")
        return builder.build() to keysInRowsField.get(builder) as ArrayList<ArrayList<KeyParams>>
    }
}

@Implements(ProximityInfo::class)
class ShadowProximityInfo {
    @Implementation
    fun createNativeProximityInfo(tpc: TouchPositionCorrection): Long = 0
}
