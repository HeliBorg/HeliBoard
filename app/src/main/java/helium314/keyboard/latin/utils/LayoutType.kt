package helium314.keyboard.latin.utils

import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue
import java.io.File
import java.util.EnumMap

enum class LayoutType {
    MAIN, SYMBOLS, MORE_SYMBOLS, FUNCTIONAL, NUMBER, NUMBER_ROW, NUMPAD,
    NUMPAD_LANDSCAPE, PHONE, PHONE_SYMBOLS, EMOJI_BOTTOM, CLIPBOARD_BOTTOM,
    SHORTCUT_TOP, SHORTCUT_BOTTOM;

    companion object {
        fun EnumMap<LayoutType, String>.toExtraValue() = map { it.key.name + Separators.KV + it.value }.joinToString(Separators.ENTRY)

        fun getLayoutMap(string: String?): EnumMap<LayoutType, String> {
            val map = EnumMap<LayoutType, String>(LayoutType::class.java)
            string?.split(Separators.ENTRY)?.forEach {
                val s = it.split(Separators.KV)
                runCatching { map[LayoutType.valueOf(s[0])] = s[1] }
            }
            return map
        }

        val LayoutType.folder get() = "layouts${File.separator}${name.lowercase()}"

        val LayoutType.displayNameId get() = when (this) {
            MAIN -> R.string.subtype_no_language
            SYMBOLS -> R.string.layout_symbols
            MORE_SYMBOLS -> R.string.layout_symbols_shifted
            FUNCTIONAL -> R.string.layout_functional_keys
            NUMBER -> R.string.layout_number
            NUMBER_ROW -> R.string.layout_number_row
            NUMPAD -> R.string.layout_numpad
            NUMPAD_LANDSCAPE -> R.string.layout_numpad_landscape
            PHONE -> R.string.layout_phone
            PHONE_SYMBOLS -> R.string.layout_phone_symbols
            EMOJI_BOTTOM -> R.string.layout_emoji_bottom_row
            CLIPBOARD_BOTTOM -> R.string.layout_clip_bottom_row
            SHORTCUT_TOP -> R.string.layout_shortcut_top
            SHORTCUT_BOTTOM -> R.string.layout_shortcut_bottom
        }

        fun getMainLayoutFromExtraValue(extraValue: String): String? {
            val value = extraValue.split(",")
                .firstOrNull { it.startsWith("${ExtraValue.KEYBOARD_LAYOUT_SET}=") }?.substringAfter("=")
            if (value == null) return null
            return getLayoutMap(value)[MAIN]
        }
    }
}
