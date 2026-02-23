package helium314.keyboard.keyboard.internal.keyboard_parser.floris

import android.view.inputmethod.EditorInfo
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.internal.KeyboardCodesSet
import helium314.keyboard.keyboard.internal.KeyboardIconsSet
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyData.Companion.replaceIconWithLabelIfNoDrawable
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.InputTypeUtils
import helium314.keyboard.latin.utils.ToolbarKey
import helium314.keyboard.latin.utils.getCodeForToolbarKey
import helium314.keyboard.latin.utils.toolbarKeyStrings
import java.util.Locale

/** labels for functional / special keys */
object KeyLabel {
    const val COM = "com"
    const val LANGUAGE_SWITCH = "language_switch"
    const val ACTION = "action"
    const val DELETE = "delete"
    const val SHIFT = "shift"
    const val NUMPAD = "numpad"
    const val SYMBOL = "symbol"
    const val ALPHA = "alpha"
    const val SYMBOL_ALPHA = "symbol_alpha"
    const val PERIOD = "period"
    const val COMMA = "comma"
    const val SPACE = "space"
    const val ZWNJ = "zwnj"
    const val CURRENCY = "$$$"
    const val CURRENCY1 = "$$$1"
    const val CURRENCY2 = "$$$2"
    const val CURRENCY3 = "$$$3"
    const val CURRENCY4 = "$$$4"
    const val CURRENCY5 = "$$$5"
    const val CTRL = "ctrl"
    const val ALT = "alt"
    const val FN = "fn"
    const val META = "meta"
    const val TAB = "tab"
    const val ESCAPE = "esc"
    const val TIMESTAMP = "timestamp"
    const val EMOJI_SEARCH = "emoji_search"

    /** to make sure a FlorisBoard label works when reading a JSON layout */
    // resulting special labels should be names of FunctionalKey enum, case insensitive
    fun String.convertFlorisLabel(): String = when (this) {
        "view_characters" -> ALPHA
        "view_symbols" -> SYMBOL
        "view_numeric_advanced" -> NUMPAD
        "view_phone" -> ALPHA // phone keyboard is treated like alphabet, just with different layout
        "view_phone2" -> SYMBOL // phone symbols
        "ime_ui_mode_media" -> toolbarKeyStrings[ToolbarKey.EMOJI]!!
        "ime_ui_mode_clipboard" -> toolbarKeyStrings[ToolbarKey.CLIPBOARD]!!
        "ime_ui_mode_text" -> ALPHA
        "currency_slot_1" -> CURRENCY
        "currency_slot_2" -> CURRENCY1
        "currency_slot_3" -> CURRENCY2
        "currency_slot_4" -> CURRENCY3
        "currency_slot_5" -> CURRENCY4
        "currency_slot_6" -> CURRENCY5
        "enter" -> ACTION
        "half_space" -> ZWNJ
        else -> this
    }

    fun String.rtlLabel(params: KeyboardParams): String {
        if (!params.mId.mSubtype.isRtlSubtype || params.mId.isNumberLayout) return this
        return when (this) {
            "{" -> "{|}"
            "}" -> "}|{"
            "(" -> "(|)"
            ")" -> ")|("
            "[" -> "[|]"
            "]" -> "]|["
            "<" -> "<|>"
            ">" -> ">|<"
            "≤" -> "≤|≥"
            "≥" -> "≥|≤"
            "«" -> "«|»"
            "»" -> "»|«"
            "‹" -> "‹|›"
            "›" -> "›|‹"
            "﴾" -> "﴾|﴿"
            "﴿" -> "﴿|﴾"
            else -> this
        }
    }

    fun keyLabelToActualLabel(label: String, params: KeyboardParams): String {
        val newLabel = when (label) {
            SYMBOL_ALPHA -> if (params.mId.isAlphabetKeyboard) params.mLocaleKeyboardInfos.labelSymbol else params.mLocaleKeyboardInfos.labelAlphabet
            SYMBOL -> params.mLocaleKeyboardInfos.labelSymbol
            ALPHA -> params.mLocaleKeyboardInfos.labelAlphabet
            COMMA -> params.mLocaleKeyboardInfos.labelComma
            PERIOD -> getPeriodLabel(params)
            SPACE -> getSpaceLabel(params)
            ACTION -> "${getActionKeyLabel(params)}|${getActionKeyCode(params)}"
            DELETE -> "!icon/delete_key|!code/key_delete"
            SHIFT -> "${getShiftLabel(params)}|!code/key_shift"
            COM -> params.mLocaleKeyboardInfos.tlds.first()
            LANGUAGE_SWITCH -> "!icon/language_switch_key|!code/key_language_switch"
            ZWNJ -> "!icon/zwnj_key|\u200C"
            CURRENCY -> params.mLocaleKeyboardInfos.currencyKey.first
            CURRENCY1 -> params.mLocaleKeyboardInfos.currencyKey.second[0]
            CURRENCY2 -> params.mLocaleKeyboardInfos.currencyKey.second[1]
            CURRENCY3 -> params.mLocaleKeyboardInfos.currencyKey.second[2]
            CURRENCY4 -> params.mLocaleKeyboardInfos.currencyKey.second[3]
            CURRENCY5 -> params.mLocaleKeyboardInfos.currencyKey.second[4]
            CTRL, ALT, FN, META, ESCAPE -> label.uppercase(Locale.US)
            TAB -> "!icon/tab_key|!code/${KeyCode.TAB}"
            TIMESTAMP -> "⌚"
            EMOJI_SEARCH -> "!icon/search_key|!code/key_emoji_search"
            else -> if (label in toolbarKeyStrings.values)
                "!icon/$label|!code/${getCodeForToolbarKey(ToolbarKey.valueOf(label.uppercase(Locale.US)))}"
            else label
        }
        val code = when (label) { // maybe a bit lazy to not assemble the entire string above
            SYMBOL_ALPHA -> KeyCode.SYMBOL_ALPHA
            SYMBOL       -> KeyCode.SYMBOL
            ALPHA        -> KeyCode.ALPHA
            CTRL         -> KeyCode.CTRL
            ALT          -> KeyCode.ALT
            FN           -> KeyCode.FN
            META         -> KeyCode.META
            ESCAPE       -> KeyCode.ESCAPE
            TIMESTAMP    -> KeyCode.TIMESTAMP
            else         -> null
        }
        return if (code == null) newLabel
        else "$newLabel|!code/$code"
    }

    private fun getShiftLabel(params: KeyboardParams) = when (params.mId.mElementId) {
        KeyboardId.ELEMENT_SYMBOLS_SHIFTED -> params.mLocaleKeyboardInfos.labelSymbol
        KeyboardId.ELEMENT_SYMBOLS -> params.mLocaleKeyboardInfos.getShiftSymbolLabel(
            Settings.getInstance().isTablet)
        KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED, KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED -> "!icon/${KeyboardIconsSet.NAME_SHIFT_KEY_SHIFTED}"
        KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCKED, KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED -> "!icon/${KeyboardIconsSet.NAME_SHIFT_KEY_LOCKED}"

        else -> "!icon/${KeyboardIconsSet.NAME_SHIFT_KEY}"
    }

    // todo (later): try avoiding this weirdness
    //  maybe just remove it and if users want it they can use custom functional layouts?
    //  but it has been like this "forever" and actually seems to make sense
    private fun getPeriodLabel(params: KeyboardParams): String {
        if (params.mId.isNumberLayout) return "."
        if (params.mId.isAlphabetKeyboard || params.mId.locale.language in listOf("ar", "fa"))
            return params.mLocaleKeyboardInfos.labelPeriod
        return "."
    }

    private fun getSpaceLabel(params: KeyboardParams): String =
        if (params.mId.isAlphaOrSymbolKeyboard || params.mId.isEmojiClipBottomRow)
            "!icon/space_key|!code/key_space"
        else "!icon/space_key_for_number_layout|!code/key_space"

    // todo (later): should this be handled with metaState? but metaState shift would require LOTS of changes...
    private fun getActionKeyCode(params: KeyboardParams): String {
        params.mId.mInternalAction?.let { return "${KeyboardCodesSet.PREFIX_CODE}${it.code()}" }
        return if (params.mId.isMultiLine && (params.mId.mElementId == KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED || params.mId.mElementId == KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED))
            "!code/key_shift_enter"
        else "!code/key_enter"
    }

    private fun getActionKeyLabel(params: KeyboardParams): String {
        params.mId.mInternalAction?.let { return it.label() }
        if (params.mId.isMultiLine && (params.mId.mElementId == KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED || params.mId.mElementId == KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED))
            return "!icon/enter_key"
        val iconName = when (params.mId.imeAction()) {
            EditorInfo.IME_ACTION_GO               -> KeyboardIconsSet.NAME_GO_KEY
            EditorInfo.IME_ACTION_SEARCH           -> KeyboardIconsSet.NAME_SEARCH_KEY
            EditorInfo.IME_ACTION_SEND             -> KeyboardIconsSet.NAME_SEND_KEY
            EditorInfo.IME_ACTION_NEXT             -> KeyboardIconsSet.NAME_NEXT_KEY
            EditorInfo.IME_ACTION_DONE             -> KeyboardIconsSet.NAME_DONE_KEY
            EditorInfo.IME_ACTION_PREVIOUS         -> KeyboardIconsSet.NAME_PREVIOUS_KEY
            InputTypeUtils.IME_ACTION_CUSTOM_LABEL -> return params.mId.mCustomActionLabel
            else                                   -> return "!icon/enter_key"
        }
        val replacement = iconName.replaceIconWithLabelIfNoDrawable(params)
        return if (iconName == replacement) // i.e. icon exists
            "!icon/$iconName"
        else
            replacement
    }
}
