// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.internal.KeySpecParser
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.PopupKeySpec
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyLabel.rtlLabel
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.PopupSet
import helium314.keyboard.latin.common.Constants.Separators

const val POPUP_KEYS_NUMBER = "number"
private const val POPUP_KEYS_LANGUAGE_PRIORITY = "language_priority"
const val POPUP_KEYS_LAYOUT = "layout"
private const val POPUP_KEYS_SYMBOLS = "symbols"
private const val POPUP_KEYS_LANGUAGE = "language"
const val POPUP_KEYS_LABEL_DEFAULT = POPUP_KEYS_NUMBER + Separators.KV + true + Separators.ENTRY + POPUP_KEYS_LANGUAGE_PRIORITY +
        Separators.KV + false + Separators.ENTRY + POPUP_KEYS_LAYOUT + Separators.KV + true + Separators.ENTRY +
        POPUP_KEYS_SYMBOLS + Separators.KV + true + Separators.ENTRY + POPUP_KEYS_LANGUAGE + Separators.KV + false
const val POPUP_KEYS_ORDER_DEFAULT = POPUP_KEYS_LANGUAGE_PRIORITY + Separators.KV + true + Separators.ENTRY + POPUP_KEYS_NUMBER +
        Separators.KV + true + Separators.ENTRY + POPUP_KEYS_SYMBOLS + Separators.KV + true + Separators.ENTRY +
        POPUP_KEYS_LAYOUT + Separators.KV + true + Separators.ENTRY + POPUP_KEYS_LANGUAGE + Separators.KV + true

private val allPopupKeyTypes = listOf(POPUP_KEYS_NUMBER, POPUP_KEYS_LAYOUT, POPUP_KEYS_SYMBOLS, POPUP_KEYS_LANGUAGE, POPUP_KEYS_LANGUAGE_PRIORITY)

fun createPopupKeysArray(popupSet: PopupSet<*>?, params: KeyboardParams, label: String): Array<String>? {
    // often PopupKeys are empty, so we want to avoid unnecessarily creating sets
    val popupKeysDelegate = lazy { mutableSetOf<String>() }
    val popupKeys by popupKeysDelegate
    val types = if (params.mId.isAlphabetKeyboard) params.mPopupKeyOrder else allPopupKeyTypes
    types.forEach { type ->
        when (type) {
            POPUP_KEYS_NUMBER -> popupSet?.numberLabel?.let { popupKeys.add(it) }
            POPUP_KEYS_LAYOUT -> popupSet?.getPopupKeyLabels(params)?.let { popupKeys.addAll(it) }
            POPUP_KEYS_SYMBOLS -> popupSet?.symbol?.let { popupKeys.add(it) }
            POPUP_KEYS_LANGUAGE -> params.mLocaleKeyboardInfos.getPopupKeys(label)?.let { popupKeys.addAll(it) }
            POPUP_KEYS_LANGUAGE_PRIORITY -> params.mLocaleKeyboardInfos.getPriorityPopupKeys(label)?.let { popupKeys.addAll(it) }
        }
    }
    if (!popupKeysDelegate.isInitialized() || popupKeys.isEmpty())
        return null
    val fco = popupKeys.firstOrNull { it.startsWith(Key.POPUP_KEYS_FIXED_COLUMN_ORDER) }
    if (fco != null && fco.substringAfter(Key.POPUP_KEYS_FIXED_COLUMN_ORDER).toIntOrNull() != popupKeys.size - 1) {
        val fcoExpected = popupKeys.size - popupKeys.count { it.startsWith("!") && it.endsWith("!") } - 1
        if (fco.substringAfter(Key.POPUP_KEYS_FIXED_COLUMN_ORDER).toIntOrNull() != fcoExpected)
            popupKeys.remove(fco) // maybe rather adjust the number instead of remove?
    }
    if (popupKeys.size > 1 && (label == "(" || label == ")")) { // add fixed column order for that case (typically other variants of brackets / parentheses
        // not really fast, but no other way to add first in a LinkedHashSet
        val tmp = popupKeys.toList()
        popupKeys.clear()
        popupKeys.add("${Key.POPUP_KEYS_FIXED_COLUMN_ORDER}${tmp.size}")
        popupKeys.addAll(tmp)
    }
    // autoColumnOrder should be fine

    val array = popupKeys.toTypedArray()
    for (i in array.indices) {
        array[i] = transformLabel(array[i], params)
    }
    return array
}

fun getHintLabel(popupSet: PopupSet<*>?, params: KeyboardParams, label: String): String? {
    val hintLabel = getHintText(popupSet, params, label) ?: return null
    if (hintLabel in toolbarKeyStrings.values || hintLabel.isEmpty())
        return null // better show nothing instead of the toolbar key label

    return KeySpecParser.getLabel(transformLabel(hintLabel, params))
        // avoid e.g. !autoColumnOrder! as label
        //  this will avoid having labels on comma and period keys
        ?.takeIf { label -> !label.startsWith("!") || label.count { it == '!' } != 2 } // excluding the special labels
}

fun getHintIcon(popupSet: PopupSet<*>?, params: KeyboardParams, label: String): String? =
    KeySpecParser.getIconName(getHintText(popupSet, params, label))

private fun getHintText(popupSet: PopupSet<*>?, params: KeyboardParams, label: String): String? {
    var hintLabel: String? = null
    for (type in params.mPopupKeyHintOrder) {
        when (type) {
            POPUP_KEYS_NUMBER -> popupSet?.numberLabel?.let { hintLabel = it }
            POPUP_KEYS_LAYOUT -> popupSet?.getPopupKeyLabels(params)?.let { hintLabel = it.firstOrNull() }
            POPUP_KEYS_SYMBOLS -> popupSet?.symbol?.let { hintLabel = it }
            POPUP_KEYS_LANGUAGE -> params.mLocaleKeyboardInfos.getPopupKeys(label)?.let { hintLabel = it.firstOrNull() }
            POPUP_KEYS_LANGUAGE_PRIORITY -> params.mLocaleKeyboardInfos.getPriorityPopupKeys(label)?.let { hintLabel = it.firstOrNull() }
        }
        if (hintLabel != null) return hintLabel
    }
    return null
}

private fun transformLabel(label: String, params: KeyboardParams): String =
    if (label.startsWith("$$$")) { // currency keys, todo: handling is similar to textKeyData, could it be merged?
        if (label == "$$$") {
            if (params.mId.passwordInput()) "$"
            else params.mLocaleKeyboardInfos.currencyKey.first
        } else {
            val index = label.substringAfter("$$$").toIntOrNull()
            if (index != null && index in 1..5)
                params.mLocaleKeyboardInfos.currencyKey.second[index - 1]
            else label
        }
    } else if (params.mId.mSubtype.isRtlSubtype) {
        label.rtlLabel(params)
    } else label

/** returns a list of enabled popup keys */
fun getEnabledPopupKeys(string: String): List<String> {
    return string.split(Separators.ENTRY).mapNotNull {
        val split = it.split(Separators.KV)
        if (split.last() == "true") split.first() else null
    }
}

// keep old label / icon, or switch to first found label or icon
// this is not really correct because it does not respect hint label order (fixing that is a little more effort)
fun findPopupHintLabelOrIcon(popupKeys: Array<PopupKeySpec>?, oldHintLabel: String?, oldHintIconName: String?): Pair<String?, String?> {
    if (popupKeys == null || (oldHintLabel == null && oldHintIconName == null)) return null to null
    if (oldHintLabel != null && popupKeys.any { it.mLabel == oldHintLabel }) return oldHintLabel to null
    if (oldHintIconName != null && popupKeys.any { it.mIconName == oldHintIconName }) return null to oldHintIconName
    val first = popupKeys.firstOrNull { it.mLabel != null || it.mIconName != null }
    return first?.mLabel to first?.mIconName
}
