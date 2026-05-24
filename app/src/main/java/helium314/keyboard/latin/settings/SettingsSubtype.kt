// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.settings

import android.content.SharedPreferences
import android.os.Build
import android.view.inputmethod.InputMethodSubtype
import helium314.keyboard.keyboard.internal.keyboard_parser.POPUP_KEYS_ALL
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue.KEYBOARD_LAYOUT_SET
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Defaults.default
import helium314.keyboard.latin.utils.LayoutType
import helium314.keyboard.latin.utils.LayoutType.Companion.toExtraValue
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.POPUP_KEYS_ORDER_DEFAULT
import helium314.keyboard.latin.utils.ScriptUtils
import helium314.keyboard.latin.utils.ScriptUtils.script
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional
import helium314.keyboard.latin.utils.locale
import java.util.Locale

// some kind of intermediate between the string stored in preferences and an InputMethodSubtype
// todo: consider using a hashMap or sortedMap instead of a string if we run into comparison issues once again
data class SettingsSubtype(val locale: Locale, val extraValues: String) {

    fun toPref() = locale.toLanguageTag() + Separators.SET + extraValues

    /** Creates an additional subtype from the SettingsSubtype.
     *  Resulting InputMethodSubtypes are equal if SettingsSubtypes are equal */
    fun toAdditionalSubtype(): InputMethodSubtype {
        val asciiCapable = locale.script() == ScriptUtils.SCRIPT_LATIN
        return SubtypeUtilsAdditional.createAdditionalSubtype(locale, extraValues, asciiCapable, true)
    }

    fun mainLayoutName() = LayoutType.getMainLayoutFromExtraValue(extraValues)

    fun layoutName(type: LayoutType) = LayoutType.getLayoutMap(getExtraValueOf(KEYBOARD_LAYOUT_SET) ?: "")[type]

    fun with(extraValueKey: String, extraValue: String? = null): SettingsSubtype {
        val newList = extraValues.split(",")
            .filterNot { it.isBlank() || it.startsWith("$extraValueKey=") || it == extraValueKey }
        val newValue = if (extraValue == null) extraValueKey else "$extraValueKey=$extraValue"
        val newValues = (newList + newValue).sorted().joinToString(",")
        return copy(extraValues = newValues)
    }

    fun without(extraValueKey: String): SettingsSubtype {
        val newValues = extraValues.split(",")
            .filterNot { it.isBlank() || it.startsWith("$extraValueKey=") || it == extraValueKey }
            .joinToString(",")
        return copy(extraValues = newValues)
    }

    fun getExtraValueOf(extraValueKey: String): String? = extraValues.getExtraValueOf(extraValueKey)

    fun hasExtraValueOf(extraValueKey: String): Boolean = extraValues.hasExtraValueOf(extraValueKey)

    fun withLayout(type: LayoutType, name: String): SettingsSubtype {
        val map = LayoutType.getLayoutMap(getExtraValueOf(KEYBOARD_LAYOUT_SET) ?: "")
        map[type] = name
        return with(KEYBOARD_LAYOUT_SET, map.toExtraValue())
    }

    fun withoutLayout(type: LayoutType): SettingsSubtype {
        val map = LayoutType.getLayoutMap(getExtraValueOf(KEYBOARD_LAYOUT_SET) ?: "")
        map.remove(type)
        return if (map.isEmpty()) without(KEYBOARD_LAYOUT_SET)
        else with(KEYBOARD_LAYOUT_SET, map.toExtraValue())
    }

    fun isAdditionalSubtype(prefs: SharedPreferences) =
        prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, Defaults.PREF_ADDITIONAL_SUBTYPES)!!
            .split(Separators.SETS).contains(toPref())

    fun isSameAsDefault() = SubtypeSettings.getResourceSubtypesForLocale(locale).any { it.toSettingsSubtype() == this.toPref().toSettingsSubtype() }

    fun toEnabledSubtype(): InputMethodSubtype? =
        SubtypeSettings.getEnabledSubtypes().firstOrNull { it.toSettingsSubtype() == this }

    companion object {
        fun String.toSettingsSubtype(): SettingsSubtype =
            SettingsSubtype(
                substringBefore(Separators.SET).constructLocale(),
                // we want a sorted string for reliable comparison
                substringAfter(Separators.SET).split(",").sorted().joinToString(",")
            )

        fun String.getExtraValueOf(extraValueKey: String) = split(",")
            .firstOrNull { it.startsWith("$extraValueKey=") }?.substringAfter("$extraValueKey=")

        fun String.hasExtraValueOf(extraValueKey: String) = split(",")
            .any { it.startsWith("$extraValueKey=") || it == extraValueKey }

        /** Creates a SettingsSubtype from the given InputMethodSubtype.
         *  Will strip some extra values that are set when creating the InputMethodSubtype from SettingsSubtype */
        fun InputMethodSubtype.toSettingsSubtype(): SettingsSubtype {
            if (DebugFlags.DEBUG_ENABLED && Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && locale().toLanguageTag() == "und") {
                @Suppress("deprecation") // it's debug logging, better get all information
                (Log.e(
                    SettingsSubtype::class.simpleName,
                    "unknown language, should not happen ${locale}, $languageTag, $extraValue, ${hashCode()}, $nameResId"
                ))
            }
            val filteredExtraValue = extraValue.split(",").filterNot {
                it.isBlank()
                        || it == ExtraValue.ASCII_CAPABLE
                        || it == ExtraValue.EMOJI_CAPABLE
                        || it == ExtraValue.IS_ADDITIONAL_SUBTYPE
                        || it.startsWith(ExtraValue.UNTRANSLATABLE_STRING_IN_SUBTYPE_NAME)
            }.sorted().joinToString(",")
            require(!filteredExtraValue.contains(Separators.SETS) && !filteredExtraValue.contains(Separators.SET))
            { "extra value contains not allowed characters $filteredExtraValue" }
            return SettingsSubtype(locale(), filteredExtraValue)
        }

        // qwerty with all diacritics and all popups enabled
        val fallbackSubtype = SettingsSubtype("zz".constructLocale(), "").let {
            var subtype = SettingsSubtype("zz".constructLocale(), "")
                .with(ExtraValue.MORE_POPUPS, POPUP_KEYS_ALL)
                .with(ExtraValue.POPUP_ORDER, POPUP_KEYS_ORDER_DEFAULT)
            LayoutType.entries.forEach { subtype = subtype.withLayout(it, it.default) }
            subtype
        }
    }
}
