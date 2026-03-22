package helium314.keyboard.latin

import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import helium314.keyboard.compat.isDeviceLocked
import helium314.keyboard.compat.isUserLocked
import helium314.keyboard.keyboard.ColorSetting
import helium314.keyboard.keyboard.KeyboardTheme
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode.checkAndConvertCode
import helium314.keyboard.latin.common.ColorType
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.common.encodeBase36
import helium314.keyboard.latin.database.ClipboardDao
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsSubtype
import helium314.keyboard.latin.settings.SettingsSubtype.Companion.toSettingsSubtype
import helium314.keyboard.latin.settings.createPrefKeyForBooleanSettings
import helium314.keyboard.latin.utils.DeviceProtectedUtils
import helium314.keyboard.latin.utils.DictionaryInfoUtils
import helium314.keyboard.latin.utils.DictionaryInfoUtils.USER_DICTIONARY_SUFFIX
import helium314.keyboard.latin.utils.LayoutType
import helium314.keyboard.latin.utils.LayoutType.Companion.folder
import helium314.keyboard.latin.utils.LayoutUtilsCustom
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.ScriptUtils.SCRIPT_LATIN
import helium314.keyboard.latin.utils.ScriptUtils.script
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional
import helium314.keyboard.latin.utils.ToolbarKey
import helium314.keyboard.latin.utils.defaultPinnedToolbarPref
import helium314.keyboard.latin.utils.getResourceSubtypes
import helium314.keyboard.latin.utils.locale
import helium314.keyboard.latin.utils.mainLayoutNameOrQwerty
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.protectedPrefs
import helium314.keyboard.latin.utils.upgradeToolbarPrefs
import helium314.keyboard.latin.utils.writeCustomKeyCodes
import helium314.keyboard.settings.screens.colorPrefsAndResIds
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.util.EnumMap
import kotlin.collections.component1
import kotlin.collections.component2
import kotlin.collections.set

object AppUpgrade {
    fun checkVersionUpgrade(context: Context) {
        val prefs = context.prefs()
        val oldVersion = prefs.getInt(Settings.PREF_VERSION_CODE, 0)
        if (oldVersion == BuildConfig.VERSION_CODE)
            return
        // clear extracted dictionaries, in case updated version contains newer ones
        DictionaryInfoUtils.getCacheDirectories(context).forEach {
            for (file in it.listFiles()!!) {
                if (!file.name.endsWith(USER_DICTIONARY_SUFFIX))
                    file.delete()
            }
        }
        if (oldVersion <= 1000) { // upgrade old custom layouts name
            val oldShiftSymbolsFile = getCustomLayoutFile("custom.shift_symbols", context)
            if (oldShiftSymbolsFile.exists()) {
                oldShiftSymbolsFile.renameTo(getCustomLayoutFile("custom.symbols_shifted", context))
            }
            if (prefs.contains("enabled_input_styles")) {
                // rename subtype setting, and clean old subtypes that might remain in some cases
                val subtypesPref = prefs.getString("enabled_input_styles", "")!!
                    .split(";").filter { it.isNotEmpty() }
                    .map {
                        val localeAndLayout = it.split(":").toMutableList()
                        localeAndLayout[0] = localeAndLayout[0].constructLocale().toLanguageTag()
                        localeAndLayout.joinToString(":")
                    }.toSet().joinToString(";")
                prefs.edit { remove("enabled_input_styles").putString(Settings.PREF_ENABLED_SUBTYPES, subtypesPref) }
            }
            if (prefs.contains("selected_input_style")) {
                val selectedSubtype = prefs.getString("selected_input_style", "")
                prefs.edit { remove("selected_input_style").putString(Settings.PREF_SELECTED_SUBTYPE, selectedSubtype) }
            }
        }
        if (oldVersion <= 2000) {
            // upgrade pinned toolbar keys pref
            val oldPinnedKeysPref = prefs.getString(Settings.PREF_PINNED_TOOLBAR_KEYS, "")!!
            val pinnedKeys = oldPinnedKeysPref.split(";").mapNotNull {
                try {
                    ToolbarKey.valueOf(it)
                } catch (_: IllegalArgumentException) {
                    null
                }
            }
            val newPinnedKeysPref = (pinnedKeys.map { "${it.name},true" } + defaultPinnedToolbarPref.split(";"))
                .distinctBy { it.split(",").first() }
                .joinToString(";")
            prefs.edit { putString(Settings.PREF_PINNED_TOOLBAR_KEYS, newPinnedKeysPref) }

            // enable language switch key if it was enabled previously
            if (prefs.contains(Settings.PREF_LANGUAGE_SWITCH_KEY) && prefs.getString(Settings.PREF_LANGUAGE_SWITCH_KEY, "") != "off")
                prefs.edit { putBoolean(Settings.PREF_SHOW_LANGUAGE_SWITCH_KEY, true) }
        }
        if (oldVersion <= 2100) {
            if (prefs.contains("show_more_colors")) {
                val moreColors = prefs.getInt("show_more_colors", 0)
                prefs.edit {
                    putInt("theme_color_show_more_colors", moreColors)
                    if (prefs.getBoolean(Settings.PREF_THEME_DAY_NIGHT, false))
                        putInt("theme_dark_color_show_more_colors", moreColors)
                    remove("show_more_colors")
                }
            }
        }
        if (oldVersion <= 2201) {
            val additionalSubtypeString = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, Defaults.PREF_ADDITIONAL_SUBTYPES)!!
            if (additionalSubtypeString.contains(".")) { // means there are custom layouts
                val subtypeStrings = additionalSubtypeString.split(";")
                val newSubtypeStrings = subtypeStrings.mapNotNull {
                    if ("." !in it) // not a custom subtype, nothing to do
                        return@mapNotNull it
                    val split = it.split(":").toMutableList()
                    if (split.size < 2) return@mapNotNull null // should never happen
                    val oldName = split[1]
                    val newName = oldName.substringBeforeLast(".") + "."
                    if (oldName == newName) return@mapNotNull split.joinToString(":") // should never happen
                    val oldFile = getCustomLayoutFile(oldName, context)
                    val newFile = getCustomLayoutFile(newName, context)
                    if (!oldFile.exists()) return@mapNotNull null // should never happen
                    if (newFile.exists()) newFile.delete() // should never happen
                    oldFile.renameTo(newFile)
                    val enabledSubtypes = prefs.getString(Settings.PREF_ENABLED_SUBTYPES, "")!!
                    if (enabledSubtypes.contains(oldName))
                        prefs.edit { putString(Settings.PREF_ENABLED_SUBTYPES, enabledSubtypes.replace(oldName, newName)) }
                    val selectedSubtype = prefs.getString(Settings.PREF_SELECTED_SUBTYPE, "")!!
                    if (selectedSubtype.contains(oldName))
                        prefs.edit { putString(Settings.PREF_SELECTED_SUBTYPE, selectedSubtype.replace(oldName, newName)) }
                    split[1] = newName
                    split.joinToString(":")
                }
                prefs.edit { putString(Settings.PREF_ADDITIONAL_SUBTYPES, newSubtypeStrings.joinToString(";")) }
            }
            // rename other custom layouts
            LayoutUtilsCustom.onLayoutFileChanged()
            File(DeviceProtectedUtils.getFilesDir(context), "layouts").listFiles()?.forEach {
                val newFile = getCustomLayoutFile(it.name.substringBeforeLast(".") + ".", context)
                if (newFile.name == it.name) return@forEach
                if (newFile.exists()) newFile.delete() // should never happen
                it.renameTo(newFile)
            }
        }
        if (oldVersion <= 2301) {
            // upgrade and remove old color prefs
            fun readAllColorsMap(isNight: Boolean): EnumMap<ColorType, Int> {
                val prefPrefix = if (isNight) "theme_dark_color_" else "theme_color_"
                val colorsString = prefs.getString(prefPrefix + "all_colors", "") ?: ""
                val colorMap = EnumMap<ColorType, Int>(ColorType::class.java)
                colorsString.split(";").forEach {
                    val ct = try {
                        ColorType.valueOf(it.substringBefore(",").uppercase())
                    } catch (_: Exception) {
                        return@forEach
                    }
                    val i = it.substringAfter(",").toIntOrNull() ?: return@forEach
                    colorMap[ct] = i
                }
                return colorMap
            }
            // day colors
            val themeNameDay = context.getString(R.string.theme_name_user)
            val colorsDay = colorPrefsAndResIds.map {
                val pref = "theme_color_" + it.first
                val color = if (prefs.contains(pref)) prefs.getInt(pref, 0) else null
                val result = ColorSetting(it.first, prefs.getBoolean(pref + "_auto", true), color)
                prefs.edit { remove(pref).remove(pref + "_auto") }
                result
            }
            if (colorsDay.any { it.color != null }) {
                KeyboardTheme.writeUserColors(prefs, themeNameDay, colorsDay)
            }
            if (prefs.contains("theme_color_show_more_colors")) {
                val moreColorsDay = prefs.getInt("theme_color_show_more_colors", 0)
                prefs.edit { remove("theme_color_show_more_colors") }
                KeyboardTheme.writeUserMoreColors(prefs, themeNameDay, moreColorsDay)
            }
            if (prefs.contains("theme_color_all_colors")) {
                val allColorsDay = readAllColorsMap(false)
                prefs.edit { remove("theme_color_all_colors") }
                KeyboardTheme.writeUserAllColors(prefs, themeNameDay, allColorsDay)
            }
            if (prefs.getString(Settings.PREF_THEME_COLORS, Defaults.PREF_THEME_COLORS) == "user")
                prefs.edit { putString(Settings.PREF_THEME_COLORS, themeNameDay) }

            // same for night colors
            val themeNameNight = "User-Defined (night)"
            val colorsNight = colorPrefsAndResIds.map {
                val pref = "theme_dark_color_" + it.first
                val color = if (prefs.contains(pref)) prefs.getInt(pref, 0) else null
                val result = ColorSetting(it.first, prefs.getBoolean(pref + "_auto", true), color)
                prefs.edit { remove(pref).remove(pref + "_auto") }
                result
            }
            if (colorsNight.any { it.color!= null }) {
                KeyboardTheme.writeUserColors(prefs, themeNameNight, colorsNight)
            }
            if (prefs.contains("theme_dark_color_show_more_colors")) {
                val moreColorsNight = prefs.getInt("theme_dark_color_show_more_colors", 0)
                prefs.edit { remove("theme_dark_color_show_more_colors") }
                KeyboardTheme.writeUserMoreColors(prefs, themeNameNight, moreColorsNight)
            }
            if (prefs.contains("theme_dark_color_all_colors")) {
                val allColorsNight = readAllColorsMap(true)
                prefs.edit { remove("theme_dark_color_all_colors") }
                KeyboardTheme.writeUserAllColors(prefs, themeNameNight, allColorsNight)
            }
            if (prefs.getString(Settings.PREF_THEME_COLORS_NIGHT, Defaults.PREF_THEME_COLORS_NIGHT) == "user_night")
                prefs.edit { putString(Settings.PREF_THEME_COLORS_NIGHT, themeNameNight) }
        }
        if (oldVersion <= 2302) {
            fun readCustomKeyCodes(setting: String) =
                prefs.getString(setting, "")!!
                    .split(";").filter { it.isNotEmpty()}.associate {
                        val code = runCatching { it.substringAfter(",").toIntOrNull()?.checkAndConvertCode() }.getOrNull()
                        it.substringBefore(",") to code
                    }
            val customCodes = readCustomKeyCodes("toolbar_custom_key_codes")
            val customLongpressCodes = readCustomKeyCodes("toolbar_custom_longpress_codes")
            prefs.edit { remove("toolbar_custom_longpress_codes").remove("toolbar_custom_key_codes") }
            val combined = EnumMap<ToolbarKey, Pair<Int?, Int?>>(ToolbarKey::class.java)
            customCodes.forEach { runCatching {
                val key = ToolbarKey.valueOf(it.key)
                combined[key] = (combined[key] ?: (null to null)).copy(first = it.value)
            } }
            customLongpressCodes.forEach { runCatching {
                val key = ToolbarKey.valueOf(it.key)
                combined[key] = (combined[key] ?: (null to null)).copy(second = it.value)
            } }
            writeCustomKeyCodes(prefs, combined)
        }
        if (oldVersion <= 2303) {
            File(DeviceProtectedUtils.getFilesDir(context), "layouts").listFiles()?.forEach { file ->
                val folder = DeviceProtectedUtils.getFilesDir(context)
                if (file.isDirectory) return@forEach
                when (file.name) {
                    "custom.symbols." -> {
                        val dir = File(folder, LayoutType.SYMBOLS.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("symbols")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.SYMBOLS.name, name) }
                    }
                    "custom.symbols_shifted." -> {
                        val dir = File(folder, LayoutType.MORE_SYMBOLS.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("symbols_shifted")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.MORE_SYMBOLS.name, name) }
                    }
                    "custom.symbols_arabic." -> {
                        val dir = File(folder, LayoutType.SYMBOLS.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("symbols_arabic")}."
                        file.renameTo(File(dir, name))
                    }
                    "custom.numpad." -> {
                        val dir = File(folder, LayoutType.NUMPAD.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("numpad")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.NUMPAD.name, name) }
                    }
                    "custom.numpad_landscape." -> {
                        val dir = File(folder, LayoutType.NUMPAD_LANDSCAPE.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("numpad_landscape")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.NUMPAD_LANDSCAPE.name, name) }
                    }
                    "custom.number." -> {
                        val dir = File(folder, LayoutType.NUMBER.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("number")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.NUMBER.name, name) }
                    }
                    "custom.phone." -> {
                        val dir = File(folder, LayoutType.PHONE.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("phone")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.PHONE.name, name) }
                    }
                    "custom.phone_symbols." -> {
                        val dir = File(folder, LayoutType.PHONE_SYMBOLS.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("phone_symbols")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.PHONE_SYMBOLS.name, name) }
                    }
                    "custom.number_row." -> {
                        val dir = File(folder, LayoutType.NUMBER_ROW.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("number_row")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.NUMBER_ROW.name, name) }
                    }
                    "custom.emoji_bottom_row." -> {
                        val dir = File(folder, LayoutType.EMOJI_BOTTOM.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("emoji_bottom_row")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.EMOJI_BOTTOM.name, name) }
                    }
                    "custom.clip_bottom_row." -> {
                        val dir = File(folder, LayoutType.CLIPBOARD_BOTTOM.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("clip_bottom_row")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.CLIPBOARD_BOTTOM.name, name) }
                    }
                    "custom.functional_keys." -> {
                        val dir = File(folder, LayoutType.FUNCTIONAL.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("functional_keys")}."
                        file.renameTo(File(dir, name))
                        prefs.edit { putString(Settings.PREF_LAYOUT_PREFIX + LayoutType.FUNCTIONAL.name, name) }
                    }
                    "custom.functional_keys_symbols." -> {
                        val dir = File(folder, LayoutType.FUNCTIONAL.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("functional_keys_symbols")}."
                        file.renameTo(File(dir, name))
                    }
                    "custom.functional_keys_symbols_shifted." -> {
                        val dir = File(folder, LayoutType.FUNCTIONAL.folder)
                        dir.mkdirs()
                        val name = "custom.${encodeBase36("functional_keys_symbols_shifted")}."
                        file.renameTo(File(dir, name))
                    }
                    else -> {
                        // main layouts
                        val dir = File(folder, LayoutType.MAIN.folder)
                        dir.mkdirs()
                        file.renameTo(File(dir, file.name))
                    }
                }
            }
            if (prefs.contains(Settings.PREF_ADDITIONAL_SUBTYPES))
                prefs.edit {
                    putString(
                        Settings.PREF_ADDITIONAL_SUBTYPES, prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, "")!!
                            .replace(":", Separators.SET)
                    )
                }
        }
        if (oldVersion <= 2304) {
            // rename layout files for latin scripts, and adjust layouts stored in prefs accordingly
            LayoutUtilsCustom.getLayoutFiles(LayoutType.MAIN, context).forEach {
                val locale = it.name.substringAfter("custom.").substringBefore(".").constructLocale()
                if (locale.script() != SCRIPT_LATIN) return@forEach
                // change language tag to SCRIPT_LATIN, but
                //  avoid overwriting if 2 layouts have a different language tag, but the same name
                val layoutDisplayName = LayoutUtilsCustom.getDisplayName(it.name)
                var newFile = File(it.parentFile!!, LayoutUtilsCustom.getLayoutName(layoutDisplayName, LayoutType.MAIN, locale))
                var i = 1
                while (newFile.exists()) // make sure name is not already in use, e.g. custom.en.abcd. and custom.it.abcd. would both be custom.Latn.abcd
                    newFile = File(it.parentFile!!, LayoutUtilsCustom.getLayoutName(layoutDisplayName + i++, LayoutType.MAIN, locale))
                it.renameTo(newFile)
                // modify prefs
                listOf(Settings.PREF_ENABLED_SUBTYPES, Settings.PREF_SELECTED_SUBTYPE, Settings.PREF_ADDITIONAL_SUBTYPES).forEach { key ->
                    val value = prefs.getString(key, "")!!
                    if (it.name in value)
                        prefs.edit { putString(key, value.replace(it.name, newFile.name)) }
                }
                LayoutUtilsCustom.onLayoutFileChanged()
            }
        }
        if (oldVersion <= 2305) {
            (prefs.all.keys.filter { it.startsWith(Settings.PREF_POPUP_KEYS_ORDER) || it.startsWith(Settings.PREF_POPUP_KEYS_LABELS_ORDER) } +
                listOf(Settings.PREF_TOOLBAR_KEYS, Settings.PREF_PINNED_TOOLBAR_KEYS, Settings.PREF_CLIPBOARD_TOOLBAR_KEYS)).forEach {
                if (!prefs.contains(it)) return@forEach
                val newValue = prefs.getString(it, "")!!.replace(",", Separators.KV).replace(";", Separators.ENTRY)
                prefs.edit { putString(it, newValue) }
            }
            listOf(Settings.PREF_ENABLED_SUBTYPES, Settings.PREF_SELECTED_SUBTYPE, Settings.PREF_ADDITIONAL_SUBTYPES).forEach {
                if (!prefs.contains(it)) return@forEach
                val value = prefs.getString(it, "")!!.replace(":", Separators.SET)
                prefs.edit { putString(it, value) }
            }
            prefs.all.keys.filter { it.startsWith("secondary_locales_") }.forEach {
                val newValue = prefs.getString(it, "")!!.replace(";", Separators.KV)
                prefs.edit { putString(it, newValue) }
            }
        }
        if (oldVersion <= 2306) {
            // upgrade additional, enabled, and selected subtypes to same format of locale and (filtered) extra value
            if (prefs.contains(Settings.PREF_ADDITIONAL_SUBTYPES)) {
                val new = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, "")!!
                    .split(Separators.SETS)
                    .filter { it.isNotEmpty() }
                    .mapNotNull { pref ->
                        val oldSplit = pref.split(Separators.SET)
                        if (oldSplit.size < 3) return@mapNotNull null
                        val languageTag = oldSplit[0]
                        val mainLayoutName = oldSplit[1]
                        val extraValue = oldSplit[2]
                        SettingsSubtype(
                            languageTag.constructLocale(),
                            ExtraValue.KEYBOARD_LAYOUT_SET + "=MAIN" + Separators.KV + mainLayoutName + "," + extraValue
                        ).toAdditionalSubtype().toSettingsSubtype().toPref()
                    }.joinToString(Separators.SETS)
                prefs.edit { putString(Settings.PREF_ADDITIONAL_SUBTYPES, new) }
            }
            listOf(Settings.PREF_ENABLED_SUBTYPES, Settings.PREF_SELECTED_SUBTYPE).forEach { key ->
                if (!prefs.contains(key)) return@forEach
                val resourceSubtypes = getResourceSubtypes(context.resources)
                val additionalSubtypeString = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, Defaults.PREF_ADDITIONAL_SUBTYPES)!!
                val additionalSubtypes = SubtypeUtilsAdditional.createAdditionalSubtypes(additionalSubtypeString)
                val new = prefs.getString(key, "")!!.split(Separators.SETS).filter { it.isNotEmpty() }.joinToString(Separators.SETS) { pref ->
                    val oldSplit = pref.split(Separators.SET)
                    val languageTag = oldSplit[0]
                    if (oldSplit.size == 1)
                        return@joinToString resourceSubtypes.first { it.locale().language == languageTag.constructLocale().language }
                            .toSettingsSubtype().toPref()
                    val mainLayoutName = oldSplit[1]
                    // we now need more information than just locale and main layout name, get it from existing subtypes
                    val filtered = additionalSubtypes.filter {
                        it.locale().toLanguageTag() == languageTag && (it.mainLayoutNameOrQwerty()) == mainLayoutName
                    }
                    if (filtered.isNotEmpty())
                        return@joinToString filtered.first().toSettingsSubtype().toPref()
                    // find best matching resource subtype
                    val goodMatch = resourceSubtypes.filter {
                        it.locale().toLanguageTag() == languageTag && (it.mainLayoutNameOrQwerty()) == mainLayoutName
                    }
                    if (goodMatch.isNotEmpty())
                        return@joinToString goodMatch.first().toSettingsSubtype().toPref()
                    // not sure how we can get here, but better deal with it
                    val okMatch = resourceSubtypes.filter {
                        it.locale().language == languageTag.constructLocale().language && (it.mainLayoutNameOrQwerty()) == mainLayoutName
                    }
                    if (okMatch.isNotEmpty())
                        okMatch.first().toSettingsSubtype().toPref()
                    else resourceSubtypes.first { it.locale().language == languageTag.constructLocale().language }
                        .toSettingsSubtype().toPref()
                }
                prefs.edit { putString(key, new) }
            }
        }
        if (oldVersion <= 2307) {
            prefs.all.keys.forEach {
                if (!it.startsWith(Settings.PREF_POPUP_KEYS_ORDER) && !it.startsWith(Settings.PREF_POPUP_KEYS_LABELS_ORDER))
                    return@forEach
                prefs.edit { putString(it, prefs.getString(it, "")!!.replace("popup_keys_", "")) }
            }
        }
        if (oldVersion <= 2308) {
            SubtypeSettings.reloadEnabledSubtypes(context)
            prefs.all.keys.toList().forEach { key ->
                if (key.startsWith(Settings.PREF_POPUP_KEYS_ORDER+"_")) {
                    val locale = key.substringAfter(Settings.PREF_POPUP_KEYS_ORDER+"_").constructLocale()
                    SubtypeSettings.getEnabledSubtypes().forEach {
                        if (it.locale() == locale && !SubtypeSettings.isAdditionalSubtype(it)) {
                            SubtypeUtilsAdditional.changeAdditionalSubtype(it.toSettingsSubtype(), it.toSettingsSubtype(), context)
                        }
                    }
                    val additional = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, "")!!
                    additional.split(Separators.SETS).filter { it.isNotEmpty() }.forEach inner@{
                        val subtype = it.toSettingsSubtype()
                        if (subtype.locale != locale) return@inner
                        val newSubtype = subtype.with(ExtraValue.POPUP_ORDER, prefs.getString(key, ""))
                        SubtypeUtilsAdditional.changeAdditionalSubtype(subtype, newSubtype, context)
                    }
                    prefs.edit { remove(key) }
                }
                if (key.startsWith(Settings.PREF_POPUP_KEYS_LABELS_ORDER+"_")) {
                    val locale = key.substringAfter(Settings.PREF_POPUP_KEYS_LABELS_ORDER+"_").constructLocale()
                    SubtypeSettings.getEnabledSubtypes().forEach {
                        if (it.locale() == locale && !SubtypeSettings.isAdditionalSubtype(it)) {
                            SubtypeUtilsAdditional.changeAdditionalSubtype(it.toSettingsSubtype(), it.toSettingsSubtype(), context)
                        }
                    }
                    val additional = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, "")!!
                    additional.split(Separators.SETS).filter { it.isNotEmpty() }.forEach inner@{
                        val subtype = it.toSettingsSubtype()
                        if (subtype.locale != locale) return@inner
                        val newSubtype = subtype.with(ExtraValue.HINT_ORDER, prefs.getString(key, ""))
                        SubtypeUtilsAdditional.changeAdditionalSubtype(subtype, newSubtype, context)
                    }
                    prefs.edit { remove(key) }
                }
                if (key.startsWith("secondary_locales_")) {
                    val locale = key.substringAfter("secondary_locales_").constructLocale()
                    SubtypeSettings.getEnabledSubtypes().forEach {
                        if (it.locale() == locale && !SubtypeSettings.isAdditionalSubtype(it)) {
                            SubtypeUtilsAdditional.changeAdditionalSubtype(it.toSettingsSubtype(), it.toSettingsSubtype(), context)
                        }
                    }
                    val additional = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, "")!!
                    val secondaryLocales = prefs.getString(key, "")!!.split(Separators.KV).filter { it.isNotBlank() }.joinToString(Separators.KV)
                    additional.split(Separators.SETS).filter { it.isNotEmpty() }.forEach inner@{
                        val subtype = it.toSettingsSubtype()
                        if (subtype.locale != locale) return@inner
                        val newSubtype = subtype.with(ExtraValue.SECONDARY_LOCALES, secondaryLocales)
                        SubtypeUtilsAdditional.changeAdditionalSubtype(subtype, newSubtype, context)
                    }
                    prefs.edit { remove(key) }
                }
            }
        }
        if (oldVersion <= 2309) {
            if (prefs.contains("auto_correction_confidence")) {
                val value = when (prefs.getString("auto_correction_confidence", "0")) {
                    "1" -> 0.067f
                    "2" -> -1f
                    else -> 0.185f
                }
                prefs.edit { remove("auto_correction_confidence").putFloat(Settings.PREF_AUTO_CORRECT_THRESHOLD, value) }
            }
        }
        if (oldVersion <= 2310) {
            listOf(
                Settings.PREF_ENABLED_SUBTYPES,
                Settings.PREF_SELECTED_SUBTYPE,
                Settings.PREF_ADDITIONAL_SUBTYPES
            ).forEach { key ->
                val value = prefs.getString(key, "")!!
                if ("bengali," in value) {
                    prefs.edit { putString(key, value.replace("bengali,", "bengali_inscript,")) }
                }
            }
        }
        if (oldVersion <= 3001 && prefs.getInt(Settings.PREF_CLIPBOARD_HISTORY_RETENTION_TIME, Defaults.PREF_CLIPBOARD_HISTORY_RETENTION_TIME) <= 0) {
            prefs.edit { putInt(Settings.PREF_CLIPBOARD_HISTORY_RETENTION_TIME, 121) }
        }
        if (oldVersion <= 3002) {
            prefs.all.filterKeys { it.startsWith(Settings.PREF_USER_ALL_COLORS_PREFIX) }.forEach {
                val oldValue = prefs.getString(it.key, "")!!
                if ("KEY_PREVIEW" !in oldValue) return@forEach
                val newValue = oldValue.replace("KEY_PREVIEW", "KEY_PREVIEW_BACKGROUND")
                prefs.edit { putString(it.key, newValue) }
            }
        }
        if (oldVersion <= 3101) {
            @SuppressLint("UseKtx") val e = prefs.edit()
            prefs.all.toMap().forEach { (key, value) ->
                if (key == "side_padding_scale") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_SIDE_PADDING_SCALE_PREFIX, 0, 2), value as Float)
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_SIDE_PADDING_SCALE_PREFIX, 2, 2), value)
                } else if (key == "side_padding_scale_landscape") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_SIDE_PADDING_SCALE_PREFIX, 1, 2), value as Float)
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_SIDE_PADDING_SCALE_PREFIX, 3, 2), value)
                } else if (key == "bottom_padding_scale") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_BOTTOM_PADDING_SCALE_PREFIX, 0, 1), value as Float)
                } else if (key == "bottom_padding_scale_landscape") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_BOTTOM_PADDING_SCALE_PREFIX, 1, 1), value as Float)
                } else if (key == "split_spacer_scale") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_SPLIT_SPACER_SCALE_PREFIX, 0, 1), value as Float)
                } else if (key == "split_spacer_scale_landscape") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_SPLIT_SPACER_SCALE_PREFIX, 1, 1), value as Float)
                } else if (key == "one_handed_mode_enabled_p_true") {
                    e.putBoolean(createPrefKeyForBooleanSettings(Settings.PREF_ONE_HANDED_MODE_PREFIX, 0, 2), value as Boolean)
                } else if (key == "one_handed_mode_enabled_p_false") {
                    e.putBoolean(createPrefKeyForBooleanSettings(Settings.PREF_ONE_HANDED_MODE_PREFIX, 1, 2), value as Boolean)
                } else if (key == "one_handed_mode_scale_p_true") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_ONE_HANDED_SCALE_PREFIX, 0, 2), value as Float)
                } else if (key == "one_handed_mode_scale_p_false") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_ONE_HANDED_SCALE_PREFIX, 1, 2), value as Float)
                } else if (key == "one_handed_mode_gravity_p_true") {
                    e.putInt(createPrefKeyForBooleanSettings(Settings.PREF_ONE_HANDED_GRAVITY_PREFIX, 0, 2), value as Int)
                } else if (key == "one_handed_mode_gravity_p_false") {
                    e.putInt(createPrefKeyForBooleanSettings(Settings.PREF_ONE_HANDED_GRAVITY_PREFIX, 1, 2), value as Int)
                } else if (key == "keyboard_height_scale") {
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_KEYBOARD_HEIGHT_SCALE_PREFIX, 1, 1), value as Float)
                    e.putFloat(createPrefKeyForBooleanSettings(Settings.PREF_KEYBOARD_HEIGHT_SCALE_PREFIX, 1, 1), value)
                } else {
                    if (key == Settings.PREF_ADDITIONAL_SUBTYPES || key == Settings.PREF_ENABLED_SUBTYPES) {
                        val subtypes = prefs.getString(key, "")!!.split(Separators.SETS).filter { it.isNotEmpty() }.map {
                            val st = it.toSettingsSubtype()
                            if (st.locale.language == "ko") st.with(ExtraValue.COMBINING_RULES, "hangul")
                            else st
                        }
                        e.putString(key, subtypes.joinToString(Separators.SETS) { it.toPref() })
                    } else if (key == Settings.PREF_SELECTED_SUBTYPE) {
                        val subtype = prefs.getString(key, "")!!.toSettingsSubtype()
                        if (subtype.locale.language == "ko")
                            e.putString(key, subtype.with(ExtraValue.COMBINING_RULES, "hangul").toPref())
                    }
                    return@forEach
                }
                e.remove(key)
            }
            e.apply()
        }
        if (oldVersion <= 3201) {
            prefs.edit {
                putBoolean(Settings.PREF_SUGGEST_PUNCTUATION,
                    !prefs.getBoolean(Settings.PREF_BIGRAM_PREDICTIONS, Defaults.PREF_BIGRAM_PREDICTIONS))
            }
        }
        if (prefs.contains(Settings.PREF_VIBRATE_ON)) {
            prefs.edit {
                if (prefs.getBoolean(Settings.PREF_VIBRATE_ON, false))
                    if (prefs.getInt(Settings.PREF_VIBRATION_DURATION_SETTINGS, Defaults.PREF_VIBRATION_DURATION_SETTINGS) == -1) {
                        putString(Settings.PREF_VIBRATION_TYPE, AudioAndHapticFeedbackManager.VibrationType.SYSTEM.toString())
                        putInt(Settings.PREF_VIBRATION_DURATION_SETTINGS, Defaults.PREF_VIBRATION_DURATION_SETTINGS)
                    }
                    else
                        putString(Settings.PREF_VIBRATION_TYPE, AudioAndHapticFeedbackManager.VibrationType.CUSTOM.toString())
                else putString(Settings.PREF_VIBRATION_TYPE, AudioAndHapticFeedbackManager.VibrationType.OFF.toString())
                remove(Settings.PREF_VIBRATE_ON)
            }
        }
        upgradeToolbarPrefs(prefs)
        LayoutUtilsCustom.onLayoutFileChanged() // just to be sure
        prefs.edit { putInt(Settings.PREF_VERSION_CODE, BuildConfig.VERSION_CODE) }
    }

    // not only on upgrade, because this might also be called when db is locked
    fun transferOldPinnedClips(context: Context) {
        @Serializable
        data class OldClipboardHistoryEntry (
            var timeStamp: Long,
            val content: String,
            var isPinned: Boolean = false
        )
        if (isUserLocked(context) || isDeviceLocked(context)) return
        try {
            val pinnedClipString = context.protectedPrefs().getString("pinned_clips", "")
            if (pinnedClipString.isNullOrBlank())
                return
            val pinnedClips: List<OldClipboardHistoryEntry> = Json.decodeFromString(pinnedClipString)
            val dao = ClipboardDao.getInstance(context) ?: return
            pinnedClips.forEach { dao.addClip(it.timeStamp, it.isPinned, it.content) }
            context.protectedPrefs().edit { remove("pinned_clips") }
        } catch (e: Throwable) {
            Log.e("upgrade", "error transferring old pinned clips", e)
        }
    }

    // old variant for old folder structure
    private fun getCustomLayoutFile(layoutName: String, context: Context): File =
        File(File(DeviceProtectedUtils.getFilesDir(context), "layouts"), layoutName)
}
