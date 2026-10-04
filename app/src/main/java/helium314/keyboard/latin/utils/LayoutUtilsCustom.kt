// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Context
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.KeyboardElement
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.keyboard_parser.LayoutParser
import helium314.keyboard.keyboard.internal.keyboard_parser.LocaleKeyboardInfos
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue.KEYBOARD_LAYOUT_SET
import helium314.keyboard.latin.common.decodeBase36
import helium314.keyboard.latin.common.encodeBase36
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsSubtype.Companion.toSettingsSubtype
import helium314.keyboard.latin.utils.LayoutType.Companion.folder
import helium314.keyboard.latin.utils.ScriptUtils.script
import kotlinx.serialization.SerializationException
import java.io.File
import java.util.EnumMap
import java.util.Locale

object LayoutUtilsCustom {

    fun checkLayout(layoutContent: String, context: Context): Boolean {
        if (Settings.getValues() == null)
            Settings.getInstance().loadSettings(context)
        val params = KeyboardParams()
        params.mId = KeyboardLayoutSet.getFakeKeyboardId(KeyboardElement.ALPHABET)
        params.mPopupKeyOrder.add(POPUP_KEYS_LAYOUT)
        LocaleKeyboardInfos.addLocaleKeyTextsToParams(context, params, LocaleKeyboardInfos.POPUP_KEYS_NORMAL)
        try {
            if (layoutContent.trimStart().startsWith("[") || layoutContent.trimStart().startsWith("//")) {
                val keys = LayoutParser.parseJsonString(layoutContent).map { row -> row.mapNotNull { it.compute(params)?.toKeyParams(params) } }
                return checkKeys(keys)
            }
        } catch (e: SerializationException) {
            Log.w(TAG, "json parsing error", e)
            if (layoutContent.trimEnd().endsWith("]") && layoutContent.contains("},"))
                return false // we're sure enough it's a json
        } catch (e: Exception) {
            Log.w(TAG, "json layout parsed, but considered invalid", e)
            return false
        }
        try {
            val keys = LayoutParser.parseSimpleString(layoutContent).map { row -> row.map { it.toKeyParams(params) } }
            return checkKeys(keys)
        } catch (e: Exception) { Log.w(TAG, "error parsing custom simple layout", e) }
        if (layoutContent.trimStart().startsWith("[") && layoutContent.trimEnd().endsWith("]")) {
            // layout can't be loaded, assume it's json -> load json layout again because the error message shown to the user is from the most recent error
            try {
                LayoutParser.parseJsonString(layoutContent).map { row -> row.mapNotNull { it.compute(params)?.toKeyParams(params) } }
            } catch (e: Exception) { Log.w(TAG, "json parsing error", e) }
        }
        return false
    }

    fun checkKeys(keys: List<List<Key.KeyParams>>): Boolean {
        if (keys.isEmpty() || keys.any { it.isEmpty() }) {
            Log.w(TAG, "empty rows")
            return false
        }
        if (keys.size > 8) {
            Log.w(TAG, "too many rows")
            return false
        }
        if (keys.any { row -> row.size > 20 }) {
            Log.w(TAG, "too many keys in one row")
            return false
        }
        if (keys.any { row -> row.any {
                if ((it.mLabel?.length ?: 0) > 20) {
                    Log.w(TAG, "too long text on key: ${it.mLabel}")
                    true
                } else false
            } }) {
            return false
        }
        if (keys.any { row -> row.any {
                if ((it.mPopupKeys?.size ?: 0) > 20) {
                    Log.w(TAG, "too many popup keys on key ${it.mLabel}")
                    true
                } else false
            } }) {
            return false
        }
        if (keys.any { row -> row.any { true == it.mPopupKeys?.any { popupKey ->
                if ((popupKey.mLabel?.length ?: 0) > 10) {
                    Log.w(TAG, "too long text on popup key: ${popupKey.mLabel}")
                    true
                } else false
            } } }) {
            return false
        }
        return true
    }

    fun getLayoutFiles(layoutType: LayoutType, context: Context, locale: Locale? = null): List<File> {
        val layouts = customLayoutMap.getOrPut(layoutType) {
            File(DeviceProtectedUtils.getFilesDir(context), layoutType.folder).listFiles()?.toList() ?: emptyList()
        }
        if (layoutType != LayoutType.MAIN || locale == null)
            return layouts
        if (locale.script() == ScriptUtils.SCRIPT_LATIN)
            return layouts.filter { it.name.startsWith(CUSTOM_LAYOUT_PREFIX + ScriptUtils.SCRIPT_LATIN + ".") }
        return layouts.filter { it.name.startsWith(CUSTOM_LAYOUT_PREFIX + locale.toLanguageTag() + ".") }
    }

    fun onLayoutFileChanged() {
        customLayoutMap.clear()
    }

    fun deleteLayout(layoutName: String, layoutType: LayoutType, context: Context) {
        getLayoutFile(layoutName, layoutType, context).delete()
        onLayoutFileChanged()
        SubtypeSettings.onRenameLayout(layoutType, layoutName, null, context)
        KeyboardSwitcher.getInstance().setThemeNeedsReload()
    }

    fun getDisplayName(layoutName: String) =
        try {
            if (layoutName.count { it == '.' } == 3) // main layout: "custom.<locale or script>.<name>.", other: custom.<name>.
                decodeBase36(layoutName.substringAfter(CUSTOM_LAYOUT_PREFIX).substringAfter(".").substringBeforeLast("."))
            else decodeBase36(layoutName.substringAfter(CUSTOM_LAYOUT_PREFIX).substringBeforeLast("."))
        } catch (_: NumberFormatException) {
            layoutName
        }

    /** @return layoutName for given [displayName]. If [layoutType ]is MAIN, non-null [locale] must be supplied */
    fun getLayoutName(displayName: String, layoutType: LayoutType, locale: Locale? = null): String {
        if (layoutType != LayoutType.MAIN)
            return CUSTOM_LAYOUT_PREFIX + encodeBase36(displayName) + "."
        if (locale == null) throw IllegalArgumentException("locale for main layout not specified")
        return if (locale.script() == ScriptUtils.SCRIPT_LATIN)
            CUSTOM_LAYOUT_PREFIX + ScriptUtils.SCRIPT_LATIN + "." + encodeBase36(displayName) + "."
        else CUSTOM_LAYOUT_PREFIX + locale.toLanguageTag() + "." + encodeBase36(displayName) + "."
    }

    fun isCustomLayout(layoutName: String) = layoutName.startsWith(CUSTOM_LAYOUT_PREFIX)

    fun getLayoutFile(layoutName: String, layoutType: LayoutType, context: Context): File {
        val file = File(DeviceProtectedUtils.getFilesDir(context), layoutType.folder + File.separator + layoutName)
        file.parentFile?.mkdirs()
        return file
    }

    // remove layouts without a layout file from custom subtypes and settings
    // should not be necessary, but better fall back to default instead of crashing when encountering a bug
    fun removeMissingLayouts(context: Context) {
        val prefs = context.prefs()
        fun remove(type: LayoutType, name: String) {
            Log.w(TAG, "removing custom layout ${getDisplayName(name)} / $name without file")
            SubtypeSettings.onRenameLayout(type, name, null, context)
        }
        LayoutType.entries.forEach { type ->
            val name = Settings.readDefaultLayoutName(type, prefs)
            if (!isCustomLayout(name) || getLayoutFiles(type, context).any { it.name.startsWith(name) })
                return@forEach
            remove(type, name)
        }
        prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, Defaults.PREF_ADDITIONAL_SUBTYPES)!!
            .split(Separators.SETS).forEach outer@{
                val subtype = it.toSettingsSubtype()
                LayoutType.getLayoutMap(subtype.getExtraValueOf(KEYBOARD_LAYOUT_SET) ?: "").forEach { (type, name) ->
                    if (!isCustomLayout(name) || getLayoutFiles(type, context).any { it.name.startsWith(name) })
                        return@forEach
                    remove(type, name)
                    // recursive call: additional subtypes must have changed, so we repeat until nothing needs to be deleted
                    removeMissingLayouts(context)
                    return
                }
            }
    }

    // this goes into prefs and file names, so do not change!
    const val CUSTOM_LAYOUT_PREFIX = "custom."
    private const val TAG = "LayoutUtilsCustom"
    private val customLayoutMap = EnumMap<LayoutType, List<File>>(LayoutType::class.java)
}
