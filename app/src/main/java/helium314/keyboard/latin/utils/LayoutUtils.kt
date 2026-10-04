package helium314.keyboard.latin.utils

import android.content.Context
import helium314.keyboard.keyboard.internal.keyboard_parser.LocaleKeyboardInfos
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.SimplePopups
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.Defaults.default
import helium314.keyboard.latin.utils.LayoutType.Companion.folder
import helium314.keyboard.latin.utils.ScriptUtils.script
import java.io.File
import java.util.Locale

// for layouts provided by the app
object LayoutUtils {
    fun getAvailableLayouts(layoutType: LayoutType, context: Context, locale: Locale? = null): Collection<String> {
        if (layoutType != LayoutType.MAIN)
            return context.assets.list(layoutType.folder)?.map { it.substringBefore(".") }.orEmpty()
        if (locale == null)
            return SubtypeSettings.getAllAvailableSubtypes()
                .mapTo(HashSet()) { it.mainLayoutNameOrQwerty().substringBefore("+") }
                .apply { addAll(context.resources.getStringArray(R.array.predefined_layouts)) }
        val layouts = SubtypeSettings.getResourceSubtypesForLocale(locale).mapNotNullTo(mutableSetOf()) { it.mainLayoutName() }
        if (locale.script() == ScriptUtils.SCRIPT_LATIN)
            layouts.addAll(context.resources.getStringArray(R.array.predefined_layouts))
        return layouts
    }

    fun getLMainLayoutsForLocales(locales: List<Locale>, context: Context): Collection<String> =
        locales.flatMapTo(HashSet()) { getAvailableLayouts(LayoutType.MAIN, context, it) }.sorted()

    /** gets content for built-in (non-custom) layout [layoutName], with fallback to qwerty */
    fun getContent(layoutType: LayoutType, layoutName: String, context: Context): String {
        val layouts = context.assets.list(layoutType.folder)!!
        layouts.firstOrNull { it.startsWith("$layoutName.") }
            ?.let { return context.assets.open(layoutType.folder + File.separator + it).reader().readText() }
        val fallback = layouts.first { it.startsWith(layoutType.default) } // must exist!
        return context.assets.open(layoutType.folder + File.separator + fallback).reader().readText()
    }

    fun getContentWithPlus(mainLayoutName: String, locale: Locale, context: Context): String {
        val content = getContent(LayoutType.MAIN, mainLayoutName, context)
        if (!mainLayoutName.endsWith("+"))
            return content
        // the stuff below will not work if we add "+" layouts in json format
        // ideally we should serialize keyData to json to solve this
        val rows = getSimpleRowStrings(content)
        val localeKeyboardInfos = LocaleKeyboardInfos.getOrCreate(context, locale)
        return rows.mapIndexed { i, row ->
            val extraKeys = localeKeyboardInfos.getExtraKeys(i + 1) ?: return@mapIndexed row
            val rowList = row.split("\n").filterNot { it.isEmpty() }.toMutableList()
            extraKeys.forEach { key ->
                val popups = (key.popup as? SimplePopups)?.popupKeys?.joinToString(" ")
                    ?.takeIf { it.isNotEmpty() }?.let { " $it" } ?: ""
                rowList.add(key.label + popups)
            }
            rowList.joinToString("\n")
        }.joinToString("\n\n")
    }

    fun getSimpleRowStrings(layoutContent: String): List<String> =
        layoutContent.replace("\r\n", "\n").split("\\n\\s*\\n".toRegex()).filter { it.isNotBlank() }
}
