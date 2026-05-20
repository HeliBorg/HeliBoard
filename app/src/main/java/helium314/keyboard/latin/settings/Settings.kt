// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.settings

import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.provider.Settings.Global
import androidx.core.content.edit
import helium314.keyboard.latin.utils.prefs
import kotlinx.serialization.json.Json

fun customIconNames(prefs: SharedPreferences) = runCatching {
    Json.decodeFromString<Map<String, String>>(prefs.getString(Settings.PREF_CUSTOM_ICON_NAMES, Defaults.PREF_CUSTOM_ICON_NAMES)!!)
}.getOrElse { emptyMap() }

@SuppressLint("DiscouragedApi")
fun customIconIds(context: Context, prefs: SharedPreferences) = customIconNames(prefs)
    .mapNotNull { entry ->
        val id = runCatching { context.resources.getIdentifier(entry.value, "drawable", context.packageName) }.getOrNull()
        id?.let { entry.key to it }
    }

/** Derive an index from a number of boolean [settingValues], used to access the matching default value in a defaults arraY */
fun findIndexOfDefaultSetting(vararg settingValues: Boolean): Int {
    var i = -1
    return settingValues.sumOf { i++; if (it) 1.shl(i) else 0 }
}

/** Create pref key that is derived from a [number] of boolean conditions. The [index] is as created by [findIndexOfDefaultSetting]. */
fun createPrefKeyForBooleanSettings(prefix: String, index: Int, number: Int): String =
    "${prefix}_${Array(number) { index.shr(it) % 2 == 1 }.joinToString("_")}"

fun getTransitionAnimationScale(context: Context) =
    Global.getFloat(context.contentResolver, Global.TRANSITION_ANIMATION_SCALE, 1f)

fun isFloatingKeyboardEnabled(context: Context) =
    context.prefs().getBoolean(Settings.PREF_FLOATING_ENABLED_PREFIX + context.resources.displayMetrics.widthPixels, false)

fun setFloatingKeyboardEnabled(context: Context, enabled: Boolean) =
    context.prefs().edit { putBoolean(Settings.PREF_FLOATING_ENABLED_PREFIX + context.resources.displayMetrics.widthPixels, enabled) }

fun readFloatingHeight(context: Context): Int {
    val screenWidth = context.resources.displayMetrics.widthPixels
    val key = Settings.PREF_FLOATING_HEIGHT_PREFIX + screenWidth
    return context.prefs().getInt(key, context.resources.displayMetrics.heightPixels / 3)
}

fun readFloatingWidth(context: Context): Int {
    val screenWidth = context.resources.displayMetrics.widthPixels
    val key = Settings.PREF_FLOATING_WIDTH_PREFIX + screenWidth
    return context.prefs().getInt(key, screenWidth / 2)
}

fun setFloatingSize(context: Context, width: Int, height: Int) {
    val screenWidth = context.resources.displayMetrics.widthPixels
    context.prefs().edit {
        putInt(Settings.PREF_FLOATING_WIDTH_PREFIX + screenWidth, width)
        putInt(Settings.PREF_FLOATING_HEIGHT_PREFIX + screenWidth, height)
    }
}
