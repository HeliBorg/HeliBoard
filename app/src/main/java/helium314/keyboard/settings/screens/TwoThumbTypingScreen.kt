// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.Context
import androidx.compose.material3.Text
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.content.edit
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.JniUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.SettingsWithoutKey
import helium314.keyboard.settings.Theme
import helium314.keyboard.settings.dialogs.ListPickerDialog
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.preferences.ListPreference
import helium314.keyboard.settings.preferences.Preference
import helium314.keyboard.settings.preferences.SliderPreference
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.settings.previewDark

@Composable
fun TwoThumbTypingScreen(
    onClickBack: () -> Unit,
) {
    val prefs = LocalContext.current.prefs()
    // Same recomposition-trigger trick as the other screens — pref changes update via
    // SettingsActivity.prefChanged so conditional-visibility groups re-render.
    val b = (LocalContext.current.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")

    val hasGestureLib = JniUtils.sHaveGestureLib
    val gestureEnabled = hasGestureLib && prefs.getBoolean(Settings.PREF_GESTURE_INPUT, Defaults.PREF_GESTURE_INPUT)
    val spacingMode = currentSpacingMode(prefs)
    val autospaceMode = spacingMode == SPACING_MODE_AUTOSPACE
    val nonNormalSpacing = spacingMode != SPACING_MODE_NORMAL
    val dualThumbHinting = prefs.getBoolean(Settings.PREF_GESTURE_DUAL_THUMB_HINTING, Defaults.PREF_GESTURE_DUAL_THUMB_HINTING)

    val items = buildList {
        if (!gestureEnabled) {
            add(R.string.two_thumb_typing_requires_gesture)
            return@buildList
        }

        add(R.string.settings_category_two_thumb_typing_words)
        add(SettingsWithoutKey.TWO_THUMB_SPACING_MODE)
        if (autospaceMode) {
            add(Settings.PREF_COMBINING_GRACE_MS)
            add(Settings.PREF_COMBINING_TAP_EXTRA_MS)
            add(Settings.PREF_COMBINING_AUTOCORRECT_ON_AUTOSPACE)
            add(Settings.PREF_COMBINING_AUTOSPACE_SUGGESTIONS)
        }
        if (nonNormalSpacing) {
            add(SettingsWithoutKey.TWO_THUMB_BACKSPACE_BEHAVIOR)
        }

        add(R.string.settings_category_two_thumb_typing_recognition)
        add(Settings.PREF_GESTURE_DUAL_THUMB_HINTING)
        if (dualThumbHinting) {
            add(Settings.PREF_GESTURE_DUAL_THUMB_MIDLINE_PCT)
        }

        add(R.string.settings_category_two_thumb_typing_troubleshooting)
        add(Settings.PREF_GESTURE_DEBUG_DRAW_POINTS)
    }

    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.settings_screen_two_thumb_typing),
        settings = items,
    )
}

fun createTwoThumbTypingSettings(context: Context) = listOf(
    Setting(context, SettingsWithoutKey.TWO_THUMB_SPACING_MODE,
        R.string.two_thumb_spacing_mode, R.string.two_thumb_spacing_mode_summary) {
        TwoThumbSpacingModePreference(it)
    },
    Setting(context, Settings.PREF_COMBINING_GRACE_MS,
        R.string.two_thumb_autospace_duration, R.string.two_thumb_autospace_duration_summary) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_COMBINING_GRACE_MS,
            range = 0f..1000f,
            description = {
                if (it <= 0) stringResource(R.string.gesture_autospace_grace_off)
                else stringResource(R.string.abbreviation_unit_milliseconds, it.toString())
            }
        )
    },
    Setting(context, Settings.PREF_COMBINING_AUTOCORRECT_ON_AUTOSPACE,
        R.string.combining_autocorrect_on_autospace, R.string.combining_autocorrect_on_autospace_summary) {
        SwitchPreference(it, Defaults.PREF_COMBINING_AUTOCORRECT_ON_AUTOSPACE)
    },
    Setting(context, Settings.PREF_COMBINING_AUTOSPACE_SUGGESTIONS,
        R.string.combining_autospace_suggestions, R.string.combining_autospace_suggestions_summary) { def ->
        val items = listOf(
            stringResource(R.string.combining_autospace_suggestions_next) to "next_word",
            stringResource(R.string.combining_autospace_suggestions_keep) to "keep_alternatives",
            stringResource(R.string.combining_autospace_suggestions_keep_then_next) to "alternatives_then_next_word",
        )
        ListPreference(def, items, Defaults.PREF_COMBINING_AUTOSPACE_SUGGESTIONS)
    },
    Setting(context, Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD,
        R.string.combining_backspace_deletes_gesture_word,
        R.string.combining_backspace_deletes_gesture_word_summary) {
        SwitchPreference(it, Defaults.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD)
    },
    Setting(context, Settings.PREF_COMBINING_TAP_EXTRA_MS,
        R.string.two_thumb_tap_autospace_grace, R.string.two_thumb_tap_autospace_grace_summary) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_COMBINING_TAP_EXTRA_MS,
            range = 0f..1000f,
            description = {
                if (it <= 0) stringResource(R.string.gesture_autospace_grace_off)
                else stringResource(R.string.abbreviation_unit_milliseconds, it.toString())
            }
        )
    },
    Setting(context, SettingsWithoutKey.TWO_THUMB_BACKSPACE_BEHAVIOR,
        R.string.two_thumb_backspace_behavior, R.string.two_thumb_backspace_behavior_summary) {
        TwoThumbBackspaceBehaviorPreference(it)
    },
    Setting(context, Settings.PREF_GESTURE_DUAL_THUMB_HINTING,
        R.string.two_thumb_point_hinting, R.string.two_thumb_point_hinting_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_DUAL_THUMB_HINTING)
    },
    Setting(context, Settings.PREF_GESTURE_DUAL_THUMB_MIDLINE_PCT, R.string.gesture_dual_thumb_midline) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_GESTURE_DUAL_THUMB_MIDLINE_PCT,
            range = 30f..70f,
            description = { value -> "${value.toInt()}%" }
        )
    },
    Setting(context, Settings.PREF_GESTURE_DEBUG_DRAW_POINTS,
        R.string.gesture_debug_draw_points, R.string.gesture_debug_draw_points_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_DEBUG_DRAW_POINTS)
    },
)

private const val SPACING_MODE_NORMAL = "normal"
private const val SPACING_MODE_MANUAL = "manual"
private const val SPACING_MODE_AUTOSPACE = "autospace"
private const val DEFAULT_AUTOSPACE_GRACE_MS = 500

private const val BACKSPACE_NORMAL = "normal"
private const val BACKSPACE_FRAGMENT = "fragment"
private const val BACKSPACE_WORD = "word"

private fun currentSpacingMode(prefs: android.content.SharedPreferences): String = when {
    prefs.getBoolean(Settings.PREF_GESTURE_MANUAL_SPACING, Defaults.PREF_GESTURE_MANUAL_SPACING) -> SPACING_MODE_MANUAL
    prefs.getInt(Settings.PREF_COMBINING_GRACE_MS, Defaults.PREF_COMBINING_GRACE_MS) > 0 -> SPACING_MODE_AUTOSPACE
    else -> SPACING_MODE_NORMAL
}

@Composable
private fun TwoThumbSpacingModePreference(setting: Setting) {
    val prefs = LocalContext.current.prefs()
    var showDialog by rememberSaveable { mutableStateOf(false) }
    val items = listOf(
        stringResource(R.string.two_thumb_spacing_mode_normal) to SPACING_MODE_NORMAL,
        stringResource(R.string.two_thumb_spacing_mode_manual) to SPACING_MODE_MANUAL,
        stringResource(R.string.two_thumb_spacing_mode_autospace) to SPACING_MODE_AUTOSPACE,
    )
    val selected = currentSpacingMode(prefs)
    Preference(
        name = setting.title,
        description = items.first { it.second == selected }.first,
        onClick = { showDialog = true },
    )
    if (showDialog) {
        ListPickerDialog(
            onDismissRequest = { showDialog = false },
            items = items,
            onItemSelected = { item ->
                val mode = item.second
                prefs.edit {
                    putBoolean(Settings.PREF_GESTURE_MANUAL_SPACING, mode == SPACING_MODE_MANUAL)
                    when (mode) {
                        SPACING_MODE_NORMAL, SPACING_MODE_MANUAL -> putInt(Settings.PREF_COMBINING_GRACE_MS, 0)
                        SPACING_MODE_AUTOSPACE -> if (prefs.getInt(Settings.PREF_COMBINING_GRACE_MS, 0) <= 0) {
                            putInt(Settings.PREF_COMBINING_GRACE_MS, DEFAULT_AUTOSPACE_GRACE_MS)
                        }
                    }
                }
            },
            selectedItem = items.first { it.second == selected },
            title = { Text(setting.title) },
            getItemName = { it.first },
        )
    }
}

private fun currentBackspaceBehavior(prefs: android.content.SharedPreferences): String = when {
    prefs.getBoolean(Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD,
        Defaults.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD) -> BACKSPACE_WORD
    prefs.getBoolean(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE,
        Defaults.PREF_GESTURE_FRAGMENT_BACKSPACE) -> BACKSPACE_FRAGMENT
    else -> BACKSPACE_NORMAL
}

@Composable
private fun TwoThumbBackspaceBehaviorPreference(setting: Setting) {
    val prefs = LocalContext.current.prefs()
    var showDialog by rememberSaveable { mutableStateOf(false) }
    val items = listOf(
        stringResource(R.string.two_thumb_backspace_normal) to BACKSPACE_NORMAL,
        stringResource(R.string.two_thumb_backspace_fragment) to BACKSPACE_FRAGMENT,
        stringResource(R.string.two_thumb_backspace_word) to BACKSPACE_WORD,
    )
    val selected = currentBackspaceBehavior(prefs)
    Preference(
        name = setting.title,
        description = items.first { it.second == selected }.first,
        onClick = { showDialog = true },
    )
    if (showDialog) {
        ListPickerDialog(
            onDismissRequest = { showDialog = false },
            items = items,
            onItemSelected = { item ->
                when (item.second) {
                    BACKSPACE_NORMAL -> prefs.edit {
                        putBoolean(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE, false)
                        putBoolean(Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD, false)
                    }
                    BACKSPACE_FRAGMENT -> prefs.edit {
                        putBoolean(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE, true)
                        putBoolean(Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD, false)
                    }
                    BACKSPACE_WORD -> prefs.edit {
                        putBoolean(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE, false)
                        putBoolean(Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD, true)
                    }
                }
            },
            selectedItem = items.first { it.second == selected },
            title = { Text(setting.title) },
            getItemName = { it.first },
        )
    }
}

@Preview
@Composable
private fun Preview() {
    JniUtils.sHaveGestureLib = true
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            TwoThumbTypingScreen { }
        }
    }
}
