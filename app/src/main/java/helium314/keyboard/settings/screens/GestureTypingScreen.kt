// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.Context
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.JniUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.preferences.SliderPreference
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.initPreview
import helium314.keyboard.latin.utils.previewDark
import androidx.core.content.edit

@Composable
fun GestureTypingScreen(
    onClickBack: () -> Unit,
) {
    val prefs = LocalContext.current.prefs()
    val b = (LocalContext.current.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    val gestureFloatingPreviewEnabled = prefs.getBoolean(Settings.PREF_GESTURE_FLOATING_PREVIEW_TEXT, Defaults.PREF_GESTURE_FLOATING_PREVIEW_TEXT)
    val gestureEnabled = hasGestureLib && prefs.getBoolean(Settings.PREF_GESTURE_INPUT, Defaults.PREF_GESTURE_INPUT)
    
    // Always show library loader first when no library
    val items = buildList {
        add(R.string.settings_category_configuration)
        // Library loader is always first if allowed
        if (helium314.keyboard.latin.BuildConfig.BUILD_TYPE != "nouserlib") {
            add(SettingsWithoutKey.LOAD_GESTURE_LIB)
        }
        // Show all gesture settings (they will be disabled if no library)
        add(Settings.PREF_GESTURE_INPUT)

        if (hasGestureLib && gestureEnabled) {
            add(R.string.settings_category_visuals)
            add(Settings.PREF_GESTURE_PREVIEW_TRAIL)
            add(Settings.PREF_GESTURE_FLOATING_PREVIEW_TEXT)
            if (gestureFloatingPreviewEnabled)
                add(Settings.PREF_GESTURE_FLOATING_PREVIEW_DYNAMIC)
            if (prefs.getBoolean(Settings.PREF_GESTURE_PREVIEW_TRAIL, Defaults.PREF_GESTURE_PREVIEW_TRAIL) || gestureFloatingPreviewEnabled)
                add(Settings.PREF_GESTURE_TRAIL_FADEOUT_DURATION)

            add(R.string.settings_category_behavior)
            add(Settings.PREF_GESTURE_SPACE_AWARE)
            add(Settings.PREF_GESTURE_FAST_TYPING_COOLDOWN)
            // Two-thumb typing settings have moved to their own screen
            // (see TwoThumbTypingScreen / SettingsDestination.TwoThumbTyping); deliberately
            // not duplicated here to keep the gesture screen focused.
        }

        add(R.string.settings_category_gestures_advanced)
        add(Settings.PREF_SPACE_HORIZONTAL_SWIPE)
        add(Settings.PREF_SPACE_VERTICAL_SWIPE)
        add(Settings.PREF_TOUCHPAD_SENSITIVITY)
        add(Settings.PREF_DELETE_SWIPE)
        add(Settings.PREF_BACKSPACE_UP_DOWN_SWIPE)
        add(Settings.PREF_TOP_ROW_SWIPE_UP)
        add(Settings.PREF_EMOJI_KEY_SWIPE)
    }
    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.settings_screen_gesture),
        settings = items
    )
}

fun createGestureTypingSettings(context: Context) = listOf(
    Setting(context, Settings.PREF_GESTURE_INPUT, R.string.gesture_input, R.string.gesture_input_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_INPUT)
    },
    Setting(context, Settings.PREF_GESTURE_PREVIEW_TRAIL, R.string.gesture_preview_trail) {
        SwitchPreference(it, Defaults.PREF_GESTURE_PREVIEW_TRAIL)
    },
    Setting(context, Settings.PREF_GESTURE_FLOATING_PREVIEW_TEXT,
        R.string.gesture_floating_preview_static, R.string.gesture_floating_preview_static_summary)
    {
        SwitchPreference(it, Defaults.PREF_GESTURE_FLOATING_PREVIEW_TEXT)
    },
    Setting(context, Settings.PREF_GESTURE_FLOATING_PREVIEW_DYNAMIC,
        R.string.gesture_floating_preview_text, R.string.gesture_floating_preview_dynamic_summary)
    { def ->
        val ctx = LocalContext.current
        SwitchPreference(def, Defaults.PREF_GESTURE_FLOATING_PREVIEW_DYNAMIC) {
            // is this complexity and 2 pref keys for one setting really needed?
            // default value is based on system reduced motion
            val default = Settings.readGestureDynamicPreviewDefault(ctx)
            val followingSystem = it == default
            // allow the default to be overridden
            ctx.prefs().edit { putBoolean(Settings.PREF_GESTURE_DYNAMIC_PREVIEW_FOLLOW_SYSTEM, followingSystem) }
            KeyboardSwitcher.getInstance().setThemeNeedsReload()
        }
    },
    Setting(context, Settings.PREF_GESTURE_SPACE_AWARE, R.string.gesture_space_aware, R.string.gesture_space_aware_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_SPACE_AWARE)
    },
    Setting(context, Settings.PREF_GESTURE_FAST_TYPING_COOLDOWN, R.string.gesture_fast_typing_cooldown) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_GESTURE_FAST_TYPING_COOLDOWN,
            range = 0f..500f,
            description = {
                if (it <= 0) stringResource(R.string.gesture_fast_typing_cooldown_instant)
                else stringResource(R.string.abbreviation_unit_milliseconds, it.toString())
            }
        )
    },
    Setting(context, Settings.PREF_GESTURE_TRAIL_FADEOUT_DURATION, R.string.gesture_trail_fadeout_duration) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_GESTURE_TRAIL_FADEOUT_DURATION,
            range = 100f..1900f,
            description = { stringResource(R.string.abbreviation_unit_milliseconds, (it + 100).toString()) },
            stepSize = 10,
        ) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, SettingsWithoutKey.LOAD_GESTURE_LIB, R.string.load_gesture_library, R.string.load_gesture_library_summary) {
        LoadGestureLibPreference(it.title)
    },
    Setting(context, Settings.PREF_SPACE_HORIZONTAL_SWIPE, R.string.show_horizontal_space_swipe) {
        val items = listOf(
            stringResource(R.string.space_swipe_move_cursor_entry) to "move_cursor",
            stringResource(R.string.switch_language) to "switch_language",
            stringResource(R.string.space_swipe_toggle_numpad_entry) to "toggle_numpad",
            stringResource(R.string.action_none) to "none",
        )
        ListPreference(it, items, Defaults.PREF_SPACE_HORIZONTAL_SWIPE)
    },
    Setting(context, Settings.PREF_SPACE_VERTICAL_SWIPE, R.string.show_vertical_space_swipe) {
        val items = listOf(
            stringResource(R.string.space_swipe_move_cursor_entry) to "move_cursor",
            stringResource(R.string.switch_language) to "switch_language",
            stringResource(R.string.space_swipe_toggle_numpad_entry) to "toggle_numpad",
            stringResource(R.string.space_swipe_hide_keyboard_entry) to "hide_keyboard",
            stringResource(R.string.space_swipe_touchpad_mode_entry) to "touchpad_mode",
            stringResource(R.string.action_none) to "none",
        )
        ListPreference(it, items, Defaults.PREF_SPACE_VERTICAL_SWIPE)
    },
    Setting(context, Settings.PREF_TOUCHPAD_SENSITIVITY, R.string.touchpad_sensitivity) {
        SliderPreference(
            name = it.title,
            key = it.key,
            default = Defaults.PREF_TOUCHPAD_SENSITIVITY,
            range = 0f..100f,
            description = { value -> value.toInt().toString() }
        )
    },
    Setting(context, Settings.PREF_DELETE_SWIPE, R.string.delete_swipe, R.string.delete_swipe_summary) {
        SwitchPreference(it, Defaults.PREF_DELETE_SWIPE)
    },
    Setting(context, Settings.PREF_BACKSPACE_UP_DOWN_SWIPE, R.string.backspace_up_down_swipe, R.string.backspace_up_down_swipe_summary) {
        SwitchPreference(it, Defaults.PREF_BACKSPACE_UP_DOWN_SWIPE)
    },
    Setting(context, Settings.PREF_TOP_ROW_SWIPE_UP, R.string.top_row_swipe_up) {
        val items = listOf(
            stringResource(R.string.action_none) to "none",
            stringResource(R.string.undo) to "undo",
            stringResource(R.string.redo) to "redo",
            stringResource(R.string.emoji) to "emoji",
        )
        ListPreference(it, items, Defaults.PREF_TOP_ROW_SWIPE_UP)
    },
    Setting(context, Settings.PREF_EMOJI_KEY_SWIPE, R.string.emoji_key_swipe, R.string.emoji_key_swipe_summary) {
        SwitchPreference(it, Defaults.PREF_EMOJI_KEY_SWIPE)
    },
)

@Preview
@Composable
private fun Preview() {
    JniUtils.sHaveGestureLib = true
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            GestureTypingScreen { }
        }
    }
}
