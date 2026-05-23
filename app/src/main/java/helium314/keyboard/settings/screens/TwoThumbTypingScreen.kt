// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.Context
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
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
import helium314.keyboard.settings.Theme
import helium314.keyboard.settings.initPreview
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
    val combiningGraceMs = prefs.getInt(Settings.PREF_COMBINING_GRACE_MS, Defaults.PREF_COMBINING_GRACE_MS)
    val combiningEnabled = combiningGraceMs > 0
    val multiPartEnabled = prefs.getBoolean(
        Settings.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING,
        Defaults.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING,
    )
    val manualSpacing = prefs.getBoolean(Settings.PREF_GESTURE_MANUAL_SPACING, Defaults.PREF_GESTURE_MANUAL_SPACING)
    val tapDuringSwipe = prefs.getBoolean(Settings.PREF_GESTURE_TAP_DURING_SWIPE, Defaults.PREF_GESTURE_TAP_DURING_SWIPE)
    val dualThumbHinting = prefs.getBoolean(Settings.PREF_GESTURE_DUAL_THUMB_HINTING, Defaults.PREF_GESTURE_DUAL_THUMB_HINTING)

    val items = buildList {
        if (!gestureEnabled) {
            add(R.string.two_thumb_typing_requires_gesture)
            return@buildList
        }

        add(R.string.settings_category_two_thumb_typing_words)
        add(Settings.PREF_COMBINING_GRACE_MS)
        if (combiningEnabled) {
            add(Settings.PREF_COMBINING_TAP_EXTRA_MS)
            add(Settings.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING)
            if (multiPartEnabled) {
                add(Settings.PREF_MULTIPART_FULL_WORD_SUGGESTIONS)
                add(Settings.PREF_MULTIPART_TAP_SEED_GESTURE)
                add(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE)
            }
            add(Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD)
            add(Settings.PREF_COMBINING_AUTOCORRECT_ON_AUTOSPACE)
            add(Settings.PREF_COMBINING_AUTOSPACE_SUGGESTIONS)
        }

        add(R.string.settings_category_two_thumb_typing_manual)
        add(Settings.PREF_GESTURE_MANUAL_SPACING)
        if (manualSpacing && !(combiningEnabled && multiPartEnabled)) {
            add(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE)
        } else {
            // Mutually exclusive with manual spacing: when the user opts into the explicit
            // "no autospace ever" mode, the timer-based grace period is meaningless.
            add(Settings.PREF_GESTURE_AUTOSPACE_GRACE_MS)
        }

        add(R.string.settings_category_two_thumb_typing_two_finger)
        add(Settings.PREF_GESTURE_TAP_DURING_SWIPE)
        if (tapDuringSwipe) {
            add(Settings.PREF_GESTURE_TAP_AS_SWIPE_WINDOW_MS)
        }
        add(Settings.PREF_GESTURE_TAP_PROMOTION_MS)

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

/**
 * Factory for all two-thumb typing {@link Setting} entries. Registered with
 * {@link helium314.keyboard.settings.SettingsContainer} so the entries are picked up by the
 * screen above AND remain searchable globally. Moved out of {@link #createGestureTypingSettings}
 * so the parent gesture screen stays uncluttered.
 */
fun createTwoThumbTypingSettings(context: Context) = listOf(
    Setting(context, Settings.PREF_COMBINING_GRACE_MS,
        R.string.two_thumb_combine_grace, R.string.two_thumb_combine_grace_summary) { def ->
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
        R.string.two_thumb_tap_extra, R.string.two_thumb_tap_extra_summary) { def ->
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
    Setting(context, Settings.PREF_GESTURE_MANUAL_SPACING,
        R.string.gesture_manual_spacing, R.string.gesture_manual_spacing_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_MANUAL_SPACING)
    },
    Setting(context, Settings.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING,
        R.string.two_thumb_multi_part_words, R.string.two_thumb_multi_part_words_summary) {
        SwitchPreference(it, Defaults.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING)
    },
    Setting(context, Settings.PREF_MULTIPART_FULL_WORD_SUGGESTIONS,
        R.string.multipart_full_word_suggestions, R.string.multipart_full_word_suggestions_summary) {
        SwitchPreference(it, Defaults.PREF_MULTIPART_FULL_WORD_SUGGESTIONS)
    },
    Setting(context, Settings.PREF_MULTIPART_TAP_SEED_GESTURE,
        R.string.two_thumb_typed_prefix_swipe, R.string.two_thumb_typed_prefix_swipe_summary) {
        SwitchPreference(it, Defaults.PREF_MULTIPART_TAP_SEED_GESTURE)
    },
    Setting(context, Settings.PREF_GESTURE_FRAGMENT_BACKSPACE,
        R.string.gesture_fragment_backspace, R.string.gesture_fragment_backspace_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_FRAGMENT_BACKSPACE)
    },
    Setting(context, Settings.PREF_GESTURE_AUTOSPACE_GRACE_MS, R.string.gesture_autospace_grace) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_GESTURE_AUTOSPACE_GRACE_MS,
            range = 0f..500f,
            description = {
                if (it <= 0) stringResource(R.string.gesture_autospace_grace_off)
                else stringResource(R.string.abbreviation_unit_milliseconds, it.toString())
            }
        )
    },
    Setting(context, Settings.PREF_GESTURE_TAP_DURING_SWIPE,
        R.string.two_thumb_tap_during_swipe, R.string.two_thumb_tap_during_swipe_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_TAP_DURING_SWIPE)
    },
    Setting(context, Settings.PREF_GESTURE_TAP_AS_SWIPE_WINDOW_MS,
        R.string.two_thumb_tap_during_swipe_duration, R.string.two_thumb_tap_during_swipe_duration_summary) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_GESTURE_TAP_AS_SWIPE_WINDOW_MS,
            range = 0f..200f,
            description = { stringResource(R.string.abbreviation_unit_milliseconds, it.toString()) }
        )
    },
    Setting(context, Settings.PREF_GESTURE_DUAL_THUMB_HINTING,
        R.string.gesture_dual_thumb_hinting, R.string.gesture_dual_thumb_hinting_summary) {
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
