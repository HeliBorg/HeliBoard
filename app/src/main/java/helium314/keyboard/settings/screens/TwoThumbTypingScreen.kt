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
import helium314.keyboard.settings.preferences.ListPreference
import helium314.keyboard.settings.preferences.SliderPreference
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.settings.previewDark

/**
 * Dedicated screen for the experimental two-thumb typing features (HeliBoard issue #291).
 * Every option here defaults to off / 0 — turning the whole screen into a no-op until the
 * user opts in. Subsections group features around the user-facing behaviour (manual vs.
 * autospace-grace, tap+swipe interactions, point hinting, debug overlay) rather than around
 * the underlying code paths.
 *
 * The screen is gated on {@code JniUtils.sHaveGestureLib} + {@code PREF_GESTURE_INPUT}: with
 * either of those off, none of the toggles have any effect, so we surface a single hint row
 * and skip everything else. That keeps the experimental options out of the user's way until
 * they're actually relevant.
 */
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

    val items = buildList {
        if (!gestureEnabled) {
            // Nothing useful to show — every toggle is a no-op until gesture typing is on.
            // We deliberately don't even show the section headers; the user lands on an empty
            // screen with just the search box, which is the clearest signal "you need to enable
            // gesture typing first".
            return@buildList
        }

        // --- Combining mode (unified replacement for autospace grace + tap promotion + flash) ---
        add(R.string.settings_category_two_thumb_typing_spacing)
        add(Settings.PREF_COMBINING_GRACE_MS)
        if (prefs.getInt(Settings.PREF_COMBINING_GRACE_MS, Defaults.PREF_COMBINING_GRACE_MS) > 0) {
            add(Settings.PREF_COMBINING_TAP_EXTRA_MS)
            add(Settings.PREF_COMBINING_AUTOCORRECT_ON_AUTOSPACE)
            add(Settings.PREF_COMBINING_AUTOSPACE_SUGGESTIONS)
            add(Settings.PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD)
            // Multi-part word composition (only meaningful when combining grace > 0).
            add(Settings.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING)
            if (prefs.getBoolean(Settings.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING,
                    Defaults.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING)) {
                add(Settings.PREF_MULTIPART_FULL_WORD_SUGGESTIONS)
                add(Settings.PREF_MULTIPART_TAP_SEED_GESTURE)
                add(Settings.PREF_MULTIPART_JOIN_KEY_MODE)
                // Show fragment-backspace alongside multi-part: backspace popping the last
                // joined fragment is the natural "undo a bad join" gesture.
                add(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE)
            }
        }
        add(Settings.PREF_GESTURE_MANUAL_SPACING)
        val manualSpacing = prefs.getBoolean(Settings.PREF_GESTURE_MANUAL_SPACING, Defaults.PREF_GESTURE_MANUAL_SPACING)
        if (manualSpacing) {
            // Sub-option of manual spacing too. Multi-part also shows it above; we don't
            // double-add since SettingsContainer dedupes by key.
            add(Settings.PREF_GESTURE_FRAGMENT_BACKSPACE)
        }

        // --- Tap / swipe interaction tweaks (#1.3) ---
        add(R.string.settings_category_two_thumb_typing_taps)
        add(Settings.PREF_GESTURE_TAP_DURING_SWIPE)
        if (prefs.getBoolean(Settings.PREF_GESTURE_TAP_DURING_SWIPE, Defaults.PREF_GESTURE_TAP_DURING_SWIPE)) {
            add(Settings.PREF_GESTURE_TAP_AS_SWIPE_WINDOW_MS)
        }

        // --- Layout-side (#2.3): apostrophe key for contractions. Layout edit still owed to
        // the user; the toggle is here so it's discoverable once that's in place. ---
        add(R.string.settings_category_two_thumb_typing_layout)
        add(Settings.PREF_GESTURE_APOSTROPHE_KEY)

        // --- Point hinting + debug overlay (#2.1). The midline only matters when hinting is
        // on, so it's nested under the toggle. ---
        add(R.string.settings_category_two_thumb_typing_hinting)
        add(Settings.PREF_GESTURE_DUAL_THUMB_HINTING)
        if (prefs.getBoolean(Settings.PREF_GESTURE_DUAL_THUMB_HINTING, Defaults.PREF_GESTURE_DUAL_THUMB_HINTING)) {
            add(Settings.PREF_GESTURE_DUAL_THUMB_MIDLINE_PCT)
        }
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
        R.string.combining_grace, R.string.combining_grace_summary) { def ->
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
        R.string.combining_tap_extra, R.string.combining_tap_extra_summary) { def ->
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
        R.string.multipart_auto_extend_in_combining, R.string.multipart_auto_extend_in_combining_summary) {
        SwitchPreference(it, Defaults.PREF_MULTIPART_AUTO_EXTEND_IN_COMBINING)
    },
    Setting(context, Settings.PREF_MULTIPART_FULL_WORD_SUGGESTIONS,
        R.string.multipart_full_word_suggestions, R.string.multipart_full_word_suggestions_summary) {
        SwitchPreference(it, Defaults.PREF_MULTIPART_FULL_WORD_SUGGESTIONS)
    },
    Setting(context, Settings.PREF_MULTIPART_TAP_SEED_GESTURE,
        R.string.multipart_tap_seed_gesture, R.string.multipart_tap_seed_gesture_summary) {
        SwitchPreference(it, Defaults.PREF_MULTIPART_TAP_SEED_GESTURE)
    },
    Setting(context, Settings.PREF_MULTIPART_JOIN_KEY_MODE,
        R.string.multipart_join_key_mode, R.string.multipart_join_key_mode_summary) { def ->
        val items = listOf(
            stringResource(R.string.multipart_join_key_mode_off) to "off",
            stringResource(R.string.multipart_join_key_mode_longpress_space) to "longpress_space",
            stringResource(R.string.multipart_join_key_mode_dedicated_key) to "dedicated_key",
        )
        ListPreference(def, items, Defaults.PREF_MULTIPART_JOIN_KEY_MODE)
    },
    Setting(context, Settings.PREF_GESTURE_FRAGMENT_BACKSPACE,
        R.string.gesture_fragment_backspace, R.string.gesture_fragment_backspace_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_FRAGMENT_BACKSPACE)
    },
    Setting(context, Settings.PREF_GESTURE_TAP_DURING_SWIPE,
        R.string.gesture_tap_during_swipe, R.string.gesture_tap_during_swipe_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_TAP_DURING_SWIPE)
    },
    Setting(context, Settings.PREF_GESTURE_TAP_AS_SWIPE_WINDOW_MS,
        R.string.gesture_tap_as_swipe_window, R.string.gesture_tap_as_swipe_window_summary) { def ->
        SliderPreference(
            name = def.title,
            key = def.key,
            default = Defaults.PREF_GESTURE_TAP_AS_SWIPE_WINDOW_MS,
            range = 0f..200f,
            description = { stringResource(R.string.abbreviation_unit_milliseconds, it.toString()) }
        )
    },
    Setting(context, Settings.PREF_GESTURE_APOSTROPHE_KEY,
        R.string.gesture_apostrophe_key, R.string.gesture_apostrophe_key_summary) {
        SwitchPreference(it, Defaults.PREF_GESTURE_APOSTROPHE_KEY)
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
