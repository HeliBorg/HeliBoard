// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.Manifest
import android.content.Context
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.dictionarypack.DictionaryPackConstants
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.permissions.PermissionsUtil
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.JniUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.ToolbarMode
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.NextScreenIcon
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.SettingsDestination
import helium314.keyboard.settings.SettingsWithoutKey
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.preferences.ListPreference
import helium314.keyboard.settings.preferences.Preference
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.settings.preferences.SwitchPreferenceWithEmojiDictWarning
import helium314.keyboard.latin.utils.previewDark
import androidx.core.content.edit

@Composable
fun TextCorrectionScreen(
    onClickBack: () -> Unit,
) {
    val prefs = LocalContext.current.prefs()
    val b = (LocalContext.current.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    val autocorrectEnabled = prefs.getBoolean(Settings.PREF_AUTO_CORRECTION, Defaults.PREF_AUTO_CORRECTION)
    val suggestionsVisible = Settings.readToolbarMode(prefs) in setOf(ToolbarMode.SUGGESTION_STRIP, ToolbarMode.EXPANDABLE)
    val suggestionsEnabled = suggestionsVisible && prefs.getBoolean(Settings.PREF_SHOW_SUGGESTIONS, Defaults.PREF_SHOW_SUGGESTIONS)
    val gestureEnabled = JniUtils.sHaveGestureLib && prefs.getBoolean(Settings.PREF_GESTURE_INPUT, Defaults.PREF_GESTURE_INPUT)
    val items = listOf(
        SettingsWithoutKey.EDIT_PERSONAL_DICTIONARY,
        R.string.settings_category_correction,
        Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE,
        Settings.PREF_AUTO_CORRECTION,
        if (autocorrectEnabled) Settings.PREF_MORE_AUTO_CORRECTION else null,
        if (autocorrectEnabled) Settings.PREF_AUTOCORRECT_SHORTCUTS else null,
        if (autocorrectEnabled) Settings.PREF_AUTO_CORRECT_THRESHOLD else null,
        if (autocorrectEnabled) Settings.PREF_BACKSPACE_REVERTS_AUTOCORRECT else null,
        Settings.PREF_AUTO_CAP,
        R.string.settings_category_space,
        Settings.PREF_KEY_USE_DOUBLE_SPACE_PERIOD,
        Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION,
        Settings.PREF_AUTOSPACE_AFTER_SUGGESTION,
        if (gestureEnabled) Settings.PREF_AUTOSPACE_BEFORE_GESTURE_TYPING else null,
        if (gestureEnabled) Settings.PREF_AUTOSPACE_AFTER_GESTURE_TYPING else null,
        Settings.PREF_SHIFT_REMOVES_AUTOSPACE,
        R.string.settings_category_suggestions,
        if (suggestionsVisible) Settings.PREF_SHOW_SUGGESTIONS else null,
        if (suggestionsEnabled) Settings.PREF_ALWAYS_SHOW_SUGGESTIONS else null,
        if (suggestionsEnabled && prefs.getBoolean(Settings.PREF_ALWAYS_SHOW_SUGGESTIONS, Defaults.PREF_ALWAYS_SHOW_SUGGESTIONS))
            Settings.PREF_ALWAYS_SHOW_SUGGESTIONS_EXCEPT_WEB_TEXT else null,
        if (suggestionsEnabled) Settings.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER else null,
        if (suggestionsEnabled || autocorrectEnabled) Settings.PREF_SUGGEST_EMOJIS else null,
        if (suggestionsEnabled || autocorrectEnabled) Settings.PREF_INLINE_EMOJI_SEARCH else null,
        Settings.PREF_KEY_USE_PERSONALIZED_DICTS,
        Settings.PREF_BIGRAM_PREDICTIONS,
        Settings.PREF_SUGGEST_PUNCTUATION,
        Settings.PREF_SUGGEST_CLIPBOARD_CONTENT,
        Settings.PREF_USE_CONTACTS,
        Settings.PREF_USE_APPS,
        if (prefs.getBoolean(Settings.PREF_KEY_USE_PERSONALIZED_DICTS, Defaults.PREF_KEY_USE_PERSONALIZED_DICTS))
            Settings.PREF_ADD_TO_PERSONAL_DICTIONARY else null
    )
    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.settings_screen_correction),
        settings = items
    )
}

fun createCorrectionSettings(context: Context) = listOf(
    Setting(context, SettingsWithoutKey.EDIT_PERSONAL_DICTIONARY, R.string.edit_personal_dictionary) {
        Preference(
            name = stringResource(R.string.edit_personal_dictionary),
            onClick = { SettingsDestination.navigateTo(SettingsDestination.PersonalDictionaries) },
        ) { NextScreenIcon() }
    },
    Setting(context, Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE,
        R.string.prefs_block_potentially_offensive_title, R.string.prefs_block_potentially_offensive_summary
    ) {
        SwitchPreference(it, Defaults.PREF_BLOCK_POTENTIALLY_OFFENSIVE)
    },
    Setting(context, Settings.PREF_AUTO_CORRECTION,
        R.string.autocorrect, R.string.auto_correction_summary
    ) {
        SwitchPreference(it, Defaults.PREF_AUTO_CORRECTION)
    },
    Setting(context, Settings.PREF_MORE_AUTO_CORRECTION,
        R.string.more_autocorrect, R.string.more_autocorrect_summary
    ) {
        SwitchPreference(it, Defaults.PREF_MORE_AUTO_CORRECTION)
    },
    Setting(context, Settings.PREF_AUTOCORRECT_SHORTCUTS,
        R.string.auto_correct_shortcuts, R.string.auto_correct_shortcuts_summary
    ) {
        SwitchPreference(it, Defaults.PREF_AUTOCORRECT_SHORTCUTS)
    },
    Setting(context, Settings.PREF_AUTO_CORRECT_THRESHOLD, R.string.auto_correction_confidence) {
        val items = listOf(
            stringResource(R.string.auto_correction_threshold_mode_modest) to 0.185f,
            stringResource(R.string.auto_correction_threshold_mode_aggressive) to 0.067f,
            stringResource(R.string.auto_correction_threshold_mode_very_aggressive) to -1f,
        )
        // todo: consider making it a slider, and maybe somehow adjust range so we can show %
        ListPreference(it, items, Defaults.PREF_AUTO_CORRECT_THRESHOLD)
    },
    Setting(context, Settings.PREF_BACKSPACE_REVERTS_AUTOCORRECT, R.string.backspace_reverts_autocorrect) {
        SwitchPreference(it, Defaults.PREF_BACKSPACE_REVERTS_AUTOCORRECT)
    },
    Setting(context, Settings.PREF_AUTO_CAP,
        R.string.auto_cap, R.string.auto_cap_summary
    ) {
        SwitchPreference(it, Defaults.PREF_AUTO_CAP)
    },
    Setting(context, Settings.PREF_KEY_USE_DOUBLE_SPACE_PERIOD,
        R.string.use_double_space_period, R.string.use_double_space_period_summary
    ) {
        SwitchPreference(it, Defaults.PREF_KEY_USE_DOUBLE_SPACE_PERIOD)
    },
    Setting(context, Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION,
        R.string.autospace_after_punctuation, R.string.autospace_after_punctuation_summary
    ) {
        SwitchPreference(it, Defaults.PREF_AUTOSPACE_AFTER_PUNCTUATION)
    },
    Setting(context, Settings.PREF_AUTOSPACE_AFTER_SUGGESTION, R.string.autospace_after_suggestion) {
        SwitchPreference(it, Defaults.PREF_AUTOSPACE_AFTER_SUGGESTION)
    },
    Setting(context, Settings.PREF_AUTOSPACE_AFTER_GESTURE_TYPING, R.string.autospace_after_gesture_typing) {
        SwitchPreference(it, Defaults.PREF_AUTOSPACE_AFTER_GESTURE_TYPING)
    },
    Setting(context, Settings.PREF_AUTOSPACE_BEFORE_GESTURE_TYPING, R.string.autospace_before_gesture_typing) {
        SwitchPreference(it, Defaults.PREF_AUTOSPACE_BEFORE_GESTURE_TYPING)
    },
    Setting(context, Settings.PREF_SHIFT_REMOVES_AUTOSPACE, R.string.shift_removes_autospace, R.string.shift_removes_autospace_summary) {
        SwitchPreference(it, Defaults.PREF_SHIFT_REMOVES_AUTOSPACE)
    },
    Setting(context, Settings.PREF_SHOW_SUGGESTIONS,
        R.string.prefs_show_suggestions, R.string.prefs_show_suggestions_summary
    ) {
        SwitchPreference(it, Defaults.PREF_SHOW_SUGGESTIONS)
    },
    Setting(context, Settings.PREF_ALWAYS_SHOW_SUGGESTIONS,
        R.string.prefs_always_show_suggestions, R.string.prefs_always_show_suggestions_summary
    ) {
        SwitchPreference(it, Defaults.PREF_ALWAYS_SHOW_SUGGESTIONS)
    },
    Setting(context, Settings.PREF_ALWAYS_SHOW_SUGGESTIONS_EXCEPT_WEB_TEXT,
        R.string.prefs_always_show_suggestions_except_web_text, R.string.prefs_always_show_suggestions_except_web_text_summary
    ) {
        SwitchPreference(it, Defaults.PREF_ALWAYS_SHOW_SUGGESTIONS_EXCEPT_WEB_TEXT)
    },
    Setting(context, Settings.PREF_KEY_USE_PERSONALIZED_DICTS,
        R.string.use_personalized_dicts, R.string.use_personalized_dicts_summary
    ) { setting ->
        var showConfirmDialog by rememberSaveable { mutableStateOf(false) }
        SwitchPreference(setting, Defaults.PREF_KEY_USE_PERSONALIZED_DICTS,
            allowCheckedChange = {
                showConfirmDialog = !it
                it
            }
        )
        if (showConfirmDialog) {
            val prefs = LocalContext.current.prefs()
            ConfirmationDialog(
                onDismissRequest = { showConfirmDialog = false },
                onConfirmed = {
                    prefs.edit { putBoolean(setting.key, false) }
                },
                content = { Text(stringResource(R.string.disable_personalized_dicts_message)) }
            )
        }

    },
    Setting(context, Settings.PREF_BIGRAM_PREDICTIONS,
        R.string.bigram_prediction, R.string.bigram_prediction_summary
    ) {
        SwitchPreference(it, Defaults.PREF_BIGRAM_PREDICTIONS) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, Settings.PREF_SUGGEST_PUNCTUATION, R.string.suggest_punctuation, R.string.suggest_punctuation_summary
    ) {
        SwitchPreference(it, Defaults.PREF_SUGGEST_PUNCTUATION) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, Settings.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER,
        R.string.center_suggestion_text_to_enter, R.string.center_suggestion_text_to_enter_summary
    ) {
        SwitchPreference(it, Defaults.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER)
    },
    Setting(context, Settings.PREF_SUGGEST_CLIPBOARD_CONTENT,
        R.string.suggest_clipboard_content, R.string.suggest_clipboard_content_summary
    ) {
        SwitchPreference(it, Defaults.PREF_SUGGEST_CLIPBOARD_CONTENT)
    },
    Setting(context, Settings.PREF_USE_CONTACTS,
        R.string.use_contacts_dict, R.string.use_contacts_dict_summary
    ) { setting ->
        val activity = LocalContext.current.getActivity() ?: return@Setting
        var granted by remember { mutableStateOf(PermissionsUtil.checkAllPermissionsGranted(activity, Manifest.permission.READ_CONTACTS)) }
        val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
            granted = it
            if (granted)
                activity.prefs().edit { putBoolean(setting.key, true) }
        }
        SwitchPreference(setting, Defaults.PREF_USE_CONTACTS,
            allowCheckedChange = {
                if (it && !granted) {
                    launcher.launch(Manifest.permission.READ_CONTACTS)
                    false
                } else true
            }
        )
    },
    Setting(context, Settings.PREF_USE_APPS,
        R.string.use_apps_dict, R.string.use_apps_dict_summary
    ) { setting ->
        SwitchPreference(setting, Defaults.PREF_USE_APPS)
    },
    Setting(
        context, Settings.PREF_SUGGEST_EMOJIS, R.string.suggest_emojis, R.string.suggest_emojis_summary
    ) {
        SwitchPreference(it, Defaults.PREF_SUGGEST_EMOJIS) {
            context.sendBroadcast(Intent(DictionaryPackConstants.NEW_DICTIONARY_INTENT_ACTION))
        }
    },
    Setting(
        context, Settings.PREF_INLINE_EMOJI_SEARCH, R.string.inline_emoji_search, R.string.inline_emoji_search_summary) {
        SwitchPreferenceWithEmojiDictWarning(it, Defaults.PREF_INLINE_EMOJI_SEARCH)
    },
    Setting(context, Settings.PREF_ADD_TO_PERSONAL_DICTIONARY,
        R.string.add_to_personal_dictionary, R.string.add_to_personal_dictionary_summary
    ) {
        SwitchPreference(it, Defaults.PREF_ADD_TO_PERSONAL_DICTIONARY)
    },
)

@Preview
@Composable
private fun PreferencePreview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            TextCorrectionScreen {  }
        }
    }
}
