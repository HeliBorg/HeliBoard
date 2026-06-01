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
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.previewDark
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.settings.preferences.TextInputPreference

@Composable
fun VoiceInputScreen(
    onClickBack: () -> Unit,
) {
    val prefs = LocalContext.current.prefs()
    val b = (LocalContext.current.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    val enabled = prefs.getBoolean(Settings.PREF_VOICE_ENABLED, Defaults.PREF_VOICE_ENABLED)
    val items = listOf(
        Settings.PREF_VOICE_ENABLED,
        if (enabled) Settings.PREF_VOICE_SHOW_KEYS else null,
        if (enabled) R.string.voice_category_dictation else null,
        if (enabled) Settings.PREF_VOICE_STT_URL else null,
        if (enabled) Settings.PREF_VOICE_STT_MODEL else null,
        if (enabled) Settings.PREF_VOICE_STT_KEY else null,
        if (enabled) Settings.PREF_VOICE_STREAMING else null,
        if (enabled) Settings.PREF_VOICE_MAX_SECONDS else null,
        if (enabled) Settings.PREF_VOICE_TIMEOUT else null,
        if (enabled) Settings.PREF_VOICE_RETRIES else null,
        if (enabled) Settings.PREF_VOICE_STT_FALLBACK_URL else null,
        if (enabled) Settings.PREF_VOICE_STT_FALLBACK_MODEL else null,
        if (enabled) Settings.PREF_VOICE_STT_FALLBACK_KEY else null,
        if (enabled) R.string.voice_category_editing else null,
        if (enabled) Settings.PREF_VOICE_EDIT_URL else null,
        if (enabled) Settings.PREF_VOICE_EDIT_MODEL else null,
        if (enabled) Settings.PREF_VOICE_EDIT_KEY else null,
        if (enabled) R.string.voice_category_security else null,
        if (enabled) Settings.PREF_VOICE_CA_CERT else null,
    )
    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.settings_screen_voice),
        settings = items
    )
}

fun createVoiceInputSettings(context: Context) = listOf(
    Setting(context, Settings.PREF_VOICE_ENABLED, R.string.voice_ai_enable, R.string.voice_ai_enable_summary) {
        SwitchPreference(it, Defaults.PREF_VOICE_ENABLED) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, Settings.PREF_VOICE_SHOW_KEYS, R.string.voice_show_keys, R.string.voice_show_keys_summary) {
        SwitchPreference(it, Defaults.PREF_VOICE_SHOW_KEYS) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, Settings.PREF_VOICE_STREAMING, R.string.voice_streaming, R.string.voice_streaming_summary) {
        SwitchPreference(it, Defaults.PREF_VOICE_STREAMING)
    },
    Setting(context, Settings.PREF_VOICE_STT_URL, R.string.voice_stt_url) { TextInputPreference(it, Defaults.PREF_VOICE_STT_URL) },
    Setting(context, Settings.PREF_VOICE_STT_MODEL, R.string.voice_stt_model) { TextInputPreference(it, Defaults.PREF_VOICE_STT_MODEL) },
    Setting(context, Settings.PREF_VOICE_STT_KEY, R.string.voice_stt_key) { TextInputPreference(it, "") },
    Setting(context, Settings.PREF_VOICE_STT_FALLBACK_URL, R.string.voice_stt_fallback_url) { TextInputPreference(it, Defaults.PREF_VOICE_STT_FALLBACK_URL) },
    Setting(context, Settings.PREF_VOICE_STT_FALLBACK_MODEL, R.string.voice_stt_fallback_model) { TextInputPreference(it, Defaults.PREF_VOICE_STT_FALLBACK_MODEL) },
    Setting(context, Settings.PREF_VOICE_STT_FALLBACK_KEY, R.string.voice_stt_fallback_key) { TextInputPreference(it, "") },
    Setting(context, Settings.PREF_VOICE_EDIT_URL, R.string.voice_edit_url) { TextInputPreference(it, Defaults.PREF_VOICE_EDIT_URL) },
    Setting(context, Settings.PREF_VOICE_EDIT_MODEL, R.string.voice_edit_model) { TextInputPreference(it, Defaults.PREF_VOICE_EDIT_MODEL) },
    Setting(context, Settings.PREF_VOICE_EDIT_KEY, R.string.voice_edit_key) { TextInputPreference(it, "") },
    Setting(context, Settings.PREF_VOICE_CA_CERT, R.string.voice_ca_cert, R.string.voice_ca_cert_summary) {
        TextInputPreference(it, "", singleLine = false)
    },
    Setting(context, Settings.PREF_VOICE_TIMEOUT, R.string.voice_timeout) {
        TextInputPreference(it, Defaults.PREF_VOICE_TIMEOUT) { s -> s.isEmpty() || s.toIntOrNull() != null }
    },
    Setting(context, Settings.PREF_VOICE_RETRIES, R.string.voice_retries) {
        TextInputPreference(it, Defaults.PREF_VOICE_RETRIES) { s -> s.isEmpty() || s.toIntOrNull() != null }
    },
    Setting(context, Settings.PREF_VOICE_MAX_SECONDS, R.string.voice_max_seconds) {
        TextInputPreference(it, Defaults.PREF_VOICE_MAX_SECONDS) { s -> s.isEmpty() || s.toIntOrNull() != null }
    },
)

@Preview
@Composable
private fun PreferencePreview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            VoiceInputScreen { }
        }
    }
}
