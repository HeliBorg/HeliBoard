// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.DictionaryDumpBroadcastReceiver
import helium314.keyboard.latin.DictionaryFacilitator
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.DebugSettings
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.preferences.Preference
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.preferences.PreferenceCategory
import helium314.keyboard.latin.utils.previewDark
import androidx.core.content.edit

@Composable
fun DebugScreen(
    onClickBack: () -> Unit,
) {
    val ctx = LocalContext.current
    val settings = createDebugSettings(ctx)
    val items = listOfNotNull(
        if (!BuildConfig.DEBUG) DebugSettings.PREF_SHOW_DEBUG_SETTINGS else null,
        DebugSettings.PREF_DEBUG_MODE,
        DebugSettings.PREF_SHOW_SUGGESTION_INFOS,
        DebugSettings.PREF_FORCE_NON_DISTINCT_MULTITOUCH,
        DebugSettings.PREF_SLIDING_KEY_INPUT_PREVIEW,
        R.string.prefs_dump_dynamic_dicts
    ) + DictionaryFacilitator.DYNAMIC_DICTIONARY_TYPES.map { DebugSettings.PREF_KEY_DUMP_DICT_PREFIX + it }
    SearchSettingsScreen(
        onClickBack = {
            if (needsRestart) {
                val intent = Intent.makeRestartActivityTask(ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)?.component)
                intent.setPackage(ctx.packageName)
                ctx.startActivity(intent)
                Runtime.getRuntime().exit(0)
            }
            onClickBack()
        },
        title = stringResource(R.string.debug_settings_title),
        settings = emptyList()
    ) {
        // the preferences are not in SettingsContainer, so set content instead
        LazyColumn {
            items(items, key = { it }) { item ->
                if (item is Int) PreferenceCategory(stringResource(item))
                else settings.first { it.key == item }.Preference()
            }
        }
    }
}

private var needsRestart = false

private fun createDebugSettings(context: Context) = listOf(
    Setting(context, DebugSettings.PREF_SHOW_DEBUG_SETTINGS, R.string.prefs_show_debug_settings) { setting ->
        val prefs = LocalContext.current.prefs()
        SwitchPreference(setting, false)
        { if (!it) prefs.edit { putBoolean(DebugSettings.PREF_DEBUG_MODE, false) } }
    },
    Setting(context, DebugSettings.PREF_DEBUG_MODE, R.string.prefs_debug_mode) { setting ->
        val prefs = LocalContext.current.prefs()
        SwitchPreference(
            name = setting.title,
            key = setting.key,
            description = stringResource(R.string.version_text, BuildConfig.VERSION_NAME),
            default = Defaults.PREF_DEBUG_MODE,
        ) {
            if (!it) prefs.edit { putBoolean(DebugSettings.PREF_SHOW_SUGGESTION_INFOS, false) }
            needsRestart = true
        }
    },
    Setting(context, DebugSettings.PREF_SHOW_SUGGESTION_INFOS, R.string.prefs_show_suggestion_infos) {
        SwitchPreference(it, Defaults.PREF_SHOW_SUGGESTION_INFOS) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, DebugSettings.PREF_FORCE_NON_DISTINCT_MULTITOUCH, R.string.prefs_force_non_distinct_multitouch) {
        SwitchPreference(it, Defaults.PREF_FORCE_NON_DISTINCT_MULTITOUCH) { needsRestart = true }
    },
    Setting(context, DebugSettings.PREF_SLIDING_KEY_INPUT_PREVIEW, R.string.sliding_key_input_preview, R.string.sliding_key_input_preview_summary) { def ->
        SwitchPreference(def, Defaults.PREF_SLIDING_KEY_INPUT_PREVIEW)
    },
) + DictionaryFacilitator.DYNAMIC_DICTIONARY_TYPES.map { type ->
    Setting(context, DebugSettings.PREF_KEY_DUMP_DICT_PREFIX + type, R.string.button_default) {
        val ctx = LocalContext.current
        Preference(
            name = "Dump $type dictionary",
            onClick = {
                val intent = Intent(DictionaryDumpBroadcastReceiver.DICTIONARY_DUMP_INTENT_ACTION)
                intent.setPackage(context.packageName)
                intent.putExtra(DictionaryDumpBroadcastReceiver.DICTIONARY_NAME_KEY, type)
                ctx.sendBroadcast(intent)
            }
        )
    }
}

@Preview
@Composable
private fun Preview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            DebugScreen { }
        }
    }
}
