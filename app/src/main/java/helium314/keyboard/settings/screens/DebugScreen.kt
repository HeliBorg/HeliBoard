// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.BuildConfig
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
import helium314.keyboard.latin.utils.previewDark
import androidx.core.content.edit
import com.android.inputmethod.latin.BinaryDictionary
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.common.LocaleUtils.localizedDisplayName
import helium314.keyboard.latin.dictionary.AppsBinaryDictionary
import helium314.keyboard.latin.dictionary.contacts.ContactsBinaryDictionary
import helium314.keyboard.latin.dictionary.UserBinaryDictionary
import helium314.keyboard.latin.makedict.DictIO
import helium314.keyboard.latin.dictionary.history.UserHistoryDictionary
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.settings.dialogs.ListPickerDialog
import helium314.keyboard.settings.filePicker
import java.io.File
import java.util.Locale

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
        DebugSettings.PREF_EXPORT_DICTIONARY,
    )
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
                settings.first { it.key == item }.Preference()
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
    Setting(context, DebugSettings.PREF_EXPORT_DICTIONARY, R.string.prefs_export_dynamic_dicts) { setting ->
        val ctx = LocalContext.current
        var selectedDict by rememberSaveable { mutableStateOf<String?>(null) }
        val fp = filePicker { uri ->
            if (selectedDict == null) return@filePicker
            val file = File(selectedDict!!)
            // only filename actually matters
            val dict = BinaryDictionary(selectedDict, 0, file.length(), false, Locale.ROOT, "", false)
            ctx.getActivity()?.contentResolver?.openOutputStream(uri)?.bufferedWriter()?.use { w ->
                DictIO.writeCombinedDictionary(w, dict)
            }
        }
        var showDialog by remember { mutableStateOf(false) }
        Preference(setting.title, { showDialog = true })
        if (showDialog) {
            val dicts = remember {
                ctx.filesDir.listFiles { it.isDirectory && it.name.endsWith(".dict") && it.list()?.size == 2 }!!
                    .sortedBy { it.name.substringAfter('.') } // sort by language
                    .map {
                        val split = it.name.substringBefore(".dict").split('.', limit = 2)
                        if (split.size < 2) return@map Triple(it, it.name.substringBefore(".dict"), "zz") // should not happen
                        val type = when (split[0]) {
                            UserBinaryDictionary.NAME -> "personal"
                            AppsBinaryDictionary.NAME -> "apps"
                            ContactsBinaryDictionary.NAME -> "contacts"
                            UserHistoryDictionary.NAME -> "history"
                            else -> it.name.substringBefore(".dict") // should not happen
                        }
                        Triple(it, type, split[1])
                    }
            }
            ListPickerDialog(
                onDismissRequest = { showDialog = false },
                items = dicts,
                onItemSelected = {
                    selectedDict = it.first.absolutePath
                    val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        type = "*/*"
                        addCategory(Intent.CATEGORY_OPENABLE)
                        putExtra(Intent.EXTRA_TITLE, "${it.second}_${it.third}.combined")
                    }
                    fp.launch(intent)
                },
                showRadioButtons = false,
                getItemName = { "${it.third.constructLocale().localizedDisplayName(ctx.resources)}: ${it.second}" }
            )
        }
    },
)

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
