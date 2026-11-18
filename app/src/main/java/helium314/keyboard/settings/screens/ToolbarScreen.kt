// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.Context
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.GestureDataGatheringSettings.filterBackgroundGatheringToolbarKeys
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.ToolbarContent
import helium314.keyboard.latin.utils.ToolbarMode
import helium314.keyboard.latin.utils.readToolbarContents
import helium314.keyboard.latin.utils.writeToolbarContents
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.dialogs.MultiListPickerDialog
import helium314.keyboard.settings.dialogs.ToolbarKeysCustomizer
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.preferences.Preference
import helium314.keyboard.settings.preferences.ReorderSwitchPreference
import helium314.keyboard.settings.preferences.SwitchPreference
import helium314.keyboard.latin.utils.previewDark

@Composable
fun ToolbarScreen(
    onClickBack: () -> Unit,
) {
    val prefs = LocalContext.current.prefs()
    val b = (LocalContext.current.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    val toolbarMode = Settings.readToolbarMode(prefs)
    val clipboardToolbarVisible = ToolbarContent.CLIPBOARD in readToolbarContents(prefs)
    val items = listOf(
        Settings.PREF_TOOLBAR_CONTENTS,
        if (toolbarMode != ToolbarMode.HIDDEN) Settings.PREF_TOOLBAR_SWIPE_DOWN_TO_HIDE else null,
        when (toolbarMode) {
             ToolbarMode.EXPANDABLE, ToolbarMode.TOOLBAR_KEYS -> Settings.PREF_TOOLBAR_KEYS
             else -> null
        },
        when (toolbarMode) {
            ToolbarMode.EXPANDABLE, ToolbarMode.SUGGESTION_STRIP -> Settings.PREF_PINNED_TOOLBAR_KEYS
            else -> null
        },
        if (clipboardToolbarVisible) Settings.PREF_CLIPBOARD_TOOLBAR_KEYS else null,
        // custom keys can be in every toolbar
        if (toolbarMode != ToolbarMode.HIDDEN || clipboardToolbarVisible) Settings.PREF_TOOLBAR_CUSTOM_KEY_CODES else null,
        if (toolbarMode == ToolbarMode.EXPANDABLE) Settings.PREF_QUICK_PIN_TOOLBAR_KEYS else null,
        if (toolbarMode == ToolbarMode.EXPANDABLE) Settings.PREF_AUTO_SHOW_TOOLBAR else null,
        if (toolbarMode == ToolbarMode.EXPANDABLE) Settings.PREF_AUTO_HIDE_TOOLBAR else null,
        if (toolbarMode != ToolbarMode.HIDDEN) Settings.PREF_SHOW_ONLY_TOOLBAR_WITH_HARDWARE_KEYBOARD else null,
        if (toolbarMode != ToolbarMode.HIDDEN) Settings.PREF_VARIABLE_TOOLBAR_DIRECTION else null,
    )
    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.settings_screen_toolbar),
        settings = items
    )
}

private fun toolbarContentName(content: ToolbarContent, context: Context) = context.getString(when (content) {
    ToolbarContent.TOOLBAR_KEYS -> R.string.toolbar_content_keyboard
    ToolbarContent.SUGGESTIONS -> R.string.toolbar_content_suggestions
    ToolbarContent.CLIPBOARD -> R.string.toolbar_content_clipboard
    ToolbarContent.EMOJI_CATEGORIES -> R.string.toolbar_content_emoji_categories
})

fun createToolbarSettings(context: Context) = listOf(
    Setting(context, Settings.PREF_TOOLBAR_CONTENTS, R.string.toolbar_contents) { setting ->
        val ctx = LocalContext.current
        var showDialog by rememberSaveable { mutableStateOf(false) }
        val selected = readToolbarContents(ctx.prefs())
        Preference(
            name = setting.title,
            description = if (selected.isEmpty()) stringResource(R.string.toolbar_mode_hidden)
                else selected.joinToString(", ") { toolbarContentName(it, ctx) },
            onClick = { showDialog = true },
        )
        if (showDialog)
            MultiListPickerDialog(
                onDismissRequest = { showDialog = false },
                items = ToolbarContent.entries,
                onConfirmed = {
                    writeToolbarContents(ctx.prefs(), it)
                    KeyboardSwitcher.getInstance().setThemeNeedsReload()
                },
                title = { Text(setting.title) },
                initialSelection = ToolbarContent.entries.filter { it in selected },
                getItemName = { toolbarContentName(it, ctx) },
            )
    },
    Setting(context, Settings.PREF_TOOLBAR_SWIPE_DOWN_TO_HIDE, R.string.toolbar_swipe_down_to_hide, R.string.toolbar_swipe_down_to_hide_summary) {
        SwitchPreference(it, Defaults.PREF_TOOLBAR_SWIPE_DOWN_TO_HIDE)
    },
    Setting(context, Settings.PREF_TOOLBAR_KEYS, R.string.toolbar_keys) {
        val keys = Defaults.PREF_TOOLBAR_KEYS.filterBackgroundGatheringToolbarKeys(LocalContext.current.prefs())
        ReorderSwitchPreference(it, keys)
    },
    Setting(context, Settings.PREF_PINNED_TOOLBAR_KEYS, R.string.pinned_toolbar_keys) {
        val keys = Defaults.PREF_PINNED_TOOLBAR_KEYS.filterBackgroundGatheringToolbarKeys(LocalContext.current.prefs())
        ReorderSwitchPreference(it, keys)
    },
    Setting(context, Settings.PREF_CLIPBOARD_TOOLBAR_KEYS, R.string.clipboard_toolbar_keys) {
        val keys = Defaults.PREF_CLIPBOARD_TOOLBAR_KEYS.filterBackgroundGatheringToolbarKeys(LocalContext.current.prefs())
        ReorderSwitchPreference(it, keys)
    },
    Setting(context, Settings.PREF_TOOLBAR_CUSTOM_KEY_CODES, R.string.customize_toolbar_key_codes) {
        var showDialog by rememberSaveable { mutableStateOf(false) }
        Preference(
            name = it.title,
            onClick = { showDialog = true },
        )
        if (showDialog)
            ToolbarKeysCustomizer(
                key = it.key,
                onDismissRequest = { showDialog = false }
            )
    },
    Setting(context, Settings.PREF_QUICK_PIN_TOOLBAR_KEYS,
        R.string.quick_pin_toolbar_keys, R.string.quick_pin_toolbar_keys_summary)
    {
        SwitchPreference(it, Defaults.PREF_QUICK_PIN_TOOLBAR_KEYS) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
    },
    Setting(context, Settings.PREF_AUTO_SHOW_TOOLBAR, R.string.auto_show_toolbar, R.string.auto_show_toolbar_summary)
    {
        SwitchPreference(it, Defaults.PREF_AUTO_SHOW_TOOLBAR)
    },
    Setting(context, Settings.PREF_AUTO_HIDE_TOOLBAR, R.string.auto_hide_toolbar, R.string.auto_hide_toolbar_summary)
    {
        SwitchPreference(it, Defaults.PREF_AUTO_HIDE_TOOLBAR)
    },
    Setting(context, Settings.PREF_SHOW_ONLY_TOOLBAR_WITH_HARDWARE_KEYBOARD,
        R.string.toolbar_only_with_hw_keyboard, R.string.toolbar_only_with_hw_keyboard_summary)
    {
        SwitchPreference(it, Defaults.PREF_SHOW_ONLY_TOOLBAR_WITH_HARDWARE_KEYBOARD) {
            KeyboardSwitcher.getInstance().setThemeNeedsReload() // necessary for updating insets
        }
    },
    Setting(context, Settings.PREF_VARIABLE_TOOLBAR_DIRECTION,
        R.string.var_toolbar_direction, R.string.var_toolbar_direction_summary)
    {
        SwitchPreference(it, Defaults.PREF_VARIABLE_TOOLBAR_DIRECTION)
    }
)

@Preview
@Composable
private fun Preview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            ToolbarScreen { }
        }
    }
}
