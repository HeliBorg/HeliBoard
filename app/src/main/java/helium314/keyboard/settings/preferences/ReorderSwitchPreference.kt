// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.content.edit
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.internal.KeyboardIconsSet
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.utils.getStringResourceOrName
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.dialogs.ReorderDialog
import helium314.keyboard.settings.GetIconOrEmpty

@Composable
fun ReorderSwitchPreference(setting: Setting, default: String) {
    var showDialog by rememberSaveable { mutableStateOf(false) }
    Preference(
        name = setting.title,
        description = setting.description,
        onClick = { showDialog = true },
    )
    if (showDialog) {
        val ctx = LocalContext.current
        val prefs = ctx.prefs()
        val items = prefs.getString(setting.key, default)!!.split(Separators.ENTRY).map {
            val both = it.split(Separators.KV)
            KeyAndState(both.first(), both.last().toBoolean())
        }
        ReorderDialog(
            onConfirmed = { reorderedItems ->
                val value = reorderedItems.joinToString(Separators.ENTRY) { it.name + Separators.KV + it.state }
                prefs.edit { putString(setting.key, value) }
                KeyboardSwitcher.getInstance().setThemeNeedsReload()
            },
            onDismissRequest = { showDialog = false },
            onNeutral = { prefs.edit { remove(setting.key)} },
            neutralButtonText = if (prefs.contains(setting.key)) stringResource(R.string.button_default) else null,
            items = items,
            title = { Text(setting.title) },
            displayItem = { item ->
                var checked by rememberSaveable { mutableStateOf(item.state) }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    KeyboardIconsSet.instance.GetIconOrEmpty(item.name)
                    val text = item.name.lowercase().getStringResourceOrName("", ctx)
                    val actualText = if (text != item.name.lowercase()) text
                        else item.name.lowercase().getStringResourceOrName("popup_keys_", ctx)
                    Text(actualText, Modifier.weight(1f))
                    Switch(
                        checked = checked,
                        onCheckedChange = { item.state = it; checked = it }
                    )
                }
            },
            getKey = { it.name }
        )
    }
}

private class KeyAndState(var name: String, var state: Boolean)
