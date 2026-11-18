// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import androidx.compose.material3.Switch
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Links
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.htmlToAnnotated
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.withHtmlLink
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.dialogs.InfoDialog
import androidx.core.content.edit
import helium314.keyboard.latin.dictionary.DictionaryInfoUtils

@Composable
fun SwitchPreference(
    setting: Setting,
    default: Boolean,
    allowCheckedChange: (Boolean) -> Boolean = { true },
    onCheckedChange: (Boolean) -> Unit = { }
) {
    SwitchPreference(
        name = setting.title,
        description = setting.description,
        key = setting.key,
        default = default,
        allowCheckedChange = allowCheckedChange,
        onCheckedChange = onCheckedChange
    )
}

@Composable
fun SwitchPreference(
    name: String,
    modifier: Modifier = Modifier,
    key: String,
    default: Boolean,
    description: String? = null,
    allowCheckedChange: (Boolean) -> Boolean = { true }, // true means ok, usually for showing some dialog
    onCheckedChange: (Boolean) -> Unit = { },
) {
    val ctx = LocalContext.current
    val prefs = ctx.prefs()
    val b = (ctx.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    var value = prefs.getBoolean(key, default)
    fun switched(newValue: Boolean) {
        if (!allowCheckedChange(newValue)) {
            value = !newValue
            return
        }
        value = newValue
        prefs.edit { putBoolean(key, newValue) }
        onCheckedChange(newValue)
    }
    Preference(
        name = name,
        onClick = { switched(!value) },
        modifier = modifier,
        description = description
    ) {
        Switch(
            checked = value,
            onCheckedChange = { switched(it) },
        )
    }
}

@Composable
fun SwitchPreferenceWithEmojiDictWarning(setting: Setting, default: Boolean) {
    val context = LocalContext.current
    var showWarningDialog by rememberSaveable { mutableStateOf(false) }
    val hasEmojiDict = DictionaryInfoUtils.getLocalesWithEmojiDicts(context).isNotEmpty()
    SwitchPreference(setting, default && hasEmojiDict) { showWarningDialog = it && !hasEmojiDict }
    if (showWarningDialog) {
        // emoji_dictionary_required contains "%s" since we didn't supply a formatArg
        val link = stringResource(R.string.dictionary_link_text).withHtmlLink(Links.DICTIONARY_URL + Links.DICTIONARY_DOWNLOAD_SUFFIX.replace("raw", "src")
            + Links.DICTIONARY_EMOJI_CLDR_SUFFIX)
        val message = stringResource(R.string.emoji_dictionary_required, link)
        InfoDialog(message.htmlToAnnotated(), onDismissRequest = { showWarningDialog = false })
    }
}
