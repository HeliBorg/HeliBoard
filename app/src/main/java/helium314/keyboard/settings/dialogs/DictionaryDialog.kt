// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import android.content.Intent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import helium314.keyboard.compat.locale
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.LocaleUtils.localizedDisplayName
import helium314.keyboard.latin.utils.DictionaryInfoUtils
import helium314.keyboard.latin.utils.createDictionaryTextAnnotated
import helium314.keyboard.settings.DeleteButton
import helium314.keyboard.settings.ExpandButton
import helium314.keyboard.settings.Theme
import helium314.keyboard.settings.dictionaryFilePicker
import helium314.keyboard.settings.previewDark
import helium314.keyboard.settings.screens.getUserAndInternalDictionaries
import java.io.File
import java.util.Locale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalResources
import helium314.keyboard.dictionarypack.DictionaryPackConstants

@Composable
fun DictionaryDialog(
    onDismissRequest: () -> Unit,
    locale: Locale,
) {
    val ctx = LocalContext.current
    val (dictionaries, hasInternal) = getUserAndInternalDictionaries(ctx, locale)
    val mainDict = dictionaries.firstOrNull { it.name == Dictionary.TYPE_MAIN + "_" + DictionaryInfoUtils.USER_DICTIONARY_SUFFIX }
    val addonDicts = dictionaries.filterNot { it == mainDict }
    val picker = dictionaryFilePicker(locale)
    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = {},
        confirmButtonText = null,
        cancelButtonText = stringResource(R.string.dialog_close),
        title = { Text(locale.localizedDisplayName(LocalResources.current)) },
        content = {
            Column {
                if (hasInternal) {
                    val color = if (mainDict == null) MaterialTheme.typography.titleSmall.color
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f) // for disabled look
                    val bottomPadding = if (mainDict == null) 12.dp else 0.dp
                    Text(stringResource(R.string.internal_dictionary_summary),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = bottomPadding),
                        color = color,
                        style = MaterialTheme.typography.titleSmall
                    )
                }
                if (mainDict != null)
                    DictionaryDetails(mainDict)
                if (addonDicts.isNotEmpty()) {
                    HorizontalDivider()
                    Text(stringResource(R.string.dictionary_category_title),
                        modifier = Modifier.padding(vertical = 12.dp),
                        style = MaterialTheme.typography.titleSmall
                    )
                    addonDicts.forEach { DictionaryDetails(it) }
                }
                val dictString = createDictionaryTextAnnotated(locale)
                if (dictString.isNotEmpty()) {
                    HorizontalDivider()
                    Text(stringResource(R.string.dictionary_available),
                        modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
                        style = MaterialTheme.typography.titleSmall
                    )
                    Text(dictString, style = LocalTextStyle.current.merge(lineHeight = 1.8.em))
                }
            }
        },
        scrollContent = true,
        neutralButtonText = stringResource(R.string.add_new_dictionary_title),
        onNeutral = {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
                .addCategory(Intent.CATEGORY_OPENABLE)
                .setType("application/octet-stream")
            picker.launch(intent)
        }
    )
}

@Composable
private fun DictionaryDetails(dict: File) {
    val header = DictionaryInfoUtils.getDictionaryFileHeaderOrNull(dict) ?: return
    val type = header.mIdString.substringBefore(":")
    var showDeleteDialog by remember { mutableStateOf(false) }
    var showDetails by remember { mutableStateOf(false) }
    val title = if (type != DictionaryInfoUtils.DEFAULT_MAIN_DICT) type
        else stringResource(R.string.main_dictionary)
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(title, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
        DeleteButton { showDeleteDialog = true }
        ExpandButton { showDetails = !showDetails }
    }
    // default animations look better but make the dialog flash, see also MultiSliderPreference
    AnimatedVisibility(showDetails, enter = fadeIn(), exit = fadeOut()) {
        Text(
            header.info(LocalConfiguration.current.locale()),
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(start = 10.dp, top = 0.dp, end = 10.dp, bottom = 12.dp)
        )
    }
    if (showDeleteDialog) {
        val context = LocalContext.current
        ConfirmationDialog(
            onDismissRequest = { showDeleteDialog = false },
            confirmButtonText = stringResource(R.string.remove),
            onConfirmed = {
                dict.delete()
                context.sendBroadcast(Intent(DictionaryPackConstants.NEW_DICTIONARY_INTENT_ACTION))
            },
            content = { Text(stringResource(R.string.remove_dictionary_message, type)) }
        )
    }
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        DictionaryDialog({}, Locale.ENGLISH)
    }
}
