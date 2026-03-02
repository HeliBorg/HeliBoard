// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.content.ContentResolver
import android.content.Context
import android.database.Cursor
import android.provider.UserDictionary
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.LocaleUtils.localizedDisplayName
import helium314.keyboard.settings.DropDownField
import helium314.keyboard.settings.SearchScreen
import helium314.keyboard.settings.dialogs.ThreeButtonAlertDialog
import java.util.Locale

@Composable
fun PersonalDictionaryScreen(
    onClickBack: () -> Unit,
    locale: Locale?
) {
    val words = getAll(locale, LocalContext.current)
    var selectedWord: Word? by remember { mutableStateOf(null) }
    SearchScreen(
        onClickBack = onClickBack,
        title = {
            Column {
                Text(stringResource(R.string.edit_personal_dictionary))
                Text(
                    locale.getLocaleDisplayNameForUserDictSettings(LocalContext.current),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        },
        filteredItems = { term ->
            // we could maybe to this using a query and getting items by position
            // requires adjusting the SearchScreen, likely not worth the effort
            words.filter { it.word.startsWith(term, true) || it.shortcut?.startsWith(term, true) == true }
        },
        itemContent = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { selectedWord = it }
                    .padding(vertical = 6.dp, horizontal = 16.dp)
            ) {
                Column {
                    Text(it.word, style = MaterialTheme.typography.bodyLarge)
                    val details = if (it.shortcut == null) it.weight.toString() else "${it.weight}  |  ${it.shortcut}"
                    Text(details, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Icon(painterResource(R.drawable.ic_edit), stringResource(R.string.user_dict_settings_edit_dialog_title))
            }
        }
    )
    if (selectedWord != null) {
        EditWordDialog(selectedWord!!, locale) { selectedWord = null }
    }
    ExtendedFloatingActionButton(
        onClick = { selectedWord = Word("", null, null) },
        text = { Text(stringResource(R.string.user_dict_add_word_button)) },
        icon = { Icon(painter = painterResource(R.drawable.ic_edit), stringResource(R.string.user_dict_add_word_button)) },
        modifier = Modifier.wrapContentSize(Alignment.BottomEnd).padding(all = 12.dp)
            .then(Modifier.safeDrawingPadding())
    )
}

@Composable
private fun EditWordDialog(word: Word, locale: Locale?, onDismissRequest: () -> Unit) {
    val ctx = LocalContext.current
    val focusRequester = remember { FocusRequester() }
    var newWord by remember { mutableStateOf(word) }
    var newLocale by remember { mutableStateOf(locale) }
    val wordValid = (newWord.word == word.word && locale == newLocale) || !doesWordExist(newWord.word, newLocale, ctx)
    fun save() {
        if (newWord != word || locale != newLocale) {
            deleteWord(word, locale, ctx.contentResolver)
            val saveWeight = newWord.weight ?: WEIGHT_FOR_USER_DICTIONARY_ADDS
            UserDictionary.Words.addWord(ctx, newWord.word, saveWeight, newWord.shortcut, newLocale)
        }
    }
    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = { save() },
        checkOk = { newWord.word.isNotBlank() && wordValid },
        confirmButtonText = stringResource(R.string.save),
        neutralButtonText = stringResource(R.string.delete),
        onNeutral = {
            deleteWord(word, locale, ctx.contentResolver) // delete the originally selected word
            onDismissRequest()
        },
        title = {
            Column {
                Text(stringResource(R.string.user_dict_settings_edit_dialog_title))
                Text(
                    locale.getLocaleDisplayNameForUserDictSettings(ctx),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        },
        content = {
            LaunchedEffect(word) {
                if (word.word == "" && word.weight == null && word.shortcut == null)
                    focusRequester.requestFocus() // user clicked add word
            }
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                TextField(
                    value = newWord.word,
                    onValueChange = { newWord = newWord.copy(word = it) },
                    modifier = Modifier.fillMaxWidth().focusRequester(focusRequester),
                    singleLine = true,
                    label = { Text(stringResource(R.string.user_dict_settings_add_word_hint))},
                    keyboardActions = KeyboardActions {
                        if (newWord.word.isNotBlank() && wordValid) {
                            save()
                            onDismissRequest()
                        }
                    }
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.user_dict_settings_add_shortcut_option_name), Modifier.fillMaxWidth(0.3f))
                    TextField(
                        value = newWord.shortcut ?: "",
                        onValueChange = { newWord = newWord.copy(shortcut = it.ifBlank { null }) },
                        label = { Text(stringResource(R.string.user_dict_settings_add_shortcut_hint))},
                        modifier = Modifier.weight(1f),
                        singleLine = true
                    )
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.user_dict_settings_add_weight_value), Modifier.fillMaxWidth(0.3f))
                    TextField(
                        newWord.weight?.toString() ?: "",
                        {
                            if (it.isBlank())
                                newWord = newWord.copy(weight = null)
                            else if ((it.toIntOrNull() ?: -1) in 0..255)
                                newWord = newWord.copy(weight = it.toInt())
                        },
                        label = { Text(WEIGHT_FOR_USER_DICTIONARY_ADDS.toString()) },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f),
                        singleLine = true
                    )
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.user_dict_settings_add_locale_option_name), Modifier.fillMaxWidth(0.3f))
                    DropDownField(
                        items = getSpecificallySortedLocales(locale),
                        selectedItem = newLocale,
                        onSelected = { newLocale = it },
                    ) {
                        Text(it.getLocaleDisplayNameForUserDictSettings(ctx))
                    }
                }
                if (!wordValid)
                    Text(
                        stringResource(R.string.user_dict_word_already_present, newLocale.getLocaleDisplayNameForUserDictSettings(ctx)),
                        color = MaterialTheme.colorScheme.error
                    )
            }
        }
    )
}

private fun deleteWord(wordDetails: Word, locale: Locale?, resolver: ContentResolver) {
    val (word, shortcut, weightInt) = wordDetails
    val weight = weightInt.toString()
    if (shortcut.isNullOrBlank()) {
        if (locale == null) {
            resolver.delete(
                UserDictionary.Words.CONTENT_URI, DELETE_SELECTION_WITHOUT_SHORTCUT_AND_WITH_ALL_LOCALES,
                arrayOf(word, weight)
            )
        } else {
            resolver.delete( // requires use of locale string for interaction with Android system
                UserDictionary.Words.CONTENT_URI, DELETE_SELECTION_WITHOUT_SHORTCUT_AND_WITH_LOCALE,
                arrayOf(word, weight, locale.toString())
            )
        }
    } else {
        if (locale == null) {
            resolver.delete(
                UserDictionary.Words.CONTENT_URI, DELETE_SELECTION_WITH_SHORTCUT_AND_WITH_ALL_LOCALES,
                arrayOf(word, shortcut, weight)
            )
        } else {
            resolver.delete( // requires use of locale string for interaction with Android system
                UserDictionary.Words.CONTENT_URI, DELETE_SELECTION_WITH_SHORTCUT_AND_WITH_LOCALE,
                arrayOf(word, shortcut, weight, locale.toString())
            )
        }
    }
}

private fun doesWordExist(word: String, locale: Locale?, context: Context): Boolean {
    val hasWordProjection = arrayOf(UserDictionary.Words.WORD, UserDictionary.Words.LOCALE)

    val select: String
    val selectArgs: Array<String>?
    if (locale == null) {
        select = "${UserDictionary.Words.WORD}=? AND ${UserDictionary.Words.LOCALE} is null"
        selectArgs = arrayOf(word)
    } else {
        select = "${UserDictionary.Words.WORD}=? AND ${UserDictionary.Words.LOCALE}=?"
        // requires use of locale string (as opposed to more useful language tag) for interaction with Android system
        selectArgs = arrayOf(word, locale.toString())
    }
    val cursor = context.contentResolver.query(UserDictionary.Words.CONTENT_URI, hasWordProjection, select, selectArgs, null)
    cursor.use {
        if (null == it) return false
        return it.count > 0
    }
}

private fun getSpecificallySortedLocales(firstLocale: Locale?): List<Locale?> {
    val list: MutableList<Locale?> = getSortedDictionaryLocales().toMutableList()
    list.remove(firstLocale)
    list.remove(null)
    list.add(0, firstLocale)
    if (firstLocale != null)
        list.add(null)
    return list
}

fun Locale?.getLocaleDisplayNameForUserDictSettings(context: Context) =
    this?.localizedDisplayName(context.resources) ?: context.resources.getString(R.string.user_dict_settings_all_languages)

// weight is frequency but different name towards user
private data class Word(val word: String, val shortcut: String?, val weight: Int?)

// getting all words instead of reading directly cursor, because filteredItems expects a list
private fun getAll(locale: Locale?, context: Context): List<Word> {
    val cursor = createCursor(locale, context) ?: return emptyList()

    if (!cursor.moveToFirst()) return emptyList()
    val result = mutableListOf<Word>()
    val wordIndex = cursor.getColumnIndexOrThrow(UserDictionary.Words.WORD)
    val shortcutIndex = cursor.getColumnIndexOrThrow(UserDictionary.Words.SHORTCUT)
    val frequencyIndex = cursor.getColumnIndexOrThrow(UserDictionary.Words.FREQUENCY)
    while (!cursor.isAfterLast) {
        result.add(Word(cursor.getString(wordIndex), cursor.getString(shortcutIndex), cursor.getInt(frequencyIndex)))
        cursor.moveToNext()
    }
    cursor.close()
    return result
}

private fun createCursor(locale: Locale?, context: Context): Cursor? {
    // locale can be any of:
    // - An actual locale, for use of Locale#toString()
    // - The emptyLocale. This means we want a cursor returning words valid for all locales.

    // Note that this contrasts with the data inside the database, where NULL means "all
    // locales" and there should never be an empty string.
    // The confusion is called by the historical use of null for "all locales".

    val select: String
    val selectArgs: Array<String>?
    if (locale == null) {
        select = QUERY_SELECTION_ALL_LOCALES
        selectArgs = null
    } else {
        select = QUERY_SELECTION
        // requires use of locale string (as opposed to more useful language tag) for interaction with Android system
        selectArgs = arrayOf(locale.toString())
    }

    return context.contentResolver.query(
        UserDictionary.Words.CONTENT_URI, QUERY_PROJECTION, select, selectArgs, SORT_ORDER
    )
}

private val QUERY_PROJECTION =
    arrayOf(UserDictionary.Words._ID, UserDictionary.Words.WORD, UserDictionary.Words.SHORTCUT, UserDictionary.Words.FREQUENCY)
// Case-insensitive sort
private const val SORT_ORDER = "UPPER(" + UserDictionary.Words.WORD + ")"

// Either the locale is empty (means the word is applicable to all locales)
// or the word equals our current locale
private const val QUERY_SELECTION = UserDictionary.Words.LOCALE + "=?"
private const val QUERY_SELECTION_ALL_LOCALES = UserDictionary.Words.LOCALE + " is null"

private const val DELETE_SELECTION_WITH_SHORTCUT_AND_WITH_LOCALE = (UserDictionary.Words.WORD + "=? AND "
        + UserDictionary.Words.SHORTCUT + "=? AND "
        + UserDictionary.Words.FREQUENCY + "=? AND "
        + UserDictionary.Words.LOCALE + "=?")

private const val DELETE_SELECTION_WITH_SHORTCUT_AND_WITH_ALL_LOCALES = (UserDictionary.Words.WORD + "=? AND "
        + UserDictionary.Words.SHORTCUT + "=? AND "
        + UserDictionary.Words.FREQUENCY + "=? AND "
        + UserDictionary.Words.LOCALE + " is null")

private const val DELETE_SELECTION_WITHOUT_SHORTCUT_AND_WITH_LOCALE = (UserDictionary.Words.WORD + "=? AND "
        + UserDictionary.Words.SHORTCUT + " is null AND "
        + UserDictionary.Words.FREQUENCY + "=? AND "
        + UserDictionary.Words.LOCALE + "=? OR "

        + UserDictionary.Words.SHORTCUT + "='' AND "
        + UserDictionary.Words.FREQUENCY + "=? AND "
        + UserDictionary.Words.LOCALE + "=?")

private const val DELETE_SELECTION_WITHOUT_SHORTCUT_AND_WITH_ALL_LOCALES = (UserDictionary.Words.WORD + "=? AND "
        + UserDictionary.Words.SHORTCUT + " is null AND "
        + UserDictionary.Words.FREQUENCY + "=? AND "
        + UserDictionary.Words.LOCALE + " is null OR "

        + UserDictionary.Words.SHORTCUT + "='' AND "
        + UserDictionary.Words.FREQUENCY + "=? AND "
        + UserDictionary.Words.LOCALE + " is null")

private const val WEIGHT_FOR_USER_DICTIONARY_ADDS = 250
