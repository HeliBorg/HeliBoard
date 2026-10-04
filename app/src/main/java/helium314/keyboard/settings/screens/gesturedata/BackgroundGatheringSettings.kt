// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens.gesturedata

import android.graphics.drawable.Drawable
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.exclude
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.toMutableStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.LinkInteractionListener
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.fromHtml
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.withLink
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.edit
import androidx.core.graphics.drawable.toBitmap
import helium314.keyboard.latin.AppsManager
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.DeleteButton
import helium314.keyboard.latin.utils.GestureDataGatheringSettings
import helium314.keyboard.latin.utils.dpToPx
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.dialogs.ThreeButtonAlertDialog
import kotlinx.coroutines.launch
import kotlin.collections.plus
import androidx.core.graphics.toColorInt
import com.android.inputmethod.latin.BinaryDictionary
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Links
import helium314.keyboard.latin.dictionary.DictionaryFactory
import helium314.keyboard.latin.dictionary.ReadOnlyBinaryDictionary
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.GestureDataGatheringSettings.getAppExclusions
import helium314.keyboard.latin.utils.GestureDataGatheringSettings.getAppIncludeByDefault
import helium314.keyboard.latin.utils.ToolbarKey
import helium314.keyboard.latin.utils.defaultToolbarPref
import helium314.keyboard.latin.utils.getKnownDictHashes
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.dialogs.InfoDialog
import helium314.keyboard.settings.painterResourceCompat
import kotlinx.coroutines.Dispatchers
import kotlin.text.split

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

@Composable
fun BackgroundGatheringSettings() {
    val ctx = LocalContext.current
    var backgroundGathering by remember { mutableStateOf(GestureDataGatheringSettings.isBackgroundGatheringEnabled(ctx.prefs())) }
    var backgroundGatheringManuelSave by remember { mutableStateOf(GestureDataGatheringSettings.isDiscardByDefault(ctx)) }
    var showInfoDialog by remember { mutableStateOf(false) }
    var showExcludedWordsDialog by remember { mutableStateOf(false) }
    var showIncludedAppsDialog by remember { mutableStateOf(false) }
    var showFirstStartDialog by remember { mutableStateOf(false) }
    var showNoDictDialog by remember { mutableStateOf(false) }
    var packageInfos by remember { mutableStateOf(emptyList<Triple<String, String, Drawable?>>()) }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier
            .clickable { backgroundGathering = !backgroundGathering }
            .fillMaxWidth()
    ) {
        Column {
            Text(stringResource(R.string.gesture_data_background_gathering_switch))
            // only use installed apps to avoid miscounting
            val filteredAppExclusionCount = getAppExclusions(ctx).count { ex -> packageInfos.any { it.first == ex } }
            val allowedCount = if (getAppIncludeByDefault(ctx)) packageInfos.size - filteredAppExclusionCount
                else filteredAppExclusionCount
            val allowedCountText = if (packageInfos.isEmpty()) "" else allowedCount.toString()
            Text(stringResource(R.string.gesture_data_background_gathering_allowed_apps, allowedCountText), style = MaterialTheme.typography.bodySmall)
        }
        Switch(
            checked = backgroundGathering,
            onCheckedChange = {
                if (!GestureDataGatheringSettings.hasBackgroundGatheringPref(ctx.prefs()))
                    showFirstStartDialog = true
                if (it) {
                    val usedMainDicts = DictionaryFactory.getMainDictsForLocale(ctx, Settings.getValues().mLocale, false)
                    val hashes = getKnownDictHashes(ctx)
                    if (usedMainDicts.none { ((it as? BinaryDictionary)?.hash ?: (it as? ReadOnlyBinaryDictionary)?.hash) in hashes })
                        showNoDictDialog = true
                }
                backgroundGathering = it
                GestureDataGatheringSettings.setBackgroundGatheringEnabled(ctx.prefs(), it)
            }
        )
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier
            .clickable { backgroundGatheringManuelSave = !backgroundGatheringManuelSave }
            .fillMaxWidth()
    ) {
        Column {
            Text(stringResource(R.string.gesture_data_background_gathering_manual_save))
            Text(stringResource(R.string.gesture_data_background_gathering_manual_save_summary), style = MaterialTheme.typography.bodySmall)
        }
        Switch(backgroundGatheringManuelSave, { backgroundGatheringManuelSave = it; GestureDataGatheringSettings.setDiscardByDefault(ctx, it) })
    }
    ButtonWithText(stringResource(R.string.gesture_data_background_gathering_info), Modifier.fillMaxWidth()) { showInfoDialog = true }
    ButtonWithText(stringResource(R.string.gesture_data_background_excluded_words_button), Modifier.fillMaxWidth()) { showExcludedWordsDialog = true }
    ButtonWithText(stringResource(R.string.gesture_data_background_apps_button), Modifier.fillMaxWidth()) { showIncludedAppsDialog = true }
    if (showInfoDialog) {
        var indicatorInfo by remember { mutableStateOf(false) }
        var controlInfo by remember { mutableStateOf(false) }
        var reviewInfo by remember { mutableStateOf(false) }
        ThreeButtonAlertDialog(
            onDismissRequest = { showInfoDialog = false },
            title = { Text(stringResource(R.string.background_gathering)) },
            content = {
                Column {
                    Text(stringResource(R.string.gesture_data_background_gathering_info_message))
                    TextButton({ indicatorInfo = !indicatorInfo }) {
                        Text(stringResource(R.string.gesture_data_background_gathering_indicator))
                    }
                    AnimatedVisibility(indicatorInfo) {
                        val color = Color("#a00000".toColorInt()) // same as in input_view.xml
                        Column {
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                Icon(painterResourceCompat(R.drawable.btn_keyboard_key_action_normal_lxx_base, 24), null, tint = color)
                                Icon(painterResourceCompat(R.drawable.ring, 24), null, tint = color)
                            }
                            val text = stringResource(
                                R.string.gesture_data_background_gathering_indicator_message,
                                stringResource(R.string.gesture_data_background_gathering_control),
                                stringResource(R.string.gesture_data_background_gathering_manual_save),
                            )
                            Text(AnnotatedString.fromHtml(text))
                        }
                    }
                    TextButton({ controlInfo = !controlInfo }) {
                        Text(stringResource(R.string.gesture_data_background_gathering_control))
                    }
                    AnimatedVisibility(controlInfo) {
                        val text = stringResource(R.string.gesture_data_background_gathering_control_message,
                            stringResource(R.string.gesture_data_background_apps_button),
                            stringResource(R.string.gesture_data_background_excluded_words_button),
                            stringResource(R.string.gesture_data_background_gathering_manual_save),
                            stringResource(R.string.background_gathering),
                            Links.DICTIONARY_URL
                        )
                        Text(AnnotatedString.fromHtml(text))
                    }
                    TextButton({ reviewInfo = !reviewInfo }) {
                        val text = stringResource(R.string.gesture_data_review_screen_title,
                            stringResource(R.string.gesture_data_background_apps_button),
                            stringResource(R.string.gesture_data_background_excluded_words_button)
                        )
                        Text(stringResource(R.string.gesture_data_review_screen_title))
                    }
                    AnimatedVisibility(reviewInfo) {
                        val text = stringResource(
                            R.string.gesture_data_background_gathering_review_message,
                            stringResource(R.string.gesture_data_review_screen_title),
                            Links.SWIPE_O_SCOPE
                        )
                        Text(AnnotatedString.fromHtml(text))
                    }
                }
            },
            scrollContent = true,
            cancelButtonText = stringResource(android.R.string.ok),
            onConfirmed = { },
            confirmButtonText = null
        )
    }
    val scope = rememberCoroutineScope()
    LaunchedEffect(packageInfos) {
        if (packageInfos.isEmpty())
            scope.launch(Dispatchers.IO) { packageInfos = AppsManager(ctx).getPackagesWithNameAndIcon() }
    }
    if (showIncludedAppsDialog) {
        var defaultInclude by remember { mutableStateOf(getAppIncludeByDefault(ctx)) }
        var excludedPackages by remember { mutableStateOf(getAppExclusions(ctx)) }
        var sortedPackagesAndNames by remember { mutableStateOf(
            packageInfos
                .sortedWith( compareBy({ it.first !in excludedPackages }, { it.second.lowercase() }))
        ) }
        LaunchedEffect(packageInfos) {
            sortedPackagesAndNames = packageInfos
                .sortedWith( compareBy({ it.first !in excludedPackages }, { it.second.lowercase() }))
        }
        var filter by remember { mutableStateOf(TextFieldValue()) }
        ThreeButtonAlertDialog(
            title = { Text(stringResource(R.string.gesture_data_background_apps)) },
            onDismissRequest = {
                GestureDataGatheringSettings.setAppExclusions(ctx, excludedPackages)
                showIncludedAppsDialog = false
            },
            content = { Column {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(stringResource(R.string.gesture_data_background_apps_include_default))
                    Switch(checked = defaultInclude, onCheckedChange = { defaultInclude = it; GestureDataGatheringSettings.setAppIncludeByDefault(ctx, it) })
                }
                TextField(
                    value = filter,
                    onValueChange = { filter = it },
                    singleLine = true,
                    label = { Text(stringResource(R.string.label_search_key)) },
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search)
                )
                Spacer(Modifier.height(6.dp))
                LazyColumn {
                    val filtered = sortedPackagesAndNames.filter {
                        if (filter.text.lowercase() == filter.text)
                            filter.text in it.first || filter.text in it.second.lowercase()
                        else
                            filter.text in it.second
                    }
                    items(filtered, { it.first }) { (packageName, name, icon) ->
                        val included = if (defaultInclude) packageName !in excludedPackages else packageName in excludedPackages
                        Row(Modifier
                            .fillMaxWidth()
                            .clickable {
                                excludedPackages = if (included == defaultInclude) excludedPackages + packageName
                                else excludedPackages - packageName
                            }
                            .padding(vertical = 5.dp),
                            Arrangement.spacedBy(6.dp),
                            Alignment.CenterVertically
                        ) {
                            Box(Modifier.size(32.dp)) {
                                if (icon != null) {
                                    val px = 32.dpToPx(LocalResources.current)
                                    Image(icon.toBitmap(px, px).asImageBitmap(), name)
                                }
                            }
                            Column(Modifier.weight(1f)) {
                                CompositionLocalProvider(LocalTextStyle provides MaterialTheme.typography.bodyLarge) {
                                    Text(name)
                                }
                                CompositionLocalProvider(LocalTextStyle provides MaterialTheme.typography.bodyMedium) {
                                    Text(
                                        packageName,
                                        color = if (included) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                                    )
                                }
                            }
                            if (included)
                                Icon(painterResource(R.drawable.ic_setup_check), "included", Modifier.size(32.dp), MaterialTheme.colorScheme.primary)
                            else
                                Icon(painterResource(R.drawable.ic_close), "ignored", Modifier.size(32.dp), MaterialTheme.colorScheme.error)
                        }
                    }
                }
            } },
            cancelButtonText = stringResource(R.string.dialog_close),
            onConfirmed = { },
            confirmButtonText = null,
            properties = DialogProperties(dismissOnClickOutside = false, dismissOnBackPress = false)
        )
    }
    if (showExcludedWordsDialog) {
        ExcludedWordsDialog { showExcludedWordsDialog = false }
    }
    if (showFirstStartDialog) {
        ConfirmationDialog(
            onDismissRequest = { showFirstStartDialog = false },
            cancelButtonText = stringResource(R.string.dialog_close),
            confirmButtonText = stringResource(R.string.gesture_data_add_key),
            onConfirmed = {
                val toolbar = ctx.prefs().getString(Settings.PREF_TOOLBAR_KEYS, defaultToolbarPref)!!
                    .split(Separators.ENTRY)
                    .filter { ToolbarKey.BACKGROUND_GATHERING.name !in it }
                val newToolbar = toolbar + (ToolbarKey.BACKGROUND_GATHERING.name + Separators.KV + "true")
                ctx.prefs().edit { putString(Settings.PREF_TOOLBAR_KEYS, newToolbar.joinToString(Separators.ENTRY)) }
                KeyboardSwitcher.getInstance().setThemeNeedsReload()
            },
            content = {
                val splitText = stringResource(
                    R.string.gesture_data_first_enable_dialog,
                    stringResource(R.string.gesture_data_background_gathering_info),
                    stringResource(R.string.background_gathering),
                    "SPLITHERE"
                ).split("SPLITHERE")
                val text = buildAnnotatedString {
                    // there probably is an easier way for this...
                    append(AnnotatedString.fromHtml(splitText[0]))
                    if (splitText.size == 1) return@buildAnnotatedString
                    withLink(LinkAnnotation.Clickable("tag", linkInteractionListener = LinkInteractionListener { showIncludedAppsDialog = true })) {
                        append(stringResource(R.string.dictionary_link_text))
                    }
                    append(AnnotatedString.fromHtml(splitText[1]))
                }
                Text(text)
            },
        )
    }
    if (showNoDictDialog) {
        // inform user that there is no known dictionary for the current locale, so background gathering will not collect any words
        val text = stringResource(R.string.gesture_data_no_known_dict, Links.DICTIONARY_URL)
        InfoDialog(AnnotatedString.fromHtml(text)) { showNoDictDialog = false }
    }
}

@Composable fun ExcludedWordsDialog(onDismissRequest: () -> Unit) {
    val ctx = LocalContext.current
    val ignoreWords = remember { GestureDataGatheringSettings.getWordExclusions(ctx).toMutableStateList() }
    var newWord by remember { mutableStateOf(TextFieldValue()) }
    var error by remember { mutableStateOf(true) }
    LaunchedEffect(newWord) {
        // we want a letter because it should be a word, and no quote so it doesn't interfere with the simple json contains check for excluded words
        error = '"' in newWord.text || newWord.text.none { it.isLetter() }
    }
    val scroll = rememberScrollState()
    val scope = rememberCoroutineScope()
    fun addWord() {
        val word = newWord.text
        if (word.isNotBlank()) {
            ignoreWords.add(0, word.trim())
            scope.launch { scroll.scrollTo(0) }
        }
        newWord = TextFieldValue()
    }
    ThreeButtonAlertDialog(
        onDismissRequest = {
            addWord()
            GestureDataGatheringSettings.setWordExclusions(ctx, ignoreWords)
            onDismissRequest()
        },
        modifier = Modifier.windowInsetsPadding(WindowInsets.ime.exclude(WindowInsets.systemBars)),
        content = { Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextField(
                    value = newWord,
                    onValueChange = { newWord = it },
                    modifier = Modifier.weight(1f),
                    isError = error && newWord.text.isNotEmpty(),
                    singleLine = true,
                    label = { Text(stringResource(R.string.user_dict_add_word_button)) },
                    keyboardActions = KeyboardActions { addWord() }
                )
                IconButton(
                    { if (!error) addWord() },
                    Modifier.weight(0.2f)
                ) {
                    val tint = if (error) MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f) else MaterialTheme.colorScheme.onSurface
                    Icon(painterResource(R.drawable.ic_plus), stringResource(R.string.add), tint = tint)
                }
            }
            CompositionLocalProvider(
                LocalTextStyle provides MaterialTheme.typography.bodyLarge
            ) {
                Column(Modifier.verticalScroll(scroll)) {
                    ignoreWords.forEach { word ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(word)
                            DeleteButton { ignoreWords.removeAll { word == it } }
                        }
                    }
                }
            }
        } },
        onConfirmed = { },
        confirmButtonText = null,
        cancelButtonText = stringResource(android.R.string.ok),
        neutralButtonText = stringResource(android.R.string.cancel),
        onNeutral = onDismissRequest,
        properties = DialogProperties(dismissOnClickOutside = false, dismissOnBackPress = false)
    )
}
