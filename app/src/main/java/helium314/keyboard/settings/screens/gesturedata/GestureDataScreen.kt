// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens.gesturedata

import android.content.Context
import android.view.inputmethod.InputMethodManager
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.fromHtml
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.PlatformImeOptions
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.intl.LocaleList
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.android.inputmethod.latin.BinaryDictionary
import helium314.keyboard.compat.locale
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.R
import helium314.keyboard.latin.SingleDictionaryFacilitator
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Links
import helium314.keyboard.latin.common.LocaleUtils
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ChecksumCalculator
import helium314.keyboard.latin.utils.DictionaryInfoUtils
import helium314.keyboard.latin.utils.GestureDataDao
import helium314.keyboard.latin.utils.ScriptUtils
import helium314.keyboard.latin.utils.SubtypeLocaleUtils
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.SuggestionResults
import helium314.keyboard.latin.utils.UncachedInputMethodManagerUtils
import helium314.keyboard.latin.utils.WordData
import helium314.keyboard.latin.utils.dictTestImeOption
import helium314.keyboard.latin.utils.gestureDataActiveFacilitator
import helium314.keyboard.latin.utils.getExportedActiveDeletionCount
import helium314.keyboard.latin.utils.getSecondaryLocales
import helium314.keyboard.latin.utils.locale
import helium314.keyboard.settings.DropDownField
import helium314.keyboard.settings.NextScreenIcon
import helium314.keyboard.settings.Theme
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.dialogs.InfoDialog
import helium314.keyboard.settings.dialogs.ThreeButtonAlertDialog
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.isWideScreen
import helium314.keyboard.settings.previewDark
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.text.DateFormat
import java.util.Date
import java.util.Locale
import kotlin.collections.plus
import kotlin.random.Random

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished
/**
 *  Simple "settings" screen that shows up when gesture typing is enabled.
 *  Allows "active data gathering", which is input of gesture typing data,
 *  and getting results in a file / sending as mail.
 *  This mode is rather simple: the user chooses a dictionary, and get a
 *  random word. After swiping this word it gets stored along with necessary
 *  information to recreate the input and some additional data.
 *
 *  Will allow enabling "passive data gathering" later, which stores most of
 *  the words entered using gesture typing. Here the user needs the ability
 *  review and redact the data before sending, and additionally exclude some
 *  words and apps from passive gathering.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GestureDataScreen(
    onClickBack: () -> Unit,
) {
    val ctx = LocalContext.current
    val useWideLayout = isWideScreen()
    val dao = GestureDataDao.getInstance(ctx)!!

    // ideally we'd move all the active gathering stuff into a separate (non-local) function,
    // but either it has issues with the floating button positioning (if they are in the function)
    // or the keyboard flashes during recomposition if they are outside the function
    var wordFromDict by remember { mutableStateOf<String?>(null) } // some word from the dictionary
    var lastData by remember { mutableStateOf<WordData?>(null) }
    var sessionWordCount by remember { mutableIntStateOf(0) }
    var dbActiveWordCount by remember { mutableIntStateOf(dao.count(activeMode = true)) }
    var showUploadDialog by rememberSaveable { mutableStateOf(true) }
    var showEndDialog by rememberSaveable { mutableStateOf(true) }
    val focusRequester = remember { FocusRequester() }
    val keyboard = LocalSoftwareKeyboardController.current
    val words = remember { mutableListOf<Pair<String, Long>>() }
    val scope = rememberCoroutineScope()
    fun nextWord(save: Boolean) {
        if (!save) {
            lastData = null
        }
        lastData?.let { scope.launch {
            ++sessionWordCount
            it.save(ctx)
        } }
        wordFromDict = getRandomWord(words)
        lastData = null
        // reset the data
        focusRequester.requestFocus()
        keyboard?.show()
    }
    if (showEndDialog) {
        if (System.currentTimeMillis() < END_DATE_EPOCH_MILLIS - TWO_WEEKS_IN_MILLIS)
            showEndDialog = false
        val endDate = DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(END_DATE_EPOCH_MILLIS))
        if (System.currentTimeMillis() > END_DATE_EPOCH_MILLIS) {
            val message = stringResource(R.string.gesture_data_ended, endDate)
            val infos = dao.filterInfos(limit = 10000) // export no more than 10k words at once due to possibly hitting mail size limits
            ThreeButtonAlertDialog(
                onDismissRequest = onClickBack,
                content = {
                    Column {
                        Text(message)
                        if (infos.isNotEmpty())
                            ShareGestureData(infos.map { it.id })
                    }
                },
                cancelButtonText = stringResource(android.R.string.ok),
                onConfirmed = { },
                confirmButtonText = null
            )
        } else {
            val message = stringResource(R.string.gesture_data_ends_at, endDate)
            InfoDialog(message) { showEndDialog = false }
        }

    }
    if (showUploadDialog) {
        if (dao.count() < 10000)
            showUploadDialog = false
        InfoDialog(stringResource(R.string.gesture_data_much_data)) { showUploadDialog = false }
    }
    @Composable fun activeGathering() {
        val availableDicts = remember { getAvailableDictionaries(ctx) }
        val currentLocale = Settings.getValues().mLocale
        var dict by remember { mutableStateOf(
            LocaleUtils.getBestMatch(currentLocale, availableDicts) { it.locale }
        ) }
        LaunchedEffect(wordFromDict) { focusRequester.requestFocus() }
        val suggestionLogger = remember {
            object : SingleDictionaryFacilitator.Companion.SuggestionLogger {
                override fun onNewSuggestions(
                    suggestions: SuggestionResults,
                    composedData: ComposedData,
                    ngramContext: NgramContext,
                    keyboard: Keyboard,
                    inputStyle: Int
                ) {
                    if (!composedData.mIsBatchMode || inputStyle != SuggestedWords.INPUT_STYLE_TAIL_BATCH) return
                    val target = wordFromDict ?: return
                    val newData = WordData(target, suggestions, composedData, ngramContext, keyboard, inputStyle, true)
                    if (suggestions.any { it.mWord == target && it.mScore >= 0 }) { // just not negative should be fine
                        scope.launch {
                            ++sessionWordCount
                            newData.save(ctx)
                        }
                        nextWord(false)
                    } else {
                        lastData = newData // use the old flow with button
                    }
                }
            }
        }
        LaunchedEffect(dict) {
            val dict = dict ?: return@LaunchedEffect
            gestureDataActiveFacilitator?.closeDictionaries()
            gestureDataActiveFacilitator = SingleDictionaryFacilitator(dict.getDictionary(ctx))
            gestureDataActiveFacilitator?.suggestionLogger = suggestionLogger
            lastData = null
            wordFromDict = null
            scope.launch(Dispatchers.Default) {
                words.clear()
                dict.addWords(ctx, words)
            }
            scope.launch(Dispatchers.Default) {
                delay(500)
                var i = 0
                while (words.isEmpty() && i++ < 20)
                    delay(50)
                // at least a few words should be loaded now
                nextWord(false)
            }
        }

        @Composable fun dictsBox() {
            Box { // without the box the menu appears at the wrong position
                DropDownField(availableDicts, dict, { dict = it }) {
                    val locale = it?.locale?.getDisplayName(LocalConfiguration.current.locale())
                    val internal = if (it?.internal == true)
                            " (${stringResource(R.string.internal_dictionary_summary)})"
                        else ""
                    Text(locale?.let { loc -> "$loc$internal" } ?: stringResource(R.string.dictionary_load_error))
                }
            }
        }
        @Composable fun ColumnScope.texts() {
            val imm = ctx.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            val text = when {
                !UncachedInputMethodManagerUtils.isThisImeCurrent(ctx, imm) -> "please switch to HeliBoard"
                else -> stringResource(R.string.gesture_data_please_type)
            }
            if (useWideLayout) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = text,
                        modifier = Modifier.alpha(if (wordFromDict == null) 0.5f else 1f)
                    )
                    CompositionLocalProvider(
                        LocalTextStyle provides MaterialTheme.typography.titleLarge
                    ) {
                        Text(
                            text = wordFromDict ?: "",
                            modifier = Modifier.fillMaxWidth(0.4f).padding(vertical = 10.dp),
                            textAlign = TextAlign.Center
                        )
                    }
                }
            } else {
                Text(
                    text = text,
                    modifier = Modifier.alpha(if (wordFromDict == null) 0.5f else 1f)
                )
                CompositionLocalProvider(
                    LocalTextStyle provides MaterialTheme.typography.titleLarge
                ) {
                    Text(
                        text = wordFromDict ?: "",
                        modifier = Modifier.fillMaxWidth().padding(vertical = 10.dp),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
        if (!useWideLayout)
            dictsBox()
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            Column {
                texts()
                val exportedAndDeletedCount by remember { mutableIntStateOf(getExportedActiveDeletionCount(ctx)) }
                val oldActiveWords by remember {
                    sessionWordCount = 0
                    dbActiveWordCount = dao.count(activeMode = true)
                    mutableIntStateOf(dbActiveWordCount + exportedAndDeletedCount)
                }
                Text(stringResource(R.string.gesture_data_active_count, sessionWordCount, sessionWordCount + oldActiveWords, exportedAndDeletedCount))
                Box(Modifier.size(1.dp)) { // box hides the field, but we can still interact with it
                    TextField(
                        value = TextFieldValue(),
                        enabled = wordFromDict != null,
                        onValueChange = { },
                        keyboardOptions = KeyboardOptions(
                            platformImeOptions = PlatformImeOptions(privateImeOptions = dictTestImeOption),
                            imeAction = ImeAction.Next,
                            hintLocales = dict?.let { LocaleList(it.locale.toLanguageTag()) }
                        ),
                        keyboardActions = KeyboardActions { nextWord(false) },
                        modifier = Modifier.focusRequester(focusRequester),
                    )
                }
            }
        }
        if (useWideLayout)
            dictsBox()
    }

    val scrollState = rememberScrollState()
    var activeGathering by remember { mutableStateOf(false) }
    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing.only(WindowInsetsSides.Bottom),
        bottomBar = { BottomBar(sessionWordCount + dbActiveWordCount > 0) { dbActiveWordCount = dao.count(activeMode = true) } }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .verticalScroll(scrollState)
                .padding(horizontal = 12.dp)
                .then(Modifier.padding(innerPadding)),
        ) {
            var showActiveInfoDialog by remember { mutableStateOf(false) }
            var showInfoDialog by remember { mutableStateOf(false) }
            var showPrivacyDialog by remember { mutableStateOf(false) }
            TopAppBar(
                title = { Text(stringResource(R.string.gesture_data_screen)) },
                navigationIcon = {
                    IconButton(onClick = { if (activeGathering) activeGathering = false else onClickBack() }) {
                        Icon(
                            painterResource(R.drawable.ic_arrow_back),
                            stringResource(R.string.spoken_description_action_previous)
                        )
                    }
                },
            )
            BackHandler(enabled = activeGathering) { activeGathering = false }
            if (activeGathering) { // AnimatedVisibility results in buggy behavior for some reason
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    ButtonWithText(stringResource(R.string.gesture_data_how_to_use)) {
                        showActiveInfoDialog = true
                    }
                    ButtonWithText(stringResource(R.string.gesture_data_active_stop)) {
                        activeGathering = false
                    }
                }
                activeGathering()
            }
            AnimatedVisibility(!activeGathering) {
                // this part is hidden in active gathering mode because in active mode
                // neither the keyboard nor the floating buttons (!) should cover any text
                Column {
                    ButtonWithText(stringResource(R.string.gesture_data_info), Modifier.fillMaxWidth()) {
                        showInfoDialog = true
                    }
                    ButtonWithText(stringResource(R.string.gesture_data_privacy), Modifier.fillMaxWidth()) {
                        showPrivacyDialog = true
                    }
                    Spacer(Modifier.height(12.dp))
                    HorizontalDivider()
                    ButtonWithText(
                        stringResource(R.string.gesture_data_active_start),
                        Modifier.fillMaxWidth(),
                        System.currentTimeMillis() < END_DATE_EPOCH_MILLIS - TWO_WEEKS_IN_MILLIS // disabled when close to end
                    ) {
                        activeGathering = true
                        lastData = null
                        wordFromDict = null
                    }
                    ButtonWithText(stringResource(R.string.gesture_data_how_to_use), Modifier.fillMaxWidth()) {
                        showActiveInfoDialog = true
                    }
                }
            }
            if (showInfoDialog)
                InfoDialog(AnnotatedString.fromHtml(
                    stringResource(R.string.gesture_data_description,
                        DateFormat.getDateInstance(DateFormat.LONG).format(Date(END_DATE_EPOCH_MILLIS)))
                ) + AnnotatedString("\n\n" + stringResource(R.string.gesture_data_description_modes))) { showInfoDialog = false }
            if (showPrivacyDialog)
                InfoDialog(stringResource(R.string.gesture_data_description_privacy)) { showPrivacyDialog = false }
            if (showActiveInfoDialog)
                InfoDialog(AnnotatedString.fromHtml(stringResource(R.string.gesture_data_active_description, Links.DICTIONARY_URL))) { showActiveInfoDialog = false }
            Spacer(Modifier.height(12.dp))
            // PassiveGathering & Review are not finished and will be completed + enabled later
/*
            HorizontalDivider()
            PassiveGathering()
            Spacer(Modifier.height(12.dp))
            HorizontalDivider()
            // maybe move the review screen content in here if we have enough space (but landscape mode will be bad)
            TextButton(onClick = { SettingsDestination.navigateTo(SettingsDestination.DataReview) }) {
                Text(stringResource(R.string.gesture_data_review_screen_title))
            }
 */
        }
    }
    // showing at top left in preview, but correctly on device
    if (lastData != null && activeGathering)
        ExtendedFloatingActionButton(
            onClick = { nextWord(true) },
            // doesn't look good with the long text
            text = { Text(stringResource(R.string.gesture_data_next_save, lastData?.suggestions?.firstOrNull()?.word.toString())) },
            icon = { NextScreenIcon() },
            modifier = Modifier
                .wrapContentSize(Alignment.BottomEnd)
                .padding(all = 12.dp)
                .then(Modifier.safeDrawingPadding())
                .fillMaxWidth(if (useWideLayout) 0.3f else 0.5f)
        )
    if (wordFromDict != null && activeGathering)
        ExtendedFloatingActionButton(
            onClick = { nextWord(false) },
            text = { Text(stringResource(R.string.gesture_data_next)) },
            icon = { Icon(painter = painterResource(R.drawable.ic_ime_switcher), stringResource(R.string.gesture_data_next)) },
            modifier = Modifier
                .wrapContentSize(Alignment.BottomStart)
                .padding(all = 12.dp)
                .then(Modifier.safeDrawingPadding())
        )
}

@Composable
private fun BottomBar(hasWords: Boolean, onDeleted: () -> Unit) {
    var showExportDialog by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    val dao = GestureDataDao.getInstance(LocalContext.current)!!
    BottomAppBar(
        actions = {
            Row(
                horizontalArrangement = Arrangement.SpaceEvenly,
                modifier = Modifier.fillMaxWidth()
            ) {
                IconButton(
                    onClick = { showDeleteDialog = true},
                    enabled = hasWords
                ) {
                    Icon(
                        painterResource(R.drawable.ic_bin_rounded),
                        stringResource(R.string.delete),
                        Modifier.size(30.dp)
                    )
                }
                IconButton(
                    onClick = { showExportDialog = true},
                    enabled = hasWords
                ) {
                    Icon(
                        painterResource(R.drawable.ic_share),
                        "share",
                        Modifier.size(30.dp)
                    )
                }
            }
        }
    )
    if (showExportDialog) {
        val notExportedCount = dao.count(activeMode = true, exported = false)
        val totalCount = dao.count(activeMode = true)
        var shareAll by remember { mutableStateOf<Boolean?>(null) }
        ThreeButtonAlertDialog(
            onDismissRequest = { showExportDialog = false },
            content = {
                if (shareAll == null) {
                    Column {
                        ButtonWithText(stringResource(R.string.gesture_data_share_new, notExportedCount), enabled = notExportedCount > 0) {
                            shareAll = false
                        }
                        ButtonWithText(stringResource(R.string.gesture_data_share_all, totalCount), enabled = totalCount > 0) {
                            shareAll = true
                        }
                    }
                } else {
                    val toShare = dao.filterInfos(
                        activeMode = true,
                        exported = if (shareAll == true) null else false,
                        limit = 10000 // export no more than 10k words at once due to possibly hitting mail size limits
                    )
                    Column { ShareGestureData(toShare.map { it.id }) }
                }
            },
            cancelButtonText = stringResource(R.string.dialog_close),
            onConfirmed = { },
            confirmButtonText = null
        )
    }
    if (showDeleteDialog) {
        val infos = dao.filterInfos(activeMode = true)
        val exportedCount = infos.count { it.exported }
        val nonExportedCount = infos.size - exportedCount
        var showConfirmDialog by remember { mutableStateOf<String?>(null) }
        ThreeButtonAlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            content = {
                Column {
                    Text(stringResource(R.string.gesture_data_delete_dialog, nonExportedCount, exportedCount))
                    ButtonWithText(stringResource(R.string.gesture_data_delete_dialog_submitted, exportedCount)) { showConfirmDialog = "exported" }
                    ButtonWithText(stringResource(R.string.gesture_data_delete_dialog_all, exportedCount + nonExportedCount)) { showConfirmDialog = "all" }
                }
            },
            cancelButtonText = stringResource(R.string.dialog_close),
            onConfirmed = { },
            confirmButtonText = null,
        )
        if (showConfirmDialog != null) {
            val ctx = LocalContext.current
            ConfirmationDialog(
                onDismissRequest = { showDeleteDialog = false },
                onConfirmed = {
                    val ids = dao.filterInfos(activeMode = true).map { it.id }
                    dao.delete(ids, showConfirmDialog != "all", ctx)
                    onDeleted()
                },
                content = {
                    Text(stringResource(
                        R.string.delete_confirmation,
                        if (showConfirmDialog == "all") (exportedCount + nonExportedCount) else exportedCount
                    ))
                }
            )
        }
    }
}

// the text buttons look ugly, try something different
@Composable
fun ButtonWithText(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    Button(onClick, modifier, enabled) {
        Text(text)
    }
}

// we only check dictionaries for enabled locales (main + secondary)
private fun getAvailableDictionaries(context: Context): List<DictWithInfo> {
    val allowedHashes = context.assets.open("known_dict_hashes.txt")
        .use { it.reader().readLines() }.filterNot { it.isBlank() || it.startsWith("#") }
    val locales = SubtypeSettings.getEnabledSubtypes(true).flatMap {
        getSecondaryLocales(it.extraValue) + it.locale()
    }
    val languages = locales.mapTo(hashSetOf()) { it.language }
    val cached = DictionaryInfoUtils.getCacheDirectories(context)
        .filter { it.name.constructLocale().language in languages }
        .mapNotNull { dir -> dir.listFiles()?.filter {
            it.name.startsWith(DictionaryInfoUtils.DEFAULT_MAIN_DICT)
        } }.flatten().map { CacheDictWithInfo(it) }.filter { it.hash in allowedHashes }
    val assets = DictionaryInfoUtils.getAssetsDictionaryList(context).orEmpty()
        .filter { it.substringAfter("_").substringBefore(".dict").constructLocale().language in languages }
        .map { AssetsDictWithInfo(it, context) }
        .filter { dict -> cached.none { it.hash == dict.hash } && dict.hash in allowedHashes }
    return cached + assets
}

private interface DictWithInfo {
    val hash: String
    val locale: Locale
    val internal: Boolean
    fun getDictionary(context: Context): BinaryDictionary
    // not actually suspending, but makes clear that it shouldn't be called on UI thread (because it's slow)
    suspend fun addWords(context: Context, words: MutableList<Pair<String, Long>>) = getDictionary(context).addWords(words)
}

private class CacheDictWithInfo(private val file: File): DictWithInfo {
    override val locale = (DictionaryInfoUtils.getDictionaryFileHeaderOrNull(file)?.mLocaleString ?: SubtypeLocaleUtils.NO_LANGUAGE)
        .constructLocale()
    override val hash = ChecksumCalculator.checksum(file) ?: ""
    override val internal = !file.name.endsWith(DictionaryInfoUtils.USER_DICTIONARY_SUFFIX)
    override fun getDictionary(context: Context) =
        BinaryDictionary(file.absolutePath, 0, file.length(), false, locale, Dictionary.TYPE_MAIN, false)
}

private class AssetsDictWithInfo(private val name: String, context: Context): DictWithInfo {
    override val internal = true
    override val locale = DictionaryInfoUtils.extractLocaleFromAssetsDictionaryFile(name)
    override val hash = context.assets.open("dicts${File.separator}$name").use { ChecksumCalculator.checksum(it) } ?: ""
    override fun getDictionary(context: Context): BinaryDictionary {
        val file = DictionaryInfoUtils.extractAssetsDictionary(name, locale, context)!!
        return BinaryDictionary(file.absolutePath, 0, file.length(), false, locale, Dictionary.TYPE_MAIN, false)
    }
}

private fun BinaryDictionary.addWords(words: MutableList<Pair<String, Long>>) {
    var token = 0
    val hasCases = mLocale?.let { ScriptUtils.scriptSupportsUppercase(it) } ?: true
    var cumulativeWeight = /*words.lastOrNull()?.second ?:*/ 0L
    var added = false
    do {
        val result = getNextWordProperty(token)
        val word = result.mWordProperty.mWord
        if (!result.mWordProperty.mIsNotAWord
                && word.length > 1
                && !(result.mWordProperty.mIsPossiblyOffensive && Settings.getValues().mBlockPotentiallyOffensive)
                && result.mWordProperty.probability > 2 // some minimum value, as there are too many unknown / rare words down there
                && (!hasCases || word.uppercase() != word)
            ) {
            // probability actually is something like log or very high root of actual word frequency
            // we use power of 4 to shift the probabilities in favor of more frequent words, so users mostly see relatively common words, but aren't bored by tons of very common words
            cumulativeWeight += result.mWordProperty.probability * result.mWordProperty.probability * result.mWordProperty.probability * result.mWordProperty.probability
            if (added && words.isEmpty())
                return // crappy workaround for having 2 merged dictionaries when switching dicts while one is still loading
            words.add(word to cumulativeWeight)
            added = true
        }
        token = result.mNextToken
    } while (token != 0)
}

// words will be added to the list while we're choosing -> ignore the new words
// list may get cleared while we're choosing -> return null in that case
private fun getRandomWord(words: MutableList<Pair<String, Long>>): String? {
    if (words.isEmpty()) return null
    val maxIndex = words.lastIndex
    val lastCumWeight = words.getOrNull(maxIndex)?.second ?: return null
    val random = Random.nextLong(lastCumWeight + 1)
    return words.searchFirstExceedingScore(random)
}

// modified Kotlin binary search for cumulative weights
private fun <T> List<Pair<T, Long>>.searchFirstExceedingScore(scoreToExceed: Long, fromIndex: Int = 0, toIndex: Int = lastIndex): T? {
    var low = fromIndex
    var high = toIndex

    while (low <= high) {
        val mid = (low + high).ushr(1) // safe from overflows
        val midVal = getOrNull(mid) ?: return null
        val scoreBeforeMid = getOrNull(mid - 1)?.second ?: 0

        // we want midVal to be the lowest value larger than scoreToExceed, i.e. scoreBeforeMid <= scoreToExceed < midVal.score
        if (scoreBeforeMid <= scoreToExceed) {
            if (midVal.second > scoreToExceed)
                return midVal.first
            low = mid + 1
        } else {
            high = mid - 1
        }
    }
    return null
}

const val END_DATE_EPOCH_MILLIS = 1796079600000L // Dec 1st 2026
const val TWO_WEEKS_IN_MILLIS = 14L * 24 * 3600 * 1000

@Preview
@Composable
private fun Preview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            GestureDataScreen { }
        }
    }
}
