// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens.gesturedata

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonColors
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SwipeToDismissBox
import androidx.compose.material3.SwipeToDismissBoxValue
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberSwipeToDismissBoxState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.fromHtml
import androidx.compose.ui.text.input.PlatformImeOptions
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.intl.Locale
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Links
import helium314.keyboard.latin.common.decapitalize
import helium314.keyboard.latin.utils.GestureData
import helium314.keyboard.latin.utils.GestureDataDao
import helium314.keyboard.latin.utils.GestureDataGatheringSettings
import helium314.keyboard.latin.utils.GestureDataInfo
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.dialogs.InfoDialog
import helium314.keyboard.settings.dialogs.ThreeButtonAlertDialog
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.text.DateFormat
import java.util.Date

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

@Composable
fun ReviewScreen(
    onClickBack: () -> Unit,
) {
    val ctx = LocalContext.current
    val dao = GestureDataDao.getInstance(ctx)
    var selected by rememberSaveable { mutableStateOf(listOf<Long>()) }
    var filter by rememberSaveable(stateSaver = TextFieldValue.Saver) { mutableStateOf(TextFieldValue()) }
    var gestureDataInfos by remember { mutableStateOf(listOf<GestureDataInfo>()) }

    // all that filtering stuff
    var sortByName: Boolean by rememberSaveable { mutableStateOf(false) }
    var reverseSort: Boolean by rememberSaveable { mutableStateOf(true) }
    var includeActive by rememberSaveable { mutableStateOf(false) }
    var includeBackground by rememberSaveable { mutableStateOf(true) }
    var includeExported by rememberSaveable { mutableStateOf(false) }
    var startDate: Long? by rememberSaveable { mutableStateOf(null) }
    var endDate: Long? by rememberSaveable { mutableStateOf(null) }
    fun setAndSortWords(infos: List<GestureDataInfo>) {
        gestureDataInfos = if (sortByName) {
            if (reverseSort) infos.sortedWith(compareByDescending(String.CASE_INSENSITIVE_ORDER) { it.targetWord })
            else infos.sortedWith(compareBy(String.CASE_INSENSITIVE_ORDER) { it.targetWord })
        } else {
            if (reverseSort) infos.sortedByDescending { it.timestamp }
            else infos.sortedBy { it.timestamp }
        }
    }
    fun reloadGestureDataInfos() {
        val infos = if (!includeActive && !includeBackground) emptyList() else dao?.filterInfos(
            filter.text.takeIf { it.isNotEmpty() },
            startDate,
            endDate,
            if (includeExported) null else false,
            if (includeActive && includeBackground) null else includeActive
        ).orEmpty()
        selected = emptyList() // unselect on filter changes
        setAndSortWords(infos)
    }
    LifecycleResumeEffect(Unit) {
        reloadGestureDataInfos()
        onPauseOrDispose {  }
    }
    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing.only(WindowInsetsSides.Bottom),
        bottomBar = { BottomBar(
            if (selected.isNotEmpty()) selected.size else gestureDataInfos.size,
            sortByName,
            { sortByName = it },
            reverseSort,
            { reverseSort = it },
            {
                val toShare = if (selected.isEmpty()) gestureDataInfos else gestureDataInfos.filter { it.id in selected }
                val toIgnore = GestureDataGatheringSettings.getWordExclusions(ctx)
                toShare.filterNot { it.targetWord in toIgnore }.map { it.id }.take(10000)
            },
            ::reloadGestureDataInfos,
            {
                val ids = selected.ifEmpty { gestureDataInfos.map { it.id } }
                dao?.delete(ids, false, ctx)
                reloadGestureDataInfos()
            }
        )}
    ) { innerPadding ->
        @Composable fun dataColumn() {
            val wordListState = LazyListState()
            val scope = rememberCoroutineScope()
            LaunchedEffect(reverseSort, sortByName) {
                scope.launch {
                    if (gestureDataInfos.isNotEmpty())
                        wordListState.scrollToItem(0)
                }
            }
            CompositionLocalProvider(
                LocalTextStyle provides MaterialTheme.typography.bodySmall
            ) {
                Text(stringResource(R.string.gesture_data_long_press_select_hint), Modifier.padding(horizontal = 12.dp))
            }
            val deleteJobs = remember { mutableMapOf<Long, Job>() }
            LazyColumn(state = wordListState) {
                items(gestureDataInfos, { it.id }) { item ->
                    val dismissState = rememberSwipeToDismissBoxState()
                    // todo: how to prevent SwipeToDismissBox from taking the EndToStart gesture? because of this we can't swipe up nicely...
                    //  in general it should be less sensitive to swiping the wrong direction
                    SwipeToDismissBox(
                        state = dismissState,
                        enableDismissFromEndToStart = false,
                        gesturesEnabled = selected.isEmpty(),
                        onDismiss = {
                            deleteJobs[item.id] = scope.launch {
                                delay(4000)
                                dao?.delete(listOf(item.id), false, ctx)
                                gestureDataInfos = gestureDataInfos - item
                            }
                        },
                        backgroundContent = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(painterResource(R.drawable.ic_bin), stringResource(R.string.delete))
                                val undoVisible = dismissState.progress == 1f && dismissState.settledValue == SwipeToDismissBoxValue.StartToEnd
                                TextButton(
                                    onClick = {
                                        deleteJobs.remove(item.id)?.cancel()
                                        scope.launch { dismissState.reset() }
                                    },
                                    modifier = Modifier.alpha(if (undoVisible) 1f else 0f)
                                ) { Text(stringResource(R.string.undo)) }
                            }
                        },
                        content = {
                            GestureDataEntry(
                                item,
                                item.id in selected,
                                selected.isNotEmpty(),
                                { sel ->
                                    selected = if (!sel) selected.filterNot { it == item.id }
                                    else selected + item.id
                                },
                                {
                                    scope.launch {
                                        delay(20)
                                        reloadGestureDataInfos()
                                    }
                                }
                            )
                        }
                    )
                }
            }
        }
        @Composable fun controlColumn() {
            Column(Modifier.padding(horizontal = 12.dp)) {
                LaunchedEffect(filter, startDate, endDate, includeExported, reverseSort, includeActive, includeBackground) {
                    reloadGestureDataInfos()
                }
                LaunchedEffect(reverseSort, sortByName) {
                    setAndSortWords(gestureDataInfos)
                }
                TopBar(
                    onClickBack,
                    includeActive,
                    { includeActive = it },
                    includeBackground,
                    { includeBackground = it },
                    includeExported,
                    { includeExported = it }
                )
                var showDateRangePicker by remember { mutableStateOf(false) }
                val df = DateFormat.getDateInstance(DateFormat.SHORT)
                HorizontalDivider()
                Row(verticalAlignment = Alignment.CenterVertically) {
                    // allow regex in the filter?
                    TextField(
                        value = filter,
                        onValueChange = { filter = it},
                        modifier = Modifier.weight(0.7f),
                        label = { Text(stringResource(R.string.label_search_key)) },
                        keyboardOptions = KeyboardOptions(platformImeOptions = PlatformImeOptions("noBackground"))
                    )
                    Column(Modifier
                        .clickable { showDateRangePicker = true }
                        .weight(0.3f), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(stringResource(R.string.gesture_data_date_range), style = MaterialTheme.typography.bodyLarge)
                        if (startDate == null && endDate == null)
                            Text("-", style = MaterialTheme.typography.bodyMedium)
                        else {
                            Text(startDate?.let { df.format(Date(it)) }.toString(), style = MaterialTheme.typography.bodyMedium)
                            Text(endDate?.let { df.format(Date(it)) }.toString(), style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
                if (showDateRangePicker)
                    DateRangePickerModal({ startDate = it.first; endDate = it.second }) { showDateRangePicker = false }
            }
        }

        Column(Modifier.padding(innerPadding)) {
            controlColumn()
            HorizontalDivider()
            dataColumn()
        }
    }
}

@Composable
private fun GestureDataEntry(
    gestureDataInfo: GestureDataInfo,
    selected: Boolean,
    anythingSelected: Boolean,
    onSelect: (Boolean) -> Unit,
    onExcludeWord: () -> Unit,
) {
    var showDetails by remember { mutableStateOf(false) }
    val modifier = if (!anythingSelected)
        Modifier.combinedClickable(
            onClick = { showDetails = true },
            onLongClick = { onSelect(true) },
        )
    else Modifier.selectable(
        selected = selected,
        onClick = { onSelect(!selected) },
    )
    Column(modifier
        .fillMaxWidth()
        .background(MaterialTheme.colorScheme.background)
        .padding(vertical = 6.dp, horizontal = 12.dp)
    ) {
        Text(
            text = gestureDataInfo.targetWord,
            color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
        )
        val infos = listOfNotNull(
            DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.MEDIUM)
                .format(Date(gestureDataInfo.timestamp)),
            if (gestureDataInfo.activeMode) stringResource(R.string.gesture_data_active).decapitalize(Locale.current.platformLocale) else null,
            if (gestureDataInfo.exported) stringResource(R.string.gesture_data_shared) else null
        ).joinToString(", ")
        CompositionLocalProvider(
            LocalTextStyle provides MaterialTheme.typography.bodySmall,
            LocalContentColor provides if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
        ) {
            Text(text = infos, modifier = Modifier.padding(top = 2.dp))
        }
    }
    if (showDetails) {
        val ctx = LocalContext.current
        val jsonData = GestureDataDao.getInstance(ctx)?.getJsonData(listOf(gestureDataInfo.id), ctx)?.firstOrNull()
        val data = runCatching { jsonData?.let { Json.decodeFromString<GestureData>(it) } }.getOrNull()
        if (data != null) {
            ThreeButtonAlertDialog(
                onDismissRequest = { showDetails = false },
                cancelButtonText = stringResource(R.string.dialog_close),
                onConfirmed = {},
                confirmButtonText = null,
                onNeutral = {
                    GestureDataGatheringSettings.addWordExclusion(ctx, gestureDataInfo.targetWord)
                    onExcludeWord()
                    showDetails = false
                },
                neutralButtonText = stringResource(R.string.gesture_data_background_exclude_words),
                title = { Text(gestureDataInfo.targetWord) },
                content = {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(stringResource(R.string.gesture_data_background_dictionary, data.dictionaries.lastOrNull()?.let { "${it.type}:${it.language}" } ?: ""))
                        Text(stringResource(R.string.gesture_data_background_position, (data.suggestions.indexOfFirst { it.word.isNotEmpty() } + 1)))
                    }
                }
            )
        }
    }
}

@Composable
private fun BottomBar(
    wordcount: Int,
    sortByName: Boolean,
    setSortByName: (Boolean) -> Unit,
    reverseSort: Boolean,
    setReverseSort: (Boolean) -> Unit,
    getExportIds: () -> List<Long>,
    onChanged: () -> Unit,
    delete: () -> Unit
) {
    var showExportDialog by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }
    BottomAppBar(
        actions = {
            val buttonColors = ButtonColors(
                containerColor = Color.Transparent,
                contentColor = MaterialTheme.colorScheme.onSurface,
                disabledContainerColor = Color.Transparent,
                disabledContentColor = MaterialTheme.colorScheme.surfaceVariant
            )
            Row(
                horizontalArrangement = Arrangement.SpaceEvenly,
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    { showDeleteDialog = true },
                    colors = buttonColors,
                    modifier = Modifier.weight(1f),
                    enabled = wordcount > 0
                ) {
                    Column {
                        Icon(
                            painterResource(R.drawable.ic_bin),
                            stringResource(R.string.delete),
                            Modifier.align(Alignment.CenterHorizontally).size(30.dp)
                        )
                        Text(stringResource(R.string.gesture_data_words_selected, wordcount))
                    }
                }
                Button(
                    { showExportDialog = true },
                    colors = buttonColors,
                    modifier = Modifier.weight(1f),
                    enabled = wordcount > 0
                ) {
                    Column {
                        Icon(
                            painterResource(R.drawable.ic_share),
                            "share",
                            Modifier.align(Alignment.CenterHorizontally).size(30.dp)
                        )
                        Text(stringResource(R.string.gesture_data_words_selected, wordcount))
                    }
                }
                IconButton(
                    onClick = {
                        if (sortByName) setReverseSort(!reverseSort)
                        else setSortByName(true)
                    }
                ) {
                    Icon(
                        painterResource(R.drawable.ic_sort_alphabetically),
                        "sort alphabetically"
                    )
                }
                IconButton(
                    onClick = {
                        if (!sortByName) setReverseSort(!reverseSort)
                        else setSortByName(false)
                    }
                ) {
                    Icon(
                        painterResource(R.drawable.ic_sort_chronologically),
                        "sort chronologically"
                    )
                }
            }
        }
    )
    if (showExportDialog) {
        ThreeButtonAlertDialog(
            onDismissRequest = { showExportDialog = false },
            content = {
                Column {
                    val ids = getExportIds()
                    ShareGestureData(
                        ids,
                        { onChanged() },
                        { onChanged(); showExportDialog = false }
                    )
                }
            },
            cancelButtonText = stringResource(R.string.dialog_close),
            onConfirmed = { },
            confirmButtonText = null
        )
    }
    if (showDeleteDialog) {
        ConfirmationDialog(
            onDismissRequest = { showDeleteDialog = false },
            onConfirmed = delete,
            content = {
                Text(stringResource(R.string.gesture_data_delete_dialog_all, wordcount))
            }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TopBar(
    onClickBack: () -> Unit,
    includeActive: Boolean,
    setIncludeActive: (Boolean) -> Unit,
    includeBackground: Boolean,
    setIncludeBackground: (Boolean) -> Unit,
    includeExported: Boolean,
    setIncludeExported: (Boolean) -> Unit,
) {
    var infoDialog by remember { mutableStateOf(false) }
    TopAppBar( // not in the scaffold, thus will not cover data column in wide screen layout
        title = { Text(stringResource(R.string.gesture_data_review_screen_title)) },
        navigationIcon = {
            IconButton(onClick = onClickBack) {
                Icon(
                    painterResource(R.drawable.ic_arrow_back),
                    stringResource(R.string.spoken_description_action_previous)
                )
            }
        },
        actions = {
            Box {
                var showMenu by remember { mutableStateOf(false) }
                IconButton(
                    onClick = { showMenu = true }
                ) { Icon(painterResource(R.drawable.ic_arrow_left), "menu", Modifier.rotate(-90f)) }
                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false }
                ) {
                    DropdownMenuItem(
                        text = { Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(includeActive, setIncludeActive)
                            Text(stringResource(R.string.gesture_data_show_active))
                        } },
                        onClick = { showMenu = false; setIncludeActive(!includeActive) }
                    )
                    DropdownMenuItem(
                        text = { Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(includeBackground, setIncludeBackground)
                            Text(stringResource(R.string.gesture_data_show_background))
                        } },
                        onClick = { showMenu = false; setIncludeBackground(!includeBackground) }
                    )
                    DropdownMenuItem(
                        text = { Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(includeExported, setIncludeExported)
                            Text(stringResource(R.string.gesture_data_include_shared))
                        } },
                        onClick = { showMenu = false; setIncludeExported(!includeExported) }
                    )
                    DropdownMenuItem(
                        text = { Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(false, { showMenu = false; infoDialog = true }, Modifier.alpha(0f)) // just for alignment
                            Text(stringResource(R.string.gesture_data_background_gathering_info))
                        } },
                        onClick = { showMenu = false; infoDialog = true }
                    )
                }
            }
        }
    )
    if (infoDialog) {
        val text = stringResource(
            R.string.gesture_data_background_gathering_review_message,
            stringResource(R.string.gesture_data_review_screen_title),
            Links.SWIPE_O_SCOPE
        )
        InfoDialog(AnnotatedString.fromHtml(text)) { infoDialog = false }
    }
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        ReviewScreen { }
    }
}
