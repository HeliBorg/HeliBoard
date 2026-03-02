// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import android.graphics.drawable.VectorDrawable
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Icon
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import helium314.keyboard.keyboard.internal.KeyboardIconsSet
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.customIconNames
import helium314.keyboard.latin.utils.dpToPx
import helium314.keyboard.latin.utils.getStringResourceOrName
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.initPreview
import helium314.keyboard.latin.utils.previewDark
import helium314.keyboard.settings.screens.GetIcon
import kotlinx.serialization.json.Json
import androidx.core.content.edit

@Composable
fun CustomizeIconsDialog(
    prefKey: String,
    onDismissRequest: () -> Unit,
) {
    val state = rememberLazyListState()
    val ctx = LocalContext.current
    var iconsAndNames by remember { mutableStateOf(
        KeyboardIconsSet.getAllIcons(ctx).keys.map { iconName ->
            val name = iconName.getStringResourceOrName("", ctx)
            if (name == iconName) iconName to iconName.getStringResourceOrName("label_", ctx)
            else iconName to name
        }.sortedBy { it.second }
    ) }
    fun reloadItem(iconName: String) {
        iconsAndNames = iconsAndNames.map { item ->
            if (item.first == iconName) {
                item.first to if (item.second.endsWith(" ")) item.second.trimEnd() else item.second + " "
            }
            else item
        }
    }
    var showIconDialog: Pair<String, String>? by rememberSaveable { mutableStateOf(null) }
    var showDeletePrefConfirmDialog by rememberSaveable { mutableStateOf(false) }
    val prefs = ctx.prefs()
    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = { },
        confirmButtonText = null,
        cancelButtonText = stringResource(R.string.dialog_close),
        neutralButtonText = if (prefs.contains(prefKey)) stringResource(R.string.button_default) else null,
        onNeutral = { showDeletePrefConfirmDialog = true },
        title = { Text(stringResource(R.string.customize_icons)) },
        content = {
            LazyColumn(state = state) {
                items(iconsAndNames, key = { it.second }) { (iconName, displayName) ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.clickable { showIconDialog = iconName to displayName }
                    ) {
                        KeyboardIconsSet.instance.GetIcon(iconName)
                        Text(displayName, Modifier.weight(1f))
                    }
                }
            }
        },
    )
    if (showIconDialog != null) {
        val iconName = showIconDialog!!.first
        val allIcons = KeyboardIconsSet.getAllIcons(ctx)
        val iconsForName = allIcons[iconName].orEmpty()
        val iconsSet = mutableSetOf<Int>()
        iconsSet.addAll(iconsForName)
        KeyboardIconsSet.getAllIcons(ctx).forEach { iconsSet.addAll(it.value) }
        val icons = iconsSet.toList()
        val initialIcon = KeyboardIconsSet.instance.iconIds[iconName]
        var selectedIcon by rememberSaveable { mutableStateOf(initialIcon) }

        val gridState = rememberLazyGridState()
        LaunchedEffect(initialIcon) {
            val index = icons.indexOf(initialIcon)
            if (index != -1) gridState.animateScrollToItem(index, -state.layoutInfo.viewportSize.height / 3)
        }
        ThreeButtonAlertDialog(
            onDismissRequest = { showIconDialog = null },
            onConfirmed = {
                runCatching {
                    val newIcons = customIconNames(prefs).toMutableMap()
                    newIcons[iconName] = selectedIcon?.let { ctx.resources.getResourceEntryName(it) } ?: return@runCatching
                    prefs.edit { putString(prefKey, Json.encodeToString(newIcons)) }
                    KeyboardIconsSet.instance.loadIcons(ctx)
                }
                reloadItem(iconName)
            },
            neutralButtonText = if (customIconNames(prefs).contains(iconName)) stringResource(R.string.button_default) else null,
            onNeutral = {
                showIconDialog = null
                runCatching {
                    val icons2 = customIconNames(prefs).toMutableMap()
                    icons2.remove(iconName)
                    if (icons2.isEmpty()) prefs.edit { remove(prefKey) }
                    else prefs.edit { putString(prefKey, Json.encodeToString(icons2)) }
                    KeyboardIconsSet.instance.loadIcons(ctx)
                }
                reloadItem(iconName)
            },
            title = { Text(showIconDialog!!.second) },
            content = {
                LazyVerticalGrid(
                    columns = GridCells.Adaptive(minSize = 64.dp),
                    state = gridState
                ) {
                    items(icons, key = { it }) { resId ->
                        val drawable = ContextCompat.getDrawable(ctx, resId)?.mutate() ?: return@items
                        val color = if (resId == selectedIcon) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurface
                        CompositionLocalProvider(
                            LocalContentColor provides color
                        ) {
                            Box(
                                Modifier.size(40.dp).clickable { selectedIcon = resId },
                                contentAlignment = Alignment.Center
                            ) {
                                if (drawable is VectorDrawable)
                                    Icon(painterResource(resId), null, Modifier.fillMaxSize(0.8f))
                                else {
                                    val px = 40.dpToPx(ctx.resources)
                                    Icon(drawable.toBitmap(px, px).asImageBitmap(), null, Modifier.fillMaxSize(0.8f))
                                }
                            }
                        }
                    }
                }
            },
        )
    }
    if (showDeletePrefConfirmDialog) {
        ConfirmationDialog(
            onDismissRequest = { showDeletePrefConfirmDialog = false },
            onConfirmed = {
                showDeletePrefConfirmDialog = false
                onDismissRequest()
                prefs.edit { remove(prefKey) }
                KeyboardIconsSet.instance.loadIcons(ctx)
            },
            content = { Text(stringResource(R.string.customize_icons_reset_message)) }
        )
    }
}

@Preview
@Composable
private fun Preview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        CustomizeIconsDialog(
            prefKey = "",
            onDismissRequest = { },
        )
    }
}
