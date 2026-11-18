// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import helium314.keyboard.keyboard.internal.KeyboardIconsSet
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.customIconNames
import helium314.keyboard.latin.utils.getStringResourceOrName
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.initPreview
import helium314.keyboard.latin.utils.previewDark
import kotlinx.serialization.json.Json
import androidx.core.content.edit
import helium314.keyboard.latin.utils.DefaultButton
import helium314.keyboard.settings.GetIconOrEmpty
import helium314.keyboard.settings.painterResourceCompat

@Composable
fun CustomizeIconsDialog(
    prefKey: String,
    onDismissRequest: () -> Unit,
) {
    class KeyIcon(val iconName: String, val displayName: String, val isCustom: Boolean)
    val state = rememberLazyListState()
    val ctx = LocalContext.current
    val prefs = ctx.prefs()
    var showIconDialog: KeyIcon? by rememberSaveable { mutableStateOf(null) }
    var showDeletePrefConfirmDialog by rememberSaveable { mutableStateOf(false) }
    var customIconNames = remember(showIconDialog) { customIconNames(prefs) }
    var keyIcons by remember { mutableStateOf(
        KeyboardIconsSet.getAllIcons(ctx).keys.map { iconName ->
            val name = iconName.getStringResourceOrName("", ctx)
                .takeIf { it != iconName } ?: iconName.getStringResourceOrName("label_", ctx)
            val custom = iconName in customIconNames.keys
            KeyIcon(iconName, name, custom)
        }.sortedBy { it.displayName }.sortedWith(compareBy({ !it.isCustom }, { it.displayName }))
    ) }

    fun reloadItem(iconName: String) {
        customIconNames = customIconNames(prefs)
        keyIcons = keyIcons.map { item ->
            if (item.iconName == iconName) {
                // this is to trigger reload in the LazyColumn
                KeyIcon(item.iconName, if (item.displayName.endsWith(" ")) item.displayName.trimEnd() else item.displayName + " ", iconName in customIconNames)
            }
            else item
        }
    }
    fun resetIcon(iconName: String) {
        runCatching {
            val icons2 = customIconNames.toMutableMap()
            icons2.remove(iconName)
            if (icons2.isEmpty()) prefs.edit { remove(prefKey) }
            else prefs.edit { putString(prefKey, Json.encodeToString(icons2)) }
            KeyboardIconsSet.instance.loadIcons(ctx)
        }
        reloadItem(iconName)
    }

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
                items(keyIcons, key = { it.displayName }) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.clickable { showIconDialog = it }
                    ) {
                        KeyboardIconsSet.instance.GetIconOrEmpty(it.iconName)
                        Text(it.displayName, Modifier.weight(1f))
                        if (it.isCustom)
                            DefaultButton(false) { resetIcon(it.iconName) }
                    }
                }
            }
        },
    )
    if (showIconDialog != null) {
        val iconName = showIconDialog!!.iconName
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
                    val newIcons = customIconNames.toMutableMap()
                    newIcons[iconName] = selectedIcon?.let { ctx.resources.getResourceEntryName(it) } ?: return@runCatching
                    prefs.edit { putString(prefKey, Json.encodeToString(newIcons)) }
                    KeyboardIconsSet.instance.loadIcons(ctx)
                }
                reloadItem(iconName)
            },
            neutralButtonText = if (customIconNames.contains(iconName)) stringResource(R.string.button_default) else null,
            onNeutral = {
                showIconDialog = null
                resetIcon(iconName)
            },
            title = { Text(showIconDialog!!.displayName) },
            content = {
                LazyVerticalGrid(
                    columns = GridCells.Adaptive(minSize = 64.dp),
                    state = gridState
                ) {
                    items(icons, key = { it }) { resId ->
                        val color = if (resId == selectedIcon) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurface
                        CompositionLocalProvider(
                            LocalContentColor provides color
                        ) {
                            Box(
                                Modifier.size(40.dp).clickable { selectedIcon = resId },
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(painterResourceCompat(resId), null, Modifier.fillMaxSize(0.8f))
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
