// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark
import sh.calvin.reorderable.ReorderableItem
import sh.calvin.reorderable.rememberReorderableLazyListState

@Composable
fun <T: Any> ReorderDialog(
    onDismissRequest: () -> Unit,
    onConfirmed: (List<T>) -> Unit,
    items: List<T>,
    getKey: (T) -> Any, // actually it's not "Any", but "anything that can be stored in a bundle"
    displayItem: @Composable (T) -> Unit,
    modifier: Modifier = Modifier,
    title: @Composable (() -> Unit)? = null,
    onNeutral: () -> Unit = { },
    neutralButtonText: String? = null,
) {
    var reorderableItems by remember(items) { mutableStateOf(items) }
    val listState = rememberLazyListState()

    val dragDropState = rememberReorderableLazyListState(listState) { from, to ->
        reorderableItems = reorderableItems.toMutableList().apply {
            add(to.index, removeAt(from.index))
        }
    }
    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = { onConfirmed(reorderableItems) },
        onNeutral = { onDismissRequest(); onNeutral() },
        neutralButtonText = neutralButtonText,
        modifier = modifier,
        title = title,
        content = {
            LazyColumn(
                state = listState,
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                items(reorderableItems, key = getKey) { item ->
                    ReorderableItem(
                        state = dragDropState,
                        key = getKey(item)
                    ) { dragging ->
                        val elevation by animateDpAsState(if (dragging) 4.dp else 0.dp)
                        Surface(shadowElevation = elevation) {
                            Row(
                                modifier = Modifier
                                    .longPressDraggableHandle()
                                    .heightIn(min = 36.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    painterResource(R.drawable.ic_drag_indicator),
                                    "Reorder",
                                    Modifier.padding(end = 6.dp),
                                    MaterialTheme.colorScheme.onSurfaceVariant
                                )
                               displayItem(item)
                            }
                        }
                    }
                }
            }
        },
    )
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        ReorderDialog(
            onConfirmed = {},
            onDismissRequest = {},
            items = listOf(1, 2, 3),
            displayItem = { Text(it.toString(), Modifier.fillMaxWidth(), textAlign = TextAlign.Center) },
            getKey = { it.toString() }
        )
    }
}
