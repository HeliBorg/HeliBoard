// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings

import android.graphics.drawable.VectorDrawable
import androidx.annotation.DrawableRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import helium314.keyboard.latin.utils.ExpandButton
import helium314.keyboard.latin.utils.dpToPx

@Composable
fun WithSmallTitle(
    description: String,
    content: @Composable () -> Unit,
) {
    Column {
        Text(description, style = MaterialTheme.typography.titleSmall)
        content()
    }
}

@Composable
fun ActionRow(
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    verticalAlignment: Alignment.Vertical = Alignment.CenterVertically,
    content: @Composable RowScope.() -> Unit
) {
    val clickableModifier = if (onClick != null) Modifier.clickable(onClick = onClick)
    else Modifier
    Row(
        modifier = modifier
            .then(clickableModifier)
            .fillMaxWidth()
            .heightIn(min = 44.dp),
        verticalAlignment = verticalAlignment,
        content = content
    )
}

/** Icon if resource is a vector image, (bitmap) Image otherwise */
@Composable
fun IconOrImage(@DrawableRes resId: Int, name: String?, sizeDp: Int) {
    val ctx = LocalContext.current
    val drawable = ContextCompat.getDrawable(ctx, resId)
    if (drawable is VectorDrawable)
        Icon(painterResource(resId), name, Modifier.size(sizeDp.dp))
    else {
        val px = sizeDp.dpToPx(LocalResources.current)
        Image(drawable!!.toBitmap(px, px).asImageBitmap(), name)
    }
}

@Composable
fun <T>DropDownField(
    items: List<T>,
    selectedItem: T,
    onSelected: (T) -> Unit,
    extraButton: @Composable (() -> Unit)? = null,
    itemContent: @Composable (T) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(
        Modifier.clickable { expanded = !expanded }
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(start = 8.dp, bottom = 4.dp)
        ) {
            Box(Modifier.weight(1f)) {
                itemContent(selectedItem)
            }
            ExpandButton(items.size > 1) { expanded = !expanded }
            if (extraButton != null)
                extraButton()
        }
    }
    DropdownMenu(
        expanded = expanded,
        onDismissRequest = { expanded = false }
    ) {
        items.forEach {
            DropdownMenuItem(
                text = { itemContent(it) },
                onClick = { expanded = false; onSelected(it) }
            )
        }
    }
}

@Composable
fun isWideScreen(): Boolean {
    val width = LocalConfiguration.current.screenWidthDp
    val height = LocalConfiguration.current.screenHeightDp
    return height < 500 && width > height
}

val contentTextDirectionStyle = TextStyle(textDirection = TextDirection.Content)
