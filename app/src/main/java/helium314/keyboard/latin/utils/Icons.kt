// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import androidx.annotation.StringRes
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.LayoutDirection
import helium314.keyboard.latin.R

@Composable
fun NextScreenIcon() {
    Icon(
        painterResource(R.drawable.ic_arrow_left), null,
        if (LocalLayoutDirection.current == LayoutDirection.Ltr) Modifier.scale(-1f, 1f) else Modifier
    )
}

@Composable
fun EditButton(enabled: Boolean = true, onClick: () -> Unit) {
    IconButton(onClick, enabled = enabled)  { Icon(painterResource(R.drawable.ic_edit), "edit") }
}

@Composable
fun DeleteButton(onClick: () -> Unit) {
    IconButton(onClick)  { Icon(painterResource(R.drawable.ic_bin), stringResource(R.string.delete)) }
}

@Composable
fun SearchIcon() {
    Icon(painterResource(R.drawable.sym_keyboard_search_lxx), stringResource(R.string.label_search_key))
}

@Composable
fun CloseIcon(@StringRes resId: Int) {
    Icon(painterResource(R.drawable.ic_close), stringResource(resId))
}

@Composable
fun DefaultButton(isDefault: Boolean, onClick: () -> Unit) {
    IconButton(onClick = onClick, enabled = !isDefault) {
        Icon(painterResource(R.drawable.ic_settings_default), "default")
    }
}

@Composable
fun ExpandButton(enabled: Boolean = true, onClick: () -> Unit) {
    IconButton(onClick = onClick, enabled = enabled) {
        Icon(
            painterResource(R.drawable.ic_arrow_left),
            "expand",
            Modifier.rotate(-90f)
        )
    }
}

@Composable
fun BackButton(onClick: () -> Unit) {
    IconButton(onClick = onClick) {
        Icon(
            painterResource(R.drawable.ic_arrow_back),
            stringResource(R.string.spoken_description_action_previous)
        )
    }
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        Surface {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                NextScreenIcon()
                SearchIcon()
                CloseIcon(R.string.dialog_close)
                EditButton { }
                DeleteButton { }
                DefaultButton(false) { }
                ExpandButton { }
            }
        }
    }
}
