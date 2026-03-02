// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark

@Composable
fun InfoDialog(
    message: String,
    onDismissRequest: () -> Unit
) {
    InfoDialog(AnnotatedString(message), onDismissRequest)
}

@Composable
fun InfoDialog(
    message: AnnotatedString,
    onDismissRequest: () -> Unit
) {
    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        content = { Text(message) },
        scrollContent = true,
        cancelButtonText = stringResource(android.R.string.ok),
        onConfirmed = { },
        confirmButtonText = null
    )
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        InfoDialog("message") { }
    }
}
