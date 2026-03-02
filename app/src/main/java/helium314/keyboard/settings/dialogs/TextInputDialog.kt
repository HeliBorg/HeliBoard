// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.contentTextDirectionStyle
import helium314.keyboard.latin.utils.previewDark

// mostly taken from StreetComplete / SCEE
/** Dialog with which to input text. OK button is only clickable if [checkTextValid] returns true. */
@Composable
fun TextInputDialog(
    onDismissRequest: () -> Unit,
    onConfirmed: (text: String) -> Unit,
    modifier: Modifier = Modifier,
    title: @Composable (() -> Unit)? = null,
    description: @Composable (() -> Unit)? = null,
    onNeutral: () -> Unit = { },
    neutralButtonText: String? = null,
    confirmButtonText: String = stringResource(android.R.string.ok),
    initialText: String = "",
    textInputLabel: @Composable (() -> Unit)? = null,
    singleLine: Boolean = true,
    keyboardType: KeyboardType = KeyboardType.Unspecified,
    properties: DialogProperties = DialogProperties(),
    reducePadding: Boolean = false,
    checkTextValid: (text: String) -> Boolean = { it.isNotBlank() }
) {
    var value by rememberSaveable(stateSaver = TextFieldValue.Saver) {
        mutableStateOf(TextFieldValue(initialText, selection = TextRange(if (singleLine) initialText.length else 0)))
    }

    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = { onConfirmed(value.text) },
        confirmButtonText = confirmButtonText,
        checkOk = { checkTextValid(value.text) },
        neutralButtonText = neutralButtonText,
        onNeutral = { onDismissRequest(); onNeutral() },
        modifier = modifier,
        title = title,
        content = {
            Column {
                description?.let {
                    it()
                    Spacer(Modifier.height(6.dp))
                }
                val focusRequester = remember { FocusRequester() }
                OutlinedTextField(
                    value = value,
                    onValueChange = { value = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(focusRequester),
                    label = textInputLabel,
                    keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
                    singleLine = singleLine,
                    textStyle = contentTextDirectionStyle,
                )
                LaunchedEffect(Unit) { focusRequester.requestFocus() }
            }
        },
        properties = properties,
        reducePadding = reducePadding,
    )
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        TextInputDialog(
            onDismissRequest = {},
            onConfirmed = {},
            title = { Text("Title") },
            initialText = "some text\nand another line",
            singleLine = false,
            textInputLabel = { Text("fill it") },
            description = { Text("hello") }
        )
    }
}
