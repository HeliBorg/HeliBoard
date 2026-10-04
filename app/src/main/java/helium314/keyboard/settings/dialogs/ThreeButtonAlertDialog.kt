/*
 * Copyright (C) 2021 The Android Open Source Project
 * parts taken from Material3 AlertDialog.kt
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.settings.dialogs

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.contentColorFor
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark

@Composable
fun ThreeButtonAlertDialog(
    onDismissRequest: () -> Unit,
    onConfirmed: () -> Unit,
    modifier: Modifier = Modifier,
    title: @Composable (() -> Unit)? = null,
    content: @Composable (() -> Unit)? = null,
    scrollContent: Boolean = false,
    onNeutral: () -> Unit = { },
    checkOk: () -> Boolean = { true },
    confirmButtonText: String? = stringResource(android.R.string.ok),
    cancelButtonText: String = stringResource(android.R.string.cancel),
    neutralButtonText: String? = null,
    reducePadding: Boolean = false,
    properties: DialogProperties = DialogProperties()
) {
    Dialog(
        onDismissRequest = onDismissRequest,
        properties = properties
    ) {
        Box(
            modifier = modifier.widthIn(min = 280.dp, max = 560.dp),
            propagateMinConstraints = true
        ) {
            Surface(
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.surface,
                contentColor = contentColorFor(MaterialTheme.colorScheme.surface),
            ) {
                Column(modifier = Modifier.padding(
                    start = if (reducePadding) 8.dp else 16.dp,
                    end = if (reducePadding) 8.dp else 16.dp,
                    top = if (reducePadding) 8.dp else 16.dp,
                    bottom = if (reducePadding) 2.dp else 6.dp
                )) {
                    title?.let {
                        CompositionLocalProvider(LocalTextStyle provides MaterialTheme.typography.titleMedium) {
                            Box(Modifier.padding(PaddingValues(bottom = if (reducePadding) 4.dp else 16.dp))) {
                                title()
                            }
                        }
                    }
                    content?.let {
                        CompositionLocalProvider(LocalTextStyle provides MaterialTheme.typography.bodyMedium) {
                            if (scrollContent) {
                                val scrollState = rememberScrollState()
                                Box(Modifier
                                    .weight(weight = 1f, fill = false)
                                    .padding(bottom = if (reducePadding) 2.dp else 8.dp)
                                    .verticalScroll(scrollState)
                                ) {
                                    content()
                                }
                            } else {
                                Box(Modifier.weight(weight = 1f, fill = false).padding(bottom = if (reducePadding) 2.dp else 8.dp)) {
                                    content()
                                }
                            }
                        }
                    }
                    FlowRow(horizontalArrangement = Arrangement.End) {
                        if (neutralButtonText != null)
                            TextButton(
                                onClick = onNeutral
                            ) { Text(neutralButtonText) }
                        Spacer(Modifier.weight(1f))
                        TextButton(onClick = onDismissRequest) { Text(cancelButtonText) }
                        if (confirmButtonText != null)
                            TextButton(
                                enabled = checkOk(),
                                onClick = { onConfirmed(); onDismissRequest() },
                            ) { Text(confirmButtonText) }
                    }
                }
            }
        }
    }
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        ThreeButtonAlertDialog(
            onDismissRequest = {},
            onConfirmed = { },
            content = { Text("hello") },
            title = { Text("title") },
            neutralButtonText = "Default"
        )
    }
}
