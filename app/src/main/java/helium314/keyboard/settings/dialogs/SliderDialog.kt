// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark

@Composable
fun SliderDialog(
    onDismissRequest: () -> Unit,
    onDone: (Float) -> Unit,
    initialValue: Float,
    range: ClosedFloatingPointRange<Float>,
    modifier: Modifier = Modifier,
    showDefault: Boolean = false,
    onDefault: () -> Unit = { },
    onValueChanged: (Float) -> Unit = { },
    title: (@Composable () -> Unit)? = null,
    intermediateSteps: Int? = null,
    positionString: (@Composable (Float) -> String) = { it.toString() },
) {
    var sliderPosition by remember { mutableFloatStateOf(initialValue) }

    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        neutralButtonText = if (showDefault) stringResource(R.string.button_default) else null,
        onNeutral = { onDismissRequest(); onDefault() },
        onConfirmed = { onDone(sliderPosition) },
        modifier = modifier,
        title = title,
        content = {
            CompositionLocalProvider(
                LocalTextStyle provides MaterialTheme.typography.bodyLarge
            ) {
                Column {
                    if (intermediateSteps == null)
                        Slider(
                            value = sliderPosition,
                            onValueChange = { sliderPosition = it },
                            onValueChangeFinished = { onValueChanged(sliderPosition) },
                            valueRange = range,
                        )
                    else
                        Slider(
                            value = sliderPosition,
                            onValueChange = { sliderPosition = it },
                            onValueChangeFinished = { onValueChanged(sliderPosition) },
                            valueRange = range,
                            steps = intermediateSteps
                        )
                    Text(positionString(sliderPosition))
                }
            }
        },
    )
}

@Preview
@Composable
private fun PreviewSliderDialog() {
    Theme(previewDark) {
        SliderDialog(
            onDismissRequest = { },
            onDone = { },
            initialValue = 100f,
            range = 0f..500f,
            title = { Text("move it") },
            showDefault = true
        )
    }
}
