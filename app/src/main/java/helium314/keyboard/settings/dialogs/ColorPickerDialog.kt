// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.dialogs

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.PaintingStyle
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.core.graphics.toColorInt
import com.github.skydoves.colorpicker.compose.AlphaSlider
import com.github.skydoves.colorpicker.compose.BrightnessSlider
import com.github.skydoves.colorpicker.compose.HsvColorPicker
import com.github.skydoves.colorpicker.compose.rememberColorPickerController
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.previewDark
import helium314.keyboard.settings.isWideScreen

@Composable
fun ColorPickerDialog(
    onDismissRequest: () -> Unit,
    initialColor: Int,
    title: String,
    showDefault: Boolean,
    onDefault: () -> Unit,
    onConfirmed: (Int) -> Unit,
) {
    val controller = rememberColorPickerController()
    val wheelPaint = Paint().apply {
        alpha = 0.5f
        style = PaintingStyle.Stroke
        strokeWidth = 5f
        color = Color.White
    }
    controller.wheelPaint = wheelPaint
    val barHeight = 35.dp
    val initialString = initialColor.toUInt().toString(16)
    var textValue by remember { mutableStateOf(TextFieldValue(initialString, TextRange(initialString.length))) }
    var currentColor by remember { mutableStateOf(Color(initialColor)) }
    val useWideLayout = isWideScreen()
    @Composable fun topBar() {
        Row {
            Surface(
                color = Color(initialColor),
                modifier = Modifier.fillMaxWidth(0.5f)
                    .padding(start = 10.dp)
                    .height(barHeight)
            ) { }
            Surface(
                color = currentColor,
                modifier = Modifier.fillMaxWidth()
                    .padding(end = 10.dp)
                    .height(barHeight)
            ) { }
        }
    }
    @Composable fun picker() {
        HsvColorPicker(
            modifier = Modifier
                .size(300.dp)
                .padding(10.dp),
            controller = controller,
            onColorChanged = {
                if (it.fromUser)
                    textValue = TextFieldValue(it.hexCode, selection = TextRange(it.hexCode.length))
                currentColor = it.color
            },
            initialColor = Color(initialColor),
        )
    }
    @Composable fun slidersAndTextField() {
        AlphaSlider(
            modifier = Modifier
                .fillMaxWidth()
                .padding(10.dp)
                .height(barHeight),
            controller = controller,
            initialColor = Color(initialColor),
            wheelPaint = wheelPaint
        )
        BrightnessSlider(
            modifier = Modifier
                .fillMaxWidth()
                .padding(10.dp)
                .height(barHeight),
            controller = controller,
            initialColor = Color(initialColor),
            wheelPaint = wheelPaint
        )
        TextField(
            value = textValue,
            keyboardOptions = KeyboardOptions(
                autoCorrectEnabled = false,
                keyboardType = KeyboardType.Password, // todo: KeyboardType.Password is a crappy way of avoiding suggestions... is there really no way in compose?
                imeAction = ImeAction.Done,
            ),
            keyboardActions = KeyboardActions(onDone = { onDismissRequest(); onConfirmed(controller.selectedColor.value.toArgb()) }),
            onValueChange = {
                textValue = it
                val androidColor = runCatching { "#${it.text}".toColorInt() }.getOrNull()
                if (androidColor != null)
                    controller.selectByColor(Color(androidColor), false)
            }
        )
    }
    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = { onConfirmed(controller.selectedColor.value.toArgb()) },
        title = { Text(title) },
        content = {
            if (useWideLayout)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    picker()
                    Column {
                        topBar()
                        slidersAndTextField()
                    }
                }
            else
                Column {
                    topBar()
                    picker()
                    slidersAndTextField()
                }
        },
        neutralButtonText = if (showDefault) stringResource(R.string.button_default) else null,
        onNeutral = onDefault,
        properties = DialogProperties(usePlatformDefaultWidth = !useWideLayout)
    )
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        ColorPickerDialog({}, -0x0f4488aa, "color name", true, {}, {})
    }
}

// for some reason this is cut of while both previews are shown
@Preview(device = "spec:orientation=landscape,width=400dp,height=780dp")
@Composable
private fun WidePreview() {
    Theme(previewDark) {
        ColorPickerDialog({}, -0x0f4488aa, "color name", true, {}, {})
    }
}
