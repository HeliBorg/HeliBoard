// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.createPrefKeyForBooleanSettings
import helium314.keyboard.latin.utils.FoldableUtils
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.WithSmallTitle
import helium314.keyboard.settings.dialogs.ThreeButtonAlertDialog
import helium314.keyboard.latin.utils.previewDark
import androidx.core.content.edit

// too specialized for using a more generic dialog
// actual key for each setting is baseKey with one _true/_false appended per dimension (need to keep order!)
// todo: possible adjustments, maybe depending on user feedback
//  should dimension checkboxes have any other effect than just showing / hiding sliders?
//   one could argue that e.g. when disabling the split checkbox, then split mode should not affect the setting
//  store checkbox states?
//   if so, per setting or global?
//  show a description? currently commented because it could get long, even without showing the variations
//   maybe if we store the checkbox state in a setting, we could use it for determining what to show
@Composable
fun MultiSliderPreference(
    name: String,
    baseKey: String,
    dimensions: List<String>,
    defaults: Array<Float>,
    range:  ClosedFloatingPointRange<Float>,
    description: (Float) -> String,
    onDone: () -> Unit
) {
    if (defaults.size != 1.shl(dimensions.size))
        throw ArithmeticException("defaults size does not match with dimensions, expected ${1.shl(dimensions.size)}, got ${defaults.size}")
    var showDialog by remember { mutableStateOf(false) }
    //val (_, keys) = remember { createVariantsAndKeys(dimensions, baseKey) }
    //val prefs = LocalContext.current.prefs()
    Preference(
        name = name,
        onClick = { showDialog = true },
        //description = keys.mapIndexed { i, it -> description(prefs.getFloat(it, defaults[i])) }.joinToString(" $SPLIT ")
    )
    if (showDialog)
        MultiSliderDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(name) },
            baseKey = baseKey,
            onDone = onDone,
            defaultValues = defaults,
            range = range,
            dimensions = dimensions,
            positionString = description
        )
}

// SliderDialog, but for multiple sliders with same range, each with a different setting and title
@Composable
private fun MultiSliderDialog(
    onDismissRequest: () -> Unit,
    title: @Composable () -> Unit,
    baseKey: String,
    onDone: () -> Unit,
    defaultValues: Array<Float>,
    range: ClosedFloatingPointRange<Float>,
    dimensions: List<String>,
    modifier: Modifier = Modifier,
    positionString: (Float) -> String,
) {
    val (variants, keys) = createVariantsAndKeys(dimensions, baseKey)
    val foldedString = stringResource(R.string.folded)
    val ctx = LocalContext.current
    var checked by remember { mutableStateOf(dimensions.map { FoldableUtils.isFoldable(ctx) || !it.contains(foldedString) }) }
    val prefs = ctx.prefs()
    val done = remember { mutableMapOf<String, () -> Unit>() }

    ThreeButtonAlertDialog(
        onDismissRequest = onDismissRequest,
        onConfirmed = { done.values.forEach { it.invoke() }; onDone() },
        modifier = modifier,
        title = title,
        content = {
            CompositionLocalProvider(
                LocalTextStyle provides MaterialTheme.typography.bodyLarge
            ) {
                val state = rememberScrollState()
                Column(Modifier.verticalScroll(state)) {
                    if (dimensions.size > 1) {
                        dimensions.forEachIndexed { i, dimension ->
                            DimensionCheckbox(checked[i], dimension) {
                                checked = checked.mapIndexed { j, c -> if (i == j) it else c }
                            }
                        }
                    }
                    variants.forEachIndexed { i, variant ->
                        val key = keys[i]
                        var sliderPosition by remember { mutableFloatStateOf(prefs.getFloat(key, defaultValues[i])) }
                        if (!done.contains(variant))
                            done[variant] = {
                                if (sliderPosition == defaultValues[i])
                                    prefs.edit { remove(key) }
                                else
                                    prefs.edit { putFloat(key, sliderPosition) }
                            }
                        val forbiddenDimensions = dimensions.filterIndexed { index, _ -> !checked[index] }
                        val visible = variant.split(SPLIT).none { it in forbiddenDimensions }
                        // default animations make the dialog flash (see also DictionaryDialog)
                        AnimatedVisibility(visible, exit = fadeOut(), enter = fadeIn()) {
                            WithSmallTitle(variant.ifEmpty { stringResource(R.string.button_default) }) {
                                Slider(
                                    value = sliderPosition,
                                    onValueChange = { sliderPosition = it },
                                    valueRange = range,
                                )
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(positionString(sliderPosition))
                                    TextButton({ sliderPosition = defaultValues[i] }) { Text(stringResource(R.string.button_default)) }
                                }
                                Spacer(Modifier.height(6.dp))
                            }
                        }
                    }
                }
            }
        },
    )
}

@Composable
private fun DimensionCheckbox(checked: Boolean, dimension: String, onCheckedChange: (Boolean) -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth().clickable { onCheckedChange(!checked) }
    ) {
        Checkbox(
            checked = checked,
            onCheckedChange = { onCheckedChange(it) }
        )
        Text(dimension)
    }
}

private fun createVariantsAndKeys(dimensions: List<String>, baseKey: String): Pair<List<String>, List<String>> {
    val variants = mutableListOf("")
    val keys = mutableListOf(createPrefKeyForBooleanSettings(baseKey, 0, dimensions.size))
    var i = 1
    dimensions.forEach { dimension ->
        variants.toList().forEach { variant ->
            if (variant.isEmpty()) variants.add(dimension)
            else variants.add(variant + SPLIT + dimension)
            keys.add(createPrefKeyForBooleanSettings(baseKey, i, dimensions.size))
            i++
        }
    }
    return variants to keys
}

private const val SPLIT = " / "

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        MultiSliderDialog(
            onDismissRequest = { },
            onDone = { },
            positionString = { "${it.toInt()}%"},
            defaultValues = Array(8) { 100f - it % 2 * 50f },
            range = 0f..500f,
            title = { Text("bottom padding scale") },
            dimensions = listOf("landscape", "unfolded", "split"),
            baseKey = ""
        )
    }
}
