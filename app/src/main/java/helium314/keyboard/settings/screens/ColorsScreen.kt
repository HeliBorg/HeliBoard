// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import helium314.keyboard.keyboard.ColorSetting
import helium314.keyboard.keyboard.KeyboardTheme
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.ColorType
import helium314.keyboard.latin.common.default
import helium314.keyboard.latin.common.encodeBase36
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.CloseIcon
import helium314.keyboard.settings.SearchScreen
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.contentTextDirectionStyle
import helium314.keyboard.settings.dialogs.ColorPickerDialog
import helium314.keyboard.latin.utils.previewDark
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Composable
fun ColorsScreen(
    isNight: Boolean,
    theme: String?,
    onClickBack: () -> Unit
) {
    val ctx = LocalContext.current
    val prefs = ctx.prefs()
    val b = (ctx.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")

    val themeName = theme ?: if (isNight) prefs.getString(Settings.PREF_THEME_COLORS_NIGHT, Defaults.PREF_THEME_COLORS_NIGHT)!!
        else prefs.getString(Settings.PREF_THEME_COLORS, Defaults.PREF_THEME_COLORS)!!
    var newThemeName by rememberSaveable(stateSaver = TextFieldValue.Saver) { mutableStateOf(TextFieldValue(themeName)) }

    // is there really no better way of only setting forceOpposite while the screen is shown (and not paused)?
    // lifecycle stuff is weird, there is no pause and similar when activity is paused
    DisposableEffect(isNight) {
        onDispose { // works on pressing back
            (ctx.getActivity() as? SettingsActivity)?.setForceTheme(null, null)
        }
    }
    val lifecycleOwner = LocalLifecycleOwner.current
    val lifecycleState by lifecycleOwner.lifecycle.currentStateFlow.collectAsState()
    LaunchedEffect(lifecycleState) {
        if (lifecycleState == Lifecycle.State.RESUMED) {
            (ctx.getActivity() as? SettingsActivity)?.setForceTheme(newThemeName.text, isNight)
        }
    }

    val moreColors = KeyboardTheme.readUserMoreColors(prefs, newThemeName.text)
    val userColors = KeyboardTheme.readUserColors(prefs, newThemeName.text)
    val shownColors = if (moreColors == 2) {
        val fallbackColors = KeyboardTheme.readUserColorTheme(
            prefs.getString(Settings.PREF_THEME_STYLE, Defaults.PREF_THEME_STYLE)!!,
            prefs.getBoolean(Settings.PREF_THEME_KEY_BORDERS, Defaults.PREF_THEME_KEY_BORDERS),
        userColors, ctx, isNight, null
        )
        val allColors = KeyboardTheme.readUserAllColors(prefs, newThemeName.text, fallbackColors)
        ColorType.entries.map {
            ColorSetting(it.name, null, allColors[it] ?: it.default())
        }
    } else {
        val toDisplay = colorPrefsAndResIds.map { (colorName, resId) ->
            val cs = userColors.firstOrNull { it.name == colorName } ?: ColorSetting(colorName, true, null)
            cs.displayName = stringResource(resId)
            cs
        }
        val colorsToHide = getColorPrefsToHideInitially(prefs)
        if (moreColors == 1) toDisplay
        else toDisplay.filter { it.color != null || it.name !in colorsToHide }
    }
    fun ColorSetting.displayColor() = if (auto == true) KeyboardTheme.determineUserColor(userColors, ctx, name, isNight)
        else color ?: KeyboardTheme.determineUserColor(userColors, ctx, name, isNight)

    var chosenColorString: String by rememberSaveable { mutableStateOf("") }
    val chosenColor = runCatching { Json.decodeFromString<ColorSetting?>(chosenColorString) }.getOrNull()
    val saveLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@rememberLauncherForActivityResult
        val uri = result.data?.data ?: return@rememberLauncherForActivityResult
        ctx.getActivity()?.contentResolver?.openOutputStream(uri)?.writer()?.use { it.write(getColorString(prefs, newThemeName.text)) }
    }
    SearchScreen(
        title = {
            var nameValid by rememberSaveable { mutableStateOf(true) }
            var nameField by rememberSaveable(stateSaver = TextFieldValue.Saver) { mutableStateOf(newThemeName) }
            TextField(
                value = nameField,
                onValueChange = {
                    nameValid = KeyboardTheme.renameUserColors(newThemeName.text, it.text, prefs)
                    if (nameValid) {
                        newThemeName = it
                        SettingsActivity.forceTheme = newThemeName.text
                    }
                    nameField = it
                },
                isError = !nameValid,
//                supportingText = { if (!nameValid) Text(stringResource(R.string.name_invalid)) } // todo: this is cutting off bottom half of the actual text...
                trailingIcon = { if (!nameValid) CloseIcon(R.string.name_invalid) },
                singleLine = true,
                textStyle = contentTextDirectionStyle,
            )
        },
        menu = listOf(
            stringResource(R.string.main_colors) to { KeyboardTheme.writeUserMoreColors(prefs, newThemeName.text, 0) },
            stringResource(R.string.more_colors) to { KeyboardTheme.writeUserMoreColors(prefs, newThemeName.text, 1) },
            stringResource(R.string.all_colors) to { KeyboardTheme.writeUserMoreColors(prefs, newThemeName.text, 2) },
            stringResource(R.string.button_save_file) to {
                val intent = Intent(Intent.ACTION_CREATE_DOCUMENT)
                    .addCategory(Intent.CATEGORY_OPENABLE)
                    .putExtra(Intent.EXTRA_TITLE,"${newThemeName.text}.json")
                    .setType("application/json")
                saveLauncher.launch(intent)
            },
            stringResource(R.string.copy_to_clipboard) to {
                val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("HeliBoard theme", getColorString(prefs, newThemeName.text)))
            },
        ),
        onClickBack = onClickBack,
        filteredItems = { search ->
            val result = shownColors.filter { color ->
                color.displayName.split(" ", "_").any { it.startsWith(search, true) }
            }
            if (moreColors == 2) result.toMutableList<ColorSetting?>().apply { add(0, null) }
            else result
        },
        itemContent = { colorSetting ->
            if (colorSetting == null)
                Text( // not a colorSetting, but still best done as part of the list
                    stringResource(R.string.all_colors_warning),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            else
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                        .clickable { chosenColorString = Json.encodeToString(colorSetting) }
                ) {
                    Spacer(
                        modifier = Modifier
                            .background(Color(colorSetting.displayColor()), shape = CircleShape)
                            .size(50.dp)
                    )
                    Column(Modifier
                        .weight(1f)
                        .padding(horizontal = 16.dp)) {
                        Text(colorSetting.displayName)
                        if (colorSetting.auto == true)
                            CompositionLocalProvider(
                                LocalTextStyle provides MaterialTheme.typography.bodyMedium,
                                LocalContentColor provides MaterialTheme.colorScheme.onSurfaceVariant
                            ) {
                                Text(stringResource(R.string.auto_user_color))
                            }
                    }
                    if (colorSetting.auto != null)
                        Switch(colorSetting.auto, onCheckedChange = { checked ->
                            val oldUserColors = KeyboardTheme.readUserColors(prefs, newThemeName.text)
                            val newUserColors = (oldUserColors + ColorSetting(colorSetting.name, checked, colorSetting.color))
                                .reversed().distinctBy { it.displayName }
                            KeyboardTheme.writeUserColors(prefs, newThemeName.text, newUserColors)
                        })
                }
        }
    )
    if (chosenColor != null) {
        val oldAllColors = KeyboardTheme.readUserAllColors(prefs, newThemeName.text, null)
        ColorPickerDialog(
            onDismissRequest = { chosenColorString = "" },
            initialColor = chosenColor.displayColor(),
            title = chosenColor.displayName,
            showDefault = moreColors == 2 && oldAllColors.contains(ColorType.valueOf(chosenColor.name)),
            onDefault = {
                // clear the color
                oldAllColors.remove(ColorType.valueOf(chosenColor.name))
                KeyboardTheme.writeUserAllColors(prefs, newThemeName.text, oldAllColors)
            }
        ) { color ->
            if (moreColors == 2) {
                oldAllColors[ColorType.valueOf(chosenColor.name)] = color
                KeyboardTheme.writeUserAllColors(prefs, newThemeName.text, oldAllColors)
            } else {
                val oldUserColors = KeyboardTheme.readUserColors(prefs, newThemeName.text)
                val newUserColors = (oldUserColors + ColorSetting(chosenColor.name, false, color))
                    .reversed().distinctBy { it.displayName }
                KeyboardTheme.writeUserColors(prefs, newThemeName.text, newUserColors)
            }
        }
    }
}

private fun getColorString(prefs: SharedPreferences, themeName: String): String {
    val moreColors = KeyboardTheme.readUserMoreColors(prefs, themeName)
    if (moreColors == 2) {
        val colors = KeyboardTheme.readUserAllColors(prefs, themeName, null).map { it.key.name to it.value }
        return Json.encodeToString((colors + (encodeBase36(themeName) to 0)).toMap()) // put theme name in here too
    }
    val colors = KeyboardTheme.readUserColors(prefs, themeName).associate { it.name to (it.color to (it.auto == true)) }
    return Json.encodeToString(SaveThoseColors(themeName, moreColors, colors))
}

@Serializable
data class SaveThoseColors(val name: String? = null, val moreColors: Int, val colors: Map<String, Pair<Int?, Boolean>>)

val colorPrefsAndResIds = listOf(
    KeyboardTheme.COLOR_BACKGROUND to R.string.select_color_background,
    KeyboardTheme.COLOR_KEYS to R.string.select_color_key_background,
    KeyboardTheme.COLOR_FUNCTIONAL_KEYS to R.string.select_color_functional_key_background,
    KeyboardTheme.COLOR_SPACEBAR to R.string.select_color_spacebar_background,
    KeyboardTheme.COLOR_TEXT to R.string.select_color_key,
    KeyboardTheme.COLOR_HINT_TEXT to R.string.select_color_key_hint,
    KeyboardTheme.COLOR_SUGGESTION_TEXT to R.string.select_color_suggestion,
    KeyboardTheme.COLOR_SPACEBAR_TEXT to R.string.select_color_spacebar_text,
    KeyboardTheme.COLOR_ACCENT to R.string.select_color_accent,
    KeyboardTheme.COLOR_GESTURE to R.string.select_color_gesture,
)

private fun getColorPrefsToHideInitially(prefs: SharedPreferences): List<String> {
    return listOf(KeyboardTheme.COLOR_SUGGESTION_TEXT, KeyboardTheme.COLOR_SPACEBAR_TEXT, KeyboardTheme.COLOR_GESTURE) +
            if (prefs.getBoolean(Settings.PREF_THEME_KEY_BORDERS, false)) listOf(KeyboardTheme.COLOR_SPACEBAR_TEXT)
            else listOf(KeyboardTheme.COLOR_FUNCTIONAL_KEYS)
}

@Preview
@Composable
private fun Preview() {
    Theme(previewDark) {
        Surface {
            ColorsScreen(false, null) { }
        }
    }
}
