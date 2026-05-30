// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.JniUtils
import helium314.keyboard.latin.utils.SubtypeLocaleUtils.displayName
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.NextScreenIcon
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.initPreview
import helium314.keyboard.settings.preferences.Preference
import helium314.keyboard.latin.utils.previewDark
import helium314.keyboard.settings.screens.gesturedata.END_DATE_EPOCH_MILLIS
import helium314.keyboard.settings.screens.gesturedata.TWO_WEEKS_IN_MILLIS

@Composable
fun MainSettingsScreen(
    onClickAbout: () -> Unit,
    onClickTextCorrection: () -> Unit,
    onClickPreferences: () -> Unit,
    onClickToolbar: () -> Unit,
    onClickGestureTyping: () -> Unit,
    onClickTwoThumbTyping: () -> Unit,
    onClickAdvanced: () -> Unit,
    onClickAppearance: () -> Unit,
    onClickLanguage: () -> Unit,
    onClickLayouts: () -> Unit,
    onClickDictionaries: () -> Unit,
    onClickBack: () -> Unit,
) {
    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.ime_settings),
        settings = emptyList(),
    ) {
        val enabledSubtypes = SubtypeSettings.getEnabledSubtypes(true)
        Scaffold(contentWindowInsets = WindowInsets.safeDrawing.only(WindowInsetsSides.Bottom)) { innerPadding ->
            Column(
                Modifier.verticalScroll(rememberScrollState()).then(Modifier.padding(innerPadding))
            ) {
                // Group 1: General (AI, Languages, Preferences, Appearance, Toolbar)
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer
                    )
                ) {
                    Column {
                        if (BuildConfig.FLAVOR != "offlinelite") {
                            Preference(
                                name = stringResource(R.string.settings_screen_ai_integration),
                                onClick = onClickAIIntegration,
                                icon = R.drawable.ic_proofread
                            ) { NextScreenIcon() }
                        }
                        Preference(
                            name = stringResource(R.string.libraries_hub_title),
                            onClick = onClickLibraries,
                            icon = R.drawable.ic_emoji_objects
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.language_and_layouts_title),
                            description = enabledSubtypes.joinToString(", ") { it.displayName() },
                            onClick = onClickLanguage,
                            icon = R.drawable.ic_settings_languages
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.settings_screen_preferences),
                            onClick = onClickPreferences,
                            icon = R.drawable.ic_settings_preferences
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.settings_screen_appearance),
                            onClick = onClickAppearance,
                            icon = R.drawable.ic_settings_appearance
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.settings_screen_toolbar),
                            onClick = onClickToolbar,
                            icon = R.drawable.ic_settings_toolbar
                        ) { NextScreenIcon() }
                    }
                }

                // Group 2: Typing (Gesture, Correction, Secondary Layouts, Dictionaries)
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer
                    )
                ) {
                    Column {
                        Preference(
                            name = stringResource(R.string.settings_screen_gesture),
                            onClick = onClickGestureTyping,
                            icon = R.drawable.ic_settings_gesture
                        ) { NextScreenIcon() }
                        // Dedicated entry for the experimental two-thumb typing features so
                        // they're discoverable without spelunking through Gesture Typing's
                        // sub-options. The screen itself short-circuits to empty when gesture
                        // typing is off / the lib isn't loaded — see TwoThumbTypingScreen.
                        Preference(
                            name = stringResource(R.string.settings_screen_two_thumb_typing),
                            description = stringResource(R.string.settings_screen_two_thumb_typing_summary),
                            onClick = onClickTwoThumbTyping,
                            icon = R.drawable.ic_settings_gesture
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.settings_screen_correction),
                            onClick = onClickTextCorrection,
                            icon = R.drawable.ic_settings_correction
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.settings_screen_secondary_layouts),
                            onClick = onClickLayouts,
                            icon = R.drawable.ic_ime_switcher
                        ) { NextScreenIcon() }
                    }
                }

                // Group 3: Other (Advanced, About)
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer
                    )
                ) {
                    Column {
                        Preference(
                            name = stringResource(R.string.settings_screen_advanced),
                            onClick = onClickAdvanced,
                            icon = R.drawable.ic_settings_advanced
                        ) { NextScreenIcon() }
                        Preference(
                            name = stringResource(R.string.settings_screen_about),
                            onClick = onClickAbout,
                            icon = R.drawable.ic_settings_about
                        ) { NextScreenIcon() }
                    }
                }
            }
        }
    }
}

@Preview
@Composable
private fun PreviewScreen() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            MainSettingsScreen(
                onClickAbout = {},
                onClickTextCorrection = {},
                onClickPreferences = {},
                onClickToolbar = {},
                onClickGestureTyping = {},
                onClickTwoThumbTyping = {},
                onClickAdvanced = {},
                onClickAppearance = {},
                onClickLanguage = {},
                onClickLayouts = {},
                onClickDictionaries = {},
                onClickBack = {},
            )
        }
    }
}
