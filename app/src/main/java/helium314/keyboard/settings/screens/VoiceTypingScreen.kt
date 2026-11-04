package helium314.keyboard.settings.screens

import android.annotation.SuppressLint
import android.content.Context
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.previewDark
import helium314.keyboard.settings.SearchSettingsScreen
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.initPreview

@Composable
fun VoiceTypingScreen(
    onClickBack: () -> Unit,
) {
    val b = (LocalContext.current.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0)
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    val items = listOf(
        Settings.PREF_ALWAYS_INCOGNITO_MODE,
    )
    SearchSettingsScreen(
        onClickBack = onClickBack,
        title = stringResource(R.string.voice),
        settings = items
    )
}

@SuppressLint("ApplySharedPref")
fun createVoiceTypingSettings(context: Context) = listOf<Setting>(
//    Setting(context, Settings.PREF_ALWAYS_INCOGNITO_MODE,
//        R.string.incognito, R.string.prefs_force_incognito_mode_summary)
//    {
//        SwitchPreference(it, Defaults.PREF_ALWAYS_INCOGNITO_MODE) { KeyboardSwitcher.getInstance().setThemeNeedsReload() }
//    },
)

@Preview
@Composable
private fun Preview() {
    initPreview(LocalContext.current)
    Theme(previewDark) {
        Surface {
            VoiceTypingScreen { }
        }
    }
}
