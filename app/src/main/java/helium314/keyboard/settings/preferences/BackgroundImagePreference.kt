// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.FileUtils
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.IntentUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.dialogs.InfoDialog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@Composable
fun BackgroundImagePref(setting: Setting, isLandscape: Boolean) {
    var showDayNightDialog by rememberSaveable { mutableStateOf(false) }
    var showSelectionDialog by rememberSaveable { mutableStateOf(false) }
    var showErrorDialog by rememberSaveable { mutableStateOf(false) }
    var isNight by rememberSaveable { mutableStateOf(false) }
    val ctx = LocalContext.current
    fun getFile() = Settings.getCustomBackgroundFile(ctx, isNight, isLandscape)
    val b = (ctx.getActivity() as? SettingsActivity)?.prefChanged?.collectAsState()
    if ((b?.value ?: 0) < 0) // necessary to reload dayNightPref
        Log.v("irrelevant", "stupid way to trigger recomposition on preference change")
    val dayNightPref = ctx.prefs().getBoolean(Settings.PREF_THEME_DAY_NIGHT, Defaults.PREF_THEME_DAY_NIGHT)
    if (!dayNightPref)
        isNight = false
    val scope = rememberCoroutineScope { Dispatchers.IO }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@rememberLauncherForActivityResult
        val uri = result.data?.data ?: return@rememberLauncherForActivityResult
        showSelectionDialog = false
        showDayNightDialog = false
        scope.launch(Dispatchers.IO) {
            if (!setBackgroundImage(ctx, uri, isNight, isLandscape))
                showErrorDialog = true
        }
    }
    val intent = IntentUtils.getResolvableTypeIntent(ctx, "image/*")
        .setAction(Intent.ACTION_OPEN_DOCUMENT)
    Preference(
        name = setting.title,
        onClick = {
            if (dayNightPref) {
                showDayNightDialog = true
            } else if (!getFile().exists()) {
                launcher.launch(intent)
            } else {
                showSelectionDialog = true
            }
        }
    )
    if (showDayNightDialog) {
        ConfirmationDialog(
            onDismissRequest = { showDayNightDialog = false },
            onConfirmed = {
                isNight = false
                if (getFile().exists())
                    showSelectionDialog = true
                else launcher.launch(intent)
            },
            confirmButtonText = stringResource(R.string.day_or_night_day),
            cancelButtonText = "",
            onNeutral = {
                isNight = true
                if (getFile().exists())
                    showSelectionDialog = true
                else launcher.launch(intent)
            },
            neutralButtonText = stringResource(R.string.day_or_night_night),
            title = { Text(stringResource(R.string.day_or_night_image)) },
        )
    }
    if (showSelectionDialog) {
        ConfirmationDialog(
            onDismissRequest = { showSelectionDialog = false },
            title = { Text(stringResource(R.string.customize_background_image)) },
            confirmButtonText = stringResource(R.string.button_load_custom),
            onConfirmed = { launcher.launch(intent) },
            neutralButtonText = stringResource(R.string.delete),
            onNeutral = {
                getFile().delete()
                Settings.clearCachedBackgroundImages()
                KeyboardSwitcher.getInstance().setThemeNeedsReload()
                showSelectionDialog = false
            }
        )
    }
    if (showErrorDialog) {
        InfoDialog(stringResource(R.string.file_read_error)) { showErrorDialog = false }
    }
}

private fun setBackgroundImage(ctx: Context, uri: Uri, isNight: Boolean, isLandscape: Boolean): Boolean {
    val imageFile = Settings.getCustomBackgroundFile(ctx, isNight, isLandscape)
    FileUtils.copyContentUriToNewFile(uri, ctx, imageFile)
    KeyboardSwitcher.getInstance().setThemeNeedsReload()
    try {
        BitmapFactory.decodeFile(imageFile.absolutePath)
    } catch (_: Exception) {
        imageFile.delete()
        return false
    }
    Settings.clearCachedBackgroundImages()
    return true
}
