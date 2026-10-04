// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Build
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.FileUtils
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ChecksumCalculator
import helium314.keyboard.latin.utils.JniUtils
import helium314.keyboard.latin.utils.protectedPrefs
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.filePicker
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import androidx.core.content.edit
import helium314.keyboard.latin.utils.IntentUtils

@SuppressLint("ApplySharedPref")
@Composable
fun LoadGestureLibPreference(setting: Setting) {
    var showDialog by rememberSaveable { mutableStateOf(false) }
    val ctx = LocalContext.current
    val prefs = ctx.protectedPrefs()
    val abi = Build.SUPPORTED_ABIS[0]
    val libFile = File(ctx.filesDir?.absolutePath + File.separator + JniUtils.JNI_LIB_IMPORT_FILE_NAME)
    fun renameToLibFileAndRestart(file: File, checksum: String) {
        libFile.setWritable(true)
        libFile.delete()
        // store checksum in default preferences (see JniUtils)
        prefs.edit(commit = true) { putString(Settings.PREF_LIBRARY_CHECKSUM, checksum) }
        file.copyTo(libFile)
        libFile.setReadOnly()
        file.delete()
        Runtime.getRuntime().exit(0) // exit will restart the app, so library will be loaded
    }
    var tempFilePath: String? by rememberSaveable { mutableStateOf(null) }
    val launcher = filePicker { uri ->
        val tmpfile = File(ctx.filesDir.absolutePath + File.separator + "tmplib")
        try {
            val otherTemporaryFile = File(ctx.filesDir.absolutePath + File.separator + "tmpfile")
            FileUtils.copyContentUriToNewFile(uri, ctx, otherTemporaryFile)
            val inputStream = FileInputStream(otherTemporaryFile)
            val outputStream = FileOutputStream(tmpfile)
            outputStream.use {
                tmpfile.setReadOnly() // as per recommendations in https://developer.android.com/about/versions/14/behavior-changes-14#safer-dynamic-code-loading
                FileUtils.copyStreamToOtherStream(inputStream, it)
            }
            otherTemporaryFile.delete()

            val checksum = ChecksumCalculator.checksum(tmpfile) ?: ""
            if (checksum == JniUtils.expectedDefaultChecksum()) {
                renameToLibFileAndRestart(tmpfile, checksum)
            } else {
                tempFilePath = tmpfile.absolutePath
            }
        } catch (e: IOException) {
            tmpfile.delete()
            // should inform user, but probably the issues will only come when reading the library
        }
    }
    Preference(
        name = setting.title,
        onClick = { showDialog = true }
    )
    if (showDialog) {
        ConfirmationDialog(
            onDismissRequest = { showDialog = false },
            onConfirmed = {
                showDialog = false
                val intent = IntentUtils.getResolvableTypeIntent(ctx, "application/octet-stream")
                    .setAction(Intent.ACTION_OPEN_DOCUMENT)
                launcher.launch(intent)
            },
            confirmButtonText = stringResource(R.string.load_gesture_library_button_load),
            title = { Text(stringResource(R.string.load_gesture_library)) },
            content = { Text(stringResource(R.string.load_gesture_library_message, abi)) },
            neutralButtonText = if (libFile.exists()) stringResource(R.string.load_gesture_library_button_delete) else null,
            onNeutral = {
                libFile.delete()
                prefs.edit(commit = true) { remove(Settings.PREF_LIBRARY_CHECKSUM) }
                Runtime.getRuntime().exit(0)
            }
        )
    }
    if (tempFilePath != null)
        ConfirmationDialog(
            onDismissRequest = {
                File(tempFilePath!!).delete()
                tempFilePath = null
            },
            content = { Text(stringResource(R.string.checksum_mismatch_message, abi)) },
            onConfirmed = {
                val tempFile = File(tempFilePath!!)
                renameToLibFileAndRestart(tempFile, ChecksumCalculator.checksum(tempFile) ?: "")
            }
        )
}
