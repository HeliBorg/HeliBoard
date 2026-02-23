// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.screens.gesturedata

import android.content.ClipData
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import androidx.activity.compose.ManagedActivityResultLauncher
import androidx.activity.result.ActivityResult
import androidx.annotation.RequiresApi
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.content.FileProvider
import androidx.core.net.toUri
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.GestureDataDao
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.filePicker
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileInputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

@Composable
fun ShareGestureData(ids: List<Long>) {
    val ctx = LocalContext.current
    val dao = GestureDataDao.getInstance(ctx)!!
    val hasData = !dao.isEmpty() // no need to update if we have it in a dialog
    val getDataPicker = getData(ids)
    var exportStarted by remember { mutableStateOf(false) }
    var exportDone by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    LaunchedEffect(exportStarted) {
        if (exportStarted) {
            // wait until exported data is marked, then offer to delete
            gestureIdsBeingExported = ids
            scope.launch {
                while (gestureIdsBeingExported != null) {
                    delay(50)
                }
                exportDone = true
            }
        }
    }

    if (!exportDone) {
        // share file, but only to mail apps
        ButtonWithText(
            stringResource(R.string.gesture_data_send_mail),
            enabled = hasData && Intent(Intent.ACTION_SENDTO)
                .apply { data = "mailto:".toUri() }.resolveActivity(ctx.packageManager) != null
        ) {
            createZipFile(ctx, ids)
            if (zippedDataPath.isNotEmpty()) {
                exportStarted = true
                ctx.startActivity(createSendIntentChooser(ctx))
            }
        }

        // get file
        ButtonWithText(stringResource(R.string.gesture_data_get_data), enabled = hasData) {
            getDataPicker.launch(getDataIntent)
            exportStarted = true
        }

        // copy mail address to clipboard, in case user doesn't use the mail intent
        val clip = LocalClipboard.current
        ButtonWithText(stringResource(R.string.gesture_data_copy_mail)) {
            scope.launch { clip.setClipEntry(ClipEntry(ClipData.newPlainText("mail address", deobfuscateEmail(ctx)))) }
        }
        Text(stringResource(R.string.gesture_data_mail_use))
    } else {
        // this deletes the data in the dialog, but we should also have a way of deleting previously exported data
        var confirmDelete by remember { mutableStateOf(false) }
        ButtonWithText(stringResource(R.string.gesture_data_delete_dialog_submitted, ids.size), enabled = hasData) {
            confirmDelete = true
        }
        if (confirmDelete) {
            ConfirmationDialog(
                onDismissRequest = { confirmDelete = false },
                onConfirmed = { dao.delete(ids, true, ctx) },
                content = {
                    Text(stringResource(R.string.delete_confirmation, ids.size))
                }
            )
        }
    }
}

private val getDataIntent = Intent(Intent.ACTION_CREATE_DOCUMENT)
    .addCategory(Intent.CATEGORY_OPENABLE)
    .putExtra(Intent.EXTRA_TITLE, "gesture_data_${SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Calendar.getInstance().time)}.zip")
    .setType("application/zip")

private fun deobfuscateEmail(context: Context): String {
    return context.getString(R.string.gesture_data_mail)
        .map { c ->  Char(c.code-2) }
        .joinToString(separator = "")
}

private fun createSendIntentChooser(context: Context): Intent {
    // targets all apps that accept this kind of intent
    val shareFileIntent = Intent(Intent.ACTION_SEND).apply {
        type = "application/zip"
        putExtra(Intent.EXTRA_EMAIL, arrayOf(deobfuscateEmail(context)))
        putExtra(Intent.EXTRA_SUBJECT, MAIL_SUBJECT)
        putExtra(Intent.EXTRA_TEXT, context.getString(R.string.gesture_data_email_body))
        putExtra(Intent.EXTRA_STREAM, getZipFileUri(context))
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }

    val chooser = Intent.createChooser(shareFileIntent, "share it").apply {
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

        // if you are on an android older than N... you will have to just find your email in the list of apps!
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            filterIntentToOnlyIncludeEmailApps(context, shareFileIntent)
        }
    }
    return chooser
}

@RequiresApi(Build.VERSION_CODES.N)
private fun Intent.filterIntentToOnlyIncludeEmailApps(context: Context, intentToFilter: Intent) {
    val sendMailIntent = Intent(Intent.ACTION_SENDTO).apply { data = "mailto:".toUri() }
    val mailTargets = context.packageManager.queryIntentActivities(sendMailIntent, 0)
    val shareTargets = context.packageManager.queryIntentActivities(intentToFilter, 0)

    // so now we get the shareTargets that are not mailTargets and create ComponentNames...
    val mailTargetPackages = mailTargets.map { it.activityInfo.packageName }
    val nonMailShareTargets = shareTargets.filterNot { it.activityInfo.packageName in mailTargetPackages }
    val nonMailShareTargetComponentNames = nonMailShareTargets.map { ComponentName(it.activityInfo.packageName, it.activityInfo.name) }

    // ... because the chooser only allows to exclude components, but has no multi-component include functionality
    putExtra(Intent.EXTRA_EXCLUDE_COMPONENTS, nonMailShareTargetComponentNames.toTypedArray())
}

private fun getZipFileUri(context: Context) : Uri =
    FileProvider.getUriForFile(
        context,
        context.getString(R.string.gesture_data_provider_authority),
        getGestureZipFile(context))

@Composable
private fun getData(ids: List<Long>): ManagedActivityResultLauncher<Intent, ActivityResult> {
    val ctx = LocalContext.current
    // check if file exists and not size 0, otherwise don't even offer the button
    return filePicker { uri ->
        val file = getGestureDataFile(ctx)
        val dao = GestureDataDao.getInstance(ctx) ?: return@filePicker
        val data = dao.getJsonData(ids)
        file.writeText(makeJsonArrayFromEntries(data))
        ctx.getActivity()?.contentResolver?.openOutputStream(uri)?.use {
            os -> writeOutputToZipStream(file, os)
        }
        dao.markAsExported(ids, ctx)
        gestureIdsBeingExported = null
    }
}

private fun createZipFile(ctx: Context, ids: List<Long>) : File {
    zippedDataPath = ""
    val jsonFile = getGestureDataFile(ctx)
    val dao = GestureDataDao.getInstance(ctx)!!
    val data = dao.getJsonData(ids)
    jsonFile.writeText(makeJsonArrayFromEntries(data))
    val zipFile = getGestureZipFile(ctx)
    zipFile.delete()
    zipFile.outputStream().use {
        os -> writeOutputToZipStream(jsonFile, os)
    }
    zippedDataPath = zipFile.absolutePath
    return zipFile
}

private fun writeOutputToZipStream(jsonFile: File, os: OutputStream) {
    val zipStream = ZipOutputStream(os)
    zipStream.setLevel(9)
    val fileStream = FileInputStream(jsonFile).buffered()
    zipStream.putNextEntry(ZipEntry(jsonFile.name))
    fileStream.copyTo(zipStream, 1024)
    fileStream.close()
    zipStream.closeEntry()
    zipStream.close()
}

private fun makeJsonArrayFromEntries(entries: List<String>): String
    = buildString {
    val allButLast = entries.subList(0, entries.lastIndex)

    append("[\n")
    allButLast.forEach {
        append(it)
        append(",\n")
    }
    append(entries.last())
    append("\n]")
}

private fun getGestureZipFile(context: Context): File = fileGetDelegate(context, context.getString(R.string.gesture_data_zip))

private fun getGestureDataFile(context: Context): File = fileGetDelegate(context, context.getString(R.string.gesture_data_json))

private fun fileGetDelegate(context: Context, filename: String): File {
    // ensure folder exists
    val dir = File(context.filesDir, context.getString(R.string.gesture_data_directory))
    if (!dir.exists()) {
        dir.mkdirs()
    }

    return File(dir, filename)
}

// necessary for giving the mail app access to an internal file
// overrides to check when data is actually read (i.e. chooser not cancelled)
class GestureFileProvider : FileProvider() {
    override fun openFile(uri: Uri, mode: String, signal: CancellationSignal?): ParcelFileDescriptor? {
        try {
            return super.openFile(uri, mode, signal)
        } finally {
            onFileOpened()
        }
    }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor? {
        try {
            return super.openFile(uri, mode)
        } finally {
            onFileOpened()
        }
    }

    private fun onFileOpened() {
        val ctx = context
        val ids = gestureIdsBeingExported
        if (ctx != null && ids != null) {
            GestureDataDao.getInstance(ctx)?.markAsExported(ids, ctx)
        }
        gestureIdsBeingExported = null
    }
}

private var gestureIdsBeingExported: List<Long>? = null

private var zippedDataPath = "" // set after writing the file

private const val MAIL_SUBJECT = "HeliBoard ${BuildConfig.VERSION_NAME} gesture data"
