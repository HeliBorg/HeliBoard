// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings

import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.res.stringResource
import helium314.keyboard.compat.locale
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.InputAttributes
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.FileUtils
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.BackButton
import helium314.keyboard.latin.utils.DeviceProtectedUtils
import helium314.keyboard.latin.utils.ExecutorUtils
import helium314.keyboard.latin.utils.JniUtils
import helium314.keyboard.latin.utils.GestureDataPromotionReminderDialog
import helium314.keyboard.latin.utils.Theme
import helium314.keyboard.latin.utils.UncachedInputMethodManagerUtils
import helium314.keyboard.latin.utils.cleanUnusedMainDicts
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.dialogs.NewDictionaryDialog
import helium314.keyboard.settings.screens.gesturedata.END_DATE_EPOCH_MILLIS
import helium314.keyboard.settings.screens.gesturedata.TWO_WEEKS_IN_MILLIS
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

// todo: with compose, app startup is slower and UI needs some "warmup" time to be snappy
//  maybe baseline profiles help?
//  https://developer.android.com/codelabs/android-baseline-profiles-improve
//  https://developer.android.com/codelabs/jetpack-compose-performance#2
//  https://developer.android.com/topic/performance/baselineprofiles/overview
// todo: consider viewModel, at least for LanguageScreen and ColorsScreen it might help making them less awkward and complicated
open class SettingsActivity : ComponentActivity(), SharedPreferences.OnSharedPreferenceChangeListener {
    private val prefs by lazy { this.prefs() }
    val prefChanged = MutableStateFlow(0) // simple counter, as the only relevant information is that something changed
    fun prefChanged() = prefChanged.value++
    private val dictUriFlow = MutableStateFlow<Uri?>(null)
    private val cachedDictionaryFile by lazy { File(this.cacheDir.path + File.separator + "temp_dict") }
    private val crashReportFiles = MutableStateFlow<List<File>>(emptyList())
    private var paused = true

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Settings.getValues() == null) {
            val inputAttributes = InputAttributes(EditorInfo(), false, packageName)
            Settings.getInstance().loadSettings(this, resources.configuration.locale(), inputAttributes)
        }
        ExecutorUtils.getBackgroundExecutor(ExecutorUtils.KEYBOARD).execute { cleanUnusedMainDicts(this) }
        crashReportFiles.value = findCrashReports(!BuildConfig.DEBUG && !DebugFlags.DEBUG_ENABLED)
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager

        settingsContainer = SettingsContainer(this)

        val spellchecker = intent?.getBooleanExtra("spellchecker", false) ?: false

        val cv = ComposeView(context = this)
        setContentView(cv)
        cv.setContent {
            Theme {
                Surface {
                    val dictUri by dictUriFlow.collectAsState()
                    val crashReports by crashReportFiles.collectAsState()
                    val crashFilePicker = filePicker { saveCrashReports(it) }
                    var showWelcomeWizard by rememberSaveable { mutableStateOf(
                        !UncachedInputMethodManagerUtils.isThisImeCurrent(this, imm)
                                || !UncachedInputMethodManagerUtils.isThisImeEnabled(this, imm)
                    ) }
                    if (spellchecker)
                        Scaffold(contentWindowInsets = WindowInsets.safeDrawing) { innerPadding ->
                            Column(Modifier.padding(innerPadding)) {
                                TopAppBar(
                                    title = { Text(stringResource(R.string.android_spell_checker_settings)) },
                                    windowInsets = WindowInsets(0),
                                    navigationIcon = {
                                        BackButton { this@SettingsActivity.finish() }
                                    },
                                )
                                settingsContainer[Settings.PREF_USE_CONTACTS]!!.Preference()
                                settingsContainer[Settings.PREF_USE_APPS]!!.Preference()
                                settingsContainer[Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE]!!.Preference()
                            }
                        }
                    else {
                        SettingsNavHost(onClickBack = { this.finish() })
                        if (showWelcomeWizard) {
                            WelcomeWizard(close = { showWelcomeWizard = false }, finish = this::finish)
                        } else if (crashReports.isNotEmpty()) {
                            ConfirmationDialog(
                                cancelButtonText = "ignore",
                                onDismissRequest = { crashReportFiles.value = emptyList() },
                                neutralButtonText = "delete",
                                onNeutral = { crashReports.forEach { it.delete() }; crashReportFiles.value = emptyList() },
                                confirmButtonText = "get",
                                onConfirmed = {
                                    val intent = Intent(Intent.ACTION_CREATE_DOCUMENT)
                                    intent.addCategory(Intent.CATEGORY_OPENABLE)
                                    intent.putExtra(Intent.EXTRA_TITLE, "crash_reports.zip")
                                    intent.type = "application/zip"
                                    crashFilePicker.launch(intent)
                                },
                                content = { Text("Crash report files found") },
                            )
                        } else if (JniUtils.sHaveGestureLib && System.currentTimeMillis() < END_DATE_EPOCH_MILLIS + TWO_WEEKS_IN_MILLIS) {
                            GestureDataPromotionReminderDialog()
                        }
                    }
                    if (dictUri != null) {
                        NewDictionaryDialog(
                            onDismissRequest = { dictUriFlow.value = null },
                            cachedFile = cachedDictionaryFile,
                            mainLocale = null
                        )
                    }
                }
            }
        }

        if (intent?.action == Intent.ACTION_VIEW) {
            intent?.data?.let {
                cachedDictionaryFile.delete()
                FileUtils.copyContentUriToNewFile(it, this, cachedDictionaryFile)
                dictUriFlow.value = it
            }
            intent = null
        }

        enableEdgeToEdge()
    }

    override fun onStart() {
        super.onStart()
        prefs.registerOnSharedPreferenceChangeListener(this)
    }

    override fun onStop() {
        prefs.unregisterOnSharedPreferenceChangeListener(this)
        super.onStop()
    }

    override fun onPause() {
        super.onPause()
        setForceTheme(null, null)
        paused = true
    }

    override fun onResume() {
        super.onResume()
        paused = false
    }

    fun setForceTheme(theme: String?, night: Boolean?) {
        if (paused) return
        if (forceTheme == theme && forceNight == night)
            return
        forceTheme = theme
        forceNight = night
        KeyboardSwitcher.getInstance().setThemeNeedsReload()
    }

    private fun findCrashReports(onlyUnprotected: Boolean): List<File> {
        val unprotected = DeviceProtectedUtils.getFilesDir(this)?.listFiles().orEmpty()
        if (onlyUnprotected)
            return unprotected.filter { it.name.startsWith("crash_report") }

        val dir = getExternalFilesDir(null)
        val allFiles = dir?.listFiles()?.toList().orEmpty() + unprotected
        return allFiles.filter { it.name.startsWith("crash_report") }
    }

    private fun saveCrashReports(uri: Uri) {
        val files = findCrashReports(false)
        if (files.isEmpty()) return
        runCatching {
            contentResolver.openOutputStream(uri)?.use {
                val bos = BufferedOutputStream(it)
                val z = ZipOutputStream(bos)
                for (file in files) {
                    val f = FileInputStream(file)
                    z.putNextEntry(ZipEntry(file.name))
                    FileUtils.copyStreamToOtherStream(f, z)
                    f.close()
                    z.closeEntry()
                }
                z.close()
                bos.close()
                for (file in files) {
                    file.delete()
                }
            }
        }
    }

    companion object {
        // public write so compose previews can show the screens
        // having it in a companion object is not ideal as it will stay in memory even after settings are closed
        // but it's small enough to not care
        lateinit var settingsContainer: SettingsContainer

        var forceNight: Boolean? = null
        var forceTheme: String? = null
    }

    override fun onSharedPreferenceChanged(prefereces: SharedPreferences?, key: String?) {
        prefChanged()
    }
}

// duplicate of SettingsActivity so we can launch it when the app icon is disabled in Android 9 and older
class SettingsActivity2 : SettingsActivity()
