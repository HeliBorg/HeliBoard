// SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
package helium314.keyboard.latin

import android.app.Application
import android.content.Context
import android.os.Build
import helium314.keyboard.keyboard.emoji.SupportedEmojis
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.dictionary.ExpandableDictionaryUtils
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.FoldableUtils
import helium314.keyboard.latin.utils.LayoutUtilsCustom
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.upgradeToolbarPrefs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        app = this
        val ctx: Context = this
        DebugFlags.init(this) // needed by several other inits, so do it first

        val scope = CoroutineScope(Dispatchers.Default)
        // other inits can be in parallel, except what depends on SubtypeSettings
        val jobs = listOf(
            scope.launch {
                SubtypeSettings.init(ctx)
                RichInputMethodManager.init(ctx)
                checkVersionUpgrade(ctx)
            },
            scope.launch { FoldableUtils.init(ctx) },
            scope.launch { Settings.init(ctx) },
            // upgrade toolbar on every debug apk start because we may work on adding a new toolbar key
            scope.launch { if (BuildConfig.DEBUG) upgradeToolbarPrefs(prefs()) },
            scope.launch { transferOldPinnedClips(ctx) }, // todo: remove in a few months, maybe end 2026
            scope.launch { Defaults.initDynamicDefaults(ctx) }
        )
        runBlocking { jobs.joinAll() } // word should be finished before anything else is started

        scope.launch { // do some uncritical work in background for faster startup
            SupportedEmojis.load(this@App)
            ExpandableDictionaryUtils.initUpToDateDicts(this@App)
            LayoutUtilsCustom.removeMissingLayouts(this@App)
            val packageInfo = packageManager.getPackageInfo(packageName, 0)
            @Suppress("DEPRECATION")
            Log.i(
                "startup", "Starting ${applicationInfo.processName} version ${packageInfo.versionName} (${
                    packageInfo.versionCode
                }) on Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})"
            )
        }
    }

    companion object {
        // used so JniUtils can access application once
        private var app: App? = null
        fun getApp(): App? {
            val application = app
            app = null
            return application
        }
    }
}
