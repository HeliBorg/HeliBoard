// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.pm.ResolveInfo

class AppsManager(val context: Context) : BroadcastReceiver() {
    private val mPackageManager: PackageManager = context.packageManager
    private var listener: AppsChangedListener? = null

    /**
     * Returns all app labels associated with a launcher icon, sorted arbitrarily.
     */
    fun getNames(): HashSet<String> {
        val filter = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        // activities with an entry/icon for the launcher
        val launcherApps: List<ResolveInfo> = mPackageManager.queryIntentActivities(filter, 0)

        return launcherApps.mapTo(HashSet(launcherApps.size)) {
            it.activityInfo.loadLabel(mPackageManager).toString()
        }
    }

    fun registerForUpdates(listener: AppsChangedListener) {
        this.listener = listener
        val packageFilter = IntentFilter()
        packageFilter.addAction(Intent.ACTION_PACKAGE_ADDED)
        packageFilter.addAction(Intent.ACTION_PACKAGE_REMOVED)
        packageFilter.addDataScheme("package")
        context.registerReceiver(this, packageFilter)
    }

    fun close() {
        context.unregisterReceiver(this)
        listener = null
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (!intent.getBooleanExtra(Intent.EXTRA_REPLACING, false))
            listener?.onAppsChanged()
    }

    interface AppsChangedListener {
        fun onAppsChanged()
    }

    fun getPackagesAndNames(): Collection<Pair<String, String>> {
        val filter = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        return mPackageManager.queryIntentActivities(filter, 0).distinctBy { it.activityInfo.packageName }.map {
            it.activityInfo.packageName to it.activityInfo.loadLabel(mPackageManager).toString()
        }
    }
}
