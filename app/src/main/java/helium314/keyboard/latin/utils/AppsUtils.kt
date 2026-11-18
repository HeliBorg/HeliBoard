// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ResolveInfo
import android.graphics.drawable.Drawable
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.dictionary.AppsBinaryDictionary
import helium314.keyboard.latin.dictionary.ExpandableDictionaryUtils

object AppsUtils {
    /** Returns all app labels associated with a launcher icon, sorted arbitrarily. */
    private fun getCurrentNames(packageManager: PackageManager): HashSet<String> {
        val filter = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        // activities with an entry/icon for the launcher
        val launcherApps: List<ResolveInfo> = packageManager.queryIntentActivities(filter, 0)

        return launcherApps.mapTo(HashSet(launcherApps.size)) {
            it.activityInfo.loadLabel(packageManager).toString()
        }
    }

    fun getPackagesWithNameAndIcon(packageManager: PackageManager): List<Triple<String, String, Drawable?>> {
        val filter = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        return packageManager.queryIntentActivities(filter, 0).distinctBy { it.activityInfo.packageName }.map {
            Triple(it.activityInfo.packageName, it.activityInfo.loadLabel(packageManager).toString(), it.activityInfo.loadIcon(packageManager))
        }
    }

    private var currentNames = hashSetOf<String>()
    fun getNames(packageManager: PackageManager): Collection<String> {
        if (currentNames.isEmpty())
            currentNames = getCurrentNames(packageManager)
        return currentNames
    }

    fun onAppsChanged(packageManager: PackageManager) {
        if (DebugFlags.DEBUG_ENABLED)
            Log.d("AppsUtils", "reloading app names")
        val newNames = getCurrentNames(packageManager)
        if (newNames == currentNames) return
        currentNames = newNames
        ExpandableDictionaryUtils.setNeedsUpdate(AppsBinaryDictionary.NAME)
    }
}
