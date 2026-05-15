// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.compat

import android.app.KeyguardManager
import android.content.Context
import android.os.Build
import android.os.UserManager

fun isDeviceLocked(context: Context): Boolean {
    val keyguardManager = runCatching { context.getSystemService(Context.KEYGUARD_SERVICE) as KeyguardManager }
        .getOrNull() ?: return false // necessary so things can work in preview mode
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP_MR1)
        keyguardManager.isDeviceLocked
    else
        keyguardManager.isKeyguardLocked
}

fun isUserLocked(context: Context): Boolean {
    val userManager = context.getSystemService(Context.USER_SERVICE) as UserManager
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N)
        !userManager.isUserUnlocked
    else
        false
}
