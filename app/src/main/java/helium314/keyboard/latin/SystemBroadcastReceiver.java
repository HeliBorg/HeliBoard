/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Process;

import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.Log;
import android.view.inputmethod.InputMethodManager;

import helium314.keyboard.keyboard.KeyboardLayoutSet;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.UncachedInputMethodManagerUtils;
import helium314.keyboard.settings.SettingsActivity;

/**
 * This class detects the {@link Intent#ACTION_MY_PACKAGE_REPLACED} broadcast intent when this IME
 * package has been replaced by a newer version of the same package. This class also detects
 * {@link Intent#ACTION_BOOT_COMPLETED} and {@link Intent#ACTION_USER_INITIALIZE} broadcast intent.
 * <p>
 * If this IME has already been installed in the system image and a new version of this IME has
 * been installed, {@link Intent#ACTION_MY_PACKAGE_REPLACED} is received by this receiver and it
 * will hide the setup wizard's icon.
 * <p>
 * If this IME has already been installed in the data partition and a new version of this IME has
 * been installed, {@link Intent#ACTION_MY_PACKAGE_REPLACED} is received by this receiver but it
 * will not hide the setup wizard's icon, and the icon will appear on the launcher.
 * <p>
 * If this IME hasn't been installed yet and has been newly installed, no
 * {@link Intent#ACTION_MY_PACKAGE_REPLACED} will be sent and the setup wizard's icon will appear
 * on the launcher.
 * <p>
 * When the device has been booted, {@link Intent#ACTION_BOOT_COMPLETED} is received by this
 * receiver and it checks whether the setup wizard's icon should be appeared or not on the launcher
 * depending on which partition this IME is installed.
 * <p>
 * When the system locale has been changed, {@link Intent#ACTION_LOCALE_CHANGED} is received by
 * this receiver and the {@link KeyboardLayoutSet}'s cache is cleared.
 */
public final class SystemBroadcastReceiver extends BroadcastReceiver {
    private static final String TAG = SystemBroadcastReceiver.class.getSimpleName();

    @Override
    public void onReceive(final Context context, final Intent intent) {
        final String intentAction = intent.getAction();
        if (Intent.ACTION_MY_PACKAGE_REPLACED.equals(intentAction)) {
            Log.i(TAG, "Package has been replaced: " + context.getPackageName());
            toggleAppIcon(context);
        } else if (Intent.ACTION_BOOT_COMPLETED.equals(intentAction)) {
            Log.i(TAG, "Boot has been completed");
            toggleAppIcon(context);
        } else if (Intent.ACTION_LOCKED_BOOT_COMPLETED.equals(intentAction)) {
            Log.i(TAG, "Locked boot has been completed");
        } else if (Intent.ACTION_LOCALE_CHANGED.equals(intentAction)) {
            Log.i(TAG, "System locale changed");
            KeyboardLayoutSet.onSystemLocaleChanged();
        }

        // The process that hosts this broadcast receiver is invoked and remains alive even after
        // 1) the package has been re-installed,
        // 2) the device has just booted,
        // 3) a new user has been created.
        // There is no good reason to keep the process alive if this IME isn't a current IME.
        final InputMethodManager imm = (InputMethodManager)
                context.getSystemService(Context.INPUT_METHOD_SERVICE);
        // Called to check whether this IME has been triggered by the current user or not
        final boolean isInputMethodManagerValidForUserOfThisProcess =
                !imm.getInputMethodList().isEmpty();
        final boolean isCurrentImeOfCurrentUser = isInputMethodManagerValidForUserOfThisProcess
                && UncachedInputMethodManagerUtils.isThisImeCurrent(context, imm);
        if (!isCurrentImeOfCurrentUser) {
            final int myPid = Process.myPid();
            Log.i(TAG, "Killing my process: pid=" + myPid);
            Process.killProcess(myPid);
        }
    }

    public static void toggleAppIcon(final Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
            return; // can't change visibility in Android 10 and above
        final SharedPreferences prefs = KtxKt.prefs(context);
        context.getPackageManager().setComponentEnabledSetting(
                new ComponentName(context, SettingsActivity.class),
                Settings.readShowSetupWizardIcon(prefs, context)
                        ? PackageManager.COMPONENT_ENABLED_STATE_ENABLED
                        : PackageManager.COMPONENT_ENABLED_STATE_DISABLED,
                PackageManager.DONT_KILL_APP);
    }
}
