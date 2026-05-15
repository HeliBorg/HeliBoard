/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.content.Context;
import android.net.Uri;
import android.widget.Toast;

import java.lang.ref.WeakReference;

import helium314.keyboard.latin.LatinIME;
import helium314.keyboard.latin.utils.Log;

public final class MediaInsertionDispatcher {
    private static WeakReference<LatinIME> sLatinIme = new WeakReference<>(null);

    private MediaInsertionDispatcher() {
    }

    public static void register(final LatinIME latinIme) {
        sLatinIme = new WeakReference<>(latinIme);
    }

    public static void unregister(final LatinIME latinIme) {
        final LatinIME registered = sLatinIme.get();
        if (registered == latinIme) {
            sLatinIme.clear();
        }
    }

    public static boolean dispatch(final Context context, final Uri uri, final String mime,
            final String label) {
        final LatinIME latinIme = sLatinIme.get();
        if (latinIme == null) {
            Log.d(MediaPluginContract.LOG_TAG, "No active IME for media insertion");
            Toast.makeText(context, "Open a text field first", Toast.LENGTH_SHORT).show();
            return false;
        }
        latinIme.onExternalMediaRequested(uri, mime, label);
        return true;
    }
}
