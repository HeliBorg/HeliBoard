/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Parcelable;
import android.widget.Toast;

import androidx.annotation.Nullable;

import helium314.keyboard.latin.settings.Defaults;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.Log;

public final class MediaReceiverActivity extends Activity {
    @Override
    protected void onCreate(@Nullable final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        handleIntent(getIntent());
        finish();
        overridePendingTransition(0, 0);
    }

    private void handleIntent(final Intent intent) {
        if (intent == null) {
            return;
        }
        final String action = intent.getAction();
        Log.d(MediaPluginContract.LOG_TAG, "received intent action=" + action);

        if (!areMediaPluginsEnabled()) {
            Log.d(MediaPluginContract.LOG_TAG, "Ignoring media intent because media plugins are disabled");
            Toast.makeText(this, "Media plugins disabled", Toast.LENGTH_SHORT).show();
            return;
        }

        final Uri uri = getMediaUri(intent);
        final String mime = getMediaMime(intent, uri);
        final String label = getMediaLabel(intent);
        Log.d(MediaPluginContract.LOG_TAG, "received uri=" + uri + " mime=" + mime);

        if (uri == null || !ContentResolver.SCHEME_CONTENT.equals(uri.getScheme())) {
            Log.w(MediaPluginContract.LOG_TAG, "Ignoring media intent without content URI");
            Toast.makeText(this, "Invalid media", Toast.LENGTH_SHORT).show();
            return;
        }
        if (!MediaPluginContract.isAcceptedMimeType(mime)) {
            Log.w(MediaPluginContract.LOG_TAG, "Ignoring unsupported media MIME type=" + mime);
            Toast.makeText(this, "Unsupported media type", Toast.LENGTH_SHORT).show();
            return;
        }

        tryTakePersistableGrant(intent, uri);
        MediaInsertionDispatcher.dispatch(this, uri, mime, label);
    }

    private boolean areMediaPluginsEnabled() {
        return KtxKt.prefs(this).getBoolean(Settings.PREF_ENABLE_MEDIA_PLUGINS,
                Defaults.PREF_ENABLE_MEDIA_PLUGINS);
    }

    @Nullable
    private Uri getMediaUri(final Intent intent) {
        final Parcelable extraUri = intent.getParcelableExtra(MediaPluginContract.EXTRA_MEDIA_URI);
        if (extraUri instanceof Uri) {
            return (Uri) extraUri;
        }
        final Uri data = intent.getData();
        if (data != null) {
            return data;
        }
        final Parcelable stream = intent.getParcelableExtra(Intent.EXTRA_STREAM);
        if (stream instanceof Uri) {
            return (Uri) stream;
        }
        return null;
    }

    @Nullable
    private String getMediaMime(final Intent intent, @Nullable final Uri uri) {
        String mime = intent.getStringExtra(MediaPluginContract.EXTRA_MEDIA_MIME);
        if (mime == null) {
            mime = intent.getType();
        }
        if (mime == null && uri != null) {
            mime = getContentResolver().getType(uri);
        }
        return mime;
    }

    private String getMediaLabel(final Intent intent) {
        String label = intent.getStringExtra(MediaPluginContract.EXTRA_MEDIA_LABEL);
        if (label == null) {
            label = intent.getStringExtra(Intent.EXTRA_TITLE);
        }
        return label == null ? MediaPluginContract.DEFAULT_MEDIA_LABEL : label;
    }

    private void tryTakePersistableGrant(final Intent intent, final Uri uri) {
        final int flags = intent.getFlags()
                & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        if ((flags & Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION) == 0
                || (flags & Intent.FLAG_GRANT_READ_URI_PERMISSION) == 0) {
            return;
        }
        try {
            getContentResolver().takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (Throwable t) {
            Log.d(MediaPluginContract.LOG_TAG, "takePersistableUriPermission failed: " + t);
        }
    }
}
