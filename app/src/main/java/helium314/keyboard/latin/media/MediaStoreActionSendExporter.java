/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.webkit.MimeTypeMap;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;

import helium314.keyboard.latin.utils.DeviceProtectedUtils;
import helium314.keyboard.latin.utils.Log;

import org.json.JSONArray;
import org.json.JSONObject;

public final class MediaStoreActionSendExporter {
    // Public/user-visible compatibility fallback for apps such as AOSP/Graphene Messaging
    // that reject private content:// URIs with "Cannot send private file content://...".
    // Callers must gate this behind Settings.PREF_MEDIA_PUBLIC_STORAGE_FALLBACK. Inserted
    // MediaStore files require no storage permission on Android Q+; cleanup is best-effort.
    private static final String PREF_MEDIASTORE_EXPORTS = "media_plugin_mediastore_exports";
    private static final long EXPORT_TTL_MILLIS = 24L * 60L * 60L * 1000L;
    private static final int COPY_BUFFER_SIZE = 64 * 1024;
    private static final String IMAGE_RELATIVE_PATH = "Pictures/HeliBoard Shared";
    private static final String VIDEO_RELATIVE_PATH = "Movies/HeliBoard Shared";

    private MediaStoreActionSendExporter() {
    }

    public static Uri exportForActionSendIfNeeded(final Context context, final Uri sourceUri,
            final String mimeType, final String displayName, final long declaredSizeBytes,
            final long maxBytes) {
        cleanupOldExports(context);
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            Log.d(MediaPluginContract.LOG_TAG, "ACTION_SEND MediaStore export skipped api="
                    + Build.VERSION.SDK_INT + " sourceUri=" + sourceUri);
            return sourceUri;
        }
        if (mimeType == null || sourceUri == null) {
            return sourceUri;
        }

        final Uri collectionUri = getCollectionUri(mimeType);
        if (collectionUri == null) {
            Log.d(MediaPluginContract.LOG_TAG, "ACTION_SEND MediaStore export skipped unsupported"
                    + " mime=" + mimeType + " sourceUri=" + sourceUri);
            return sourceUri;
        }

        final ContentResolver resolver = context.getContentResolver();
        final String safeDisplayName = ensureExtension(displayName, mimeType);
        final MediaSizeValidator.Result sizeResult =
                MediaSizeValidator.validate(context, sourceUri, declaredSizeBytes, maxBytes);
        if (!sizeResult.valid) {
            Log.w(MediaPluginContract.LOG_TAG, "ACTION_SEND MediaStore export skipped oversized"
                    + " sourceUri=" + sourceUri
                    + " mime=" + mimeType
                    + " size=" + sizeResult.sizeBytes
                    + " maxBytes=" + maxBytes);
            return sourceUri;
        }
        final ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, safeDisplayName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, mimeType);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH,
                mimeType.toLowerCase(Locale.ROOT).startsWith("video/")
                        ? VIDEO_RELATIVE_PATH : IMAGE_RELATIVE_PATH);
        values.put(MediaStore.MediaColumns.IS_PENDING, 1);

        Uri exportedUri = null;
        try {
            exportedUri = resolver.insert(collectionUri, values);
            if (exportedUri == null) {
                Log.w(MediaPluginContract.LOG_TAG,
                        "ACTION_SEND MediaStore export insert returned null");
                return sourceUri;
            }
            try (InputStream inputStream = resolver.openInputStream(sourceUri);
                    OutputStream outputStream = resolver.openOutputStream(exportedUri, "w")) {
                if (inputStream == null || outputStream == null) {
                    throw new IllegalStateException("Could not open MediaStore export streams");
                }
                final byte[] buffer = new byte[COPY_BUFFER_SIZE];
                long copiedBytes = 0;
                int read;
                while ((read = inputStream.read(buffer)) != -1) {
                    copiedBytes += read;
                    if (copiedBytes > maxBytes) {
                        throw new IllegalStateException("MediaStore export exceeds max bytes: "
                                + copiedBytes + " > " + maxBytes);
                    }
                    outputStream.write(buffer, 0, read);
                }
            }

            final ContentValues complete = new ContentValues();
            complete.put(MediaStore.MediaColumns.IS_PENDING, 0);
            resolver.update(exportedUri, complete, null, null);
            rememberExport(context, exportedUri, safeDisplayName, mimeType);
            Log.d(MediaPluginContract.LOG_TAG, "ACTION_SEND MediaStore export used=true"
                    + " api=" + Build.VERSION.SDK_INT
                    + " sourceUri=" + sourceUri
                    + " exportedUri=" + exportedUri
                    + " mime=" + mimeType
                    + " size=" + sizeResult.sizeBytes);
            return exportedUri;
        } catch (Throwable t) {
            Log.w(MediaPluginContract.LOG_TAG, "ACTION_SEND MediaStore export failed"
                    + " sourceUri=" + sourceUri
                    + " insertedUri=" + exportedUri
                    + " mime=" + mimeType
                    + " size=" + sizeResult.sizeBytes, t);
            if (exportedUri != null) {
                try {
                    resolver.delete(exportedUri, null, null);
                } catch (Throwable deleteError) {
                    Log.d(MediaPluginContract.LOG_TAG,
                            "Could not delete failed MediaStore export: " + deleteError);
                }
            }
            return sourceUri;
        }
    }

    private static Uri getCollectionUri(final String mimeType) {
        final String lowerMime = mimeType.toLowerCase(Locale.ROOT);
        if (lowerMime.startsWith("image/")) {
            return MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
        }
        if (lowerMime.startsWith("video/")) {
            return MediaStore.Video.Media.EXTERNAL_CONTENT_URI;
        }
        return null;
    }

    private static void cleanupOldExports(final Context context) {
        final SharedPreferences prefs = DeviceProtectedUtils.getSharedPreferences(context);
        final String serialized = prefs.getString(PREF_MEDIASTORE_EXPORTS, "[]");
        final JSONArray retained = new JSONArray();
        int deleted = 0;
        final long now = System.currentTimeMillis();
        try {
            final JSONArray exports = new JSONArray(serialized);
            for (int i = 0; i < exports.length(); i++) {
                final JSONObject export = exports.optJSONObject(i);
                if (export == null) {
                    continue;
                }
                final String uriString = export.optString("uri", null);
                final long created = export.optLong("created", 0);
                if (uriString == null || created <= 0) {
                    continue;
                }
                if (now - created > EXPORT_TTL_MILLIS) {
                    try {
                        context.getContentResolver().delete(Uri.parse(uriString), null, null);
                        deleted++;
                    } catch (Throwable t) {
                        retained.put(export);
                        Log.d(MediaPluginContract.LOG_TAG,
                                "Could not delete old MediaStore export uri=" + uriString
                                        + " error=" + t);
                    }
                } else {
                    retained.put(export);
                }
            }
        } catch (Throwable t) {
            Log.d(MediaPluginContract.LOG_TAG,
                    "Could not parse MediaStore export cleanup records: " + t);
        }
        prefs.edit().putString(PREF_MEDIASTORE_EXPORTS, retained.toString()).apply();
        Log.d(MediaPluginContract.LOG_TAG, "ACTION_SEND MediaStore cleanup deleted=" + deleted);
    }

    private static void rememberExport(final Context context, final Uri uri,
            final String displayName, final String mimeType) {
        final SharedPreferences prefs = DeviceProtectedUtils.getSharedPreferences(context);
        JSONArray exports;
        try {
            exports = new JSONArray(prefs.getString(PREF_MEDIASTORE_EXPORTS, "[]"));
        } catch (Throwable t) {
            exports = new JSONArray();
        }
        final JSONObject export = new JSONObject();
        try {
            export.put("uri", uri.toString());
            export.put("created", System.currentTimeMillis());
            export.put("displayName", displayName);
            export.put("mimeType", mimeType);
            exports.put(export);
            prefs.edit().putString(PREF_MEDIASTORE_EXPORTS, exports.toString()).apply();
        } catch (Throwable t) {
            Log.d(MediaPluginContract.LOG_TAG, "Could not record MediaStore export: " + t);
        }
    }

    private static String ensureExtension(final String displayName, final String mimeType) {
        final String fallbackName = MediaPluginContract.DEFAULT_MEDIA_LABEL;
        final String baseName = displayName == null || displayName.trim().isEmpty()
                ? fallbackName : displayName.trim();
        final int slash = baseName.lastIndexOf('/');
        final String sanitized = (slash >= 0 ? baseName.substring(slash + 1) : baseName)
                .replaceAll("[\\\\/:*?\"<>|]", "_");
        final int dot = sanitized.lastIndexOf('.');
        if (dot > 0 && dot < sanitized.length() - 1) {
            return sanitized;
        }
        final String extension = MimeTypeMap.getSingleton().getExtensionFromMimeType(mimeType);
        if (extension == null || extension.isEmpty()) {
            return sanitized;
        }
        return sanitized + "." + extension;
    }
}
