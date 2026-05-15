/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.webkit.MimeTypeMap;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.UUID;

import helium314.keyboard.latin.media.provider.MediaProviderItem;
import helium314.keyboard.latin.utils.Log;

public final class CacheActionSendExporter {
    private static final String CACHE_DIR_NAME = "media_share_cache";
    private static final long EXPORT_TTL_MILLIS = 60L * 60L * 1000L;
    private static final int COPY_BUFFER_SIZE = 64 * 1024;

    private CacheActionSendExporter() {
    }

    public static Export exportForActionSend(final Context context, final MediaProviderItem item,
            final long maxBytes) throws IOException {
        if (item == null) {
            throw new IOException("Missing media item");
        }
        return exportForActionSend(context, item.contentUri, item.mime, item.label,
                item.sizeBytes, maxBytes);
    }

    public static Export exportForActionSend(final Context context, final Uri sourceUri,
            final String mimeType, final String displayName, final long declaredSizeBytes,
            final long maxBytes) throws IOException {
        if (sourceUri == null || mimeType == null) {
            throw new IOException("Missing media source");
        }
        cleanupOldExports(context);
        final long sizeBytes = declaredSizeBytes >= 0
                ? declaredSizeBytes : getSizeIfKnown(context, sourceUri);
        if (sizeBytes > maxBytes) {
            throw new IOException("Media exceeds max bytes: " + sizeBytes + " > " + maxBytes);
        }

        final File cacheDir = getCacheDir(context);
        if (!cacheDir.exists() && !cacheDir.mkdirs()) {
            throw new IOException("Could not create media cache directory");
        }
        final String safeDisplayName = createGeneratedDisplayName(mimeType);
        final File outputFile = new File(cacheDir, safeDisplayName);
        boolean success = false;
        long copiedBytes = 0;
        try (InputStream inputStream = context.getContentResolver().openInputStream(sourceUri);
                FileOutputStream outputStream = new FileOutputStream(outputFile)) {
            if (inputStream == null) {
                throw new IOException("Could not open source media");
            }
            final byte[] buffer = new byte[COPY_BUFFER_SIZE];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                copiedBytes += read;
                if (copiedBytes > maxBytes) {
                    throw new IOException("Copied media exceeds max bytes: "
                            + copiedBytes + " > " + maxBytes);
                }
                outputStream.write(buffer, 0, read);
            }
            success = true;
        } finally {
            if (!success && outputFile.exists() && !outputFile.delete()) {
                Log.d(MediaPluginContract.LOG_TAG,
                        "Could not delete partial cache export: " + outputFile);
            }
        }

        final Uri uri = FileProvider.getUriForFile(context,
                context.getPackageName() + ".mediafileprovider", outputFile);
        Log.d(MediaPluginContract.LOG_TAG, "ACTION_SEND cache export used=true"
                + " sourceUri=" + sourceUri
                + " cacheUri=" + uri
                + " mime=" + mimeType
                + " size=" + copiedBytes);
        return new Export(uri, mimeType, safeDisplayName);
    }

    public static void cleanupOldExports(final Context context) {
        final File cacheDir = getCacheDir(context);
        final File[] files = cacheDir.listFiles();
        if (files == null) {
            return;
        }
        final long cutoff = System.currentTimeMillis() - EXPORT_TTL_MILLIS;
        int deleted = 0;
        for (final File file : files) {
            if (file.isFile() && file.lastModified() < cutoff && file.delete()) {
                deleted++;
            }
        }
        Log.d(MediaPluginContract.LOG_TAG, "ACTION_SEND cache cleanup deleted=" + deleted);
    }

    private static File getCacheDir(final Context context) {
        return new File(context.getCacheDir(), CACHE_DIR_NAME);
    }

    private static long getSizeIfKnown(final Context context, final Uri uri) {
        try (Cursor cursor = context.getContentResolver().query(uri,
                new String[] { OpenableColumns.SIZE }, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                final int index = cursor.getColumnIndex(OpenableColumns.SIZE);
                if (index >= 0 && !cursor.isNull(index)) {
                    return cursor.getLong(index);
                }
            }
        } catch (Throwable t) {
            Log.d(MediaPluginContract.LOG_TAG, "Could not query cache export source size: " + t);
        }
        return -1;
    }

    private static String createGeneratedDisplayName(final String mimeType) {
        final String extension = MimeTypeMap.getSingleton().getExtensionFromMimeType(mimeType);
        final String suffix = extension == null || extension.isEmpty()
                ? "" : "." + extension.toLowerCase(Locale.ROOT);
        return "heliboard_media_" + System.currentTimeMillis()
                + "_" + UUID.randomUUID() + suffix;
    }

    public static final class Export {
        public final Uri uri;
        public final String mimeType;
        public final String displayName;

        private Export(final Uri uri, final String mimeType, final String displayName) {
            this.uri = uri;
            this.mimeType = mimeType;
            this.displayName = displayName;
        }
    }
}
