/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.content.Context;
import android.net.Uri;
import android.widget.Toast;

import java.io.InputStream;

import helium314.keyboard.latin.utils.Log;

public final class MediaSizeValidator {
    private static final int COPY_BUFFER_SIZE = 64 * 1024;

    private MediaSizeValidator() {
    }

    public static Result validate(final Context context, final Uri uri,
            final long declaredSizeBytes, final long maxBytes) {
        if (uri == null) {
            return reject(context, "Missing media URI");
        }
        if (maxBytes < 0) {
            return reject(context, "Invalid media size limit");
        }
        if (declaredSizeBytes > maxBytes) {
            Log.w(MediaPluginContract.LOG_TAG, "Rejecting declared oversized media"
                    + " uri=" + uri + " size=" + declaredSizeBytes + " maxBytes=" + maxBytes);
            return reject(context, "Media is too large");
        }
        if (maxBytes == Long.MAX_VALUE) {
            return new Result(true, declaredSizeBytes);
        }
        try (InputStream inputStream = context.getContentResolver().openInputStream(uri)) {
            if (inputStream == null) {
                Log.w(MediaPluginContract.LOG_TAG, "Media size validation returned null stream"
                        + " uri=" + uri);
                return reject(context, "Media unavailable");
            }
            final byte[] buffer = new byte[COPY_BUFFER_SIZE];
            long countedBytes = 0;
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                countedBytes += read;
                if (countedBytes > maxBytes) {
                    Log.w(MediaPluginContract.LOG_TAG, "Rejecting actual oversized media"
                            + " uri=" + uri + " size=" + countedBytes
                            + " declaredSize=" + declaredSizeBytes
                            + " maxBytes=" + maxBytes);
                    return reject(context, "Media is too large");
                }
            }
            Log.d(MediaPluginContract.LOG_TAG, "media size validated"
                    + " uri=" + uri + " size=" + countedBytes
                    + " declaredSize=" + declaredSizeBytes
                    + " maxBytes=" + maxBytes);
            return new Result(true, countedBytes);
        } catch (Throwable t) {
            Log.w(MediaPluginContract.LOG_TAG, "Could not validate media size uri=" + uri, t);
            return reject(context, "Media unavailable");
        }
    }

    private static Result reject(final Context context, final String toastText) {
        Toast.makeText(context, toastText, Toast.LENGTH_SHORT).show();
        return new Result(false, -1);
    }

    public static final class Result {
        public final boolean valid;
        public final long sizeBytes;

        private Result(final boolean valid, final long sizeBytes) {
            this.valid = valid;
            this.sizeBytes = sizeBytes;
        }
    }
}
