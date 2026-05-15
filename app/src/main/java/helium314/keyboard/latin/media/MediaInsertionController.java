/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.app.Service;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.ClipDescription;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.provider.Telephony;
import android.telephony.CarrierConfigManager;
import android.telephony.SubscriptionManager;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.core.view.inputmethod.EditorInfoCompat;
import androidx.core.view.inputmethod.InputConnectionCompat;
import androidx.core.view.inputmethod.InputContentInfoCompat;

import java.util.Arrays;

import helium314.keyboard.latin.media.provider.MediaProviderItem;
import helium314.keyboard.latin.settings.Defaults;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.Log;

public final class MediaInsertionController {
    private static final int FALLBACK_MIN_MMS_CAP_BYTES = 300 * 1024;
    private static final double MMS_MEDIA_SAFETY_FACTOR = 0.85;

    private final Service mService;

    public MediaInsertionController(final Service service) {
        mService = service;
    }

    public long getPreferredMediaMaxBytes(@Nullable final EditorInfo editorInfo) {
        final String defaultSmsPackage = getDefaultSmsPackage();
        final String targetPackage = editorInfo == null ? null : editorInfo.packageName;
        if (editorInfo != null && defaultSmsPackage != null
                && defaultSmsPackage.equals(editorInfo.packageName)) {
            final long maxBytes = Math.max(FALLBACK_MIN_MMS_CAP_BYTES,
                    (long) (getMmsMaxBytes() * MMS_MEDIA_SAFETY_FACTOR));
            Log.d(MediaPluginContract.LOG_TAG, "media target package=" + targetPackage
                    + " defaultSmsPackage=" + defaultSmsPackage
                    + " applying MMS max bytes=" + maxBytes);
            return maxBytes;
        }
        Log.d(MediaPluginContract.LOG_TAG, "media target package=" + targetPackage
                + " defaultSmsPackage=" + defaultSmsPackage
                + " using unrestricted media size");
        return Long.MAX_VALUE;
    }

    public void insertMedia(@Nullable final InputConnection inputConnection,
            @Nullable final EditorInfo editorInfo, final MediaProviderItem item,
            @Nullable final String targetPackage, final long maxBytes) {
        if (item == null) {
            Toast.makeText(mService, "Media unavailable", Toast.LENGTH_SHORT).show();
            return;
        }
        insertMedia(inputConnection, editorInfo, item.contentUri, item.mime, item.label,
                item.sizeBytes, targetPackage, maxBytes);
    }

    public void insertMedia(@Nullable final InputConnection inputConnection,
            @Nullable final EditorInfo editorInfo, final Uri uri, final String mime,
            final String label, final long declaredSizeBytes,
            @Nullable final String targetPackage, final long maxBytes) {
        final String safeLabel = label == null ? MediaPluginContract.DEFAULT_MEDIA_LABEL : label;
        if (inputConnection == null || editorInfo == null) {
            Log.d(MediaPluginContract.LOG_TAG, "No active input connection for media insertion");
            Toast.makeText(mService, "Open a text field first", Toast.LENGTH_SHORT).show();
            return;
        }
        if (uri == null || mime == null || !MediaPluginContract.isAcceptedMimeType(mime)) {
            Toast.makeText(mService, "Unsupported media type", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            inputConnection.finishComposingText();
        } catch (Throwable t) {
            Log.w(MediaPluginContract.LOG_TAG, "finishComposingText failed: " + t);
        }

        final MediaSizeValidator.Result sizeResult =
                MediaSizeValidator.validate(mService, uri, declaredSizeBytes, maxBytes);
        if (!sizeResult.valid) {
            return;
        }

        final String[] supportedMimeTypes = EditorInfoCompat.getContentMimeTypes(editorInfo);
        Log.d(MediaPluginContract.LOG_TAG, "supported editor MIME types="
                + Arrays.toString(supportedMimeTypes));
        final boolean hasSupportedMimeTypes =
                supportedMimeTypes != null && supportedMimeTypes.length > 0;
        boolean commitContentResult = false;
        if (hasSupportedMimeTypes && editorSupportsMimeType(supportedMimeTypes, mime)) {
            final ClipDescription description = new ClipDescription(safeLabel, new String[] { mime });
            final InputContentInfoCompat inputContentInfo =
                    new InputContentInfoCompat(uri, description, null);
            try {
                commitContentResult = InputConnectionCompat.commitContent(inputConnection, editorInfo,
                        inputContentInfo,
                        InputConnectionCompat.INPUT_CONTENT_GRANT_READ_URI_PERMISSION, null);
            } catch (Throwable t) {
                Log.e(MediaPluginContract.LOG_TAG, "commitContent threw", t);
            }
            Log.d(MediaPluginContract.LOG_TAG, "commitContent result=" + commitContentResult);
        } else {
            Log.d(MediaPluginContract.LOG_TAG,
                    hasSupportedMimeTypes
                            ? "Editor MIME types do not accept " + mime
                            : "Editor declares no rich-content MIME types");
        }

        if (!commitContentResult) {
            launchMediaSendFallback(uri, mime, safeLabel, sizeResult.sizeBytes, maxBytes,
                    targetPackage);
        }
    }

    private void launchMediaSendFallback(final Uri uri, final String mime, final String label,
            final long mediaSize, final long maxBytes, @Nullable final String targetPackage) {
        final String resolvedTargetPackage = getMediaSendTargetPackage(targetPackage);
        final boolean privateUriRejectedTarget =
                MediaShareTargetCompatibility.isPrivateUriRejectedTarget(resolvedTargetPackage);
        if (!privateUriRejectedTarget) {
            try {
                if (!MediaSizeValidator.validate(mService, uri, mediaSize, maxBytes).valid) {
                    return;
                }
                // ACTION_SEND targets are arbitrary external apps. Keep plugin content URIs
                // private to the HeliBoard <-> plugin hop by re-hosting through HeliBoard's
                // own FileProvider before launching the share intent.
                final CacheActionSendExporter.Export cacheExport =
                        CacheActionSendExporter.exportForActionSend(mService, uri, mime, label,
                                mediaSize, maxBytes);
                if (MediaSizeValidator.validate(mService, cacheExport.uri, mediaSize, maxBytes).valid
                        && tryLaunchMediaSendFallback(uri, cacheExport.uri, cacheExport.mimeType,
                        cacheExport.displayName, resolvedTargetPackage, "cache-fileprovider")) {
                    return;
                }
            } catch (Throwable t) {
                Log.w(MediaPluginContract.LOG_TAG, "ACTION_SEND cache FileProvider fallback failed"
                        + " sourceUri=" + uri + " mime=" + mime + " size=" + mediaSize, t);
            }
        } else {
            Log.d(MediaPluginContract.LOG_TAG, "Skipping private ACTION_SEND URI fallback for "
                    + resolvedTargetPackage);
        }

        if (!isPublicMediaFallbackEnabled()) {
            Log.d(MediaPluginContract.LOG_TAG, "Public MediaStore fallback disabled"
                    + " targetPackage=" + resolvedTargetPackage
                    + " privateUriRejectedTarget=" + privateUriRejectedTarget);
            Toast.makeText(mService,
                    "Sharing to this app requires public media fallback. Enable it in HeliBoard settings.",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        if (!MediaSizeValidator.validate(mService, uri, mediaSize, maxBytes).valid) {
            return;
        }
        // Compatibility fallback for MMS/share targets such as AOSP/Graphene Messaging and
        // other apps that reject private content:// URIs during ACTION_SEND. This is broader
        // than a strict known-app allowlist by design: some failures are only observable after
        // private URI methods fail. The explicit setting defaults off because this creates
        // temporary public, user-visible MediaStore files; cleanup is best-effort only.
        final Uri mediaStoreUri = MediaStoreActionSendExporter.exportForActionSendIfNeeded(mService,
                uri, mime, label, mediaSize, maxBytes);
        if (!uri.equals(mediaStoreUri)
                && MediaSizeValidator.validate(mService, mediaStoreUri, mediaSize, maxBytes).valid
                && tryLaunchMediaSendFallback(uri, mediaStoreUri, mime, label,
                resolvedTargetPackage, "mediastore-last-resort")) {
            return;
        }

        Toast.makeText(mService, "No app available to share media", Toast.LENGTH_SHORT).show();
    }

    private boolean isPublicMediaFallbackEnabled() {
        return KtxKt.prefs(mService).getBoolean(Settings.PREF_MEDIA_PUBLIC_STORAGE_FALLBACK,
                Defaults.PREF_MEDIA_PUBLIC_STORAGE_FALLBACK);
    }

    private boolean tryLaunchMediaSendFallback(final Uri sourceUri, final Uri shareUri,
            final String mime, final String label, @Nullable final String targetPackage,
            final String fallbackKind) {
        final Intent send = new Intent(Intent.ACTION_SEND);
        send.setType(mime);
        send.putExtra(Intent.EXTRA_STREAM, shareUri);
        send.setClipData(ClipData.newUri(mService.getContentResolver(), label, shareUri));
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_NEW_TASK);

        if (targetPackage != null && !mService.getPackageName().equals(targetPackage)) {
            send.setPackage(targetPackage);
            grantMediaUriToTarget(targetPackage, shareUri);
        }

        try {
            Log.d(MediaPluginContract.LOG_TAG, "fallback ACTION_SEND sourceUri=" + sourceUri
                    + " shareUri=" + shareUri
                    + " kind=" + fallbackKind
                    + " api=" + Build.VERSION.SDK_INT
                    + " mime=" + mime
                    + " targetPackage=" + send.getPackage());
            if (send.getPackage() != null) {
                mService.startActivity(send);
            } else {
                final Intent chooser = Intent.createChooser(send, "Share media");
                chooser.setClipData(send.getClipData());
                chooser.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                        | Intent.FLAG_ACTIVITY_NEW_TASK);
                mService.startActivity(chooser);
            }
            Log.d(MediaPluginContract.LOG_TAG, "fallback ACTION_SEND launched sourceUri="
                    + sourceUri + " shareUri=" + shareUri + " mime=" + mime
                    + " kind=" + fallbackKind);
            return true;
        } catch (ActivityNotFoundException e) {
            Log.e(MediaPluginContract.LOG_TAG, "No activity found for ACTION_SEND kind="
                    + fallbackKind, e);
            return false;
        } catch (Throwable t) {
            Log.e(MediaPluginContract.LOG_TAG, "ACTION_SEND launch failed kind="
                    + fallbackKind, t);
            return false;
        }
    }

    @Nullable
    private String getMediaSendTargetPackage(@Nullable final String editorPackage) {
        final String defaultSmsPackage = getDefaultSmsPackage();
        if (defaultSmsPackage != null && defaultSmsPackage.equals(editorPackage)) {
            return defaultSmsPackage;
        }
        return editorPackage;
    }

    private void grantMediaUriToTarget(final String targetPackage, final Uri uri) {
        try {
            mService.grantUriPermission(targetPackage, uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (Throwable t) {
            Log.d(MediaPluginContract.LOG_TAG, "grantUriPermission failed: " + t);
        }
    }

    private int getMmsMaxBytes() {
        try {
            final int subId = SubscriptionManager.getDefaultSubscriptionId();
            if (!SubscriptionManager.isValidSubscriptionId(subId)) {
                Log.d(MediaPluginContract.LOG_TAG,
                        "No valid default subscription; using conservative MMS cap");
                return FALLBACK_MIN_MMS_CAP_BYTES;
            }
            final CarrierConfigManager carrierConfigManager =
                    (CarrierConfigManager) mService.getSystemService(Context.CARRIER_CONFIG_SERVICE);
            if (carrierConfigManager != null) {
                final android.os.PersistableBundle bundle =
                        carrierConfigManager.getConfigForSubId(subId);
                if (bundle != null) {
                    final int value = bundle.getInt(
                            CarrierConfigManager.KEY_MMS_MAX_MESSAGE_SIZE_INT,
                            FALLBACK_MIN_MMS_CAP_BYTES);
                    if (value >= FALLBACK_MIN_MMS_CAP_BYTES && value < 10 * 1024 * 1024) {
                        return value;
                    }
                }
            }
        } catch (Throwable t) {
            Log.d(MediaPluginContract.LOG_TAG,
                    "Could not read carrier MMS cap; using conservative fallback: " + t);
        }
        return FALLBACK_MIN_MMS_CAP_BYTES;
    }

    @Nullable
    private String getDefaultSmsPackage() {
        try {
            return Telephony.Sms.getDefaultSmsPackage(mService);
        } catch (Throwable t) {
            return null;
        }
    }

    private boolean editorSupportsMimeType(final String[] supportedMimeTypes, final String mime) {
        for (final String supportedMimeType : supportedMimeTypes) {
            if (ClipDescription.compareMimeTypes(mime, supportedMimeType)
                    || ClipDescription.compareMimeTypes(supportedMimeType, mime)) {
                return true;
            }
        }
        return false;
    }
}
