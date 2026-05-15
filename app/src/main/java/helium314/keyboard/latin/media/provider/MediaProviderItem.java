/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media.provider;

import android.net.Uri;
import android.os.Bundle;

import helium314.keyboard.latin.media.MediaPluginContract;

public final class MediaProviderItem {
    public final String id;
    public final String title;
    public final String mime;
    public final String label;
    public final Uri previewUri;
    public final Uri contentUri;
    public final long sizeBytes;
    public final boolean isFolder;

    private MediaProviderItem(final String id, final String title, final String mime,
            final String label, final Uri previewUri, final Uri contentUri, final long sizeBytes,
            final boolean isFolder) {
        this.id = id;
        this.title = title;
        this.mime = mime;
        this.label = label;
        this.previewUri = previewUri;
        this.contentUri = contentUri;
        this.sizeBytes = sizeBytes;
        this.isFolder = isFolder;
    }

    public static MediaProviderItem fromBundle(final Bundle bundle) {
        if (bundle == null) {
            return null;
        }
        final String id = bundle.getString(MediaPluginContract.ITEM_ID);
        final String mime = bundle.getString(MediaPluginContract.ITEM_MIME);
        if (id == null || mime == null) {
            return null;
        }
        final String preview = bundle.getString(MediaPluginContract.ITEM_PREVIEW_URI);
        final String content = bundle.getString(MediaPluginContract.ITEM_CONTENT_URI);
        return new MediaProviderItem(id,
                bundle.getString(MediaPluginContract.ITEM_TITLE),
                mime,
                bundle.getString(MediaPluginContract.ITEM_LABEL, MediaPluginContract.DEFAULT_MEDIA_LABEL),
                preview == null ? null : Uri.parse(preview),
                content == null ? null : Uri.parse(content),
                bundle.getLong(MediaPluginContract.ITEM_SIZE_BYTES, -1),
                MediaPluginContract.MIME_FOLDER.equals(mime));
    }
}
