/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import android.content.ClipDescription;

public final class MediaPluginContract {
    /*
     * Minimal host-owned plugin contract.
     *
     * preview_uri must be readable by HeliBoard for thumbnail display.
     * content_uri must be readable by HeliBoard and remain stable long enough for
     * commitContent() or HeliBoard-owned ACTION_SEND re-hosting. External share targets should
     * receive HeliBoard cache/MediaStore URIs, not direct plugin content_uri values.
     *
     * Providers should honor BUNDLE_MAX_BYTES, but HeliBoard validates size before committing,
     * sharing, caching, or exporting media and does not trust ITEM_SIZE_BYTES by itself.
     */
    public static final String ACTION_MEDIA_PROVIDER =
            "com.heliboard.intent.MEDIA_PROVIDER";
    public static final String ACTION_SEND_MEDIA_TO_IME =
            "helium314.keyboard.action.SEND_MEDIA_TO_IME";

    public static final String EXTRA_MEDIA_URI = "helium314.keyboard.extra.MEDIA_URI";
    public static final String EXTRA_MEDIA_MIME = "helium314.keyboard.extra.MEDIA_MIME";
    public static final String EXTRA_MEDIA_LABEL = "helium314.keyboard.extra.MEDIA_LABEL";
    public static final String EXTRA_MEDIA_MAX_BYTES =
            "helium314.keyboard.extra.MEDIA_MAX_BYTES";
    public static final String EXTRA_TARGET_PACKAGE =
            "helium314.keyboard.extra.TARGET_PACKAGE";

    public static final String MIME_IMAGE_GIF = "image/gif";
    public static final String MIME_IMAGE_PNG = "image/png";
    public static final String MIME_IMAGE_JPEG = "image/jpeg";
    public static final String MIME_IMAGE_WEBP = "image/webp";
    public static final String MIME_VIDEO_MP4 = "video/mp4";
    public static final String MIME_FOLDER = "vnd.android.document/directory";

    public static final String[] ACCEPTED_MIME_TYPES = new String[] {
            MIME_IMAGE_GIF,
            MIME_IMAGE_PNG,
            MIME_IMAGE_JPEG,
            MIME_IMAGE_WEBP,
            MIME_VIDEO_MP4
    };

    public static final String DEFAULT_MEDIA_LABEL = "media";
    public static final String LOG_TAG = "MediaPlugin";
    public static final String BUNDLE_SUPPORTS_SEARCH = "supports_search";
    public static final String BUNDLE_SUPPORTS_BROWSE = "supports_browse";
    public static final String BUNDLE_ITEMS = "items";
    public static final String BUNDLE_NEXT_PAGE_TOKEN = "next_page_token";
    public static final String BUNDLE_QUERY = "query";
    public static final String BUNDLE_PAGE_TOKEN = "page_token";
    public static final String BUNDLE_LIMIT = "limit";
    public static final String BUNDLE_MAX_BYTES = "max_bytes";
    public static final String ITEM_ID = "id";
    public static final String ITEM_TITLE = "title";
    public static final String ITEM_MIME = "mime";
    public static final String ITEM_WIDTH = "width";
    public static final String ITEM_HEIGHT = "height";
    public static final String ITEM_SIZE_BYTES = "size_bytes";
    public static final String ITEM_DURATION_MILLIS = "duration_millis";
    public static final String ITEM_PREVIEW_URI = "preview_uri";
    public static final String ITEM_CONTENT_URI = "content_uri";
    public static final String ITEM_LABEL = "label";

    private MediaPluginContract() {
    }

    public static boolean isAcceptedMimeType(final String mimeType) {
        if (mimeType == null) {
            return false;
        }
        for (final String accepted : ACCEPTED_MIME_TYPES) {
            if (ClipDescription.compareMimeTypes(mimeType, accepted)) {
                return true;
            }
        }
        return false;
    }
}
