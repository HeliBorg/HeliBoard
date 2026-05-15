/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media.provider;

public final class MediaProviderInfo {
    public final String key;
    public final String label;
    public final String packageName;
    public final String serviceName;
    public final boolean supportsSearch;
    public final boolean supportsBrowse;

    public MediaProviderInfo(final String key, final String label, final String packageName,
            final String serviceName) {
        this(key, label, packageName, serviceName, true, false);
    }

    public MediaProviderInfo(final String key, final String label, final String packageName,
            final String serviceName, final boolean supportsSearch, final boolean supportsBrowse) {
        this.key = key;
        this.label = label;
        this.packageName = packageName;
        this.serviceName = serviceName;
        this.supportsSearch = supportsSearch;
        this.supportsBrowse = supportsBrowse;
    }

    public MediaProviderInfo withCapabilities(final boolean supportsSearch,
            final boolean supportsBrowse) {
        return new MediaProviderInfo(key, label, packageName, serviceName,
                supportsSearch, supportsBrowse);
    }
}
