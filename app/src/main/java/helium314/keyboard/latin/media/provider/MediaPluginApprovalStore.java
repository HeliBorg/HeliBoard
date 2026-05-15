/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media.provider;

import android.content.Context;
import android.content.SharedPreferences;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public final class MediaPluginApprovalStore {
    private static final String PREFS_NAME = "media_provider_prefs";
    private static final String PREF_ENABLED_PROVIDERS = "enabled_providers";

    private MediaPluginApprovalStore() {
    }

    public static boolean isEnabled(final Context context, final MediaProviderInfo provider) {
        return enabledKeys(context).contains(provider.key);
    }

    public static void setEnabled(final Context context, final MediaProviderInfo provider,
            final boolean enabled) {
        final HashSet<String> keys = new HashSet<>(enabledKeys(context));
        if (enabled) {
            keys.add(provider.key);
        } else {
            keys.remove(provider.key);
        }
        prefs(context).edit().putStringSet(PREF_ENABLED_PROVIDERS, keys).apply();
    }

    public static Set<String> enabledKeys(final Context context) {
        final Set<String> keys = prefs(context).getStringSet(PREF_ENABLED_PROVIDERS,
                Collections.emptySet());
        return keys == null ? Collections.emptySet() : new HashSet<>(keys);
    }

    static boolean isKnownEnabledKey(final Context context, final String key) {
        return enabledKeys(context).contains(key);
    }

    private static SharedPreferences prefs(final Context context) {
        return context.getApplicationContext()
                .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }
}
