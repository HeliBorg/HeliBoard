/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.settings;

/**
 * Debug settings for the application.
 */
public final class DebugSettings {
    public static final String PREF_DEBUG_MODE = "debug_mode";
    public static final String PREF_FORCE_NON_DISTINCT_MULTITOUCH = "force_non_distinct_multitouch";
    public static final String PREF_SLIDING_KEY_INPUT_PREVIEW = "sliding_key_input_preview";
    public static final String PREF_SHOW_DEBUG_SETTINGS = "show_debug_settings";
    public static final String PREF_EXPORT_DICTIONARY = "export_dictionary";

    public static final String PREF_SHOW_SUGGESTION_INFOS = "show_suggestion_infos";
    private DebugSettings() {
        // This class is not publicly instantiable.
    }
}
