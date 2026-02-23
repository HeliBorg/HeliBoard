/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.Constants;

import java.util.HashMap;

public final class KeyboardCodesSet {
    public static final String PREFIX_CODE = "!code/";

    private static final HashMap<String, Integer> sNameToIdMap = new HashMap<>();

    private KeyboardCodesSet() {
        // This utility class is not publicly instantiable.
    }

    public static int getCode(final String name) {
        Integer id = sNameToIdMap.get(name);
        if (id == null) {
            try {
                return KeyCode.INSTANCE.checkAndConvertCode(Integer.parseInt(name));
            } catch (final Exception e) {
                throw new RuntimeException("Unknown key code: " + name);
            }
        }
        return DEFAULT[id];
    }

    private static final String[] ID_TO_NAME = {
        "key_tab",
        "key_enter",
        "key_space",
        "key_shift",
        "key_capslock",
        "key_switch_alpha_symbol",
        "key_switch_alpha",
        "key_switch_symbol",
        "key_output_text",
        "key_delete",
        "key_settings",
        "key_voice_input",
        "key_action_next",
        "key_action_previous",
        "key_shift_enter",
        "key_language_switch",
        "key_emoji",
        "key_unspecified",
        "key_clipboard",
        "key_toggle_onehanded",
        "key_start_onehanded", // keep name to avoid breaking custom layouts
        "key_stop_onehanded", // keep name to avoid breaking custom layouts
        "key_switch_onehanded",
        "key_emoji_search"
    };

    private static final int[] DEFAULT = {
        Constants.CODE_TAB,
        Constants.CODE_ENTER,
        Constants.CODE_SPACE,
        KeyCode.SHIFT,
        KeyCode.CAPS_LOCK,
        KeyCode.SYMBOL_ALPHA,
        KeyCode.ALPHA,
        KeyCode.SYMBOL,
        KeyCode.MULTIPLE_CODE_POINTS,
        KeyCode.DELETE,
        KeyCode.SETTINGS,
        KeyCode.VOICE_INPUT,
        KeyCode.ACTION_NEXT,
        KeyCode.ACTION_PREVIOUS,
        KeyCode.SHIFT_ENTER,
        KeyCode.LANGUAGE_SWITCH,
        KeyCode.EMOJI,
        KeyCode.NOT_SPECIFIED,
        KeyCode.CLIPBOARD,
        KeyCode.TOGGLE_ONE_HANDED_MODE,
        KeyCode.TOGGLE_ONE_HANDED_MODE,
        KeyCode.TOGGLE_ONE_HANDED_MODE,
        KeyCode.SWITCH_ONE_HANDED_MODE,
        KeyCode.EMOJI_SEARCH
    };

    static {
        for (int i = 0; i < ID_TO_NAME.length; i++) {
            sNameToIdMap.put(ID_TO_NAME[i], i);
        }
    }
}
