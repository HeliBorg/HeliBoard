/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.common;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.BuildConfig;

public final class Constants {

    public static final class Color {
        /**
         * The alpha value for fully opaque.
         */
        public final static int ALPHA_OPAQUE = 255;
    }

    public static final class ImeOption {
        /**
         * The private IME option used to indicate that no microphone should be shown for a given
         * text field. For instance, this is specified by the search dialog when the dialog is
         * already showing a voice search button.
         */
        public static final String NO_MICROPHONE = "noMicrophoneKey";

        /**
         * The private IME option used to suppress the floating gesture preview for a given text
         * field. This overrides the corresponding keyboard settings preference.
         * {@link helium314.keyboard.latin.settings.SettingsValues#mGestureFloatingPreviewTextEnabled}
         */
        public static final String NO_FLOATING_GESTURE_PREVIEW = "noGestureFloatingPreview";

        private ImeOption() {
            // This utility class is not publicly instantiable.
        }
    }

    public static final class Subtype {
        /** The subtype mode used to indicate that the subtype is a keyboard. */
        public static final String KEYBOARD_MODE = "keyboard";

        // some extra values:
        //  TrySuppressingImeSwitcher: not documented, but used in Android source
        //  SupportTouchPositionCorrection: never read, never used outside AOSP keyboard -> can be removed?
        public static final class ExtraValue {
            /** Indicates that this subtype is capable of entering ASCII characters (not used, but recommended for Android 9 and older). */
            public static final String ASCII_CAPABLE = "AsciiCapable";

            /** Indicates that this subtype is enabled when the default subtype is not marked as ascii capable (used where?). */
            public static final String ENABLED_WHEN_DEFAULT_IS_NOT_ASCII_CAPABLE = "EnabledWhenDefaultIsNotAsciiCapable";

            /** Indicates that this subtype is capable of entering emoji characters (always set?). */
            public static final String EMOJI_CAPABLE = "EmojiCapable";

            /** Indicates that the subtype does not have a shift key */
            public static final String NO_SHIFT_KEY = "NoShiftKey";

            /** Indicates that for this subtype corrections should not be based on proximity of keys for when shifted */
            public static final String NO_SHIFT_PROXIMITY_CORRECTION = "NoShiftProximityCorrection";

            /**
             * The subtype extra value used to indicate that the display name of this subtype
             * contains a "%s" for printf-like replacement and it should be replaced by
             * this extra value.
             * This extra value is supported on JellyBean and later.
             */
            public static final String UNTRANSLATABLE_STRING_IN_SUBTYPE_NAME = "UntranslatableReplacementStringInSubtypeName";

            /** Contains the layouts used by this subtype. This extra value is private to LatinIME.*/
            public static final String KEYBOARD_LAYOUT_SET = "KeyboardLayoutSet";

            /** Indicates that this subtype is an additional subtype that the user defined. This extra value is private to LatinIME. */
            public static final String IS_ADDITIONAL_SUBTYPE = "isAdditionalSubtype";

            /** The subtype extra value used to specify the combining rules. */
            public static final String COMBINING_RULES = "CombiningRules";

            /** Overrides the general popup order setting */
            public static final String POPUP_ORDER = "PopupOrder";

            /** Overrides the general hint order / priority setting */
            public static final String HINT_ORDER = "HintOrder";

            /** Language tags indicating enabled secondary locales */
            public static final String SECONDARY_LOCALES = "SecondaryLocales";

            /** Overrides the general "more popups" setting */
            public static final String MORE_POPUPS = "MorePopups";

            /** Overrides the general "localized number row" setting */
            public static final String LOCALIZED_NUMBER_ROW = "LocalizedNumberRow";

            private ExtraValue() {
                // This utility class is not publicly instantiable.
            }
        }

        private Subtype() {
            // This utility class is not publicly instantiable.
        }
    }

    /** Separators for use in extra values and preferences. Notably cannot be = and , as they are already used in extra values */
    public static final class Separators {
        /** key-value separator (to be used in subtype extra values) */
        public static final String KV = ":";
        /** separator between entries that might be key-value pairs (to be used in subtype extra values) */
        public static final String ENTRY = "|";
        /** separator between sets of entries (to be used for storing data for additional subtypes) */
        public static final String SET = "§";
        /** separator for sets (to be used for storing multiple extra additional subtypes in prefs) */
        public static final String SETS = ";";
    }

    public static final class TextUtils {
        /**
         * Capitalization mode for {@link android.text.TextUtils#getCapsMode}: don't capitalize
         * characters.  This value may be used with
         * {@link android.text.TextUtils#CAP_MODE_CHARACTERS},
         * {@link android.text.TextUtils#CAP_MODE_WORDS}, and
         * {@link android.text.TextUtils#CAP_MODE_SENTENCES}.
         */
        // TODO: Straighten this out. It's bizarre to have to use android.text.TextUtils.CAP_MODE_*
        // except for OFF that is in Constants.TextUtils.
        public static final int CAP_MODE_OFF = 0;

        private TextUtils() {
            // This utility class is not publicly instantiable.
        }
    }

    public static final int NOT_A_CODE = -1;
    public static final int NOT_A_CURSOR_POSITION = -1;
    // TODO: replace the following constants with state in InputTransaction?
    public static final int NOT_A_COORDINATE = -1;
    public static final int SUGGESTION_STRIP_COORDINATE = -2;
    public static final int EXTERNAL_KEYBOARD_COORDINATE = -4;

    // A hint on how many characters to cache from the TextView. A good value of this is given by
    // how many characters we need to be able to almost always find the caps mode.
    public static final int EDITOR_CONTENTS_CACHE_SIZE = 1024;
    // How many characters we accept for the recapitalization functionality. This needs to be
    // large enough for all reasonable purposes, but avoid purposeful attacks. 100k sounds about
    // right for this.
    public static final int MAX_CHARACTERS_FOR_RECAPITALIZATION = 1024 * 100;

    // Key events coming any faster than this are long-presses.
    public static final int LONG_PRESS_MILLISECONDS = 200;
    // TODO: Set this value appropriately.
    public static final int GET_SUGGESTED_WORDS_TIMEOUT = BuildConfig.DEBUG ? 500 : 200; // debug build is slow, and timeout is annoying for testing
    // How many continuous deletes at which to start deleting at a higher speed.
    public static final int DELETE_ACCELERATE_AT = 20;

    public static final String WORD_SEPARATOR = " ";

    public static boolean isValidCoordinate(final int coordinate) {
        // Detect {@link NOT_A_COORDINATE}, {@link SUGGESTION_STRIP_COORDINATE},
        // and {@link SPELL_CHECKER_COORDINATE}.
        return coordinate >= 0;
    }

    /**
     * Custom request code used in
     * {@link helium314.keyboard.keyboard.KeyboardActionListener#onCustomRequest(int)}.
     */
    // The code to show input method picker.
    public static final int CUSTOM_CODE_SHOW_INPUT_METHOD_PICKER = 1;

    /**
     * Some common keys code. Must be positive.
     */
    public static final int CODE_ENTER = '\n';
    public static final int CODE_TAB = '\t';
    public static final int CODE_SPACE = ' ';
    public static final int CODE_PERIOD = '.';
    public static final int CODE_COMMA = ',';
    public static final int CODE_DASH = '-';
    public static final int CODE_SINGLE_QUOTE = '\'';
    public static final int CODE_DOUBLE_QUOTE = '"';
    public static final int CODE_SLASH = '/';
    public static final int CODE_BACKSLASH = '\\';
    public static final int CODE_VERTICAL_BAR = '|';
    public static final int CODE_COMMERCIAL_AT = '@';
    public static final int CODE_PLUS = '+';
    public static final int CODE_PERCENT = '%';
    public static final int CODE_CLOSING_PARENTHESIS = ')';
    public static final int CODE_CLOSING_SQUARE_BRACKET = ']';
    public static final int CODE_CLOSING_CURLY_BRACKET = '}';
    public static final int CODE_CLOSING_ANGLE_BRACKET = '>';
    public static final int CODE_INVERTED_QUESTION_MARK = '¿';
    public static final int CODE_INVERTED_EXCLAMATION_MARK = '¡';
    public static final int CODE_GRAVE_ACCENT = '`';
    public static final int CODE_CIRCUMFLEX_ACCENT = '^';
    public static final int CODE_TILDE = '~';
    public static final int RECENTS_TEMPLATE_KEY_CODE_0 = 0x30;
    public static final int RECENTS_TEMPLATE_KEY_CODE_1 = 0x31;

    public static final String REGEXP_PERIOD = "\\.";
    public static final String STRING_SPACE = " ";

    public static boolean isLetterCode(final int code) {
        return code >= CODE_SPACE;
    }

    @NonNull
    public static String printableCode(final int code) {
        switch (code) {
        case KeyCode.SHIFT: return "shift";
        case KeyCode.CAPS_LOCK: return "capslock";
        case KeyCode.SYMBOL_ALPHA: return "symbol_alpha";
        case KeyCode.ALPHA: return "alpha";
        case KeyCode.SYMBOL: return "symbol";
        case KeyCode.MULTIPLE_CODE_POINTS: return "text";
        case KeyCode.DELETE: return "delete";
        case KeyCode.SETTINGS: return "settings";
        case KeyCode.VOICE_INPUT: return "shortcut";
        case KeyCode.ACTION_NEXT: return "actionNext";
        case KeyCode.ACTION_PREVIOUS: return "actionPrevious";
        case KeyCode.LANGUAGE_SWITCH: return "languageSwitch";
        case KeyCode.EMOJI: return "emoji";
        case KeyCode.CLIPBOARD: return "clipboard";
        case KeyCode.SHIFT_ENTER: return "shiftEnter";
        case KeyCode.NOT_SPECIFIED: return "unspec";
        case CODE_TAB: return "tab";
        case CODE_ENTER: return "enter";
        case CODE_SPACE: return "space";
        case KeyCode.TOGGLE_ONE_HANDED_MODE: return "toggleOneHandedMode";
        case KeyCode.SWITCH_ONE_HANDED_MODE: return "switchOneHandedMode";
        case KeyCode.SPLIT_LAYOUT: return "splitLayout";
        case KeyCode.NUMPAD: return "numpad";
        case KeyCode.EMOJI_SEARCH: return "emojiSearch";
        default:
            if (code < CODE_SPACE) return String.format("\\u%02X", code);
            if (code < 0x100) return String.format("%c", code);
            if (code < 0x10000) return String.format("\\u%04X", code);
            return String.format("\\U%05X", code);
        }
    }

    /**
     * Screen metrics (a.k.a. Device form factor) constants of
     * {@link helium314.keyboard.latin.R.integer#config_screen_metrics}.
     */
    public static final int SCREEN_METRICS_SMALL_PHONE = 0;
    public static final int SCREEN_METRICS_LARGE_PHONE = 1;
    public static final int SCREEN_METRICS_LARGE_TABLET = 2;
    public static final int SCREEN_METRICS_SMALL_TABLET = 3;

    /**
     * Default capacity of gesture points container.
     * This constant is used by {@link helium314.keyboard.keyboard.internal.BatchInputArbiter}
     * and etc. to preallocate regions that contain gesture event points.
     */
    public static final int DEFAULT_GESTURE_POINTS_CAPACITY = 128;

    public static final int MAX_IME_DECODER_RESULTS = 20;
    public static final int DECODER_SCORE_SCALAR = 1000000;
    public static final int DECODER_MAX_SCORE = 1000000000;

    public static final int EVENT_BACKSPACE = 1;
    public static final int EVENT_REJECTION = 2;
    public static final int EVENT_REVERT = 3;

    private Constants() {
        // This utility class is not publicly instantiable.
    }
}
