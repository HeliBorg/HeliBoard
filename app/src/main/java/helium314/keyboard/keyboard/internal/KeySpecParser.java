/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.define.DebugFlags;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * The string parser of the key specification.
 * <p>
 * Each key specification is one of the following:
 * - Label optionally followed by keyOutputText (keyLabel|keyOutputText).
 * - Label optionally followed by code point (keyLabel|!code/code_name).
 * - Icon followed by keyOutputText (!icon/icon_name|keyOutputText).
 * - Icon followed by code point (!icon/icon_name|!code/code_name).
 * Label and keyOutputText are literal strings.
 * Icon is represented by (!icon/icon_name), see {@link KeyboardIconsSet}.
 * Code is one of the following:
 * - Code point presented by hexadecimal string prefixed with "0x"
 * - Code reference represented by (!code/code_name), see {@link KeyboardCodesSet}.
 * Special character, comma ',' backslash '\', and bar '|' can be escaped by '\' character.
 * Note that the '\' is also parsed by XML parser and {@link PopupKeySpec#splitKeySpecs(String)}
 * as well.
 */
// TODO: Rename to KeySpec and make this class to the key specification object.
public final class KeySpecParser {
    // Constants for parsing.
    private static final char BACKSLASH = Constants.CODE_BACKSLASH;
    private static final char VERTICAL_BAR = Constants.CODE_VERTICAL_BAR;
    private static final String PREFIX_HEX = "0x";

    private KeySpecParser() {
        // Intentional empty constructor for utility class.
    }

    private static boolean hasIcon(@NonNull final String keySpec) {
        return keySpec.startsWith(KeyboardIconsSet.PREFIX_ICON);
    }

    private static boolean hasCode(@NonNull final String keySpec, final int labelEnd) {
        if (labelEnd <= 0 || labelEnd + 1 >= keySpec.length()) {
            return false;
        }
        if (keySpec.startsWith(KeyboardCodesSet.PREFIX_CODE, labelEnd + 1)) {
            return true;
        }
        // This is a workaround to have a key that has a supplementary code point. We can't put a
        // string in resource as a XML entity of a supplementary code point or a surrogate pair.
        return keySpec.startsWith(PREFIX_HEX, labelEnd + 1);
    }

    @NonNull
    private static String parseEscape(@NonNull final String text) {
        if (text.indexOf(BACKSLASH) < 0) {
            return text;
        }
        final int length = text.length();
        final StringBuilder sb = new StringBuilder();
        for (int pos = 0; pos < length; pos++) {
            final char c = text.charAt(pos);
            if (c == BACKSLASH && pos + 1 < length) {
                // Skip escape char
                pos++;
                sb.append(text.charAt(pos));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static int indexOfLabelEnd(@NonNull final String keySpec) {
        final int length = keySpec.length();
        if (keySpec.indexOf(BACKSLASH) < 0) {
            final int labelEnd = keySpec.lastIndexOf(VERTICAL_BAR);
            if (labelEnd == 0) {
                if (length == 1) {
                    // Treat a sole vertical bar as a special case of key label.
                    return -1;
                }
                if (DebugFlags.DEBUG_ENABLED)
                    throw new KeySpecParserError("Empty label");
                else return -1;
            }
            return labelEnd;
        }
        for (int pos = length - 1; pos >= 0; pos--) {
            final char c = keySpec.charAt(pos);
            if (c != VERTICAL_BAR) continue;
            if (pos > 0 && keySpec.charAt(pos - 1) == BACKSLASH) {
                pos--; // Skip escape char
            } else {
                return pos;
            }
        }
        return -1;
    }

    @NonNull
    private static String getBeforeLabelEnd(@NonNull final String keySpec, final int labelEnd) {
        return (labelEnd < 0) ? keySpec : keySpec.substring(0, labelEnd);
    }

    @NonNull
    private static String getAfterLabelEnd(@NonNull final String keySpec, final int labelEnd) {
        return keySpec.substring(labelEnd + /* VERTICAL_BAR */1);
    }

    private static void checkDoubleLabelEnd(@NonNull final String keySpec, final int labelEnd) {
        if (indexOfLabelEnd(getAfterLabelEnd(keySpec, labelEnd)) < 0) {
            return;
        }
        if (DebugFlags.DEBUG_ENABLED)
            throw new KeySpecParserError("Multiple " + VERTICAL_BAR + ": " + keySpec);
    }

    @Nullable
    public static String getLabel(@Nullable final String keySpec) {
        if (keySpec == null) {
            // TODO: Throw {@link KeySpecParserError} once Key.keyLabel attribute becomes mandatory.
            return null;
        }
        if (hasIcon(keySpec)) {
            return null;
        }
        final int labelEnd = indexOfLabelEnd(keySpec);
        final String label = parseEscape(getBeforeLabelEnd(keySpec, labelEnd));
        if (label.isEmpty() && DebugFlags.DEBUG_ENABLED) {
            throw new KeySpecParserError("Empty label: " + keySpec);
        }
        return label;
    }

    @Nullable
    private static String getOutputTextInternal(@NonNull final String keySpec, final int labelEnd) {
        if (labelEnd <= 0) {
            return null;
        }
        checkDoubleLabelEnd(keySpec, labelEnd);
        return parseEscape(getAfterLabelEnd(keySpec, labelEnd));
    }

    @Nullable
    public static String getOutputText(@Nullable final String keySpec, final int code) {
        if (keySpec == null) {
            // TODO: Throw {@link KeySpecParserError} once Key.keyLabel attribute becomes mandatory.
            return null;
        }
        final int labelEnd = indexOfLabelEnd(keySpec);
        if (hasCode(keySpec, labelEnd)) {
            return null;
        }
        final String outputText = getOutputTextInternal(keySpec, labelEnd);
        // don't set output text if there is an actual code, because the text will not be used in that case
        if (outputText != null && (code == KeyCode.UNSPECIFIED || code == KeyCode.MULTIPLE_CODE_POINTS)) {
            if (StringUtils.codePointCount(outputText) == 1) {
                // If output text is one code point, it should be treated as a code.
                // See {@link #getCode(Resources, String)}.
                return null;
            }
            // also empty output texts are acceptable
            return outputText;
        }
        final String label = getLabel(keySpec);
        if (label == null) {
            if (keySpec.startsWith(KeyboardIconsSet.PREFIX_ICON) && code != KeyCode.UNSPECIFIED && code != KeyCode.MULTIPLE_CODE_POINTS)
                return null; // allow empty label in case of icon & actual code
            throw new KeySpecParserError("Empty label: " + keySpec);
        }
        // Code is automatically generated for one letter label. See {@link getCode()}.
        return (StringUtils.codePointCount(label) == 1) ? null : label;
    }

    public static int getCode(@Nullable final String keySpec) {
        if (keySpec == null) {
            // TODO: Throw {@link KeySpecParserError} once Key.keyLabel attribute becomes mandatory.
            return KeyCode.NOT_SPECIFIED;
        }
        final int labelEnd = indexOfLabelEnd(keySpec);
        if (hasCode(keySpec, labelEnd)) {
            checkDoubleLabelEnd(keySpec, labelEnd);
            return parseCode(getAfterLabelEnd(keySpec, labelEnd), KeyCode.NOT_SPECIFIED);
        }
        final String outputText = getOutputTextInternal(keySpec, labelEnd);
        if (outputText != null) {
            // If output text is one code point, it should be treated as a code.
            // See {@link #getOutputText(String)}.
            if (StringUtils.codePointCount(outputText) == 1) {
                return outputText.codePointAt(0);
            }
            return KeyCode.MULTIPLE_CODE_POINTS;
        }
        final String label = getLabel(keySpec);
        if (label == null) {
            if (DebugFlags.DEBUG_ENABLED)
                throw new KeySpecParserError("Empty label: " + keySpec);
            else return KeyCode.MULTIPLE_CODE_POINTS;
        }
        // Code is automatically generated for one letter label.
        return (StringUtils.codePointCount(label) == 1) ? label.codePointAt(0) : KeyCode.MULTIPLE_CODE_POINTS;
    }

    public static int parseCode(@Nullable final String text, final int defaultCode) {
        if (text == null) {
            return defaultCode;
        }
        if (text.startsWith(KeyboardCodesSet.PREFIX_CODE)) {
            return KeyboardCodesSet.getCode(text.substring(KeyboardCodesSet.PREFIX_CODE.length()));
        }
        // This is a workaround to have a key that has a supplementary code point. We can't put a
        // string in resource as a XML entity of a supplementary code point or a surrogate pair.
        if (text.startsWith(PREFIX_HEX)) {
            return Integer.parseInt(text.substring(PREFIX_HEX.length()), 16);
        }
        return defaultCode;
    }

    @Nullable
    public static String getIconName(@Nullable final String keySpec) {
        if (keySpec == null) {
            // TODO: Throw {@link KeySpecParserError} once Key.keyLabel attribute becomes mandatory.
            return null;
        }
        if (!hasIcon(keySpec)) {
            return null;
        }
        final int labelEnd = indexOfLabelEnd(keySpec);
        return getBeforeLabelEnd(keySpec, labelEnd).substring(KeyboardIconsSet.PREFIX_ICON.length()).intern();
    }

    public static final class KeySpecParserError extends RuntimeException {
        public KeySpecParserError(final String message) {
            super(message);
        }
    }
}
