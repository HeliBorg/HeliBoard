/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.common;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.utils.ScriptUtils;

import java.util.Arrays;
import java.util.Locale;

public final class StringUtils {

    public static final int CAPITALIZE_NONE = 0;  // No caps, or mixed case
    public static final int CAPITALIZE_FIRST = 1; // First only
    public static final int CAPITALIZE_ALL = 2;   // All caps

    private static final char CHAR_LINE_FEED = 0X000A;
    private static final char CHAR_VERTICAL_TAB = 0X000B;
    private static final char CHAR_FORM_FEED = 0X000C;
    private static final char CHAR_CARRIAGE_RETURN = 0X000D;
    private static final char CHAR_NEXT_LINE = 0X0085;
    private static final char CHAR_LINE_SEPARATOR = 0X2028;
    private static final char CHAR_PARAGRAPH_SEPARATOR = 0X2029;

    private StringUtils() {
        // This utility class is not publicly instantiable.
    }

    // Taken from android.text.TextUtils. We are extensively using this method in many places,
    // some of which don't have the android libraries available.

    /**
     * Returns true if the string is null or 0-length.
     *
     * @param str the string to be examined
     * @return true if str is null or zero length
     */
    public static boolean isEmpty(@Nullable final CharSequence str) {
        return (str == null || str.length() == 0);
    }

    public static int codePointCount(@Nullable final CharSequence text) {
        if (isEmpty(text)) {
            return 0;
        }
        return Character.codePointCount(text, 0, text.length());
    }

    @NonNull
    public static String newSingleCodePointString(final int codePoint) {
        if (Character.charCount(codePoint) == 1) {
            // Optimization: avoid creating a temporary array for characters that are
            // represented by a single char value
            return String.valueOf((char) codePoint);
        }
        // For surrogate pair
        return new String(Character.toChars(codePoint));
    }

    @NonNull
    public static String capitalizeFirstCodePoint(@NonNull final String s,
                                                  @NonNull final Locale locale) {
        if (s.length() <= 1) {
            return s.toUpperCase(getLocaleUsedForToTitleCase(locale));
        }
        // Please refer to the comment below in
        // {@link #capitalizeFirstAndDowncaseRest(String,Locale)} as this has the same shortcomings
        final int cutoff = s.offsetByCodePoints(0, 1);
        return s.substring(0, cutoff).toUpperCase(getLocaleUsedForToTitleCase(locale))
                + s.substring(cutoff);
    }

    @NonNull
    public static String capitalizeFirstAndDowncaseRest(@NonNull final String s,
                                                        @NonNull final Locale locale) {
        if (s.length() <= 1) {
            return s.toUpperCase(getLocaleUsedForToTitleCase(locale));
        }
        // TODO: fix the bugs below
        // - It does not work for Serbian, because it fails to account for the "lj" character,
        // which should be "Lj" in title case and "LJ" in upper case.
        // - It does not work for Dutch, because it fails to account for the "ij" digraph when it's
        // written as two separate code points. They are two different characters but both should
        // be capitalized as "IJ" as if they were a single letter in most words (not all). If the
        // unicode char for the ligature is used however, it works.
        final int cutoff = s.offsetByCodePoints(0, 1);
        return s.substring(0, cutoff).toUpperCase(getLocaleUsedForToTitleCase(locale))
                + s.substring(cutoff).toLowerCase(locale);
    }

    @NonNull
    public static int[] toCodePointArray(@NonNull final CharSequence charSequence) {
        return toCodePointArray(charSequence, 0, charSequence.length());
    }

    @NonNull
    private static final int[] EMPTY_CODEPOINTS = {};

    /**
     * Converts a range of a string to an array of code points.
     *
     * @param charSequence the source string.
     * @param startIndex   the start index inside the string in java chars, inclusive.
     * @param endIndex     the end index inside the string in java chars, exclusive.
     * @return a new array of code points. At most endIndex - startIndex, but possibly less.
     */
    @NonNull
    public static int[] toCodePointArray(@NonNull final CharSequence charSequence,
                                         final int startIndex, final int endIndex) {
        final int length = charSequence.length();
        if (length <= 0) {
            return EMPTY_CODEPOINTS;
        }
        final int[] codePoints =
                new int[Character.codePointCount(charSequence, startIndex, endIndex)];
        copyCodePointsAndReturnCodePointCount(codePoints, charSequence, startIndex, endIndex,
                false /* downCase */);
        return codePoints;
    }

    /**
     * Copies the codepoints in a CharSequence to an int array.
     * <p>
     * This method assumes there is enough space in the array to store the code points. The size
     * can be measured with Character#codePointCount(CharSequence, int, int) before passing to this
     * method. If the int array is too small, an ArrayIndexOutOfBoundsException will be thrown.
     * Also, this method makes no effort to be thread-safe. Do not modify the CharSequence while
     * this method is running, or the behavior is undefined.
     * This method can optionally downcase code points before copying them, but it pays no attention
     * to locale while doing so.
     *
     * @param destination  the int array.
     * @param charSequence the CharSequence.
     * @param startIndex   the start index inside the string in java chars, inclusive.
     * @param endIndex     the end index inside the string in java chars, exclusive.
     * @param downCase     if this is true, code points will be downcased before being copied.
     * @return the number of copied code points.
     */
    public static int copyCodePointsAndReturnCodePointCount(@NonNull final int[] destination,
                                                            @NonNull final CharSequence charSequence, final int startIndex, final int endIndex,
                                                            final boolean downCase) {
        int destIndex = 0;
        for (int index = startIndex; index < endIndex;
             index = Character.offsetByCodePoints(charSequence, index, 1)) {
            final int codePoint = Character.codePointAt(charSequence, index);
            // TODO: stop using this, as it's not aware of the locale and does not always do
            // the right thing.
            destination[destIndex] = downCase ? Character.toLowerCase(codePoint) : codePoint;
            destIndex++;
        }
        return destIndex;
    }

    @NonNull
    public static int[] toSortedCodePointArray(@NonNull final String string) {
        final int[] codePoints = toCodePointArray(string);
        Arrays.sort(codePoints);
        return codePoints;
    }

    /**
     * Construct a String from a code point array
     *
     * @param codePoints a code point array that is null terminated when its logical length is
     *                   shorter than the array length.
     * @return a string constructed from the code point array.
     */
    @NonNull
    public static String getStringFromNullTerminatedCodePointArray(
            @NonNull final int[] codePoints) {
        int stringLength = codePoints.length;
        for (int i = 0; i < codePoints.length; i++) {
            if (codePoints[i] == 0) {
                stringLength = i;
                break;
            }
        }
        return new String(codePoints, 0 /* offset */, stringLength);
    }

    // This method assumes the text is not null. For the empty string, it returns CAPITALIZE_NONE.
    public static int getCapitalizationType(@NonNull final String text) {
        // If the first char is not uppercase, then the word is either all lower case or
        // camel case, and in either case we return CAPITALIZE_NONE.
        final int len = text.length();
        int index = 0;
        for (; index < len; index = text.offsetByCodePoints(index, 1)) {
            if (Character.isLetter(text.codePointAt(index))) {
                break;
            }
        }
        if (index == len) return CAPITALIZE_NONE;
        if (!Character.isUpperCase(text.codePointAt(index))) {
            return CAPITALIZE_NONE;
        }
        int capsCount = 1;
        int letterCount = 1;
        for (index = text.offsetByCodePoints(index, 1); index < len;
             index = text.offsetByCodePoints(index, 1)) {
            if (1 != capsCount && letterCount != capsCount) break;
            final int codePoint = text.codePointAt(index);
            if (Character.isUpperCase(codePoint)) {
                ++capsCount;
                ++letterCount;
            } else if (Character.isLetter(codePoint)) {
                // We need to discount non-letters since they may not be upper-case, but may
                // still be part of a word (e.g. single quote or dash, as in "IT'S" or "FULL-TIME")
                ++letterCount;
            }
        }
        // We know the first char is upper case. So we want to test if either every letter other
        // than the first is lower case, or if they are all upper case. If the string is exactly
        // one char long, then we will arrive here with letterCount 1, and this is correct, too.
        if (1 == capsCount) return CAPITALIZE_FIRST;
        return (letterCount == capsCount ? CAPITALIZE_ALL : CAPITALIZE_NONE);
    }

    public static boolean isIdenticalAfterUpcase(@NonNull final String text) {
        final int length = text.length();
        int i = 0;
        while (i < length) {
            final int codePoint = text.codePointAt(i);
            if (Character.isLetter(codePoint) && !Character.isUpperCase(codePoint)) {
                return false;
            }
            i += Character.charCount(codePoint);
        }
        return true;
    }

    public static boolean isIdenticalAfterDowncase(@NonNull final String text) {
        final int length = text.length();
        int i = 0;
        while (i < length) {
            final int codePoint = text.codePointAt(i);
            if (Character.isLetter(codePoint) && !Character.isLowerCase(codePoint)) {
                return false;
            }
            i += Character.charCount(codePoint);
        }
        return true;
    }

    public static boolean isIdenticalAfterCapitalizeEachWord(@NonNull final String text,
                                                             @NonNull final int[] sortedSeparators) {
        boolean needsCapsNext = true;
        final int len = text.length();
        for (int i = 0; i < len; i = text.offsetByCodePoints(i, 1)) {
            final int codePoint = text.codePointAt(i);
            if (Character.isLetter(codePoint)) {
                if ((needsCapsNext && !Character.isUpperCase(codePoint))
                        || (!needsCapsNext && !Character.isLowerCase(codePoint))) {
                    return false;
                }
            }
            // We need a capital letter next if this is a separator.
            needsCapsNext = (Arrays.binarySearch(sortedSeparators, codePoint) >= 0);
        }
        return true;
    }

    // TODO: like capitalizeFirst*, this does not work perfectly for Dutch because of the IJ digraph
    // which should be capitalized together in *some* cases.
    @NonNull
    public static String capitalizeEachWord(@NonNull final String text,
                                            @NonNull final int[] sortedSeparators, @NonNull final Locale locale) {
        final StringBuilder builder = new StringBuilder();
        boolean needsCapsNext = true;
        final int len = text.length();
        for (int i = 0; i < len; i = text.offsetByCodePoints(i, 1)) {
            final String nextChar = text.substring(i, text.offsetByCodePoints(i, 1));
            if (needsCapsNext) {
                builder.append(nextChar.toUpperCase(locale));
            } else {
                builder.append(nextChar.toLowerCase(locale));
            }
            // We need a capital letter next if this is a separator.
            needsCapsNext = (Arrays.binarySearch(sortedSeparators, nextChar.codePointAt(0)) >= 0);
        }
        return builder.toString();
    }

    /**
     * Approximates whether the text before the cursor looks like a URL.
     * <p>
     * This is not foolproof, but it should work well in the practice.
     * Essentially it walks backward from the cursor until it finds something that's not a letter,
     * digit, or common URL symbol like underscore. If it hasn't found a period yet, then it
     * does not look like a URL.
     * If the text:
     * - starts with www and contains a period
     * - starts with a slash preceded by either a slash, whitespace, or start-of-string
     * Then it looks like a URL and we return true. Otherwise, we return false.
     * <p>
     * Note: this method is called quite often, and should be fast.
     * <p>
     * TODO: This will return that "abc./def" and ".abc/def" look like URLs to keep down the
     * code complexity, but ideally it should not. It's acceptable for now.
     */
    public static boolean lastPartLooksLikeURL(@NonNull final CharSequence text) {
        int i = text.length();
        if (0 == i) {
            return false;
        }
        int wCount = 0;
        int slashCount = 0;
        boolean hasSlash = false;
        boolean hasPeriod = false;
        int codePoint = 0;
        while (i > 0) {
            codePoint = Character.codePointBefore(text, i);
            if (codePoint < Constants.CODE_PERIOD || codePoint > 'z') {
                // Handwavy heuristic to see if that's a URL character. Anything between period
                // and z. This includes all lower- and upper-case ascii letters, period,
                // underscore, arrobase, question mark, equal sign. It excludes spaces, exclamation
                // marks, double quotes...
                // Anything that's not a URL-like character causes us to break from here and
                // evaluate normally.
                break;
            }
            if (Constants.CODE_PERIOD == codePoint) {
                hasPeriod = true;
            }
            if (Constants.CODE_SLASH == codePoint) {
                hasSlash = true;
                if (2 == ++slashCount) {
                    return true;
                }
            } else {
                slashCount = 0;
            }
            if ('w' == codePoint) {
                ++wCount;
            } else {
                wCount = 0;
            }
            i = Character.offsetByCodePoints(text, i, -1);
        }
        // End of the text run.
        // If it starts with www and includes a period, then it looks like a URL.
        if (wCount >= 3 && hasPeriod) {
            return true;
        }
        // If it starts with a slash, and the code point before is whitespace, it looks like an URL.
        if (1 == slashCount && (0 == i || Character.isWhitespace(codePoint))) {
            return true;
        }
        // If it has both a period and a slash, it looks like an URL.
        return hasPeriod && hasSlash;
        // Otherwise, it doesn't look like an URL.
    }

    /**
     * Examines the string and returns whether we're inside a double quote.
     * <p>
     * This is used to decide whether we should put an automatic space before or after a double
     * quote character. If we're inside a quotation, then we want to close it, so we want a space
     * after and not before. Otherwise, we want to open the quotation, so we want a space before
     * and not after. Exception: after a digit, we never want a space because the "inch" or
     * "minutes" use cases is dominant after digits.
     * In the practice, we determine whether we are in a quotation or not by finding the previous
     * double quote character, and looking at whether it's followed by whitespace. If so, that
     * was a closing quotation mark, so we're not inside a double quote. If it's not followed
     * by whitespace, then it was an opening quotation mark, and we're inside a quotation.
     * However, on the way to the double quote we can determine, some double quotes might be
     * ignored, e.g. because they are followed by punctuation. These double quotes are counted and
     * taken into account.
     *
     * @param text the text to examine.
     * @return whether we're inside a double quote.
     */
    public static boolean isInsideDoubleQuoteOrAfterDigit(@NonNull final CharSequence text) {
        int i = text.length();
        if (0 == i) {
            return false;
        }
        int codePoint = Character.codePointBefore(text, i);
        if (Character.isDigit(codePoint)) {
            return true;
        }
        int prevCodePoint = 0;
        int ignoredDoubleQuoteCount = 0;
        while (i > 0) {
            codePoint = Character.codePointBefore(text, i);
            if (Constants.CODE_DOUBLE_QUOTE == codePoint) {
                // If we see a double quote followed by whitespace, then that
                // was a closing quote.
                if (Character.isWhitespace(prevCodePoint)) {
                    return ignoredDoubleQuoteCount % 2 == 1;
                }
            }
            if (Character.isWhitespace(codePoint) && Constants.CODE_DOUBLE_QUOTE == prevCodePoint) {
                // If we see a double quote preceded by whitespace, then that
                // was an opening quote. No need to continue seeking.
                return ignoredDoubleQuoteCount % 2 == 0;
            }
            if (Constants.CODE_DOUBLE_QUOTE == prevCodePoint) {
                ignoredDoubleQuoteCount++;
            }
            i -= Character.charCount(codePoint);
            prevCodePoint = codePoint;
        }
        // We reached the start of text. If the first char is a double quote, then we're inside
        // a double quote. Otherwise we're not.
        if (ignoredDoubleQuoteCount % 2 == 0)
            return Constants.CODE_DOUBLE_QUOTE == codePoint;
        else
            return Constants.CODE_DOUBLE_QUOTE != codePoint;
    }

    public static boolean isEmptyStringOrWhiteSpaces(@NonNull final String s) {
        final int N = codePointCount(s);
        for (int i = 0; i < N; ++i) {
            if (!Character.isWhitespace(s.codePointAt(i))) {
                return false;
            }
        }
        return true;
    }

    private static final String LANGUAGE_GREEK = "el";

    @NonNull
    private static Locale getLocaleUsedForToTitleCase(@NonNull final Locale locale) {
        // In Greek locale {@link String#toUpperCase(Locale)} eliminates accents from its result.
        // In order to get accented upper case letter, {@link Locale#ROOT} should be used.
        if (LANGUAGE_GREEK.equals(locale.getLanguage())) {
            return Locale.ROOT;
        }
        return locale;
    }

    @Nullable
    public static String toTitleCaseOfKeyLabel(@Nullable final String label,
                                               @NonNull final Locale locale) {
        if (label == null || !ScriptUtils.scriptSupportsUppercase(locale)) {
            return label;
        }
        if (label.equals("ß"))
            return "ẞ"; // upcasing of standalone ß, SS is not useful as s is on the keyboard anyway

        return label.toUpperCase(getLocaleUsedForToTitleCase(locale));
    }

    public static int toTitleCaseOfKeyCode(final int code, @NonNull final Locale locale) {
        if (!Constants.isLetterCode(code)) {
            return code;
        }
        final String label = newSingleCodePointString(code);
        final String titleCaseLabel = toTitleCaseOfKeyLabel(label, locale);
        return codePointCount(titleCaseLabel) == 1
                ? titleCaseLabel.codePointAt(0) : KeyCode.NOT_SPECIFIED;
    }

    public static int getTrailingSingleQuotesCount(@NonNull final CharSequence charSequence) {
        final int lastIndex = charSequence.length() - 1;
        int i = lastIndex;
        while (i >= 0 && charSequence.charAt(i) == Constants.CODE_SINGLE_QUOTE) {
            --i;
        }
        return lastIndex - i;
    }

    /**
     * Returns whether the last composed word contains line-breaking character (e.g. CR or LF).
     *
     * @param text the text to be examined.
     * @return {@code true} if the last composed word contains line-breaking separator.
     */
    public static boolean hasLineBreakCharacter(@Nullable final String text) {
        if (isEmpty(text)) {
            return false;
        }
        for (int i = text.length() - 1; i >= 0; --i) {
            final char c = text.charAt(i);
            switch (c) {
                case CHAR_LINE_FEED:
                case CHAR_VERTICAL_TAB:
                case CHAR_FORM_FEED:
                case CHAR_CARRIAGE_RETURN:
                case CHAR_NEXT_LINE:
                case CHAR_LINE_SEPARATOR:
                case CHAR_PARAGRAPH_SEPARATOR:
                    return true;
            }
        }
        return false;
    }

    // unicode blocks that contain emojis
    // very fast check, but there are very few blocks that exclusively contain emojis,
    public static boolean mightBeEmoji(final int c) {
        return (0x200D <= c && c <= 0x2BFF) // unicode blocks from General Punctuation to Miscellaneous Symbols and Arrows
                || (0x1F000 <= c && c <= 0x1FAFF) // unicode blocks from Mahjong Tiles to Symbols and Pictographs Extended-A
                || (0xE0000 <= c && c <= 0xE007F) // unicode block Tags
                || c == 0xFE0F; // variation selector emoji with color
    }

    public static boolean isLowerCaseAscii(final String s) {
        final int length = s.length();
        for (int i = 0; i < length; i++) {
            final int c = s.charAt(i);
            if (c < 97 || c > 122) return false;
        }
        return true;
    }

    public static int charIndexOfFirstWhitespace(final CharSequence s) {
        for (int i = 0; i < s.length() - 1; i++) {
            final char c = s.charAt(i);
            if (Character.isWhitespace(c)) {
                return i + 1;
            }
        }
        return -1;
    }

    public static int charIndexOfLastWhitespace(final CharSequence s) {
        for (int i = s.length() - 1; i >= 0; i--) {
            final char c = s.charAt(i);
            if (Character.isWhitespace(c)) {
                return i + 1;
            }
        }
        return -1;
    }
}
