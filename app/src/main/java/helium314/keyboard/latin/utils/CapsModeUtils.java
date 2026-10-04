/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import android.text.InputType;
import android.text.TextUtils;

import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.settings.SpacingAndPunctuations;

import java.util.ArrayList;

public final class CapsModeUtils {
    private CapsModeUtils() {
        // This utility class is not publicly instantiable.
    }
    /**
     * Helper method to find out if a code point is starting punctuation.
     * <p>
     * This include the Unicode START_PUNCTUATION category, but also some other symbols that are
     * starting, like the inverted question mark or the double quote.
     *
     * @param codePoint the code point
     * @return true if it's starting punctuation, false otherwise.
     */
    private static boolean isStartPunctuation(final int codePoint) {
        return (codePoint == Constants.CODE_DOUBLE_QUOTE || codePoint == Constants.CODE_SINGLE_QUOTE
                || codePoint == Constants.CODE_INVERTED_QUESTION_MARK
                || codePoint == Constants.CODE_INVERTED_EXCLAMATION_MARK
                || Character.getType(codePoint) == Character.START_PUNCTUATION);
    }

    /**
     * Determine what caps mode should be in effect at the current offset in
     * the text. Only the mode bits set in <var>reqModes</var> will be
     * checked. Note that the caps mode flags here are explicitly defined
     * to match those in {@link InputType}.
     * <p>
     * This code is a straight copy of TextUtils.getCapsMode (modulo namespace and formatting
     * issues). This will change in the future as we simplify the code for our use and fix bugs.
     *
     * @param cs The text that should be checked for caps modes.
     * @param reqModes The modes to be checked: may be any combination of
     * {@link TextUtils#CAP_MODE_CHARACTERS}, {@link TextUtils#CAP_MODE_WORDS}, and
     * {@link TextUtils#CAP_MODE_SENTENCES}.
     * @param spacingAndPunctuations The current spacing and punctuations settings.
     * @param hasSpaceBefore Whether we should consider there is a space inserted at the end of cs
     *
     * @return Returns the actual capitalization modes that can be in effect
     * at the current position, which is any combination of
     * {@link TextUtils#CAP_MODE_CHARACTERS}, {@link TextUtils#CAP_MODE_WORDS}, and
     * {@link TextUtils#CAP_MODE_SENTENCES}.
     */
    public static int getCapsMode(final CharSequence cs, final int reqModes,
            final SpacingAndPunctuations spacingAndPunctuations, final boolean hasSpaceBefore) {
        // Quick description of what we want to do:
        // CAP_MODE_CHARACTERS is always on.
        // CAP_MODE_WORDS is on if there is some whitespace before the cursor.
        // CAP_MODE_SENTENCES is on if there is some whitespace before the cursor, and the end
        //   of a sentence just before that.
        // We ignore opening parentheses and the like just before the cursor for purposes of
        // finding whitespace for WORDS and SENTENCES modes.
        // The end of a sentence ends with a period, question mark or exclamation mark. If it's
        // a period, it also needs not to be an abbreviation, which means it also needs to either
        // be immediately preceded by punctuation, or by a string of only letters with single
        // periods interleaved.

        // Step 1 : check for cap MODE_CHARACTERS. If it's looked for, it's always on.
        if ((reqModes & (TextUtils.CAP_MODE_WORDS | TextUtils.CAP_MODE_SENTENCES)) == 0) {
            // Here we are not looking for MODE_WORDS or MODE_SENTENCES, so since we already
            // evaluated MODE_CHARACTERS, we can return.
            return TextUtils.CAP_MODE_CHARACTERS & reqModes;
        }

        // Step 2 : Skip (ignore at the end of input) any opening punctuation. This includes
        // opening parentheses, brackets, opening quotes, everything that *opens* a span of
        // text in the linguistic sense. In RTL languages, this is still an opening sign, although
        // it may look like a right parenthesis for example. We also include double quote and
        // single quote since they aren't start punctuation in the unicode sense, but should still
        // be skipped for English. TODO: does this depend on the language?
        int i;
        if (hasSpaceBefore) {
            i = cs.length() + 1;
        } else {
            for (i = cs.length(); i > 0; i--) {
                final char c = cs.charAt(i - 1);
                if (!isStartPunctuation(c)) {
                    break;
                }
            }
        }

        // We are now on the character that precedes any starting punctuation, so in the most
        // frequent case this will be whitespace or a letter, although it may occasionally be a
        // start of line, or some symbol.

        // Step 3 : Search for the start of a paragraph. From the starting point computed in step 2,
        // we go back over any space or tab char sitting there. We find the start of a paragraph
        // if the first char that's not a space or tab is a start of line (as in \n, start of text,
        // or some other similar characters).
        int j = i;
        char prevChar = Constants.CODE_SPACE;
        if (hasSpaceBefore) --j;
        while (j > 0) {
            prevChar = cs.charAt(j - 1);
            if (!Character.isSpaceChar(prevChar) && prevChar != Constants.CODE_TAB) break;
            j--;
        }
        if (j <= 0 || Character.isWhitespace(prevChar)) {
            if (spacingAndPunctuations.mUsesGermanRules) {
                // In German typography rules, there is a specific case that the first character
                // of a new line should not be capitalized if the previous line ends in a comma.
                boolean hasNewLine = false;
                while (--j >= 0 && Character.isWhitespace(prevChar)) {
                    if (Constants.CODE_ENTER == prevChar) {
                        hasNewLine = true;
                    }
                    prevChar = cs.charAt(j);
                }
                if (Constants.CODE_COMMA == prevChar && hasNewLine) {
                    return (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS) & reqModes;
                }
            }
            // There are only spacing chars between the start of the paragraph and the cursor,
            // defined as a isWhitespace() char that is neither a isSpaceChar() nor a tab. Both
            // MODE_WORDS and MODE_SENTENCES should be active.
            return (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS
                    | TextUtils.CAP_MODE_SENTENCES) & reqModes;
        }
        if (i == j) {
            // If we don't have whitespace before index i, it means neither MODE_WORDS
            // nor mode sentences should be on so we can return right away.
            return TextUtils.CAP_MODE_CHARACTERS & reqModes;
        }
        if ((reqModes & TextUtils.CAP_MODE_SENTENCES) == 0) {
            // Here we know we have whitespace before the cursor (if not, we returned in the above
            // if i == j clause), so we need MODE_WORDS to be on. And we don't need to evaluate
            // MODE_SENTENCES so we can return right away.
            return (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS) & reqModes;
        }
        // Please note that because of the reqModes & CAP_MODE_SENTENCES test a few lines above,
        // we know that MODE_SENTENCES is being requested.

        // Step 4 : Search for MODE_SENTENCES.
        // English is a special case in that "American typography" rules, which are the most common
        // in English, state that a sentence terminator immediately following a quotation mark
        // should be swapped with it and de-duplicated (included in the quotation mark),
        // e.g. <<Did he say, "let's go home?">>
        // No other language has such a rule as far as I know, instead putting inside the quotation
        // mark as the exact thing quoted and handling the surrounding punctuation independently,
        // e.g. <<Did he say, "let's go home"?>>
        if (spacingAndPunctuations.mUsesAmericanTypography) {
            for (; j > 0; j--) {
                // Here we look to go over any closing punctuation. This is because in dominant
                // variants of English, the final period is placed within double quotes and maybe
                // other closing punctuation signs. This is generally not true in other languages.
                final char c = cs.charAt(j - 1);
                if (c != Constants.CODE_DOUBLE_QUOTE && c != Constants.CODE_SINGLE_QUOTE
                        && Character.getType(c) != Character.END_PUNCTUATION) {
                    break;
                }
            }
        }

        if (j <= 0) return TextUtils.CAP_MODE_CHARACTERS & reqModes;
        char c = cs.charAt(--j);

        // We found the next interesting chunk of text ; next we need to determine if it's the
        // end of a sentence. If we have a sentence terminator (typically a question mark or an
        // exclamation mark), then it's the end of a sentence; however, we treat the abbreviation
        // marker specially because usually is the same char as the sentence separator (the
        // period in most languages) and in this case we need to apply a heuristic to determine
        // in which of these senses it's used.
        if (spacingAndPunctuations.isSentenceTerminator(c)
                && !spacingAndPunctuations.isAbbreviationMarker(c)) {
            return (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS
                    | TextUtils.CAP_MODE_SENTENCES) & reqModes;
        }
        // If we reach here, we know we have whitespace before the cursor and before that there
        // is something that either does not terminate the sentence, or a symbol preceded by the
        // start of the text, or it's the sentence separator AND it happens to be the same code
        // point as the abbreviation marker.
        // If it's a symbol or something that does not terminate the sentence, then we need to
        // return caps for MODE_CHARACTERS and MODE_WORDS, but not for MODE_SENTENCES.
        if (!spacingAndPunctuations.isSentenceSeparator(c) || j <= 0) {
            return (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS) & reqModes;
        }

        // We found out that we have a period. We need to determine if this is a full stop or
        // otherwise sentence-ending period, or an abbreviation like "e.g.". An abbreviation
        // looks like (\w\.){2,}. Moreover, in German, you put periods after digits for dates
        // and some other things, and in German specifically we need to not go into autocaps after
        // a whitespace-digits-period sequence.
        // To find out, we will have a simple state machine with the following states :
        // START, WORD, PERIOD, ABBREVIATION, NUMBER
        // On START : (just before the first period)
        //           letter => WORD
        //           digit => NUMBER if German; end with caps otherwise
        //           whitespace => end with no caps (it was a stand-alone period)
        //           otherwise => end with caps (several periods/symbols in a row)
        // On WORD : (within the word just before the first period)
        //           letter => WORD
        //           period => PERIOD
        //           otherwise => end with caps (it was a word with a full stop at the end)
        // On PERIOD : (period within a potential abbreviation)
        //           letter => LETTER
        //           otherwise => end with caps (it was not an abbreviation)
        // On LETTER : (letter within a potential abbreviation)
        //           letter => LETTER
        //           period => PERIOD
        //           otherwise => end with no caps (it was an abbreviation)
        // On NUMBER : (period immediately preceded by one or more digits)
        //           digit => NUMBER
        //           letter => LETTER (promote to word)
        //           otherwise => end with no caps (it was a whitespace-digits-period sequence,
        //            or a punctuation-digits-period sequence like "11.11.")
        // "Not an abbreviation" in the above chart essentially covers cases like "...yes.". This
        // should capitalize.

        final int START = 0;
        final int WORD = 1;
        final int PERIOD = 2;
        final int LETTER = 3;
        final int NUMBER = 4;
        final int caps = (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS
                | TextUtils.CAP_MODE_SENTENCES) & reqModes;
        final int noCaps = (TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS) & reqModes;
        int state = START;
        while (j > 0) {
            c = cs.charAt(--j);
            switch (state) {
            case START:
                if (Character.isLetter(c)) {
                    state = WORD;
                } else if (Character.isWhitespace(c)) {
                    return noCaps;
                } else if (Character.isDigit(c) && spacingAndPunctuations.mUsesGermanRules) {
                    state = NUMBER;
                } else {
                    return caps;
                }
                break;
            case WORD:
                if (Character.isLetter(c)) {
                    state = WORD;
                } else if (spacingAndPunctuations.isSentenceSeparator(c)) {
                    state = PERIOD;
                } else {
                    return caps;
                }
                break;
            case PERIOD:
                if (Character.isLetter(c)) {
                    state = LETTER;
                } else {
                    return caps;
                }
                break;
            case LETTER:
                if (Character.isLetter(c)) {
                    state = LETTER;
                } else if (spacingAndPunctuations.isSentenceSeparator(c)) {
                    state = PERIOD;
                } else {
                    return noCaps;
                }
                break;
            case NUMBER:
                if (Character.isLetter(c)) {
                    state = WORD;
                } else if (Character.isDigit(c)) {
                    state = NUMBER;
                } else {
                    return noCaps;
                }
            }
        }
        // Here we arrived at the start of the line. This should behave exactly like whitespace.
        return (START == state || LETTER == state) ? noCaps : caps;
    }

    /**
     * Convert capitalize mode flags into human readable text.
     *
     * @param capsFlags The modes flags to be converted. It may be any combination of
     * {@link TextUtils#CAP_MODE_CHARACTERS}, {@link TextUtils#CAP_MODE_WORDS}, and
     * {@link TextUtils#CAP_MODE_SENTENCES}.
     * @return the text that describe the <code>capsMode</code>.
     */
    public static String flagsToString(final int capsFlags) {
        final int capsFlagsMask = TextUtils.CAP_MODE_CHARACTERS | TextUtils.CAP_MODE_WORDS
                | TextUtils.CAP_MODE_SENTENCES;
        if ((capsFlags & ~capsFlagsMask) != 0) {
            return "unknown<0x" + Integer.toHexString(capsFlags) + ">";
        }
        final ArrayList<String> builder = new ArrayList<>();
        if ((capsFlags & android.text.TextUtils.CAP_MODE_CHARACTERS) != 0) {
            builder.add("characters");
        }
        if ((capsFlags & android.text.TextUtils.CAP_MODE_WORDS) != 0) {
            builder.add("words");
        }
        if ((capsFlags & android.text.TextUtils.CAP_MODE_SENTENCES) != 0) {
            builder.add("sentences");
        }
        return builder.isEmpty() ? "none" : TextUtils.join("|", builder);
    }
}
