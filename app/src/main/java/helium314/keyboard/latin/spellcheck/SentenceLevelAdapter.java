/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.spellcheck;

import android.content.res.Resources;
import android.view.textservice.SentenceSuggestionsInfo;
import android.view.textservice.SuggestionsInfo;
import android.view.textservice.TextInfo;

import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.settings.SpacingAndPunctuations;
import helium314.keyboard.latin.utils.RunInLocaleKt;

import java.util.ArrayList;
import java.util.Locale;

/**
 * This code is mostly lifted directly from android.service.textservice.SpellCheckerService in
 * the framework; maybe that should be protected instead, so that implementers don't have to
 * rewrite everything for any small change.
 */
public class SentenceLevelAdapter {
    private static class EmptySentenceSuggestionsInfosInitializationHolder {
        public static final SentenceSuggestionsInfo[] EMPTY_SENTENCE_SUGGESTIONS_INFOS =
                new SentenceSuggestionsInfo[]{};
    }
    private static final SuggestionsInfo EMPTY_SUGGESTIONS_INFO = new SuggestionsInfo(0, null);

    public static SentenceSuggestionsInfo[] getEmptySentenceSuggestionsInfo() {
        return EmptySentenceSuggestionsInfosInitializationHolder.EMPTY_SENTENCE_SUGGESTIONS_INFOS;
    }

    /**
     * Container for split TextInfo parameters
     */
    public static class SentenceWordItem {
        public final TextInfo mTextInfo;
        public final int mStart;
        public final int mLength;
        public SentenceWordItem(TextInfo ti, int start, int end) {
            mTextInfo = ti;
            mStart = start;
            mLength = end - start;
        }
    }

    /**
     * Container for originally queried TextInfo and parameters
     */
    public static class SentenceTextInfoParams {
        final TextInfo mOriginalTextInfo;
        final ArrayList<SentenceWordItem> mItems;
        final int mSize;
        public SentenceTextInfoParams(TextInfo ti, ArrayList<SentenceWordItem> items) {
            mOriginalTextInfo = ti;
            mItems = items;
            mSize = items.size();
        }
    }

    private static class WordIterator {
        private final SpacingAndPunctuations mSpacingAndPunctuations;
        public WordIterator(Resources res, Locale locale) {
            mSpacingAndPunctuations = RunInLocaleKt.runInLocale(res, locale, r -> new SpacingAndPunctuations(r, false, locale));
        }

        public int getEndOfWord(final CharSequence sequence, final int fromIndex) {
            final int length = sequence.length();
            int index = fromIndex < 0 ? 0 : Character.offsetByCodePoints(sequence, fromIndex, 1);
            while (index < length) {
                final int codePoint = Character.codePointAt(sequence, index);
                if (mSpacingAndPunctuations.isWordSeparator(codePoint)) {
                    // If it's a period, we want to stop here only if it's followed by another
                    // word separator. In all other cases we stop here.
                    if (Constants.CODE_PERIOD == codePoint) {
                        final int indexOfNextCodePoint =
                                index + Character.charCount(Constants.CODE_PERIOD);
                        if (indexOfNextCodePoint < length
                                && mSpacingAndPunctuations.isWordSeparator(
                                        Character.codePointAt(sequence, indexOfNextCodePoint))) {
                            return index;
                        }
                    } else {
                        return index;
                    }
                }
                index += Character.charCount(codePoint);
            }
            return index;
        }

        public int getBeginningOfNextWord(final CharSequence sequence, final int fromIndex) {
            final int length = sequence.length();
            if (fromIndex >= length) {
                return -1;
            }
            int index = fromIndex < 0 ? 0 : Character.offsetByCodePoints(sequence, fromIndex, 1);
            while (index < length) {
                final int codePoint = Character.codePointAt(sequence, index);
                if (!mSpacingAndPunctuations.isWordSeparator(codePoint)) {
                    return index;
                }
                index += Character.charCount(codePoint);
            }
            return -1;
        }
    }

    private final WordIterator mWordIterator;
    public SentenceLevelAdapter(final Resources res, final Locale locale) {
        mWordIterator = new WordIterator(res, locale);
    }

    public SentenceTextInfoParams getSplitWords(TextInfo originalTextInfo) {
        final WordIterator wordIterator = mWordIterator;
        final CharSequence originalText = originalTextInfo.getCharSequence();
        final int cookie = originalTextInfo.getCookie();
        final int start = -1;
        final int end = originalText.length();
        final ArrayList<SentenceWordItem> wordItems = new ArrayList<>();
        int wordStart = wordIterator.getBeginningOfNextWord(originalText, start);
        int wordEnd = wordIterator.getEndOfWord(originalText, wordStart);
        while (wordStart <= end && wordEnd != -1 && wordStart != -1) {
            if (wordEnd >= start && wordEnd > wordStart) {
                final TextInfo ti = new TextInfo(originalText, wordStart,
                        wordEnd, cookie, originalText.subSequence(wordStart, wordEnd).hashCode());
                wordItems.add(new SentenceWordItem(ti, wordStart, wordEnd));
            }
            wordStart = wordIterator.getBeginningOfNextWord(originalText, wordEnd);
            if (wordStart == -1) {
                break;
            }
            wordEnd = wordIterator.getEndOfWord(originalText, wordStart);
        }
        return new SentenceTextInfoParams(originalTextInfo, wordItems);
    }

    public static SentenceSuggestionsInfo reconstructSuggestions(
            SentenceTextInfoParams originalTextInfoParams, SuggestionsInfo[] results) {
        if (results == null || results.length == 0) {
            return null;
        }
        if (originalTextInfoParams == null) {
            return null;
        }
        final int originalCookie = originalTextInfoParams.mOriginalTextInfo.getCookie();
        final int originalSequence =
                originalTextInfoParams.mOriginalTextInfo.getSequence();

        final int querySize = originalTextInfoParams.mSize;
        final int[] offsets = new int[querySize];
        final int[] lengths = new int[querySize];
        final SuggestionsInfo[] reconstructedSuggestions = new SuggestionsInfo[querySize];
        for (int i = 0; i < querySize; ++i) {
            final SentenceWordItem item = originalTextInfoParams.mItems.get(i);
            SuggestionsInfo result = null;
            for (final SuggestionsInfo cur : results) {
                if (cur != null && cur.getSequence() == item.mTextInfo.getSequence()) {
                    result = cur;
                    result.setCookieAndSequence(originalCookie, originalSequence);
                    break;
                }
            }
            offsets[i] = item.mStart;
            lengths[i] = item.mLength;
            reconstructedSuggestions[i] = result != null ? result : EMPTY_SUGGESTIONS_INFO;
        }
        return new SentenceSuggestionsInfo(reconstructedSuggestions, offsets, lengths);
    }
}
