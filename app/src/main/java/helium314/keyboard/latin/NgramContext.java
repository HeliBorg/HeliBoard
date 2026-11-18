/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import android.text.TextUtils;

import androidx.annotation.NonNull;

import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.define.DecoderSpecificConstants;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class to represent information of previous words. This class is used to add n-gram entries
 * into binary dictionaries, to get predictions, and to get suggestions.
 */
public class NgramContext {
    @NonNull
    public static final NgramContext EMPTY_PREV_WORDS_INFO =
            new NgramContext(WordInfo.EMPTY_WORD_INFO);
    @NonNull
    public static final NgramContext BEGINNING_OF_SENTENCE =
            new NgramContext(WordInfo.BEGINNING_OF_SENTENCE_WORD_INFO);

    public static final String BEGINNING_OF_SENTENCE_TAG = "<S>";

    public static final String CONTEXT_SEPARATOR = " ";

    public static NgramContext getEmptyPrevWordsContext(int maxPrevWordCount) {
        return new NgramContext(maxPrevWordCount, WordInfo.EMPTY_WORD_INFO);
    }

    /**
     * Word information used to represent previous words information.
     */
    public static class WordInfo {
        @NonNull
        public static final WordInfo EMPTY_WORD_INFO = new WordInfo(null);
        @NonNull
        public static final WordInfo BEGINNING_OF_SENTENCE_WORD_INFO = new WordInfo();

        // This is an empty char sequence when mIsBeginningOfSentence is true.
        public final CharSequence mWord;
        // TODO: Have sentence separator.
        // Whether the current context is beginning of sentence or not. This is true when composing
        // at the beginning of an input field or composing a word after a sentence separator.
        public final boolean mIsBeginningOfSentence;

        // Beginning of sentence.
        private WordInfo() {
            mWord = "";
            mIsBeginningOfSentence = true;
        }

        public WordInfo(final CharSequence word) {
            mWord = word;
            mIsBeginningOfSentence = false;
        }

        public boolean isValid() {
            return mWord != null;
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(new Object[] { mWord, mIsBeginningOfSentence } );
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof WordInfo wordInfo)) return false;
            if (mWord == null || wordInfo.mWord == null) {
                return mWord == wordInfo.mWord
                        && mIsBeginningOfSentence == wordInfo.mIsBeginningOfSentence;
            }
            return TextUtils.equals(mWord, wordInfo.mWord)
                    && mIsBeginningOfSentence == wordInfo.mIsBeginningOfSentence;
        }
    }

    // The words immediately before the considered word. EMPTY_WORD_INFO element means we don't
    // have any context for that previous word including the "beginning of sentence context" - we
    // just don't know what to predict using the information. An example of that is after a comma.
    // For simplicity of implementation, elements may also be EMPTY_WORD_INFO transiently after the
    // WordComposer was reset and before starting a new composing word, but we should never be
    // calling getSuggetions* in this situation.
    private final WordInfo[] mPrevWordsInfo;
    private final int mPrevWordsCount;

    private final int mMaxPrevWordCount;

    // Construct from the previous word information.
    public NgramContext(final WordInfo... prevWordsInfo) {
        this(DecoderSpecificConstants.MAX_PREV_WORD_COUNT_FOR_N_GRAM, prevWordsInfo);
    }

    public NgramContext(final int maxPrevWordCount, final WordInfo... prevWordsInfo) {
        mPrevWordsInfo = prevWordsInfo;
        mPrevWordsCount = prevWordsInfo.length;
        mMaxPrevWordCount = maxPrevWordCount;
    }

    /**
     * Create next prevWordsInfo using current prevWordsInfo.
     */
    @NonNull
    public NgramContext getNextNgramContext(final WordInfo wordInfo) {
        final int nextPrevWordCount = Math.min(mMaxPrevWordCount, mPrevWordsCount + 1);
        final WordInfo[] prevWordsInfo = new WordInfo[nextPrevWordCount];
        prevWordsInfo[0] = wordInfo;
        System.arraycopy(mPrevWordsInfo, 0, prevWordsInfo, 1, nextPrevWordCount - 1);
        return new NgramContext(mMaxPrevWordCount, prevWordsInfo);
    }

    /**
     * Extracts the previous words context.
     *
     * @return a String with the previous words separated by white space.
     */
    public String extractPrevWordsContext() {
        final ArrayList<String> terms = new ArrayList<>();
        for (int i = mPrevWordsInfo.length - 1; i >= 0; --i) {
            if (mPrevWordsInfo[i] != null && mPrevWordsInfo[i].isValid()) {
                final NgramContext.WordInfo wordInfo = mPrevWordsInfo[i];
                if (wordInfo.mIsBeginningOfSentence) {
                    terms.add(BEGINNING_OF_SENTENCE_TAG);
                } else {
                    final String term = wordInfo.mWord.toString();
                    if (!term.isEmpty()) {
                        terms.add(term);
                    }
                }
            }
        }
        return TextUtils.join(CONTEXT_SEPARATOR, terms);
    }

    /**
     * Extracts the previous words context.
     *
     * @return a String array with the previous words.
     */
    public String[] extractPrevWordsContextArray() {
        final ArrayList<String> prevTermList = new ArrayList<>();
        for (int i = mPrevWordsInfo.length - 1; i >= 0; --i) {
            if (mPrevWordsInfo[i] != null && mPrevWordsInfo[i].isValid()) {
                final NgramContext.WordInfo wordInfo = mPrevWordsInfo[i];
                if (wordInfo.mIsBeginningOfSentence) {
                    prevTermList.add(BEGINNING_OF_SENTENCE_TAG);
                } else {
                    final String term = wordInfo.mWord.toString();
                    if (!term.isEmpty()) {
                        prevTermList.add(term);
                    }
                }
            }
        }
        return prevTermList.toArray(new String[0]);
    }

    public boolean isValid() {
        return mPrevWordsCount > 0 && mPrevWordsInfo[0].isValid();
    }

    public boolean isBeginningOfSentenceContext() {
        return mPrevWordsCount > 0 && mPrevWordsInfo[0].mIsBeginningOfSentence;
    }

    // n is 1-indexed.
    // TODO: Remove
    public CharSequence getNthPrevWord(final int n) {
        if (n <= 0 || n > mPrevWordsCount) {
            return null;
        }
        return mPrevWordsInfo[n - 1].mWord;
    }

    // n is 1-indexed.
    public boolean isNthPrevWordBeginningOfSentence(final int n) {
        if (n <= 0 || n > mPrevWordsCount) {
            return false;
        }
        return mPrevWordsInfo[n - 1].mIsBeginningOfSentence;
    }

    public void outputToArray(final int[][] codePointArrays,
            final boolean[] isBeginningOfSentenceArray) {
        for (int i = 0; i < mPrevWordsCount; i++) {
            final WordInfo wordInfo = mPrevWordsInfo[i];
            if (wordInfo == null || !wordInfo.isValid()) {
                codePointArrays[i] = new int[0];
                isBeginningOfSentenceArray[i] = false;
                continue;
            }
            codePointArrays[i] = StringUtils.toCodePointArray(wordInfo.mWord);
            isBeginningOfSentenceArray[i] = wordInfo.mIsBeginningOfSentence;
        }
    }

    public int getPrevWordCount() {
        return mPrevWordsCount;
    }

    @Override
    public int hashCode() {
        int hashValue = 0;
        for (final WordInfo wordInfo : mPrevWordsInfo) {
            if (!WordInfo.EMPTY_WORD_INFO.equals(wordInfo)) {
                break;
            }
            hashValue ^= wordInfo.hashCode();
        }
        return hashValue;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NgramContext prevWordsInfo)) return false;

        final int minLength = Math.min(mPrevWordsCount, prevWordsInfo.mPrevWordsCount);
        for (int i = 0; i < minLength; i++) {
            if (!mPrevWordsInfo[i].equals(prevWordsInfo.mPrevWordsInfo[i])) {
                return false;
            }
        }
        final WordInfo[] longerWordsInfo;
        final int longerWordsInfoCount;
        if (mPrevWordsCount > prevWordsInfo.mPrevWordsCount) {
            longerWordsInfo = mPrevWordsInfo;
            longerWordsInfoCount = mPrevWordsCount;
        } else {
            longerWordsInfo = prevWordsInfo.mPrevWordsInfo;
            longerWordsInfoCount = prevWordsInfo.mPrevWordsCount;
        }
        for (int i = minLength; i < longerWordsInfoCount; i++) {
            if (longerWordsInfo[i] != null
                    && !WordInfo.EMPTY_WORD_INFO.equals(longerWordsInfo[i])) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();
        for (int i = 0; i < mPrevWordsCount; i++) {
            final WordInfo wordInfo = mPrevWordsInfo[i];
            builder.append("PrevWord[");
            builder.append(i);
            builder.append("]: ");
            if (wordInfo == null) {
                builder.append("null. ");
                continue;
            }
            if (!wordInfo.isValid()) {
                builder.append("Empty. ");
                continue;
            }
            builder.append(wordInfo.mWord);
            builder.append(", isBeginningOfSentence: ");
            builder.append(wordInfo.mIsBeginningOfSentence);
            builder.append(". ");
        }
        return builder.toString();
    }
}
