/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.suggestions;

import android.text.TextUtils;
import android.view.inputmethod.CompletionInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.latin.InputStyle;
import helium314.keyboard.latin.define.DebugFlags;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

public class SuggestedWords {
    public static final int INDEX_OF_TYPED_WORD = 0;
    public static final int INDEX_OF_AUTO_CORRECTION = 1;
    public static final int NOT_A_SEQUENCE_NUMBER = -1;

    // The maximum number of suggestions available.
    public static final int MAX_SUGGESTIONS = 18;

    private static final ArrayList<SuggestedWordInfo> EMPTY_WORD_INFO_LIST = new ArrayList<>(0);
    @NonNull
    private static final SuggestedWords EMPTY = new SuggestedWords(
            EMPTY_WORD_INFO_LIST, null, null, false,
            false, false, InputStyle.NONE, NOT_A_SEQUENCE_NUMBER);

    @NonNull
    private static final SuggestedWords EMPTY_BATCH = new SuggestedWords(
            EMPTY_WORD_INFO_LIST, null, null, false,
            false, false, InputStyle.UPDATE_BATCH, NOT_A_SEQUENCE_NUMBER);

    @Nullable
    public final SuggestedWordInfo mTypedWordInfo;
    public final boolean mTypedWordValid;
    // Note: this INCLUDES cases where the word will auto-correct to itself. A good definition
    // of what this flag means would be "the top suggestion is strong enough to auto-correct",
    // whether this exactly matches the user entry or not.
    public final boolean mWillAutoCorrect;
    public final boolean mIsObsoleteSuggestions;
    // How the input for these suggested words was done by the user
    public final InputStyle mInputStyle;
    public final int mSequenceNumber; // Sequence number for auto-commit.
    @NonNull
    protected final ArrayList<SuggestedWordInfo> mSuggestedWordInfoList;
    @Nullable
    public final ArrayList<SuggestedWordInfo> mRawSuggestions;

    public SuggestedWords(@NonNull ArrayList<SuggestedWordInfo> suggestedWordInfoList,
            @Nullable ArrayList<SuggestedWordInfo> rawSuggestions, @Nullable SuggestedWordInfo typedWordInfo,
            boolean typedWordValid, boolean willAutoCorrect, boolean isObsoleteSuggestions, InputStyle inputStyle, int sequenceNumber) {
        mSuggestedWordInfoList = suggestedWordInfoList;
        mRawSuggestions = rawSuggestions;
        mTypedWordValid = typedWordValid;
        mWillAutoCorrect = willAutoCorrect;
        mIsObsoleteSuggestions = isObsoleteSuggestions;
        mInputStyle = inputStyle;
        mSequenceNumber = sequenceNumber;
        mTypedWordInfo = typedWordInfo;
    }

    public boolean isEmpty() {
        return mSuggestedWordInfoList.isEmpty();
    }

    public int size() {
        return mSuggestedWordInfoList.size();
    }

    /**
     * Get suggested word to show as suggestions to UI.
     *
     * @return the count of suggested word to show as suggestions to UI.
     */
    public int getWordCountToShow() {
        if (isPrediction()) {
            return size();
        }
        return size() - /* typed word */ 1;
    }

    /**
     * Get {@link SuggestedWordInfo} object for the typed word.
     * @return The {@link SuggestedWordInfo} object for the typed word.
     */
    public SuggestedWordInfo getTypedWordInfo() {
        return mTypedWordInfo;
    }

    /**
     * Get suggested word at <code>index</code>.
     * @param index The index of the suggested word.
     * @return The suggested word.
     */
    public String getWord(final int index) {
        return mSuggestedWordInfoList.get(index).mWord;
    }

    /**
     * Get displayed text at <code>index</code>.
     * In RTL languages, the displayed text on the suggestion strip may be different from the
     * suggested word that is returned from {@link #getWord(int)}. For example the displayed text
     * of punctuation suggestion "(" should be ")".
     * @param index The index of the text to display.
     * @return The text to be displayed.
     */
    public String getLabel(final int index) {
        return mSuggestedWordInfoList.get(index).mWord;
    }

    /**
     * Get {@link SuggestedWordInfo} object at <code>index</code>.
     * @param index The index of the {@link SuggestedWordInfo}.
     * @return The {@link SuggestedWordInfo} object.
     */
    public SuggestedWordInfo getInfo(final int index) {
        return mSuggestedWordInfoList.get(index);
    }

    /**
     * Gets the suggestion index from the suggestions list.
     * @param suggestedWordInfo The {@link SuggestedWordInfo} to find the index.
     * @return The position of the suggestion in the suggestion list.
     */
    public int indexOf(SuggestedWordInfo suggestedWordInfo) {
        return mSuggestedWordInfoList.indexOf(suggestedWordInfo);
    }

    public String getDebugString(final int pos) {
        if (!DebugFlags.DEBUG_ENABLED) {
            return null;
        }
        final SuggestedWordInfo wordInfo = getInfo(pos);
        if (wordInfo == null) {
            return null;
        }
        final String debugString = wordInfo.getDebugString();
        if (TextUtils.isEmpty(debugString)) {
            return null;
        }
        return debugString;
    }

    /**
     * The predicator to tell whether this object represents punctuation suggestions.
     * @return false if this object desn't represent punctuation suggestions.
     */
    public boolean isPunctuationSuggestions() {
        return false;
    }

    @Override
    public String toString() {
        // Pretty-print method to help debug
        return "SuggestedWords:"
                + " mTypedWordValid=" + mTypedWordValid
                + " mWillAutoCorrect=" + mWillAutoCorrect
                + " mInputStyle=" + mInputStyle
                + " words=" + Arrays.toString(mSuggestedWordInfoList.toArray());
    }

    public static ArrayList<SuggestedWordInfo> getFromApplicationSpecifiedCompletions(
            final CompletionInfo[] infos) {
        final ArrayList<SuggestedWordInfo> result = new ArrayList<>();
        for (final CompletionInfo info : infos) {
            if (null == info || null == info.getText()) {
                continue;
            }
            result.add(new SuggestedWordInfo(info));
        }
        return result;
    }

    @NonNull
    public static SuggestedWords getEmptyInstance() {
        return SuggestedWords.EMPTY;
    }

    @NonNull
    public static SuggestedWords getEmptyBatchInstance() {
        return SuggestedWords.EMPTY_BATCH;
    }

    // Should get rid of the first one (what the user typed previously) from suggestions
    // and replace it with what the user currently typed.
    public static ArrayList<SuggestedWordInfo> getTypedWordAndPreviousSuggestions(
            @NonNull final SuggestedWordInfo typedWordInfo,
            @NonNull final SuggestedWords previousSuggestions) {
        final ArrayList<SuggestedWordInfo> suggestionsList = new ArrayList<>();
        final HashSet<String> alreadySeen = new HashSet<>();
        suggestionsList.add(typedWordInfo);
        alreadySeen.add(typedWordInfo.mWord);
        final int previousSize = previousSuggestions.size();
        for (int index = 1; index < previousSize; index++) {
            final SuggestedWordInfo prevWordInfo = previousSuggestions.getInfo(index);
            final String prevWord = prevWordInfo.mWord;
            // Filter out duplicate suggestions.
            if (!alreadySeen.contains(prevWord)) {
                suggestionsList.add(prevWordInfo);
                alreadySeen.add(prevWord);
            }
        }
        return suggestionsList;
    }

    public SuggestedWordInfo getAutoCommitCandidate() {
        if (mSuggestedWordInfoList.size() <= 0) return null;
        final SuggestedWordInfo candidate = mSuggestedWordInfoList.get(0);
        return candidate.isEligibleForAutoCommit() ? candidate : null;
    }

    private static boolean isPrediction(InputStyle inputStyle) {
        return InputStyle.PREDICTION == inputStyle || InputStyle.BEGINNING_OF_SENTENCE_PREDICTION == inputStyle;
    }

    public boolean isPrediction() {
        return isPrediction(mInputStyle);
    }

    /**
     * @return the {@link SuggestedWordInfo} which corresponds to the word that is originally
     * typed by the user. Otherwise returns {@code null}. Note that gesture input is not
     * considered to be a typed word.
     */
    public SuggestedWordInfo getTypedWordInfoOrNull() {
        if (SuggestedWords.INDEX_OF_TYPED_WORD >= size()) {
            return null;
        }
        final SuggestedWordInfo info = getInfo(SuggestedWords.INDEX_OF_TYPED_WORD);
        return (info.getKind() == SuggestedWordInfo.KIND_TYPED) ? info : null;
    }
}
