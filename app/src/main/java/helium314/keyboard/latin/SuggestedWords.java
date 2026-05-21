/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import android.text.TextUtils;
import android.view.inputmethod.CompletionInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.common.StringUtilsKt;
import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.settings.Settings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

public class SuggestedWords {
    public static final int INDEX_OF_TYPED_WORD = 0;
    public static final int INDEX_OF_AUTO_CORRECTION = 1;
    public static final int NOT_A_SEQUENCE_NUMBER = -1;

    public static final int INPUT_STYLE_NONE = 0;
    public static final int INPUT_STYLE_TYPING = 1;
    public static final int INPUT_STYLE_UPDATE_BATCH = 2;
    public static final int INPUT_STYLE_TAIL_BATCH = 3;
    public static final int INPUT_STYLE_APPLICATION_SPECIFIED = 4;
    public static final int INPUT_STYLE_RECORRECTION = 5;
    public static final int INPUT_STYLE_PREDICTION = 6;
    public static final int INPUT_STYLE_BEGINNING_OF_SENTENCE_PREDICTION = 7;

    // The maximum number of suggestions available.
    public static final int MAX_SUGGESTIONS = 18;

    private static final ArrayList<SuggestedWordInfo> EMPTY_WORD_INFO_LIST = new ArrayList<>(0);
    @NonNull
    private static final SuggestedWords EMPTY = new SuggestedWords(
            EMPTY_WORD_INFO_LIST, null, null, false,
            false, false, INPUT_STYLE_NONE, NOT_A_SEQUENCE_NUMBER);

    @NonNull
    private static final SuggestedWords EMPTY_BATCH = new SuggestedWords(
            EMPTY_WORD_INFO_LIST, null, null, false,
            false, false, INPUT_STYLE_UPDATE_BATCH, NOT_A_SEQUENCE_NUMBER);

    @Nullable
    public final SuggestedWordInfo mTypedWordInfo;
    public final boolean mTypedWordValid;
    // Note: this INCLUDES cases where the word will auto-correct to itself. A good definition
    // of what this flag means would be "the top suggestion is strong enough to auto-correct",
    // whether this exactly matches the user entry or not.
    public final boolean mWillAutoCorrect;
    public final boolean mIsObsoleteSuggestions;
    // How the input for these suggested words was done by the user. Must be one of the
    // INPUT_STYLE_* constants above.
    public final int mInputStyle;
    public final int mSequenceNumber; // Sequence number for auto-commit.
    @NonNull
    protected final ArrayList<SuggestedWordInfo> mSuggestedWordInfoList;
    @Nullable
    public final ArrayList<SuggestedWordInfo> mRawSuggestions;

    public SuggestedWords(@NonNull final ArrayList<SuggestedWordInfo> suggestedWordInfoList,
            @Nullable final ArrayList<SuggestedWordInfo> rawSuggestions,
            @Nullable final SuggestedWordInfo typedWordInfo,
            final boolean typedWordValid,
            final boolean willAutoCorrect,
            final boolean isObsoleteSuggestions,
            final int inputStyle,
            final int sequenceNumber) {
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

    // non-final for testability.
    public static class SuggestedWordInfo {
        public static final int NOT_AN_INDEX = -1;
        public static final int NOT_A_CONFIDENCE = -1;
        public static final int MAX_SCORE = Integer.MAX_VALUE;

        private static final int KIND_MASK_KIND = 0xFF; // Mask to get only the kind
        public static final int KIND_TYPED = 0; // What user typed
        public static final int KIND_CORRECTION = 1; // Simple correction/suggestion
        public static final int KIND_COMPLETION = 2; // Completion (suggestion with appended chars)
        public static final int KIND_WHITELIST = 3; // Whitelisted word
        public static final int KIND_BLACKLIST = 4; // Blacklisted word
        public static final int KIND_HARDCODED = 5; // Hardcoded suggestion, e.g. punctuation
        public static final int KIND_APP_DEFINED = 6; // Suggested by the application
        public static final int KIND_SHORTCUT = 7; // A shortcut
        public static final int KIND_PREDICTION = 8; // A prediction (== a suggestion with no input)
        // KIND_RESUMED: A resumed suggestion (comes from a span, currently this type is used only
        // in java for re-correction)
        public static final int KIND_RESUMED = 9;
        public static final int KIND_OOV_CORRECTION = 10; // Most probable string correction

        public static final int KIND_FLAG_POSSIBLY_OFFENSIVE = 0x80000000;
        public static final int KIND_FLAG_EXACT_MATCH = 0x40000000;
        public static final int KIND_FLAG_EXACT_MATCH_WITH_INTENTIONAL_OMISSION = 0x20000000;
        public static final int KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION = 0x10000000;

        public final String mWord;
        public final String mPrevWordsContext;
        // The completion info from the application. Null for suggestions that don't come from
        // the application (including keyboard-computed ones, so this is almost always null)
        public final CompletionInfo mApplicationSpecifiedCompletionInfo;
        public final int mScore;
        public int mOriginalScore; // relevant only for background data gathering, todo: remove when gathering period is over
        public final int mKindAndFlags;
        public final int mCodePointCount;
        public final Dictionary mSourceDict;
        // For auto-commit. This keeps track of the index inside the touch coordinates array
        // passed to native code to get suggestions for a gesture that corresponds to the first
        // letter of the second word.
        public final int mIndexOfTouchPointOfSecondWord;
        // For auto-commit. This is a measure of how confident we are that we can commit the
        // first word of this suggestion.
        public final int mAutoCommitFirstWordConfidence;
        private String mDebugString = "";
        private Boolean mIsEmoji;

        /**
         * Create a new suggested word info.
         * @param word The string to suggest.
         * @param prevWordsContext previous words context.
         * @param score A measure of how likely this suggestion is.
         * @param kindAndFlags The kind of suggestion, as one of the above KIND_* constants with
         * flags.
         * @param sourceDict What instance of Dictionary produced this suggestion.
         * @param indexOfTouchPointOfSecondWord See mIndexOfTouchPointOfSecondWord.
         * @param autoCommitFirstWordConfidence See mAutoCommitFirstWordConfidence.
         */
        public SuggestedWordInfo(final String word, final String prevWordsContext,
                final int score, final int kindAndFlags,
                final Dictionary sourceDict, final int indexOfTouchPointOfSecondWord,
                final int autoCommitFirstWordConfidence) {
            mWord = word;
            mPrevWordsContext = prevWordsContext;
            mApplicationSpecifiedCompletionInfo = null;
            mScore = score;
            mKindAndFlags = kindAndFlags;
            mSourceDict = sourceDict;
            mCodePointCount = StringUtils.codePointCount(mWord);
            mIndexOfTouchPointOfSecondWord = indexOfTouchPointOfSecondWord;
            mAutoCommitFirstWordConfidence = autoCommitFirstWordConfidence;
        }

        /**
         * Create a new suggested word info from an application-specified completion.
         * If the passed argument or its contained text is null, this throws a NPE.
         * @param applicationSpecifiedCompletion The application-specified completion info.
         */
        public SuggestedWordInfo(final CompletionInfo applicationSpecifiedCompletion) {
            mWord = applicationSpecifiedCompletion.getText().toString();
            mPrevWordsContext = "";
            mApplicationSpecifiedCompletionInfo = applicationSpecifiedCompletion;
            mScore = SuggestedWordInfo.MAX_SCORE;
            mKindAndFlags = SuggestedWordInfo.KIND_APP_DEFINED;
            mSourceDict = Dictionary.DICTIONARY_APPLICATION_DEFINED;
            mCodePointCount = StringUtils.codePointCount(mWord);
            mIndexOfTouchPointOfSecondWord = SuggestedWordInfo.NOT_AN_INDEX;
            mAutoCommitFirstWordConfidence = SuggestedWordInfo.NOT_A_CONFIDENCE;
        }

        public boolean isEligibleForAutoCommit() {
            return (isKindOf(KIND_CORRECTION) && NOT_AN_INDEX != mIndexOfTouchPointOfSecondWord);
        }

        public int getKind() {
            return (mKindAndFlags & KIND_MASK_KIND);
        }

        public boolean isKindOf(final int kind) {
            return getKind() == kind;
        }

        public boolean isPossiblyOffensive() {
            return (mKindAndFlags & KIND_FLAG_POSSIBLY_OFFENSIVE) != 0;
        }

        public boolean isExactMatch() {
            return (mKindAndFlags & KIND_FLAG_EXACT_MATCH) != 0;
        }

        public boolean isExactMatchWithIntentionalOmission() {
            return (mKindAndFlags & KIND_FLAG_EXACT_MATCH_WITH_INTENTIONAL_OMISSION) != 0;
        }

        public boolean isAppropriateForAutoCorrection() {
            return (mKindAndFlags & KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION) != 0
                    || (isKindOf(KIND_SHORTCUT) && Settings.getValues().mAutoCorrectShortcuts);
        }

        public void setDebugString(final String str) {
            if (null == str) throw new NullPointerException("Debug info is null");
            mDebugString = str;
        }

        public String getDebugString() {
            return mDebugString;
        }

        public String getWord() {
            return mWord;
        }

        public boolean isEmoji() {
            if (mIsEmoji == null) {
                mIsEmoji = StringUtilsKt.isEmoji(mWord);
            }
            return mIsEmoji;
        }

        @Deprecated
        public Dictionary getSourceDictionary() {
            return mSourceDict;
        }

        public int codePointAt(int i) {
            return mWord.codePointAt(i);
        }

        @Override
        public String toString() {
            if (TextUtils.isEmpty(mDebugString)) {
                return mWord;
            }
            return mWord + " (" + mDebugString + ")";
        }

        /**
         * This will always remove the higher index if a duplicate is found.
         * Will also remove all occurrences of the typed word.
         *
         * @return position of typed word in the candidate list
         */
        public static int removeDupsAndTypedWord(
                @Nullable final String typedWord,
                @NonNull final ArrayList<SuggestedWordInfo> candidates) {
            if (candidates.isEmpty()) {
                return -1;
            }
            int firstOccurrenceOfWord = -1;
            if (typedWord != null && typedWord.length() > 0) {
                firstOccurrenceOfWord = removeSuggestedWordInfoFromList(
                        typedWord, candidates, -1 /* startIndexExclusive */);
            }
            for (int i = 0; i < candidates.size(); ++i) {
                removeSuggestedWordInfoFromList(
                        candidates.get(i).mWord, candidates, i /* startIndexExclusive */);
            }
            return firstOccurrenceOfWord;
        }

        private static int removeSuggestedWordInfoFromList(
                @NonNull final String word,
                @NonNull final ArrayList<SuggestedWordInfo> candidates,
                final int startIndexExclusive) {
            int firstOccurrenceOfWord = -1;
            for (int i = startIndexExclusive + 1; i < candidates.size(); ++i) {
                final SuggestedWordInfo previous = candidates.get(i);
                if (word.equals(previous.mWord)) {
                    if (firstOccurrenceOfWord == -1) {
                        firstOccurrenceOfWord = i;
                    }
                    candidates.remove(i);
                    --i;
                }
            }
            return firstOccurrenceOfWord;
        }
    }

    private static boolean isPrediction(final int inputStyle) {
        return INPUT_STYLE_PREDICTION == inputStyle
                || INPUT_STYLE_BEGINNING_OF_SENTENCE_PREDICTION == inputStyle;
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
