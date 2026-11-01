package helium314.keyboard.latin.suggestions;

import android.text.TextUtils;
import android.view.inputmethod.CompletionInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;

import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.common.StringUtilsKt;
import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.settings.Settings;

public class SuggestedWordInfo {
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
