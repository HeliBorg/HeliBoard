package helium314.keyboard.latin.suggestions

import android.view.inputmethod.CompletionInfo
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.common.isEmoji
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.settings.Settings

data class SuggestedWordInfo(
    val word: String,
    val prevWordsContext: String?,
    val score: Int,
    val kindAndFlags: Int,
    val sourceDict: Dictionary,
    // For auto-commit. This keeps track of the index inside the touch coordinates array
    // passed to native code to get suggestions for a gesture that corresponds to the first
    // letter of the second word.
    val indexOfTouchPointOfSecondWord: Int,
    // For auto-commit. This is a measure of how confident we are that we can commit the
    // first word of this suggestion.
    val autoCommitFirstWordConfidence: Int,
    // The completion info from the application. Null for suggestions that don't come from
    // the application (including keyboard-computed ones, so this is almost always null)
    val applicationSpecifiedCompletionInfo: CompletionInfo? = null
) {
    /**
     * Create a new suggested word info from an application-specified completion.
     * If the passed argument or its contained text is null, this throws a NPE.
     * @param applicationSpecifiedCompletion The application-specified completion info.
     */
    constructor(applicationSpecifiedCompletion: CompletionInfo) : this(
        applicationSpecifiedCompletion.text.toString(),
        "",
        MAX_SCORE,
        KIND_APP_DEFINED,
        Dictionary.DICTIONARY_APPLICATION_DEFINED,
        NOT_AN_INDEX,
        NOT_A_CONFIDENCE,
        applicationSpecifiedCompletion
    )

    /**
     * Word that should be replaced when this suggestíon is selected.
     * If the [wordToReplace] is null or not found at the cursor, the [word] will be committed normally, i.e. replace the current composing span.
     */
    var wordToReplace: String? = null

    var originalScore = 0 // relevant only for background data gathering, todo: remove when gathering period is over
    val codePointCount by lazy { StringUtils.codePointCount(word) }

    var debugString = ""

    val isEligibleForAutoCommit: Boolean
        get() = (isKindOf(KIND_CORRECTION) && NOT_AN_INDEX != indexOfTouchPointOfSecondWord)

    val kind: Int // kindAndFlags is returned by the native library, so we keep the original int-kind instead of an enum
        get() = (kindAndFlags and KIND_MASK_KIND)

    fun isKindOf(kind: Int): Boolean {
        return this.kind == kind
    }

    val isPossiblyOffensive get() = (kindAndFlags and KIND_FLAG_POSSIBLY_OFFENSIVE) != 0

    val isExactMatch get() = (kindAndFlags and KIND_FLAG_EXACT_MATCH) != 0

    val isExactMatchWithIntentionalOmission get() = (kindAndFlags and KIND_FLAG_EXACT_MATCH_WITH_INTENTIONAL_OMISSION) != 0

    val isAppropriateForAutoCorrection get() = (kindAndFlags and KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION) != 0
            || (isKindOf(KIND_SHORTCUT) && Settings.getValues().mAutoCorrectShortcuts)

    val isEmoji by lazy { isEmoji(word) }

    fun codePointAt(i: Int) = word.codePointAt(i)

    override fun toString() = if (debugString.isEmpty()) word else "$word ($debugString)"

    companion object {
        const val NOT_AN_INDEX = -1
        const val NOT_A_CONFIDENCE = -1
        const val MAX_SCORE = Int.MAX_VALUE

        private const val KIND_MASK_KIND = 0xFF // Mask to get only the kind
        const val KIND_TYPED = 0 // What user typed
        const val KIND_CORRECTION = 1 // Simple correction/suggestion
        const val KIND_COMPLETION = 2 // Completion (suggestion with appended chars)
        const val KIND_WHITELIST = 3 // Whitelisted word
        const val KIND_BLACKLIST = 4 // Blacklisted word
        const val KIND_HARDCODED = 5 // Hardcoded suggestion, e.g. punctuation
        const val KIND_APP_DEFINED = 6 // Suggested by the application
        const val KIND_SHORTCUT = 7 // A shortcut
        const val KIND_PREDICTION = 8 // A prediction (== a suggestion with no input)

        // KIND_RESUMED: A resumed suggestion (comes from a span, currently this type is used only
        // in java for re-correction)
        const val KIND_RESUMED = 9
        const val KIND_OOV_CORRECTION = 10 // Most probable string correction

        const val KIND_FLAG_POSSIBLY_OFFENSIVE = -0x80000000
        const val KIND_FLAG_EXACT_MATCH = 0x40000000
        const val KIND_FLAG_EXACT_MATCH_WITH_INTENTIONAL_OMISSION = 0x20000000
        const val KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION = 0x10000000

        /**
         * This will always remove the higher index if a duplicate is found.
         * Will also remove all occurrences of the typed word.
         *
         * @return position of typed word in the candidate list
         */
        fun removeDupsAndTypedWord(typedWord: String?, candidates: MutableList<SuggestedWordInfo>): Int {
            if (candidates.isEmpty()) {
                return -1
            }
            var firstOccurrenceOfWord = -1
            if (!typedWord.isNullOrEmpty()) {
                firstOccurrenceOfWord = removeSuggestedWordInfoFromList(typedWord, candidates, -1)
            }
            var i = 0
            while (i < candidates.size) {
                removeSuggestedWordInfoFromList(candidates[i].word, candidates, i)
                ++i
            }
            return firstOccurrenceOfWord
        }

        private fun removeSuggestedWordInfoFromList(word: String, candidates: MutableList<SuggestedWordInfo>, startIndexExclusive: Int): Int {
            var firstOccurrenceOfWord = -1
            var i = startIndexExclusive + 1
            while (i < candidates.size) {
                val previous = candidates[i]
                if (word == previous.word) {
                    if (firstOccurrenceOfWord == -1) {
                        firstOccurrenceOfWord = i
                    }
                    candidates.removeAt(i)
                    --i
                }
                ++i
            }
            return firstOccurrenceOfWord
        }
    }
}
