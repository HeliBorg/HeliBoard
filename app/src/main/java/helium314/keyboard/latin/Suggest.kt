/*
 * Copyright (C) 2008 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin

import android.text.TextUtils
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.define.DecoderSpecificConstants.SHOULD_AUTO_CORRECT_USING_NON_WHITE_LISTED_SUGGESTION
import helium314.keyboard.latin.define.DecoderSpecificConstants.SHOULD_REMOVE_PREVIOUSLY_REJECTED_SUGGESTION
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.suggestions.SuggestionStripView
import helium314.keyboard.latin.utils.AutoCorrectionUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.SuggestionResults
import java.util.Locale
import kotlin.math.min

/**
 * This class loads a dictionary and provides a list of suggestions for a given sequence of
 * characters. This includes corrections and completions.
 */
class Suggest(private val mDictionaryFacilitator: DictionaryFacilitator) {
    private var mAutoCorrectionThreshold = 0f
    private val mPlausibilityThreshold = 0f
    private val nextWordSuggestionsCache = HashMap<NgramContext, SuggestionResults>()

    // cache cleared whenever LatinIME.loadSettings is called, notably on changing layout and switching input fields
    fun clearNextWordSuggestionsCache() = nextWordSuggestionsCache.clear()

    /**
     * Set the normalized-score threshold for a suggestion to be considered strong enough that we
     * will auto-correct to this.
     * @param threshold the threshold
     */
    fun setAutoCorrectionThreshold(threshold: Float) {
        mAutoCorrectionThreshold = threshold
    }

    // todo: remove when InputLogic is ready
    interface OnGetSuggestedWordsCallback {
        fun onGetSuggestedWords(suggestedWords: SuggestedWords?)
    }

    fun getSuggestedWords(wordComposer: WordComposer, ngramContext: NgramContext, keyboard: Keyboard,
                          settingsValuesForSuggestion: SettingsValuesForSuggestion, isCorrectionEnabled: Boolean,
                          inputStyle: Int, sequenceNumber: Int): SuggestedWords =
        if (wordComposer.isBatchMode) {
            getSuggestedWordsForBatchInput(wordComposer, ngramContext, keyboard, settingsValuesForSuggestion,
                inputStyle, sequenceNumber)
        } else {
            getSuggestedWordsForNonBatchInput(wordComposer, ngramContext, keyboard, settingsValuesForSuggestion,
                inputStyle, isCorrectionEnabled, sequenceNumber)
        }

    // Retrieves suggestions for non-batch input (typing, recorrection, predictions...)
    // and calls the callback function with the suggestions.
    private fun getSuggestedWordsForNonBatchInput(wordComposer: WordComposer, ngramContext: NgramContext, keyboard: Keyboard,
                      settingsValuesForSuggestion: SettingsValuesForSuggestion, inputStyleIfNotPrediction: Int,
                      isCorrectionEnabled: Boolean, sequenceNumber: Int): SuggestedWords {
        val typedWordString = wordComposer.typedWord
        val resultsArePredictions = !wordComposer.isComposingWord
        val suggestionResults = if (typedWordString.isEmpty())
                getNextWordSuggestions(ngramContext, keyboard, inputStyleIfNotPrediction, settingsValuesForSuggestion)
            else mDictionaryFacilitator.getSuggestionResults(wordComposer.composedDataSnapshot, ngramContext, keyboard,
                settingsValuesForSuggestion, SESSION_ID_TYPING, inputStyleIfNotPrediction)
        val trailingSingleQuotesCount = StringUtils.getTrailingSingleQuotesCount(typedWordString)
        val suggestionsContainer = getTransformedSuggestedWordInfoList(wordComposer, suggestionResults,
            trailingSingleQuotesCount, mDictionaryFacilitator.mainLocale, keyboard)
        val keyboardShiftMode = keyboard.mId.keyboardCapsMode
        val capitalizedTypedWord = capitalize(typedWordString, keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED,
            keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFTED, mDictionaryFacilitator.mainLocale)

        // store the original SuggestedWordInfo for typed word, as it will be removed
        // we may want to re-add it in case auto-correction happens, so that the original word can at least be selected
        val typedWordFirstOccurrenceWordInfo = suggestionsContainer.firstOrNull { it.mWord == capitalizedTypedWord }
        val firstOccurrenceOfTypedWordInSuggestions = SuggestedWordInfo.removeDupsAndTypedWord(capitalizedTypedWord, suggestionsContainer)
        makeFirstTwoSuggestionsNonEmoji(suggestionsContainer)

        val (allowsToBeAutoCorrected, hasAutoCorrection) = shouldBeAutoCorrected(
            trailingSingleQuotesCount,
            capitalizedTypedWord,
            suggestionsContainer.firstOrNull(),
            {
                val first = suggestionsContainer.firstOrNull() ?: suggestionResults.first()
                val suggestions = getNextWordSuggestions(ngramContext, keyboard, inputStyleIfNotPrediction, settingsValuesForSuggestion)
                val suggestionForFirstInContainer = suggestions.firstOrNull { it.mWord == first.word }
                val suggestionForTypedWord = suggestions.firstOrNull { it.mWord == capitalizedTypedWord }
                suggestionForFirstInContainer to suggestionForTypedWord
            },
            isCorrectionEnabled,
            wordComposer,
            suggestionResults,
            firstOccurrenceOfTypedWordInSuggestions,
            typedWordFirstOccurrenceWordInfo
        )
         val typedWordInfo = SuggestedWordInfo(capitalizedTypedWord, "", SuggestedWordInfo.MAX_SCORE,
            SuggestedWordInfo.KIND_TYPED, typedWordFirstOccurrenceWordInfo?.mSourceDict ?: Dictionary.DICTIONARY_USER_TYPED,
            SuggestedWordInfo.NOT_AN_INDEX , SuggestedWordInfo.NOT_A_CONFIDENCE)
        if (!TextUtils.isEmpty(capitalizedTypedWord)) {
            suggestionsContainer.add(0, typedWordInfo)
        }
        val suggestionsList = if (SuggestionStripView.DEBUG_SUGGESTIONS && suggestionsContainer.isNotEmpty())
                getSuggestionsInfoListWithDebugInfo(capitalizedTypedWord, suggestionsContainer)
            else suggestionsContainer

        val inputStyle = if (resultsArePredictions) {
            if (suggestionResults.mIsBeginningOfSentence) SuggestedWords.INPUT_STYLE_BEGINNING_OF_SENTENCE_PREDICTION
            else SuggestedWords.INPUT_STYLE_PREDICTION
        } else {
            inputStyleIfNotPrediction
        }

        // If there is an incoming autocorrection, make sure typed word is shown, so user is able to override it.
        // Otherwise, if the relevant setting is enabled, show the typed word in the middle.
        val indexOfTypedWord = if (hasAutoCorrection) 2 else 1
        if ((hasAutoCorrection || (Settings.getValues().mCenterSuggestionTextToEnter && !wordComposer.isResumed)
                || capitalizedTypedWord != wordComposer.typedWord)
            && suggestionsList.size >= indexOfTypedWord && !TextUtils.isEmpty(capitalizedTypedWord)) {
            if (typedWordFirstOccurrenceWordInfo != null) {
                addDebugInfo(typedWordFirstOccurrenceWordInfo, capitalizedTypedWord)
                suggestionsList.add(indexOfTypedWord, typedWordFirstOccurrenceWordInfo)
            } else {
                suggestionsList.add(indexOfTypedWord,
                    SuggestedWordInfo(capitalizedTypedWord, "", 0, SuggestedWordInfo.KIND_TYPED,
                        Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE)
                )
            }
        }
        val isTypedWordValid = firstOccurrenceOfTypedWordInSuggestions > -1 || (!resultsArePredictions && !allowsToBeAutoCorrected)
        return SuggestedWords(suggestionsList, suggestionResults.mRawSuggestions,
            typedWordInfo, isTypedWordValid, hasAutoCorrection, false, inputStyle, sequenceNumber)
    }

    // returns [allowsToBeAutoCorrected, hasAutoCorrection]
    // public for testing
    fun shouldBeAutoCorrected(
        trailingSingleQuotesCount: Int,
        typedWordString: String,
        firstSuggestionInContainer: SuggestedWordInfo?,
        getEmptyWordSuggestions: () -> Pair<SuggestedWordInfo?, SuggestedWordInfo?>,
        isCorrectionEnabled: Boolean,
        wordComposer: WordComposer,
        suggestionResults: SuggestionResults,
        firstOccurrenceOfTypedWordInSuggestions: Int,
        typedWordInfo: SuggestedWordInfo?
    ): Pair<Boolean, Boolean> {
        val consideredWord = typedWordString.dropLast(trailingSingleQuotesCount)
        val firstAndTypedEmptyInfos by lazy { getEmptyWordSuggestions() }

        val scoreLimit = Settings.getValues().mScoreLimitForAutocorrect
        // We allow auto-correction if whitelisting is not required or the word is whitelisted,
        // or if the word had more than one char and was not suggested.
        val allowsToBeAutoCorrected: Boolean
        if (SHOULD_AUTO_CORRECT_USING_NON_WHITE_LISTED_SUGGESTION
                || firstSuggestionInContainer?.isKindOf(SuggestedWordInfo.KIND_WHITELIST) == true
                || (consideredWord.length > 1
                    && typedWordInfo?.mSourceDict == null // more than 1 letter and not in dictionary
                    // if the typed word contains @ or ., the suggestion also needs to contain it
                    // (avoid autocorrecting mail addresses, URLs & similar to something different)
                    && (!typedWordString.contains('@') || firstSuggestionInContainer?.mWord?.contains('@') == true)
                    && (!typedWordString.contains('.') || firstSuggestionInContainer?.mWord?.contains('.') == true)
                    )
            ) {
            allowsToBeAutoCorrected = true
        } else if (firstSuggestionInContainer != null && typedWordString.isNotEmpty()) {
            // maybe allow autocorrect, depending on scores and emptyWordSuggestions
            val first = firstAndTypedEmptyInfos.first
            val typed = firstAndTypedEmptyInfos.second
            allowsToBeAutoCorrected = when {
                firstSuggestionInContainer.mScore > scoreLimit -> true // suggestion has good score, allow
                first == null -> false // no autocorrect if first suggestion unknown in this ngram context
                typed == null -> true // allow autocorrect if typed word not known in this ngram context, todo: this may be too aggressive
                else -> first.mScore - typed.mScore > 20 // autocorrect if suggested word has clearly higher score for empty word suggestions
            }
        } else {
            allowsToBeAutoCorrected = false
        }
        // If correction is not enabled, we never auto-correct. This is for example for when
        // the setting "Auto-correction" is "off": we still suggest, but we don't auto-correct.
        val hasAutoCorrection: Boolean
        if (!isCorrectionEnabled
            // todo: can some parts be moved to isCorrectionEnabled? e.g. keyboardIdMode only depends on input type
            //  i guess then not mAutoCorrectionEnabledPerUserSettings should be read, but rather some isAutocorrectEnabled()
            // If the word does not allow to be auto-corrected, then we don't auto-correct.
            || !allowsToBeAutoCorrected // If we are doing prediction, then we never auto-correct of course
            || !wordComposer.isComposingWord // If we don't have suggestion results, we can't evaluate the first suggestion
            // for auto-correction
            || suggestionResults.isEmpty() // If the word has digits, we never auto-correct because it's likely the word
            // was type with a lot of care
            || wordComposer.hasDigits() // If the word is mostly caps, we never auto-correct because this is almost
            // certainly intentional (and careful input)
            || wordComposer.isMostlyCaps // We never auto-correct when suggestions are resumed because it would be unexpected
            || wordComposer.isResumed // If we don't have a main dictionary, we never want to auto-correct. The reason
            // for this is, the user may have a contact whose name happens to match a valid
            // word in their language, and it will unexpectedly auto-correct. For example, if
            // the user types in English with no dictionary and has a "Will" in their contact
            // list, "will" would always auto-correct to "Will" which is unwanted. Hence, no
            // main dict => no auto-correct. Also, it would probably get obnoxious quickly.
            // TODO: now that we have personalization, we may want to re-evaluate this decision
            || !mDictionaryFacilitator.hasAtLeastOneInitializedMainDictionary()
        ) {
            hasAutoCorrection = false
        } else {
            val firstSuggestion = firstSuggestionInContainer ?: suggestionResults.first()
            if (suggestionResults.mFirstSuggestionExceedsConfidenceThreshold && firstOccurrenceOfTypedWordInSuggestions != 0) {
                // mFirstSuggestionExceedsConfidenceThreshold is always set to false, so currently this branch is useless
                return true to true
            }
            if (!AutoCorrectionUtils.suggestionExceedsThreshold(firstSuggestion, consideredWord, mAutoCorrectionThreshold)) {
                // Score is too low for autocorrect
                // todo: maybe also do something here depending on ngram context?
                return true to false
            }
            // We have a high score, so we need to check if this suggestion is in the correct
            // form to allow auto-correcting to it in this language. For details of how this
            // is determined, see #isAllowedByAutoCorrectionWithSpaceFilter.
            val allowed = isAllowedByAutoCorrectionWithSpaceFilter(firstSuggestion)
            if (allowed && typedWordInfo != null && typedWordInfo.mScore > scoreLimit) {
                // typed word is valid and has good score
                // do not auto-correct if typed word is better match than first suggestion
                val dictLocale = mDictionaryFacilitator.currentLocale
                if (firstSuggestion.mScore < scoreLimit) {
                    // don't allow if suggestion has too low score
                    return true to false
                }
                if (firstSuggestion.mSourceDict.mLocale !== typedWordInfo.mSourceDict.mLocale) {
                    // dict locale different -> return the better match
                    return true to (dictLocale == firstSuggestion.mSourceDict.mLocale)
                }
                // the score difference may need tuning, but so far it seems alright
                val firstWordBonusScore =
                    ((if (firstSuggestion.isKindOf(SuggestedWordInfo.KIND_WHITELIST)) 20 else 0) // large bonus because it's wanted by dictionary
                            + (if (StringUtils.isLowerCaseAscii(typedWordString)) 5 else 0) // small bonus because typically only lower case ascii is typed (applies to latin keyboards only)
                            + if (firstSuggestion.mScore > typedWordInfo.mScore) 5 else 0) // small bonus if score is higher
                val firstScoreForEmpty = firstAndTypedEmptyInfos.first?.mScore ?: 0
                val typedScoreForEmpty = firstAndTypedEmptyInfos.second?.mScore ?: 0
                if (firstScoreForEmpty + firstWordBonusScore >= typedScoreForEmpty + 20) {
                    // first word is clearly better match for this ngram context
                    return true to true
                }
                hasAutoCorrection = false
            } else {
                hasAutoCorrection = allowed
            }
        }
        return allowsToBeAutoCorrected to hasAutoCorrection
    }

    // Retrieves suggestions for the batch input
    // and calls the callback function with the suggestions.
    private fun getSuggestedWordsForBatchInput(
        wordComposer: WordComposer,
        ngramContext: NgramContext, keyboard: Keyboard,
        settingsValuesForSuggestion: SettingsValuesForSuggestion,
        inputStyle: Int, sequenceNumber: Int
    ): SuggestedWords {
        val suggestionResults = mDictionaryFacilitator.getSuggestionResults(
            wordComposer.composedDataSnapshot, ngramContext, keyboard,
            settingsValuesForSuggestion, SESSION_ID_GESTURE, inputStyle
        )
        replaceSingleLetterFirstSuggestion(suggestionResults)

        // For transforming words that don't come from a dictionary, because it's our best bet
        val locale = mDictionaryFacilitator.mainLocale
        val suggestionsContainer = ArrayList(suggestionResults)
        val suggestionsCount = suggestionsContainer.size
        val keyboardShiftMode = keyboard.mId.keyboardCapsMode
        val shouldMakeSuggestionsOnlyFirstCharCapitalized = wordComposer.wasShiftedNoLock()
            || keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFTED
        val shouldMakeSuggestionsAllUpperCase = wordComposer.isAllUpperCase
            || keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED
        if (shouldMakeSuggestionsOnlyFirstCharCapitalized || shouldMakeSuggestionsAllUpperCase) {
            for (i in 0 until suggestionsCount) {
                val wordInfo = suggestionsContainer[i]
                val wordLocale = wordInfo!!.mSourceDict.mLocale
                val transformedWordInfo = getTransformedSuggestedWordInfo(
                    wordInfo, wordLocale ?: locale, shouldMakeSuggestionsAllUpperCase,
                    shouldMakeSuggestionsOnlyFirstCharCapitalized, 0
                )
                suggestionsContainer[i] = transformedWordInfo
            }
        }
        val rejected: SuggestedWordInfo?
        if (SHOULD_REMOVE_PREVIOUSLY_REJECTED_SUGGESTION && suggestionsContainer.size > 1 && TextUtils.equals(
                suggestionsContainer[0]!!.mWord,
                wordComposer.rejectedBatchModeSuggestion
            )
        ) {
            rejected = suggestionsContainer.removeAt(0)
            suggestionsContainer.add(1, rejected)
        } else {
            rejected = null
        }
        SuggestedWordInfo.removeDupsAndTypedWord(null, suggestionsContainer)
        makeFirstTwoSuggestionsNonEmoji(suggestionsContainer)

        // For some reason some suggestions with MIN_VALUE are making their way here.
        // TODO: Find a more robust way to detect distracters.
        for (i in suggestionsContainer.indices.reversed()) {
            if (suggestionsContainer[i]!!.mScore < SUPPRESS_SUGGEST_THRESHOLD) {
                suggestionsContainer.removeAt(i)
            }
        }

        val capitalizedTypedWord = capitalize(wordComposer.typedWord, keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED,
            keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFTED, locale)
        if (capitalizedTypedWord != wordComposer.typedWord && suggestionsContainer.drop(1).none { it.mWord == capitalizedTypedWord }) {
            suggestionsContainer.add(min(1, suggestionsContainer.size),
                SuggestedWordInfo(capitalizedTypedWord, "", 0, SuggestedWordInfo.KIND_TYPED,
                    Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE)
            )
        }

        // In the batch input mode, the most relevant suggested word should act as a "typed word"
        // (typedWordValid=true), not as an "auto correct word" (willAutoCorrect=false).
        // Note that because this method is never used to get predictions, there is no need to
        // modify inputType such in getSuggestedWordsForNonBatchInput.
        val pseudoTypedWordInfo = preferNextWordSuggestion(
            suggestionsContainer.firstOrNull(),
            suggestionsContainer, getNextWordSuggestions(ngramContext, keyboard, inputStyle, settingsValuesForSuggestion), rejected
        )
        val suggestionsList = if (SuggestionStripView.DEBUG_SUGGESTIONS && suggestionsContainer.isNotEmpty()) {
            getSuggestionsInfoListWithDebugInfo(suggestionResults.first().mWord, suggestionsContainer)
        } else {
            suggestionsContainer
        }
        return SuggestedWords(suggestionsList, suggestionResults.mRawSuggestions, pseudoTypedWordInfo, true,
            false, false, inputStyle, sequenceNumber)
    }

    /** get suggestions based on the current ngram context, with an empty typed word (that's what next word suggestions do)  */
    private fun getNextWordSuggestions(ngramContext: NgramContext, keyboard: Keyboard, inputStyle: Int,
                                       settingsValuesForSuggestion: SettingsValuesForSuggestion): SuggestionResults {
        val cachedResults = nextWordSuggestionsCache[ngramContext]
        if (cachedResults != null) return cachedResults
        val newResults = mDictionaryFacilitator.getSuggestionResults(ComposedData(InputPointers(1),
            false, ""), ngramContext, keyboard, settingsValuesForSuggestion, SESSION_ID_TYPING, inputStyle)
        nextWordSuggestionsCache[ngramContext] = newResults
        return newResults
    }

    companion object {
        private val TAG: String = Suggest::class.java.simpleName

        // Session id for {@link #getSuggestedWords(WordComposer,String,ProximityInfo,boolean,int)}.
        // We are sharing the same ID between typing and gesture to save RAM footprint.
        const val SESSION_ID_TYPING = 0
        const val SESSION_ID_GESTURE = 0

        // Close to -2**31
        private const val SUPPRESS_SUGGEST_THRESHOLD = -2000000000

        private const val MAXIMUM_AUTO_CORRECT_LENGTH_FOR_GERMAN = 12
        // TODO: should we add Finnish here?
        private val sLanguageToMaximumAutoCorrectionWithSpaceLength = hashMapOf(Locale.GERMAN.language to MAXIMUM_AUTO_CORRECT_LENGTH_FOR_GERMAN)

        private fun getTransformedSuggestedWordInfoList(
            wordComposer: WordComposer, results: SuggestionResults,
            trailingSingleQuotesCount: Int, defaultLocale: Locale, keyboard: Keyboard
        ): ArrayList<SuggestedWordInfo> {
            val keyboardShiftMode = keyboard.mId.keyboardCapsMode
            val shouldMakeSuggestionsAllUpperCase = wordComposer.isAllUpperCase && !wordComposer.isResumed
                || keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED
            val shouldMakeSuggestionsOnlyFirstCharCapitalized = wordComposer.isOrWillBeOnlyFirstCharCapitalized
                || keyboardShiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFTED
            val suggestionsContainer = ArrayList(results)
            val suggestionsCount = suggestionsContainer.size
            if (shouldMakeSuggestionsOnlyFirstCharCapitalized || shouldMakeSuggestionsAllUpperCase || 0 != trailingSingleQuotesCount) {
                for (i in 0 until suggestionsCount) {
                    val wordInfo = suggestionsContainer[i]
                    val wordLocale = wordInfo.mSourceDict.mLocale
                    val transformedWordInfo = getTransformedSuggestedWordInfo(
                        wordInfo, wordLocale ?: defaultLocale,
                        shouldMakeSuggestionsAllUpperCase, shouldMakeSuggestionsOnlyFirstCharCapitalized,
                        trailingSingleQuotesCount
                    )
                    suggestionsContainer[i] = transformedWordInfo
                }
            }
            return suggestionsContainer
        }

        private fun getSuggestionsInfoListWithDebugInfo(
            typedWord: String, suggestions: ArrayList<SuggestedWordInfo>
        ): ArrayList<SuggestedWordInfo> {
            val suggestionsSize = suggestions.size
            val suggestionsList = ArrayList<SuggestedWordInfo>(suggestionsSize)
            for (cur in suggestions) {
                addDebugInfo(cur, typedWord)
                suggestionsList.add(cur)
            }
            return suggestionsList
        }

        @JvmStatic
        fun addDebugInfo(wordInfo: SuggestedWordInfo?, typedWord: String) {
            if (!SuggestionStripView.DEBUG_SUGGESTIONS)
                return
            val normalizedScore = BinaryDictionaryUtils.calcNormalizedScore(typedWord, wordInfo.toString(), wordInfo!!.mScore)
            val scoreInfoString: String
            val dict = wordInfo.mSourceDict.mDictType + ":" + wordInfo.mSourceDict.mLocale
            scoreInfoString = if (normalizedScore > 0) {
                String.format(Locale.ROOT, "%d (%4.2f), %s", wordInfo.mScore, normalizedScore, dict)
            } else {
                String.format(Locale.ROOT, "%d, %s", wordInfo.mScore, dict)
            }
            wordInfo.debugString = scoreInfoString
        }

        /**
         * Computes whether this suggestion should be blocked or not in this language
         *
         * This function implements a filter that avoids auto-correcting to suggestions that contain
         * spaces that are above a certain language-dependent character limit. In languages like German
         * where it's possible to concatenate many words, it often happens our dictionary does not
         * have the longer words. In this case, we offer a lot of unhelpful suggestions that contain
         * one or several spaces. Ideally we should understand what the user wants and display useful
         * suggestions by improving the dictionary and possibly having some specific logic. Until
         * that's possible we should avoid displaying unhelpful suggestions. But it's hard to tell
         * whether a suggestion is useful or not. So at least for the time being we block
         * auto-correction when the suggestion is long and contains a space, which should avoid the
         * worst damage.
         * This function is implementing that filter. If the language enforces no such limit, then it
         * always returns true. If the suggestion contains no space, it also returns true. Otherwise,
         * it checks the length against the language-specific limit.
         *
         * @param info the suggestion info
         * @return whether it's fine to auto-correct to this.
         */
        private fun isAllowedByAutoCorrectionWithSpaceFilter(info: SuggestedWordInfo): Boolean {
            val locale = info.mSourceDict.mLocale ?: return true
            val maximumLengthForThisLanguage = sLanguageToMaximumAutoCorrectionWithSpaceLength[locale.language]
                ?: return true // This language does not enforce a maximum length to auto-correction
            return (info.mWord.length <= maximumLengthForThisLanguage
                    || -1 == info.mWord.indexOf(Constants.CODE_SPACE.toChar()))
        }

        // public for testing
        fun getTransformedSuggestedWordInfo(
            wordInfo: SuggestedWordInfo, locale: Locale, isAllUpperCase: Boolean,
            isOnlyFirstCharCapitalized: Boolean, trailingSingleQuotesCount: Int
        ): SuggestedWordInfo {
            var capitalizedWord = capitalize(wordInfo.mWord, isAllUpperCase, isOnlyFirstCharCapitalized, locale)
            // Appending quotes is here to help people quote words. However, it's not helpful
            // when they type words with quotes toward the end like "it's" or "didn't", where
            // it's more likely the user missed the last character (or didn't type it yet).
            val quotesToAppend = (trailingSingleQuotesCount
                    - if (-1 == wordInfo.mWord.indexOf(Constants.CODE_SINGLE_QUOTE.toChar())) 0 else 1)
            for (i in quotesToAppend - 1 downTo 0) {
                capitalizedWord = "$capitalizedWord'"
            }
            return SuggestedWordInfo(
                capitalizedWord, wordInfo.mPrevWordsContext,
                wordInfo.mScore, wordInfo.mKindAndFlags,
                wordInfo.mSourceDict, wordInfo.mIndexOfTouchPointOfSecondWord,
                wordInfo.mAutoCommitFirstWordConfidence
            )
        }

        private fun capitalize(word: String, isAllUpperCase: Boolean, isOnlyFirstCharCapitalized: Boolean, locale: Locale) =
            if (isAllUpperCase) {
                word.uppercase(locale)
            } else if (isOnlyFirstCharCapitalized) {
                StringUtils.capitalizeFirstCodePoint(word, locale)
            } else {
                word
            }

        private fun makeFirstTwoSuggestionsNonEmoji(words: MutableList<SuggestedWordInfo>) {
            for (i in 0..1) {
                if (words.size > 2 && words[i].isEmoji) {
                    val relativeIndex = words.subList(2, words.size).indexOfFirst { !it.isEmoji }
                    if (relativeIndex < 0) break
                    val firstNonEmojiIndex = relativeIndex + 2
                    if (firstNonEmojiIndex > i) {
                        words.add(i, words.removeAt(firstNonEmojiIndex))
                    }
                }
            }
        }

        /** reduces score of the first suggestion if next one is close and has more than a single letter  */
        private fun replaceSingleLetterFirstSuggestion(suggestionResults: SuggestionResults) {
            if (suggestionResults.size < 2 || suggestionResults.first().mWord.length != 1) return
            // suppress single letter suggestions if next suggestion is close and has more than one letter
            val iterator: Iterator<SuggestedWordInfo> = suggestionResults.iterator()
            val first = iterator.next()
            val second = iterator.next()
            if (second.mWord.length > 1 && second.mScore > 0.94 * first.mScore) {
                suggestionResults.remove(first) // remove and re-add with lower score
                suggestionResults.add(
                    SuggestedWordInfo(
                        first.mWord, first.mPrevWordsContext, (first.mScore * 0.93).toInt(),
                        first.mKindAndFlags, first.mSourceDict, first.mIndexOfTouchPointOfSecondWord, first.mAutoCommitFirstWordConfidence
                    )
                )
                if (DebugFlags.DEBUG_ENABLED)
                    Log.d(TAG, "reduced score of ${first.mWord} from ${first.mScore}, new first: ${suggestionResults.first().mWord} (${suggestionResults.first().mScore})")
            }
        }

        /** returns new pseudoTypedWordInfo, puts it in suggestionsContainer, modifies nextWordSuggestions */
        private fun preferNextWordSuggestion(
            pseudoTypedWordInfo: SuggestedWordInfo?,
            suggestionsContainer: ArrayList<SuggestedWordInfo>,
            nextWordSuggestions: SuggestionResults, rejected: SuggestedWordInfo?
        ): SuggestedWordInfo? {
            if (pseudoTypedWordInfo == null || !Settings.getValues().mUsePersonalizedDicts
                || pseudoTypedWordInfo.mSourceDict.mDictType != Dictionary.TYPE_MAIN || suggestionsContainer.size < 2
            ) return pseudoTypedWordInfo
            nextWordSuggestions.removeAll { info: SuggestedWordInfo -> info.mScore < 170 } // we only want reasonably often typed words, value may require tuning
            if (nextWordSuggestions.isEmpty()) return pseudoTypedWordInfo

            // for each suggestion, check whether the word was already typed in this ngram context (i.e. is nextWordSuggestion)
            for (suggestion in suggestionsContainer) {
                if (suggestion.mScore < pseudoTypedWordInfo.mScore * 0.93) break // we only want reasonably good suggestions, value may require tuning
                if (suggestion === rejected) continue  // ignore rejected suggestions
                for (nextWordSuggestion in nextWordSuggestions) {
                    if (nextWordSuggestion.mWord != suggestion.mWord) continue
                    // if we have a high scoring suggestion in next word suggestions, take it (because it's expected that user might want to type it again)
                    suggestionsContainer.remove(suggestion)
                    suggestionsContainer.add(0, suggestion)
                    if (DebugFlags.DEBUG_ENABLED)
                        Log.d(TAG, "replaced batch word $pseudoTypedWordInfo with $suggestion")
                    return suggestion
                }
            }
            return pseudoTypedWordInfo
        }
    }
}
