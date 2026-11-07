/*
 * Copyright (C) 2008 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.suggestions

import android.text.TextUtils
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils
import helium314.keyboard.compat.CompatBreakIterator
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.internal.keyboard_parser.getEmojiDefaultVersion
import helium314.keyboard.latin.CapsMode
import helium314.keyboard.latin.DictionaryFacilitator
import helium314.keyboard.latin.InputStyle
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.WordComposer
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
import helium314.keyboard.latin.utils.AutoCorrectionUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.BackgroundGatheringCache
import helium314.keyboard.latin.utils.TextAtCursor
import helium314.keyboard.latin.utils.WordData
import helium314.keyboard.latin.utils.useBackgroundGathering
import java.util.Locale
import kotlin.math.max
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
        fun onGetSuggestedWords(suggestedWords: SuggestedWords)
    }

    fun getSuggestedWords(wordComposer: WordComposer, textAtCursor: TextAtCursor?, ngramContext: NgramContext, keyboard: Keyboard,
                          settingsValuesForSuggestion: SettingsValuesForSuggestion, isCorrectionEnabled: Boolean,
                          inputStyle: InputStyle, sequenceNumber: Int): SuggestedWords =
        if (wordComposer.isBatchMode) {
            getSuggestedWordsForBatchInput(wordComposer, ngramContext, keyboard, settingsValuesForSuggestion,
                inputStyle, isCorrectionEnabled, sequenceNumber)
        } else {
            getSuggestedWordsForNonBatchInput(wordComposer, textAtCursor, ngramContext, keyboard, settingsValuesForSuggestion,
                inputStyle, isCorrectionEnabled, sequenceNumber)
        }

    // Retrieves suggestions for non-batch input (typing, recorrection, predictions...)
    // and calls the callback function with the suggestions.
    private fun getSuggestedWordsForNonBatchInput(wordComposer: WordComposer, textAtCursor: TextAtCursor?, ngramContext: NgramContext, keyboard: Keyboard,
                      settingsValuesForSuggestion: SettingsValuesForSuggestion, inputStyleIfNotPrediction: InputStyle,
                      isCorrectionEnabled: Boolean, sequenceNumber: Int): SuggestedWords {
        val typedWordString = wordComposer.typedWord
        val resultsArePredictions = !wordComposer.isComposingWord
        val suggestionResults = if (typedWordString.isEmpty())
                getNextWordSuggestions(ngramContext, keyboard, inputStyleIfNotPrediction, settingsValuesForSuggestion)
            else mDictionaryFacilitator.getSuggestionResults(wordComposer.composedDataSnapshot, ngramContext, keyboard,
                settingsValuesForSuggestion, SESSION_ID_TYPING, inputStyleIfNotPrediction)
        val trailingSingleQuotesCount = StringUtils.getTrailingSingleQuotesCount(typedWordString)
        val capsMode = getCapsModeForTyping(wordComposer, keyboard)
        val suggestionsContainer = ArrayList(suggestionResults)

        // add suggestions for the "extended word" at cursor
        // using a different iterator that also includes digits and maybe more
        val extendedWord = textAtCursor?.getWordAtCursor(CompatBreakIterator.getWordInstance(keyboard.mId.locale))?.word
        if (extendedWord != null && typedWordString.isNotEmpty() && extendedWord.length > typedWordString.length) {
            // ideally we'd extendComposed data for better suggestions, but that's for later (if ever)
            val extendedWord = textAtCursor.getWordAtCursor(CompatBreakIterator.getWordInstance(keyboard.mId.locale)).word.toString()
            val altSuggestions = mDictionaryFacilitator.getSuggestionResults(ComposedData.createForWord(extendedWord), ngramContext, keyboard,
                settingsValuesForSuggestion, SESSION_ID_TYPING, inputStyleIfNotPrediction)
            val extendingPart = extendedWord.substring(0, extendedWord.length - typedWordString.length)
            altSuggestions.forEach {
                if (it.word.startsWith(extendingPart) || (it.kind == SuggestedWordInfo.KIND_SHORTCUT && it.score > 0)) {
                    it.wordToReplace = extendedWord
                    suggestionsContainer.add(it)
                }
            }
            suggestionsContainer.sortByDescending { it.score }
        }

        capitalizeAndAddTrailingSingleQuotes(suggestionsContainer, capsMode, trailingSingleQuotesCount, mDictionaryFacilitator.mainLocale)
        val capitalizedTypedWord = capitalize(typedWordString, capsMode, mDictionaryFacilitator.mainLocale)

        // store the original SuggestedWordInfo for typed word, as it will be removed
        // we may want to re-add it in case auto-correction happens, so that the original word can at least be selected
        // we check against the capitalizedTypedWord because getTransformedSuggestedWordInfoList adjusts for capsMode
        val typedWordFirstOccurrenceWordInfo = suggestionsContainer.firstOrNull { it.word == capitalizedTypedWord }
        val firstOccurrenceOfTypedWordInSuggestions = SuggestedWordInfo.removeDupsAndTypedWord(capitalizedTypedWord, suggestionsContainer)
        makeFirstTwoSuggestionsNonEmoji(suggestionsContainer)

        val (allowsToBeAutoCorrected, hasAutoCorrection) = shouldBeAutoCorrected(
            trailingSingleQuotesCount,
            capitalizedTypedWord,
            suggestionsContainer.firstOrNull(),
            {
                val first = suggestionsContainer.firstOrNull() ?: suggestionResults.first()
                val suggestions = getNextWordSuggestions(ngramContext, keyboard, inputStyleIfNotPrediction, settingsValuesForSuggestion)
                val suggestionForFirstInContainer = suggestions.firstOrNull { it.word == first.word }
                val suggestionForTypedWord = suggestions.firstOrNull { it.word == capitalizedTypedWord }
                suggestionForFirstInContainer to suggestionForTypedWord
            },
            isCorrectionEnabled,
            wordComposer,
            suggestionResults,
            firstOccurrenceOfTypedWordInSuggestions,
            typedWordFirstOccurrenceWordInfo
        )
        val typedWordInfo = SuggestedWordInfo(typedWordString, "", SuggestedWordInfo.MAX_SCORE,
            SuggestedWordInfo.KIND_TYPED, typedWordFirstOccurrenceWordInfo?.sourceDict ?: Dictionary.DICTIONARY_USER_TYPED,
            SuggestedWordInfo.NOT_AN_INDEX , SuggestedWordInfo.NOT_A_CONFIDENCE)
        if (typedWordString.isNotEmpty()) {
            suggestionsContainer.add(0, typedWordInfo)
        }
        val suggestionsList = if (SuggestionStripView.DEBUG_SUGGESTIONS && suggestionsContainer.isNotEmpty())
                getSuggestionsInfoListWithDebugInfo(capitalizedTypedWord, suggestionsContainer)
            else suggestionsContainer

        val inputStyle = if (resultsArePredictions) {
            if (suggestionResults.ngramContext.isBeginningOfSentenceContext) InputStyle.BEGINNING_OF_SENTENCE_PREDICTION
            else InputStyle.PREDICTION
        } else {
            inputStyleIfNotPrediction
        }

        useDefaultEmojiSkinTone(suggestionsList)

        // If there is an incoming autocorrection, make sure typed word is shown, so user is able to override it.
        // Otherwise, if the relevant setting is enabled, show the typed word in the middle.
        val typedWordWasCapitalized = capitalizedTypedWord != typedWordString
        val correctToCapitalizedWord = typedWordWasCapitalized && isCorrectionEnabled && Settings.getValues().mAutoCorrectCapitalizedSuggestion
            && !wordComposer.isCursorFrontOrMiddleOfComposingWord && typedWordString.drop(1).none { it.isUpperCase() }
        val indexOfTypedWord = 1 + if (hasAutoCorrection) SuggestedWords.INDEX_OF_AUTO_CORRECTION else SuggestedWords.INDEX_OF_TYPED_WORD
        if (
            (hasAutoCorrection
                || (Settings.getValues().mCenterSuggestionTextToEnter && !wordComposer.isResumed)
                || typedWordWasCapitalized
            ) && suggestionsList.size >= indexOfTypedWord && capitalizedTypedWord.isNotEmpty()) {
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
        return SuggestedWords(suggestionsList, suggestionResults.rawSuggestions, typedWordInfo,
            isTypedWordValid, hasAutoCorrection || correctToCapitalizedWord, false, inputStyle, sequenceNumber)
    }

    // returns [allowsToBeAutoCorrected, hasAutoCorrection]
    // public for testing
    // todo: now we can do better tests, maybe make it private again and test via getSuggestedWords (and simplify if possible)
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
                    && typedWordInfo?.sourceDict == null // more than 1 letter and not in dictionary
                    // if the typed word contains @ or ., the suggestion also needs to contain it
                    // (avoid autocorrecting mail addresses, URLs & similar to something different)
                    && (!typedWordString.contains('@') || firstSuggestionInContainer?.word?.contains('@') == true)
                    && (!typedWordString.contains('.') || firstSuggestionInContainer?.word?.contains('.') == true)
                    )
            ) {
            allowsToBeAutoCorrected = true
        } else if (firstSuggestionInContainer != null && typedWordString.isNotEmpty()) {
            // maybe allow autocorrect, depending on scores and emptyWordSuggestions
            val first = firstAndTypedEmptyInfos.first
            val typed = firstAndTypedEmptyInfos.second
            allowsToBeAutoCorrected = when {
                firstSuggestionInContainer.score > scoreLimit -> true // suggestion has good score, allow
                first == null -> false // no autocorrect if first suggestion unknown in this ngram context
                typed == null -> true // allow autocorrect if typed word not known in this ngram context, todo: this may be too aggressive
                else -> first.score - typed.score > 20 // autocorrect if suggested word has clearly higher score for empty word suggestions
            }
        } else {
            allowsToBeAutoCorrected = false
        }
        // If correction is not enabled, we never auto-correct. This is for example for when
        // the setting "Auto-correction" is "off": we still suggest, but we don't auto-correct.
        val hasAutoCorrection: Boolean
        if (!isCorrectionEnabled
            || !allowsToBeAutoCorrected // If the word does not allow to be auto-corrected, then we don't auto-correct.
            || !wordComposer.isComposingWord // If we are doing prediction, then we never auto-correct of course
            || suggestionResults.isEmpty() // If we don't have suggestion results, we can't evaluate the first suggestion for auto-correction
            || wordComposer.hasDigits() // If the word has digits, we never auto-correct because it's likely the word was type with a lot of care // todo: but what if user touched the number row?
            || (wordComposer.isMostlyCaps && !wordComposer.isAllUpperCase) // If the word is mostly caps, we never auto-correct because this is almost certainly intentional
            || wordComposer.isResumed // We never auto-correct when suggestions are resumed because it would be unexpected
            // If we don't have a main dictionary, we never want to auto-correct. The reason
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
            if (suggestionResults.firstSuggestionExceedsConfidenceThreshold && firstOccurrenceOfTypedWordInSuggestions != 0) {
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
            if (allowed && typedWordInfo != null && typedWordInfo.score > scoreLimit) {
                // typed word is valid and has good score
                // do not auto-correct if typed word is better match than first suggestion
                val dictLocale = mDictionaryFacilitator.currentLocale
                if (firstSuggestion.score < scoreLimit) {
                    // don't allow if suggestion has too low score
                    return true to false
                }
                if (firstSuggestion.sourceDict.mLocale !== typedWordInfo.sourceDict.mLocale) {
                    // dict locale different -> return the better match
                    return true to (dictLocale == firstSuggestion.sourceDict.mLocale)
                }
                // the score difference may need tuning, but so far it seems alright
                val firstWordBonusScore =
                    ((if (firstSuggestion.isKindOf(SuggestedWordInfo.KIND_WHITELIST)) 20 else 0) // large bonus because it's wanted by dictionary
                            + (if (StringUtils.isLowerCaseAscii(typedWordString)) 5 else 0) // small bonus because typically only lower case ascii is typed (applies to latin keyboards only)
                            + if (firstSuggestion.score > typedWordInfo.score) 5 else 0) // small bonus if score is higher
                val firstScoreForEmpty = firstAndTypedEmptyInfos.first?.score ?: 0
                val typedScoreForEmpty = firstAndTypedEmptyInfos.second?.score ?: 0
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
        inputStyle: InputStyle, isCorrectionEnabled: Boolean, sequenceNumber: Int
    ): SuggestedWords {
        val suggestionResults = mDictionaryFacilitator.getSuggestionResults(
            wordComposer.composedDataSnapshot, ngramContext, keyboard,
            settingsValuesForSuggestion, SESSION_ID_GESTURE, inputStyle
        )

        // For transforming words that don't come from a dictionary, because it's our best bet
        val locale = mDictionaryFacilitator.mainLocale
        val capsMode = getCapsModeForGesture(wordComposer, keyboard)
        val suggestionsContainer = ArrayList(suggestionResults)
        replaceSingleLetterFirstSuggestion(suggestionsContainer)

        val rejected: SuggestedWordInfo?
        if (SHOULD_REMOVE_PREVIOUSLY_REJECTED_SUGGESTION && suggestionsContainer.size > 1 && TextUtils.equals(
                suggestionsContainer[0].word,
                wordComposer.rejectedBatchModeSuggestion
            )
        ) {
            rejected = suggestionsContainer.removeAt(0)
            suggestionsContainer.add(1, rejected)
        } else {
            rejected = null
        }
        // unchanged first suggestion, but considering adjusted order
        // we try to suppress emojis, because they will be moved to a later position if possible
        val pseudoTypedWord = suggestionsContainer.firstOrNull { !it.isEmoji } ?: suggestionsContainer.firstOrNull()

        // order is important, because capitalize may result in duplicate suggestions (e.g. frank and Frank),
        // which we want to remove before making the first 2 suggestions non-emoji
        capitalizeAndAddTrailingSingleQuotes(suggestionsContainer, capsMode, 0, locale)
        SuggestedWordInfo.removeDupsAndTypedWord(null, suggestionsContainer)
        makeFirstTwoSuggestionsNonEmoji(suggestionsContainer)

        // For some reason some suggestions with MIN_VALUE are making their way here.
        // TODO: Find a more robust way to detect distracters.
        for (i in suggestionsContainer.indices.reversed()) {
            if (suggestionsContainer[i].score < SUPPRESS_SUGGEST_THRESHOLD) {
                suggestionsContainer.removeAt(i)
            }
        }

        val capitalizedTypedWord = capitalize(wordComposer.typedWord, capsMode, locale)
        val addCapitalizedSuggestion = capitalizedTypedWord != wordComposer.typedWord && suggestionsContainer.drop(1).none { it.word == capitalizedTypedWord }
        if (addCapitalizedSuggestion) {
            suggestionsContainer.add(min(1, suggestionsContainer.size),
                SuggestedWordInfo(capitalizedTypedWord, "", 0, SuggestedWordInfo.KIND_TYPED,
                    Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE)
            )
        }

        useDefaultEmojiSkinTone(suggestionsContainer)

        // In the batch input mode, the most relevant suggested word should act as a "typed word"
        // (typedWordValid=true), not as an "auto correct word" (willAutoCorrect=false).
        // Exception is when using shift to change capitalization of suggestions.
        // Note that because this method is never used to get predictions, there is no need to
        // modify inputType such in getSuggestedWordsForNonBatchInput.
        val pseudoTypedWordInfo = preferNextWordSuggestion(
            pseudoTypedWord, suggestionsContainer,
            getNextWordSuggestions(ngramContext, keyboard, inputStyle, settingsValuesForSuggestion), rejected
        )
        val suggestionsList = if (SuggestionStripView.DEBUG_SUGGESTIONS && suggestionsContainer.isNotEmpty()) {
            getSuggestionsInfoListWithDebugInfo(suggestionResults.first().word, suggestionsContainer)
        } else {
            suggestionsContainer
        }

        if (useBackgroundGathering && inputStyle == InputStyle.TAIL_BATCH) {
            val wordData = WordData(null, suggestionResults, wordComposer.composedDataSnapshot,
                ngramContext, keyboard, inputStyle, false, pseudoTypedWordInfo)
            BackgroundGatheringCache.addWord(wordData)
        }

        val autocorrectCapitalization = addCapitalizedSuggestion && Settings.getValues().mAutoCorrectCapitalizedSuggestion && isCorrectionEnabled && !wordComposer.isCursorFrontOrMiddleOfComposingWord
        return SuggestedWords(suggestionsList, suggestionResults.rawSuggestions, pseudoTypedWordInfo, true,
            autocorrectCapitalization, false, inputStyle, sequenceNumber)
    }

    private fun useDefaultEmojiSkinTone(suggestionsList: ArrayList<SuggestedWordInfo>) {
        for (i in suggestionsList.indices) {
            suggestionsList[i] = useDefaultEmojiSkinTone(suggestionsList[i])
        }
    }

    /** get suggestions based on the current ngram context, with an empty typed word (that's what next word suggestions do)  */
    private fun getNextWordSuggestions(ngramContext: NgramContext, keyboard: Keyboard, inputStyle: InputStyle,
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

        private fun capitalizeAndAddTrailingSingleQuotes(
            suggestions: ArrayList<SuggestedWordInfo>, capsMode: CapsMode, trailingSingleQuotesCount: Int, defaultLocale: Locale
        ) {
            val suggestionsCount = suggestions.size
            if (capsMode != CapsMode.OFF || 0 != trailingSingleQuotesCount) {
                for (i in 0 until suggestionsCount) {
                    val wordInfo = suggestions[i]
                    val wordLocale = wordInfo.sourceDict.mLocale
                    val transformedWordInfo = capitalizeAndAddTrailingSingleQuotes(
                        wordInfo, wordLocale ?: defaultLocale, capsMode, trailingSingleQuotesCount
                    )
                    suggestions[i] = transformedWordInfo
                }
            }
        }

        private fun capitalizeAndAddTrailingSingleQuotes(
            wordInfo: SuggestedWordInfo, locale: Locale, capsMode: CapsMode, trailingSingleQuotesCount: Int
        ): SuggestedWordInfo {
            var capitalizedWord = capitalize(wordInfo.word, capsMode, locale)
            // Appending quotes is here to help people quote words. However, it's not helpful
            // when they type words with quotes toward the end like "it's" or "didn't", where
            // it's more likely the user missed the last character (or didn't type it yet).
            val quotesToAppend = trailingSingleQuotesCount - if (wordInfo.word.contains('\'')) 1 else 0
            repeat(max(0, quotesToAppend)) { capitalizedWord += '\'' }
            return wordInfo.copy(word = capitalizedWord)
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
            val normalizedScore = BinaryDictionaryUtils.calcNormalizedScore(typedWord, wordInfo.toString(), wordInfo!!.score)
            val scoreInfoString: String
            val dict = wordInfo.sourceDict.mDictType + ":" + wordInfo.sourceDict.mLocale
            scoreInfoString = if (normalizedScore > 0) {
                String.format(Locale.ROOT, "%d (%4.2f), %s", wordInfo.score, normalizedScore, dict)
            } else {
                String.format(Locale.ROOT, "%d, %s", wordInfo.score, dict)
            }
            wordInfo.debugString = scoreInfoString
        }

        @JvmStatic
        fun useDefaultEmojiSkinTone(suggestion: SuggestedWordInfo): SuggestedWordInfo {
            val defaultVersion = getEmojiDefaultVersion(suggestion.word)
            if (defaultVersion == suggestion.word) {
                return suggestion
            }

            return suggestion.copy(word = defaultVersion)
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
            val locale = info.sourceDict.mLocale ?: return true
            val maximumLengthForThisLanguage = sLanguageToMaximumAutoCorrectionWithSpaceLength[locale.language]
                ?: return true // This language does not enforce a maximum length to auto-correction
            return (info.word.length <= maximumLengthForThisLanguage
                    || -1 == info.word.indexOf(Constants.CODE_SPACE.toChar()))
        }

        /** returns CapsMode.MANUAL, CapsMode.MANUAL_LOCKED, or CAPS_MODE_OFF */
        private fun getCapsModeForTyping(wordComposer: WordComposer, keyboard: Keyboard): CapsMode {
            val capsMode = keyboard.mId.element.capsMode
            if (capsMode == CapsMode.MANUAL || capsMode == CapsMode.MANUAL_LOCKED)
                return capsMode
            // we have some auto-mode which we ignore
            // instead we determine mode from the typed word (that's how it was done for a long time, todo: maybe adjust if necessary?)
            if (wordComposer.isAllUpperCase && !wordComposer.isResumed) return CapsMode.MANUAL_LOCKED
            if (wordComposer.isOrWillBeOnlyFirstCharCapitalized) return CapsMode.MANUAL
            return CapsMode.OFF
        }

        /** returns CapsMode.MANUAL, CapsMode.MANUAL_LOCKED, or CAPS_MODE_OFF */
        // maybe could check the details in differences to getCapsModeForTyping and unify?
        private fun getCapsModeForGesture(wordComposer: WordComposer, keyboard: Keyboard): CapsMode {
            val capsMode = keyboard.mId.element.capsMode
            if (capsMode == CapsMode.MANUAL || capsMode == CapsMode.MANUAL_LOCKED)
                return capsMode
            if (wordComposer.isAllUpperCase) return CapsMode.MANUAL_LOCKED
            if (wordComposer.wasShiftedNoLock()) return CapsMode.MANUAL
            return CapsMode.OFF
        }

        private fun capitalize(word: String, capsMode: CapsMode, locale: Locale) = when (capsMode) {
            CapsMode.MANUAL_LOCKED -> word.uppercase(locale)
            CapsMode.MANUAL -> StringUtils.capitalizeFirstCodePoint(word, locale)
            else -> word
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

        /** reduces score of the first suggestion if next one is close and has more than a single letter */
        private fun replaceSingleLetterFirstSuggestion(suggestionResults: MutableList<SuggestedWordInfo>) {
            if (suggestionResults.size < 2 || suggestionResults.first().word.length != 1) return
            // suppress single letter suggestions if next suggestion is close and has more than one letter
            val first = suggestionResults[0]
            val second = suggestionResults[1]
            if (second.word.length > 1 && second.score > 0.94 * first.score) {
                suggestionResults.remove(first) // remove and re-add with lower score
                val modifiedFirst = first.copy(word = first.word, score = (first.score * 0.93).toInt())
                val insertIndex = suggestionResults.indexOfFirst { it.score < modifiedFirst.score }
                if (insertIndex == -1) suggestionResults.add(modifiedFirst)
                else suggestionResults.add(insertIndex, modifiedFirst)

                if (DebugFlags.DEBUG_ENABLED)
                    Log.d(TAG, "reduced score of ${first.word} from ${first.score}, new first: ${suggestionResults.first().word} (${suggestionResults.first().score})")
            }
        }

        /** returns new pseudoTypedWordInfo, puts it in suggestionsContainer, modifies nextWordSuggestions */
        private fun preferNextWordSuggestion(
            pseudoTypedWordInfo: SuggestedWordInfo?,
            suggestionsContainer: ArrayList<SuggestedWordInfo>,
            nextWordSuggestions: SuggestionResults, rejected: SuggestedWordInfo?
        ): SuggestedWordInfo? {
            if (pseudoTypedWordInfo == null || !Settings.getValues().mUsePersonalizedDicts
                || pseudoTypedWordInfo.sourceDict.mDictType != Dictionary.TYPE_MAIN || suggestionsContainer.size < 2
            ) return pseudoTypedWordInfo
            val goodNextSuggestions = nextWordSuggestions.filter { it.score >= 170 } // we only want reasonably often typed words, value may require tuning
            if (goodNextSuggestions.isEmpty()) return pseudoTypedWordInfo

            // for each suggestion, check whether the word was already typed in this ngram context (i.e. is nextWordSuggestion)
            for (suggestion in suggestionsContainer) {
                if (suggestion.score < pseudoTypedWordInfo.score * 0.93) break // we only want reasonably good suggestions, value may require tuning
                if (suggestion === rejected) continue  // ignore rejected suggestions
                for (nextWordSuggestion in goodNextSuggestions) {
                    if (nextWordSuggestion.word != suggestion.word) continue
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
