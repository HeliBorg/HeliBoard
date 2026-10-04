// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import android.content.Context
import android.util.LruCache
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.emoji.SupportedEmojis
import helium314.keyboard.latin.DictionaryFacilitator.DictionaryInitializationListener
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.dictionary.DictionaryStats
import helium314.keyboard.latin.makedict.WordProperty
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.utils.SuggestionResults
import java.util.Locale
import java.util.concurrent.TimeUnit

/** Simple DictionaryFacilitator for a single Dictionary. Has some optional special purpose functionality. */
class SingleDictionaryFacilitator(private val dict: Dictionary) : DictionaryFacilitator {
    var suggestionLogger: SuggestionLogger? = null

    /**
     * Returns combined suggestions that match any of the given words, with scores that reflect the matches against all words.
     * The combined score is calculated as the average absolute (above [Int.MIN_VALUE]) score,
     * where a non-match is considered an absolute zero.
     * Other suggestion fields of combined matches are taken from the highest-score one.
     */
    fun getSuggestions(words: List<String>): SuggestionResults {
        val suggestionResults = SuggestionResults(SuggestedWords.MAX_SUGGESTIONS, false, false)
        words.flatMap { word -> getSuggestions(word).distinctBy { it.word } } // Filter out duplicate results per word
            .groupBy { it.word }
            .forEach { (_, results) ->
                val info = results.maxBy { it.mScore }
                val score = (results.sumOf { it.mScore.toLong() - Int.MIN_VALUE } / words.size + Int.MIN_VALUE).toInt()
                suggestionResults.add(
                    SuggestedWords.SuggestedWordInfo(
                        info.word, info.mPrevWordsContext, score, info.mKindAndFlags,
                        info.mSourceDict, info.mIndexOfTouchPointOfSecondWord, info.mAutoCommitFirstWordConfidence
                    )
                )
            }
        return suggestionResults
    }

    // this will not work from spell checker if used together with a different keyboard app
    fun getSuggestions(word: String): SuggestionResults {
        val suggestionResults = getSuggestionResults(
            ComposedData.createForWord(word),
            NgramContext.getEmptyPrevWordsContext(0),
            KeyboardSwitcher.getInstance().keyboard!!, // looks like actual keyboard doesn't matter (composed data doesn't contain coordinates)
            SettingsValuesForSuggestion(false, false),
            Suggest.SESSION_ID_TYPING, SuggestedWords.INPUT_STYLE_TYPING
        )
        return suggestionResults
    }

    override fun getSuggestionResults(
        composedData: ComposedData, ngramContext: NgramContext, keyboard: Keyboard,
        settingsValuesForSuggestion: SettingsValuesForSuggestion, sessionId: Int, inputStyle: Int
    ): SuggestionResults {
        val suggestionResults = SuggestionResults(
            SuggestedWords.MAX_SUGGESTIONS, ngramContext.isBeginningOfSentenceContext,
            false
        )
        suggestionResults.addAll(
            dict.getSuggestions(composedData, ngramContext, keyboard.proximityInfo.nativeProximityInfo,
                settingsValuesForSuggestion, sessionId, 1f,
                floatArrayOf(Dictionary.NOT_A_WEIGHT_OF_LANG_MODEL_VS_SPATIAL_MODEL)
            )?.filter { !SupportedEmojis.isUnsupported(it.word) }
        )
        suggestionLogger?.onNewSuggestions(suggestionResults, composedData, ngramContext, keyboard, inputStyle)

        return suggestionResults
    }

    fun getWordProperty(word: String): WordProperty? = dict.getWordProperty(word, false)

    // ------------ dummy functionality ----------------

    override fun setValidSpellingWordReadCache(cache: LruCache<String, Boolean>) {}

    override fun setValidSpellingWordWriteCache(cache: LruCache<String, Boolean>) {}

    override fun isForLocale(locale: Locale?): Boolean = locale == dict.mLocale

    override fun onStartInput() {}

    override fun onFinishInput() {
        dict.onFinishInput()
    }

    override fun closeDictionaries() {
        dict.close()
    }

    override fun isActive(): Boolean = true

    override fun getMainLocale(): Locale = dict.mLocale

    override fun getCurrentLocale(): Locale = mainLocale

    override fun usesSameSettings(locales: List<Locale>, contacts: Boolean, apps: Boolean, personalization: Boolean): Boolean {
        return locales.singleOrNull() == mainLocale
    }

    override fun resetDictionaries(context: Context, newLocale: Locale, useContactsDict: Boolean, useAppsDict: Boolean,
                                   usePersonalizedDicts: Boolean, forceReloadMainDictionary: Boolean, dictNamePrefix: String, listener: DictionaryInitializationListener?
    ) { }

    override fun hasAtLeastOneInitializedMainDictionary(): Boolean = dict.isInitialized

    override fun hasAtLeastOneUninitializedMainDictionary(): Boolean = !dict.isInitialized

    override fun waitForLoadingMainDictionaries(timeout: Long, unit: TimeUnit) {
    }

    override fun addToUserHistory(
        suggestion: String, wasAutoCapitalized: Boolean, ngramContext: NgramContext,
        timeStampInSeconds: Long, blockPotentiallyOffensive: Boolean
    ) {}

    override fun adjustConfidences(word: String, wasAutoCapitalized: Boolean) {}

    override fun unlearnFromUserHistory(word: String, ngramContext: NgramContext,
        timeStampInSeconds: Long, event: DictionaryFacilitator.UnlearnEvent) {}

    override fun isValidSpellingWord(word: String): Boolean = dict.isValidWord(word)

    override fun isValidSuggestionWord(word: String) = isValidSpellingWord(word)

    override fun removeWord(word: String) {}

    override fun clearUserHistoryDictionary(context: Context) {}

    override fun localesAndConfidences(): String? = null

    override fun dumpDictionaryForDebug(dictName: String) {}

    override fun getDictionaryStats(context: Context): List<DictionaryStats> = emptyList()

    override fun dump(context: Context) = getDictionaryStats(context).joinToString("\n")

    companion object {
        interface SuggestionLogger {
            /** provides input data and suggestions returned by the library */
            fun onNewSuggestions(suggestions: SuggestionResults, composedData: ComposedData,
                                 ngramContext: NgramContext, keyboard: Keyboard, inputStyle: Int)
        }
    }
}
