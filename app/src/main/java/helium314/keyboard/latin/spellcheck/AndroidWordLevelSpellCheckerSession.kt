/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.spellcheck

import android.database.ContentObserver
import android.os.Binder
import android.os.Build
import android.provider.UserDictionary.Words
import android.service.textservice.SpellCheckerService
import android.text.TextUtils
import android.util.LruCache
import android.view.inputmethod.InputMethodManager
import android.view.textservice.SuggestionsInfo
import android.view.textservice.TextInfo
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.WordComposer
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.ScriptUtils
import helium314.keyboard.latin.utils.ScriptUtils.isLetterPartOfScript
import helium314.keyboard.latin.utils.ScriptUtils.script
import helium314.keyboard.latin.utils.StatsUtils
import helium314.keyboard.latin.utils.SubtypeLocaleUtils
import helium314.keyboard.latin.utils.SubtypeSettings.getSelectedSubtype
import helium314.keyboard.latin.utils.SuggestionResults
import helium314.keyboard.latin.utils.prefs
import java.util.Locale
import java.util.TreeMap
import kotlin.math.min

abstract class AndroidWordLevelSpellCheckerSession(private val mService: AndroidSpellCheckerService) :
    SpellCheckerService.Session() {
    private lateinit var locale: Locale
    private lateinit var script: String // Cache this for performance
    protected val suggestionsCache: SuggestionsCache = SuggestionsCache()
    private val observer = object : ContentObserver(null) {
        override fun onChange(self: Boolean) {
            suggestionsCache.clearCache()
        }
    }

    init {
        mService.contentResolver.registerContentObserver(Words.CONTENT_URI, true, observer)
    }

    private fun updateLocale() {
        val localeString = getLocale()
        val initialized = this::locale.isInitialized
        if (!initialized || locale.toString() != localeString) {
            Log.d(TAG, "Updating locale from ${if (initialized) locale.toString() else "null"} to $localeString")
            locale = (localeString ?: SubtypeLocaleUtils.NO_LANGUAGE).constructLocale()
            script = locale.script()
        }
    }

    override fun onCreate() {
        updateLocale()
    }

    // unfortunately this can only return a string, with the obvious issues
    override fun getLocale(): String? {
        // This function was taken from https://github.com/LineageOS/android_frameworks_base/blob/1235c24a0f092d0e41fd8e86f332f8dc03896a7b/services/core/java/com/android/server/TextServicesManagerService.java#L544 and slightly adopted.

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val imm = mService.applicationContext.getSystemService(InputMethodManager::class.java)
            val currentInputMethodSubtype = imm?.currentInputMethodSubtype
            if (currentInputMethodSubtype != null) {
                val localeString = currentInputMethodSubtype.locale
                if (localeString.isNotEmpty()) {
                    return localeString // Use keyboard locale if available in the spell checker
                }
                // localeString for this app is always empty, get it from settings if possible
                if (currentInputMethodSubtype.extraValue == "dummy") { // make sure this app is used
                    return getSelectedSubtype(mService.prefs()).locale
                }
            }
        }

        // Fallback to system locale
        return super.getLocale()
    }

    override fun onClose() {
        mService.contentResolver.unregisterContentObserver(observer)
    }

    /**
     * Helper method to test valid capitalizations of a word.
     *
     *
     * If the "text" is lower-case, we test only the exact string.
     * If the "Text" is capitalized, we test the exact string "Text" and the lower-cased
     * version of it "text".
     * If the "TEXT" is fully upper case, we test the exact string "TEXT", the lower-cased
     * version of it "text" and the capitalized version of it "Text".
     */
    private fun isInDictForAnyCapitalization(text: String, capitalizeType: Int): Boolean {
        // If the word is in there as is, then it's in the dictionary. If not, we'll test lower
        // case versions, but only if the word is not already all-lower case or mixed case.
        if (mService.isValidWord(locale, text)) return true
        if (StringUtils.CAPITALIZE_NONE == capitalizeType) return false

        // If we come here, we have a capitalized word (either First- or All-).
        // Downcase the word and look it up again. If the word is only capitalized, we
        // tested all possibilities, so if it's still negative we can return false.
        val lowerCaseText = text.lowercase(locale)
        if (mService.isValidWord(locale, lowerCaseText)) return true
        if (StringUtils.CAPITALIZE_FIRST == capitalizeType) return false

        // If the lower case version is not in the dictionary, it's still possible
        // that we have an all-caps version of a word that needs to be capitalized
        // according to the dictionary. E.g. "GERMANS" only exists in the dictionary as "Germans".
        return mService.isValidWord(locale, StringUtils.capitalizeFirstAndDowncaseRest(lowerCaseText, locale))
    }

    // Note : this must be reentrant
    /**
     * Gets a list of suggestions for a specific string. This returns a list of possible
     * corrections for the text passed as an argument. It may split or group words, and
     * even perform grammatical analysis.
     */
    private fun onGetSuggestionsInternal(textInfo: TextInfo, suggestionsLimit: Int) =
        onGetSuggestionsInternal(textInfo, null, suggestionsLimit)

    protected fun onGetSuggestionsInternal(textInfo: TextInfo, ngramContext: NgramContext?, suggestionsLimit: Int): SuggestionsInfo {
        try {
            updateLocale()
            // It's good to keep this not local specific since the standard ones may show up in other languages also.
            var text = textInfo.text
                .replace(AndroidSpellCheckerService.APOSTROPHE.toRegex(), AndroidSpellCheckerService.SINGLE_QUOTE)
                .replace(quotesStartRegex, "")
                .replace(quotesEndRegex, "")

            scriptToPunctuationRegexMap[script]?.let { text = text.replace(it, "") }

            if (!mService.hasMainDictionaryForLocale(locale))
                return AndroidSpellCheckerService.getNotInDictEmptySuggestions(false)

            // Handle special patterns like email, URI, telephone number.
            val checkability = getCheckabilityInScript(text, script)
            val capitalizeType = StringUtils.getCapitalizationType(text)
            if (Checkabiliy.CHECKABLE != checkability)
                return getSuggestionInfoForUncheckable(text, capitalizeType, checkability)

            // Handle normal words.
            if (isInDictForAnyCapitalization(text, capitalizeType)) {
                if (DebugFlags.DEBUG_ENABLED) {
                    Log.i(TAG, "onGetSuggestionsInternal() : [$text] is a valid word")
                }
                return AndroidSpellCheckerService.getInDictEmptySuggestions()
            }
            if (DebugFlags.DEBUG_ENABLED) {
                Log.i(TAG, "onGetSuggestionsInternal() : [$text] is NOT a valid word")
            }

            // unknown word -> don't show suggestions if switched off
            if (!mService.prefs().getBoolean(Settings.PREF_SPELLCHECK_SUGGEST, Defaults.PREF_SPELLCHECK_SUGGEST))
                return AndroidSpellCheckerService.getTypoNoUiSuggestions()

            val keyboard = mService.getKeyboardForLocale(locale)
            val composer = WordComposer()
            if (locale.getLanguage() == "ko")
                composer.restartCombining("hangul")
            val codePoints = StringUtils.toCodePointArray(text)
            composer.setComposingWord(codePoints, keyboard.getCoordinates(codePoints))
            // TODO: Don't gather suggestions if the limit is <= 0 unless necessary
            val suggestionResults = mService
                .getSuggestionResults(locale, composer.composedDataSnapshot, ngramContext, keyboard)
            val result = getResult(capitalizeType, locale, suggestionsLimit, mService.recommendedThreshold, text, suggestionResults)
            if (DebugFlags.DEBUG_ENABLED && !result.mSuggestions.isNullOrEmpty()) {
                val builder = StringBuilder()
                for (suggestion in result.mSuggestions) {
                    builder.append(" [")
                    builder.append(suggestion)
                    builder.append("]")
                }
                Log.i(TAG, "onGetSuggestionsInternal() : Suggestions =$builder")
            }
            // Handle word not in dictionary.
            // This is called only once per unique word, so entering multiple instances of the same word does not
            // result in more than one call to this method.
            // Also, upon changing the orientation of the device, this is called again for every unique invalid word in the text box.
            StatsUtils.onInvalidWordIdentification(text)

            val flags = SuggestionsInfo.RESULT_ATTR_LOOKS_LIKE_TYPO or
                (if (result.mHasRecommendedSuggestions) SuggestionsInfo.RESULT_ATTR_HAS_RECOMMENDED_SUGGESTIONS else 0)
            val retval = SuggestionsInfo(flags, result.mSuggestions)
            suggestionsCache.putSuggestionsToCache(text, result.mSuggestions, flags)
            return retval
        } catch (e: RuntimeException) {
            // Don't kill the keyboard if there is a bug in the spell checker
            Log.e(TAG, "Exception while spellchecking", e)
            return AndroidSpellCheckerService.getNotInDictEmptySuggestions(false)
        }
    }

    private fun getSuggestionInfoForUncheckable(text: String, capitalizeType: Int, checkability: Checkabiliy): SuggestionsInfo {
        if (checkability == Checkabiliy.FIRST_LETTER_UNCHECKABLE || checkability == Checkabiliy.TOO_MANY_NON_LETTERS)
            return AndroidSpellCheckerService.getNotInDictEmptySuggestions(false)

        // Checkabiliy.CONTAINS_PERIOD Typo should not be reported when text is a valid word followed by a single period (end of sentence).
        val periodOnlyAtLastIndex = text.indexOf(Constants.CODE_PERIOD.toChar()) == (text.length - 1)
        if (checkability == Checkabiliy.CONTAINS_PERIOD) {
            val splitText = text.split(Constants.REGEXP_PERIOD.toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
            var allWordsAreValid = true
            // Validate all words on both sides of periods, skip empty tokens due to periods at first/last index
            for (word in splitText) {
                if (!word.isEmpty() && !mService.isValidWord(locale, word) && !mService.isValidWord(locale, word.lowercase(locale))) {
                    allWordsAreValid = false
                    break
                }
            }
            if (allWordsAreValid && !periodOnlyAtLastIndex) {
                return SuggestionsInfo(
                    SuggestionsInfo.RESULT_ATTR_LOOKS_LIKE_TYPO or SuggestionsInfo.RESULT_ATTR_HAS_RECOMMENDED_SUGGESTIONS,
                    arrayOf<String?>(TextUtils.join(Constants.STRING_SPACE, splitText))
                )
            }
        }
        return if (isInDictForAnyCapitalization(text, capitalizeType))
            AndroidSpellCheckerService.getInDictEmptySuggestions()
        else
            AndroidSpellCheckerService.getNotInDictEmptySuggestions(!periodOnlyAtLastIndex)
    }

    /*
    * The spell checker acts on its own behalf. That is needed, in particular, to be able to
    * access the dictionary files, which the provider restricts to the identity of Latin IME.
    * Since it's called externally by the application, the spell checker is using the identity
    * of the application by default unless we clearCallingIdentity.
    * That's what the following method does.
    */
    override fun onGetSuggestions(textInfo: TextInfo, suggestionsLimit: Int): SuggestionsInfo {
        val ident = Binder.clearCallingIdentity()
        try {
            return onGetSuggestionsInternal(textInfo, suggestionsLimit)
        } finally {
            Binder.restoreCallingIdentity(ident)
        }
    }

    companion object {
        private val TAG: String = AndroidWordLevelSpellCheckerSession::class.java.simpleName

        val EMPTY_STRING_ARRAY: Array<String?> = arrayOfNulls(0)

        private const val QUOTES_REGEX = "([\\u0022\\u0027\\u0060\\u00B4\\u2018\\u2018\\u201C\\u201D])"
        private val quotesStartRegex = "^$QUOTES_REGEX".toRegex()
        private val quotesEndRegex = "$QUOTES_REGEX$".toRegex()

        private val scriptToPunctuationRegexMap: MutableMap<String?, Regex> = TreeMap<String?, Regex>().apply {
            // TODO: add other non-English language specific punctuation later.
            this[ScriptUtils.SCRIPT_ARMENIAN] = "([\\u0028\\u0029\\u0027\\u2026\\u055E\\u055C\\u055B\\u055D\\u058A\\u2015\\u00AB\\u00BB\\u002C\\u0589\\u2024])".toRegex()
        }

        /**
         * Finds out whether a particular string should be filtered out of spell checking.
         *
         *
         * This will loosely match URLs, numbers, symbols. To avoid always underlining words that
         * we know we will never recognize, this accepts a script identifier that should be one
         * of the SCRIPT_* constants defined above, to rule out quickly characters from very
         * different languages.
         *
         * @param text the string to evaluate.
         * @param script the identifier for the script this spell checker recognizes
         * @return one of the FILTER_OUT_* constants above.
         */
        private fun getCheckabilityInScript(text: String?, script: String): Checkabiliy {
            if (text.isNullOrEmpty()) return Checkabiliy.TOO_SHORT
            if (text.length == 1)
                return if (isLetterPartOfScript(text.codePointAt(0), script)) Checkabiliy.TOO_SHORT
                else Checkabiliy.TOO_MANY_NON_LETTERS

            // TODO: check if an equivalent processing can't be done more quickly with a compiled regexp.
            // Filter by first letter
            val firstCodePoint = text.codePointAt(0)
            // Filter out words that don't start with a letter or an apostrophe
            if (!isLetterPartOfScript(firstCodePoint, script) && '\''.code != firstCodePoint)
                return Checkabiliy.FIRST_LETTER_UNCHECKABLE

            // Filter contents
            val length = text.length
            var letterCount = 0
            var i = 0
            while (i < length) {
                val codePoint = text.codePointAt(i)
                // Any word containing a COMMERCIAL_AT is probably an e-mail address
                // Any word containing a SLASH is probably either an ad-hoc combination of two
                // words or a URI - in either case we don't want to spell check that
                if (Constants.CODE_COMMERCIAL_AT == codePoint || Constants.CODE_SLASH == codePoint) {
                    return Checkabiliy.EMAIL_OR_URL
                }
                // If the string contains a period, native returns strange suggestions (it seems
                // to return suggestions for everything up to the period only and to ignore the
                // rest), so we suppress lookup if there is a period.
                // TODO: investigate why native returns these suggestions and remove this code.
                if (Constants.CODE_PERIOD == codePoint) {
                    return Checkabiliy.CONTAINS_PERIOD
                }
                if (isLetterPartOfScript(codePoint, script)) ++letterCount
                i = text.offsetByCodePoints(i, 1)
            }
            // Guestimate heuristic: perform spell checking if at least 3/4 of the characters
            // in this word are letters
            return if (letterCount * 4 < length * 3) Checkabiliy.TOO_MANY_NON_LETTERS else Checkabiliy.CHECKABLE
        }

        private fun getResult(
            capitalizeType: Int, locale: Locale, suggestionsLimit: Int, recommendedThreshold: Float,
            originalText: String?, suggestionResults: SuggestionResults
        ): Result {
            if (suggestionResults.isEmpty() || suggestionsLimit <= 0)
                return Result(null, false)
            val suggestionsSet = LinkedHashSet<String?>()
            for (suggestedWordInfo in suggestionResults) {
                val suggestion: String? = if (StringUtils.CAPITALIZE_ALL == capitalizeType) {
                    suggestedWordInfo.mWord.uppercase(locale)
                } else if (StringUtils.CAPITALIZE_FIRST == capitalizeType) {
                    StringUtils.capitalizeFirstCodePoint(suggestedWordInfo.mWord, locale)
                } else {
                    suggestedWordInfo.mWord
                }
                suggestionsSet.add(suggestion)
            }
            val suggestions = ArrayList(suggestionsSet)
            // This returns a String[], while toArray() returns an Object[] which cannot be cast
            // into a String[].
            val gatheredSuggestionsList = suggestions.subList(0, min(suggestions.size, suggestionsLimit))
            val gatheredSuggestions = gatheredSuggestionsList.toTypedArray<String?>()

            val bestScore = suggestionResults.first().mScore
            val bestSuggestion = suggestions[0]
            val normalizedScore = BinaryDictionaryUtils.calcNormalizedScore(originalText, bestSuggestion, bestScore)
            val hasRecommendedSuggestions = normalizedScore > recommendedThreshold
            return Result(gatheredSuggestions, hasRecommendedSuggestions)
        }
    }

    private class Result(val mSuggestions: Array<String?>?, val mHasRecommendedSuggestions: Boolean)

    private enum class Checkabiliy { CHECKABLE, TOO_MANY_NON_LETTERS, CONTAINS_PERIOD, EMAIL_OR_URL, FIRST_LETTER_UNCHECKABLE, TOO_SHORT }

    // todo: this is unused...
    protected class SuggestionsParams(val mSuggestions: Array<String?>?, val mFlags: Int)

    protected class SuggestionsCache {
        private val mUnigramSuggestionsInfoCache = LruCache<String?, SuggestionsParams?>(MAX_CACHE_SIZE)

        fun getSuggestionsFromCache(query: String?) = mUnigramSuggestionsInfoCache.get(query)

        fun putSuggestionsToCache(query: String?, suggestions: Array<String?>?, flags: Int) {
            if (suggestions == null || query.isNullOrEmpty()) {
                return
            }
            mUnigramSuggestionsInfoCache.put(query, SuggestionsParams(suggestions, flags))
        }

        fun clearCache() {
            mUnigramSuggestionsInfoCache.evictAll()
        }

        companion object {
            private const val MAX_CACHE_SIZE = 50
        }
    }
}
