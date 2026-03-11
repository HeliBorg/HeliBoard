// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import androidx.core.content.edit
import helium314.keyboard.ShadowBinaryDictionaryUtils
import helium314.keyboard.ShadowInputMethodManager2
import helium314.keyboard.ShadowLocaleManagerCompat
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_SHORTCUT
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_WHITELIST
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.utils.DeviceProtectedUtils
import helium314.keyboard.latin.utils.SuggestionResults
import helium314.keyboard.latin.utils.prefs
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import org.robolectric.shadows.ShadowLog
import java.util.*
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

@Suppress("NonAsciiCharacters")
@RunWith(RobolectricTestRunner::class)
@Config(shadows = [
    ShadowLocaleManagerCompat::class,
    ShadowInputMethodManager2::class,
    ShadowBinaryDictionaryUtils::class,
    ShadowFacilitator::class,
])
class SuggestTest {
    private lateinit var latinIME: LatinIME
    private val suggest get() = latinIME.mInputLogic.mSuggest

    // values taken from the string array auto_correction_threshold_mode_indexes
    private val thresholdModest = 0.185f
    private val thresholdAggressive = 0.067f
    private val thresholdVeryAggressive = -1f

    @BeforeTest fun setUp() {
        latinIME = Robolectric.setupService(LatinIME::class.java)
        // start logging only after latinIME is created, avoids showing the stack traces if library is not found
        ShadowLog.setupLogging()
        ShadowLog.stream = System.out
        DeviceProtectedUtils.getSharedPreferences(latinIME)
            .edit { putBoolean(Settings.PREF_AUTO_CORRECTION, true) } // need to enable, off by default
    }

    @Test fun `'on' to 'in' if 'in' was used before in this context`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "on",
            listOf(suggestion("on", 1800000, locale), suggestion("in", 600000, locale)),
            suggestion("in", 240, locale),
            null, // never typed "on" in this context
            locale,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because first suggestion score is too low
    }

    @Test fun `'ill' to 'I'll' if 'ill' not used before in this context, and I'll is whitelisted`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            null,
            null,
            locale,
            thresholdModest
        )
        assert(result.last()) // should be corrected
        // correction because both empty scores are 0, which should be fine (next check is comparing empty scores)
    }

    @Test fun `not 'ill' to 'I'll' if only 'ill' was used before in this context`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            null,
            suggestion("ill", 200, locale),
            locale,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because first empty score not high enough
    }

    @Test fun `'ill' to 'I'll' if both have same ngram score`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            suggestion("I'll", 200, locale),
            suggestion("ill", 200, locale),
            locale,
            thresholdModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `no 'ill' to 'I'll' if 'ill' has somewhat better ngram score`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            suggestion("I'll", 200, locale),
            suggestion("ill", 211, locale),
            locale,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `no English 'I' for Polish 'i' when typing in Polish`() {
        val result = shouldBeAutoCorrected(
            "i",
            listOf(suggestion("I", Int.MAX_VALUE, Locale.ENGLISH), suggestion("i", 1500000, Locale("pl"))),
            null,
            null,
            Locale("pl"),
            thresholdVeryAggressive
        )
        assert(!result.last()) // should not be corrected
        // not even checking at modest and aggressive thresholds, this is a locale thing
        // if very aggressive, still no correction because locale matches with typed word only
    }

    @Test fun `English 'I' instead of Polish 'i' when typing in English`() {
        val result = shouldBeAutoCorrected(
            "i",
            listOf(suggestion("I", Int.MAX_VALUE, Locale.ENGLISH), suggestion("i", 1500000, Locale("pl"))),
            null,
            null,
            Locale.ENGLISH,
            thresholdModest
        )
        assert(result.last()) // should be corrected
        // only corrected because it's whitelisted (int max value)
        // if it wasn't whitelisted, it would never be allowed due to utoCorrectionUtils.suggestionExceedsThreshold (unless set to very aggressive)
        //  -> maybe normalizedScore needs adjustment if the only difference is upper/lowercase
        //     todo: consider special score for case-only difference?
    }

    @Test fun `no English 'in' instead of French 'un' when typing in French`() {
        val result = shouldBeAutoCorrected(
            "un",
            listOf(suggestion("in", Int.MAX_VALUE, Locale.ENGLISH), suggestion("un", 1500000, Locale.FRENCH)),
            null,
            null,
            Locale.FRENCH,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because of locale matching
    }

    @Test fun `no 'né' instead of 'ne'`() {
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, Locale.FRENCH), suggestion("né", 1900000-1, Locale.FRENCH)),
            null,
            null,
            Locale.FRENCH,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because score is lower
    }

    @Test fun `'né' instead of 'ne' if 'né' in ngram context`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, locale), suggestion("né", 1900000-1, locale)),
            suggestion("né", 200, locale),
            null,
            locale,
            thresholdModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `'né' instead of 'ne' if 'né' has clearly better score in ngram context`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, locale), suggestion("né", 1900000-1, locale)),
            suggestion("né", 215, locale),
            suggestion("ne", 200, locale),
            locale,
            thresholdModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `no 'né' instead of 'ne' if both with same score in ngram context`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, locale), suggestion("né", 1900000-1, locale)),
            suggestion("né", 200, locale),
            suggestion("ne", 200, locale),
            locale,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `no 'ne' instead of 'né'`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "né",
            listOf(suggestion("ne", 600000, locale), suggestion("né", 1600000, locale)),
            suggestion("né", 200, locale),
            suggestion("ne", 200, locale),
            locale,
            thresholdModest
        )
        assert(!result.last()) // should not be corrected
        // not even allowed to check because of low score for ne
    }

    @Test fun `shortcuts might be autocorrected by default`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 700000, locale, true)),
            null,
            null,
            locale,
            thresholdAggressive
        )
        assert(result.last()) // should be corrected

        val result2 = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 300000, locale, true)),
            null,
            null,
            locale,
            thresholdModest
        )
        assert(!result2.last()) // should not be corrected
    }

    @Test fun `shortcuts are not autocorrected when setting is off`() {
        val prefs = latinIME.prefs()
        prefs.edit { putBoolean(Settings.PREF_AUTOCORRECT_SHORTCUTS, false) }
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 12000000, locale, true)),
            null,
            null,
            locale,
            thresholdAggressive
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `quotes are added to suggestions when needed`() {
        val result = Suggest.getTransformedSuggestedWordInfo(suggestion("word", 1, Locale.ENGLISH, true),
            Locale.ENGLISH, false, false, 1)
        assertEquals("word'", result.mWord)
    }

    private fun createMockFacilitatorWithValidWords(vararg validWords: String): DictionaryFacilitator {
        val mock = org.mockito.Mockito.mock(DictionaryFacilitator::class.java)
        org.mockito.Mockito.`when`(mock.mainLocale).thenReturn(Locale.ENGLISH)
        validWords.forEach { word ->
            org.mockito.Mockito.`when`(mock.isValidSpellingWord(word)).thenReturn(true)
        }
        return mock
    }

    /**
     * Test helper for concatenated word splitting
     * @param input The typed word with accidental bottom-row char instead of space (e.g., "hellobthere")
     * @param validWords Words to mark as valid in the mock dictionary
     * @param expectedSuggestionCount Expected number of suggestions added (0 or 1)
     * @param expectedSuggestion The expected suggestion text if count > 0 (e.g., "hello there")
     * @param firstOccurrence Position of typed word in existing suggestions (-1 = not found/invalid, >=0 = already valid)
     */
    private fun testConcatenatedSplit(input: String, validWords: Array<String>,
                                      expectedSuggestionCount: Int, expectedSuggestion: String? = null,
                                      firstOccurrence: Int = -1) {
        val mockFacilitator = createMockFacilitatorWithValidWords(*validWords)
        val testSuggest = Suggest(mockFacilitator)
        val suggestions = ArrayList<SuggestedWordInfo>()
        testSuggest.tryAddConcatenatedWordSuggestions(input, suggestions, firstOccurrence)

        assertEquals(expectedSuggestionCount, suggestions.size)
        if (expectedSuggestion != null) {
            assertEquals(expectedSuggestion, suggestions[0].mWord)
        }
    }

    @Test fun `all bottom row chars trigger split`() {
        testConcatenatedSplit("hellobthere", arrayOf("hello", "there"), 1, "hello there")
        testConcatenatedSplit("goodntimes", arrayOf("good", "times"), 1, "good times")
        testConcatenatedSplit("lovevlife", arrayOf("love", "life"), 1, "love life")
        testConcatenatedSplit("bigcdog", arrayOf("big", "dog"), 1, "big dog")
        testConcatenatedSplit("somemday", arrayOf("some", "day"), 1, "some day")
    }

    @Test fun `concatenated words with multiple possible splits - only first valid`() {
        testConcatenatedSplit("hellomworld", arrayOf("hello", "world"), 1, "hello world")
    }

    @Test fun `no split if typed word already in dictionary`() {
        // "hellobthere" is already valid (e.g., custom dictionary compound word)
        // firstOccurrence=0 means it's found in suggestions at position 0
        testConcatenatedSplit("hellobthere", arrayOf("hello", "there", "hellobthere"), 0, firstOccurrence = 0)
    }

    @Test fun `no split if only one part is valid word`() {
        // "hello" is valid but "there" is not (e.g., typing in mixed languages)
        testConcatenatedSplit("hellobthere", arrayOf("hello"), 0)
    }

    @Test fun `minimum word length boundaries`() {
        // Works: 2 chars on each side (minimum)
        testConcatenatedSplit("atbcat", arrayOf("at", "cat"), 1, "at cat")
        testConcatenatedSplit("catbat", arrayOf("cat", "at"), 1, "cat at")

        // Fails: less than 2 chars before or after split
        testConcatenatedSplit("abcat", arrayOf("a", "cat"), 0)
        testConcatenatedSplit("catba", arrayOf("cat", "a"), 0)
    }

    @Test fun `no split for strings of bottom row chars only`() {
        testConcatenatedSplit("bvncm", arrayOf("b", "v", "n", "c", "m"), 0)
    }

    @Test fun `no split for very short strings`() {
        testConcatenatedSplit("ab", arrayOf("a", "b"), 0)
        testConcatenatedSplit("abc", arrayOf("a", "b", "c"), 0)
        testConcatenatedSplit("abcd", arrayOf("ab", "cd"), 0)
    }

    @Test fun `split requires exactly 2 chars on each side minimum`() {
        testConcatenatedSplit("thebcat", arrayOf("the", "cat"), 1, "the cat")
    }

    @Test fun `no false positive - words containing bottom row chars are not split`() {
        // "abacus" contains 'c' but should not split to "aba us"
        testConcatenatedSplit("abacus", arrayOf("abacus", "aba", "us"), 0, firstOccurrence = 0)
    }

    @Test fun `no false positive - abacus not split when valid`() {
        testConcatenatedSplit("abacus", arrayOf("abacus"), 0, firstOccurrence = 0)
    }

    @Test fun `no false positive - banish contains ban but should not split`() {
        testConcatenatedSplit("banish", arrayOf("banish", "ban", "ish"), 0, firstOccurrence = 0)
    }

    @Test fun `no false positive - combat contains com and bat`() {
        testConcatenatedSplit("combat", arrayOf("combat", "com", "bat"), 0, firstOccurrence = 0)
    }

    @Test fun `no false positive - mania contains bottom row chars`() {
        testConcatenatedSplit("mania", arrayOf("mania", "ma", "ia"), 0, firstOccurrence = 0)
    }

    @Test fun `split momscabacus to moms abacus`() {
        testConcatenatedSplit("momscabacus", arrayOf("moms", "abacus"), 1, "moms abacus")
    }

    @Test fun `split bannmermaids to ban mermaids`() {
        testConcatenatedSplit("bannmermaids", arrayOf("ban", "mermaids"), 1, "ban mermaids")
    }

    @Test fun `split beetlevmania to beetle mania`() {
        testConcatenatedSplit("beetlevmania", arrayOf("beetle", "mania"), 1, "beetle mania")
    }

    @Test fun `only first split for multiple concatenated words`() {
        // "thebboyboughtnthembasketball" would ideally be "the boy bought the basketball"
        // but algorithm only splits at first valid bottom-row char, giving "the boyboughtnthembasketball"
        testConcatenatedSplit("thebboyboughtnthembasketball",
            arrayOf("the", "boyboughtnthembasketball"), 1, "the boyboughtnthembasketball")
    }

    private fun shouldBeAutoCorrected(word: String, // typed word
                              suggestions: List<SuggestedWordInfo>, // suggestions ordered by score, including suggestion for typed word if in dictionary
                              firstSuggestionForEmpty: SuggestedWordInfo?, // first suggestion if typed word would be empty (null if none)
                              typedWordSuggestionForEmpty: SuggestedWordInfo?, // suggestion for actually typed word if typed word would be empty (null if none)
                              typingLocale: Locale, // used for checking whether suggestion locale is the same, relevant e.g. for English i -> I shortcut, but we want Polish i
                              autoCorrectThreshold: Float
    ): List<Boolean> {
        latinIME.prefs().edit { putFloat(Settings.PREF_AUTO_CORRECT_THRESHOLD, autoCorrectThreshold) }
        // enable "more autocorrect" so we actually have autocorrect even though we don't set a compatible input type
        latinIME.prefs().edit { putBoolean(Settings.PREF_MORE_AUTO_CORRECTION, true) }
        currentTypingLocale = typingLocale
        val suggestionsContainer = ArrayList<SuggestedWordInfo>().apply { addAll(suggestions) }
        val suggestionResults = SuggestionResults(suggestions.size, false, false)
        suggestions.forEach { suggestionResults.add(it) }

        // store the original SuggestedWordInfo for typed word, as it will be removed
        // we may want to re-add it in case auto-correction happens, so that the original word can at least be selected
        val typedWordFirstOccurrenceWordInfo: SuggestedWordInfo? = suggestionsContainer.firstOrNull { it.mWord == word }

        val firstOccurrenceOfTypedWordInSuggestions =
            SuggestedWordInfo.removeDupsAndTypedWord(word, suggestionsContainer)

        return suggest.shouldBeAutoCorrected(
            StringUtils.getTrailingSingleQuotesCount(word),
            word,
            suggestionsContainer.firstOrNull(), // todo: get from suggestions? mostly it's just removing the typed word, right?
            { firstSuggestionForEmpty to typedWordSuggestionForEmpty },
            true, // doesn't make sense otherwise
            WordComposer.getComposerForTest(false),
            suggestionResults,
            firstOccurrenceOfTypedWordInSuggestions,
            typedWordFirstOccurrenceWordInfo
        ).toList()
    }
}

private var currentTypingLocale = Locale.ENGLISH

fun suggestion(word: String, score: Int, locale: Locale, shortcut: Boolean = false) =
    SuggestedWordInfo(
        /* word */ word,
        /* prevWordsContext */ "", // irrelevant

        // typically 2B for whitelisted, 1.5M for exact match, 600k for close match
        // when previous word context is empty, scores are usually 200+ if word is known and somewhat often used, 0 if unknown
        /* score */ score,

        /* kindAndFlags */ if (score == Int.MAX_VALUE) KIND_WHITELIST
            else if (shortcut) KIND_SHORTCUT // whitelist & shortcut only counts a whitelist
            else KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION, // shortcuts seem to never have this flag
        /* sourceDict */ TestDict(locale),
        /* indexOfTouchPointOfSecondWord */ 0, // irrelevant
        /* autoCommitFirstWordConfidence */ 0 // irrelevant?
    )

@Implements(DictionaryFacilitatorImpl::class)
class ShadowFacilitator {
    @Implementation
    fun getCurrentLocale(): Locale = currentTypingLocale
    @Implementation
    fun hasAtLeastOneInitializedMainDictionary() = true // otherwise no autocorrect
}

private class TestDict(locale: Locale) : Dictionary("testDict", locale) {
    override fun getSuggestions(
        composedData: ComposedData?,
        ngramContext: NgramContext?,
        proximityInfoHandle: Long,
        settingsValuesForSuggestion: SettingsValuesForSuggestion?,
        sessionId: Int,
        weightForLocale: Float,
        inOutWeightOfLangModelVsSpatialModel: FloatArray?
    ): ArrayList<SuggestedWordInfo> {
        TODO("Not yet implemented")
    }

    override fun isInDictionary(word: String?): Boolean {
        TODO("Not yet implemented")
    }

}
