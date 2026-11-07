package helium314.keyboard.latin

import android.content.Context
import com.android.inputmethod.keyboard.ProximityInfo
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.utils.SuggestionResults
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito
import org.robolectric.RobolectricTestRunner
import java.util.ArrayList
import java.util.Locale

@RunWith(RobolectricTestRunner::class)
class DictionaryFacilitatorImplTest {

    private val localeEn = Locale.ENGLISH
    private val localeIt = Locale.ITALIAN

    @Test
    fun `Strategy A - Previous word how (English) locks context to English`() {
        val prevWord = "how"
        val typedWord = "are"
        val italianNoise = "area" 

        val facilitator = setupFacilitator(
            dictEn = listOf("how", "are"),
            dictIt = listOf("are", "area")
        )

        val prevWordInfo = NgramContext.WordInfo(prevWord)
        val ngramContext = NgramContext(prevWordInfo)

        val results = getSuggestions(facilitator, typedWord, ngramContext)

        assertTrue("Should contain English 'are'", results.any { it.word == "are" })
        assertEquals("Should filter out Italian noise 'area'", 
            0, results.filter { it.word == italianNoise }.size)
    }

    @Test
    fun `Strategy A - Next Word Prediction Hello to how`() {
        // User typed "Hello " (Space pressed).
        // Current Input: "" (Empty).
        // Context: "Hello".
        // Expectation: English dict contains "Hello", so it locks to English and shows "how".
        
        val prevWord = "Hello"
        val typedWord = "" // Empty because space was pressed
        val nextWordSuggestion = "how"
        val italianNoise = "come"

        val facilitator = setupFacilitator(
            dictEn = listOf("Hello", "how"), 
            dictIt = listOf("come") // Italian does NOT have Hello
        )

        val prevWordInfo = NgramContext.WordInfo(prevWord)
        val ngramContext = NgramContext(prevWordInfo)

        val results = getSuggestions(facilitator, typedWord, ngramContext)

        assertTrue("Should suggest 'how' based on context 'Hello'", 
            results.any { it.word == nextWordSuggestion })
            
        assertEquals("Should filter out Italian noise 'come'", 
            0, results.filter { it.word == italianNoise }.size)
    }

    @Test
    fun `Strategy A - Case Insensitive Context Capitalized The matches lowercase the`() {
        val prevWord = "The" 
        val typedWord = "end"
        val italianNoise = "endo"

        val facilitator = setupFacilitator(
            dictEn = listOf("the", "end"), 
            dictIt = listOf("endo")
        )

        val prevWordInfo = NgramContext.WordInfo(prevWord)
        val ngramContext = NgramContext(prevWordInfo)

        val results = getSuggestions(facilitator, typedWord, NgramContext.EMPTY_PREV_WORDS_INFO) // FIX: Use EMPTY here or update getSuggestions logic to use NgramContext if your code relies on it. 
        // Actually, wait, your code uses ngramContext.getNthPrevWord(1). 
        // So passing ngramContext here is required.
        val resultsWithContext = getSuggestions(facilitator, typedWord, ngramContext)

        assertTrue("Should match 'The' to 'the' and allow English suggestions", 
            resultsWithContext.any { it.word == "end" })
        assertEquals("Should filter out Italian noise", 
            0, resultsWithContext.filter { it.word == italianNoise }.size)
    }

    @Test
    fun `Edge Case - Dictionary Overlap The Hello Problem`() {
        // User types "Hel". Matches start of "Hello" (English) and "Helio" (Italian).
        // BUT "Hello" exists in BOTH dictionaries (Simulating bad dict).
        // Expectation: NO filtering.
        
        val typedWord = "Hel"
        val englishWord = "Hello"
        val italianNoise = "Helio" // Changed from "Bello" to match "Hel" prefix

        val facilitator = setupFacilitator(
            dictEn = listOf("Hello"),
            dictIt = listOf("Hello", "Helio")
        )

        val results = getSuggestions(facilitator, typedWord, NgramContext.EMPTY_PREV_WORDS_INFO)

        assertTrue("Should contain English 'Hello'", results.any { it.word == englishWord })
        assertTrue("Should preserve Italian 'Helio' because 'Hello' is ambiguous", 
            results.any { it.word == italianNoise })
    }

    @Test
    fun `Edge Case - No Match The Give Problem`() {
        // User types "Gi". 
        // English has "Givenchy". Italian has "Gia".
        // "Gi" is NOT an exact match for either.
        // Expectation: NO filtering.
        val typedWord = "Gi"
        val englishApprox = "Givenchy"
        val italianApprox = "Gia"

        val facilitator = setupFacilitator(
            dictEn = listOf("Givenchy"),
            dictIt = listOf("Gia")
        )

        val results = getSuggestions(facilitator, typedWord, NgramContext.EMPTY_PREV_WORDS_INFO)

        assertTrue("Should show English approx", results.any { it.word == englishApprox })
        assertTrue("Should show Italian approx", results.any { it.word == italianApprox })
    }

    @Test
    fun `Simulate Firefox - Main Dict fails Noise is hidden`() {
        // Strict Filtering Test
        // Context: "the" (English).
        // English Dict: Returns NOTHING for input "G" (Simulating failure).
        // Italian Dict: Returns "Già" (Noise).
        // Expectation: Empty Result (Strict filtering hides Italian).

        val prevWord = "the"
        val typedWord = "G" 
        
        val facilitator = setupFacilitator(
            dictEn = listOf("the"),
            dictIt = listOf("Già")
        )

        val prevWordInfo = NgramContext.WordInfo(prevWord)
        val ngramContext = NgramContext(prevWordInfo)

        val results = getSuggestions(facilitator, typedWord, ngramContext)

        assertEquals("Should be EMPTY. English failed, Italian filtered out.",
            0, results.size)
    }

    // --- Helper Methods ---

    private fun getSuggestions(
        facilitator: DictionaryFacilitatorImpl, 
        typedWord: String, 
        ngramContext: NgramContext
    ): SuggestionResults {
        val composedData = ComposedData(InputPointers(1), false, typedWord)
        val keyboard = Mockito.mock(Keyboard::class.java)
        val proximityInfo = Mockito.mock(ProximityInfo::class.java)
        Mockito.`when`(keyboard.proximityInfo).thenReturn(proximityInfo)
        val settings = Mockito.mock(SettingsValuesForSuggestion::class.java)

        return facilitator.getSuggestionResults(
            composedData, ngramContext, keyboard, settings, 0, SuggestedWords.INPUT_STYLE_TYPING
        )
    }

    private fun setupFacilitator(dictEn: List<String>, dictIt: List<String>): DictionaryFacilitatorImpl {
        val facilitator = DictionaryFacilitatorImpl()
        
        val stubEn = StubDictionary("main_en", localeEn, dictEn)
        val stubIt = StubDictionary("main_it", localeIt, dictIt)

        val dictGroupClass = Class.forName("helium314.keyboard.latin.DictionaryGroup")
        val dictGroupConstructor = dictGroupClass.getDeclaredConstructor(
            Locale::class.java, Dictionary::class.java, Map::class.java, Context::class.java
        )
        dictGroupConstructor.isAccessible = true

        val groupEn = dictGroupConstructor.newInstance(localeEn, stubEn, emptyMap<String, Any>(), null)
        val groupIt = dictGroupConstructor.newInstance(localeIt, stubIt, emptyMap<String, Any>(), null)

        val groupsField = facilitator.javaClass.getDeclaredField("dictionaryGroups")
        groupsField.isAccessible = true
        groupsField.set(facilitator, listOf(groupEn, groupIt))

        return facilitator
    }
}

// Smart Stub Dictionary
class StubDictionary(type: String, locale: Locale, private val wordList: List<String>) : Dictionary(type, locale) {
    override fun getSuggestions(
        composedData: ComposedData?, ngramContext: NgramContext?, proximityInfoHandle: Long, 
        settingsValuesForSuggestion: SettingsValuesForSuggestion?, sessionId: Int, weightForLocale: Float, 
        inOutWeightOfLangModelVsSpatialModel: FloatArray?
    ): ArrayList<SuggestedWordInfo> {
        val list = ArrayList<SuggestedWordInfo>()
        val typedWord = composedData?.mTypedWord ?: ""

        wordList.forEach { word ->
            // If typedWord is empty (Next Word Prediction) OR word starts with typedWord
            if (typedWord.isEmpty() || word.startsWith(typedWord, ignoreCase = true)) {
                list.add(SuggestedWordInfo(word, "", 1000, SuggestedWordInfo.KIND_TYPED, this, 0, 0))
            }
        }
        return list
    }

    override fun isValidWord(word: String?): Boolean {
        return wordList.any { it.equals(word, ignoreCase = true) }
    }

    override fun isInDictionary(word: String?): Boolean {
        return isValidWord(word)
    }
}