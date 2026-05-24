package helium314.keyboard

import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.utils.SuggestionResults
import helium314.keyboard.latin.utils.WordData
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import kotlin.test.Test

@RunWith(RobolectricTestRunner::class)
class SaveGestureDataTest {
    private val latinIME = Robolectric.setupService(LatinIME::class.java)

    @Test fun blockedWordIsFiltered() {
        val wd = wordData(suggestion("hello"), suggestion("ok", dict = "main"))
        assert(wd.filterSuggestions(listOf("hello")).none { it.mWord.equals("hello", true) })
        assert(!wd.isSavingOk(latinIME))
    }

    private fun suggestion(word: String, score: Int = 0, dict: String = "") =
        SuggestedWords.SuggestedWordInfo(word, "", score, 0, Dictionary.PhonyDictionary(dict), 0, 0)
    private fun wordData(vararg suggestions: SuggestedWords.SuggestedWordInfo) =
        WordData(
            null,
            SuggestionResults(18, false, false).apply {
                suggestions.forEach { add(it) }
            },
            cd, NgramContext.EMPTY_PREV_WORDS_INFO, kb, 0, false
        )
    private val cd = ComposedData(InputPointers(1), false, "")
    private val kb = Keyboard(KeyboardParams().apply {
        GRID_HEIGHT = 1
        GRID_WIDTH = 1
        mId = KeyboardLayoutSet.getFakeKeyboardId(0)
    })
}
