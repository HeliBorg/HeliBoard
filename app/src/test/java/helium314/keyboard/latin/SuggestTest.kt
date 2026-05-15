// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import androidx.core.content.edit
import helium314.keyboard.ShadowBinaryDictionaryUtils
import helium314.keyboard.ShadowInputMethodManager2
import helium314.keyboard.ShadowLocaleManagerCompat
import helium314.keyboard.event.Event
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_SHORTCUT
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_WHITELIST
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
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
    private val latinIME = Robolectric.setupService(LatinIME::class.java)
    private val suggest get() = latinIME.mInputLogic.mSuggest

    private val confidenceModest = 0.24f
    private val confidenceAggressive = 0.65f
    private val confidenceVeryAggressive = 0.9f

    init {
        ShadowLog.setupLogging()
        ShadowLog.stream = System.out
    }

    @BeforeTest fun reset() {
        latinIME.prefs().edit { clear() }
        currentTypingLocale = Locale.ENGLISH
        tapTypingSuggestions = suggestionResults(emptyList())
        glideTypingSuggestions = suggestionResults(emptyList())
        nextWordSuggestions = suggestionResults(emptyList())
    }

    @Test fun `'on' to 'in' if 'in' was used before in this context`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "on",
            listOf(suggestion("on", 1800000, locale), suggestion("in", 600000, locale)),
            suggestion("in", 240, locale),
            null, // never typed "on" in this context
            locale,
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceVeryAggressive
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
            confidenceModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `no English 'in' instead of French 'un' when typing in French`() {
        val result = shouldBeAutoCorrected(
            "un",
            listOf(suggestion("in", Int.MAX_VALUE, Locale.ENGLISH), suggestion("un", 1500000, Locale.FRENCH)),
            null,
            null,
            Locale.FRENCH,
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceModest
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
            confidenceAggressive
        )
        assert(result.last()) // should be corrected

        val result2 = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 300000, locale, true)),
            null,
            null,
            locale,
            confidenceModest
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
            confidenceAggressive
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `typed word is first suggestion`() { // first suggestion will not be shown to the user
        tapTypingSuggestions = suggestionResults(listOf(suggestion("hello", 123, currentTypingLocale)))
        val results = getSuggestedWords(false, "henlo", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("henlo", "hello"), results.mSuggestedWordInfoList.map { it.mWord })

        tapTypingSuggestions = suggestionResults(listOf(suggestion("hello", 123, currentTypingLocale)))
        val results2 = getSuggestedWords(false, "hello", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("hello"), results2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `typed word is second suggestion if autocorrect is pending`() {
        enableAutocorrect(confidenceVeryAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("hello", 650000, currentTypingLocale), // 600000 is the limit
            suggestion("hell", 620000, currentTypingLocale),
            suggestion("hem", 100000, currentTypingLocale),
        ))
        val results = getSuggestedWords(false, "henlo", WordComposer.CAPS_MODE_OFF)
        assert(results.mWillAutoCorrect)
        assertEquals(5, results.mSuggestedWordInfoList.size)
        assertEquals("henlo", results.mSuggestedWordInfoList[0].mWord) // typed word, not shown
        assertEquals("hello", results.mSuggestedWordInfoList[1].mWord) // autocorrection is first suggestion
        assertEquals("henlo", results.mSuggestedWordInfoList[2].mWord) // if autocorrection is pending, typed word is next suggestion
        assertEquals("hell", results.mSuggestedWordInfoList[3].mWord)
        assertEquals("hem", results.mSuggestedWordInfoList[4].mWord)
    }

    @Test fun `CenterSuggestionTextToEnter has typed text or autocorrect as first suggestion`() {
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("surge", 1000),
            suggestion("sure", 935),
            suggestion("siege", 925),
        ))
        val results = getSuggestedWords(false, "suge", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("suge", "surge", "sure", "siege"), results.mSuggestedWordInfoList.map { it.mWord })

        latinIME.prefs().edit { putBoolean(Settings.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER, true) }
        // first suggestion duplicated because the first element in the list is not shown to the user
        val results2 = getSuggestedWords(false, "suge", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("suge", "suge", "surge", "sure", "siege"), results2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `empty typed word gives next word suggestions`() {
        nextWordSuggestions = suggestionResults(listOf(
            suggestion("another", 24, currentTypingLocale),
            suggestion("next", 123, currentTypingLocale),
        ))
        val results = getSuggestedWords(false, "", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("next", "another"), results.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `single letter suggestions are suppressed dependent on score of other suggestions`() {
        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("a", 100),
            suggestion("next", 95), // more than 94% -> moved to top
        ))
        val results = getSuggestedWords(true, "", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("next", "a"), results.mSuggestedWordInfoList.map { it.mWord })
        assertEquals(2, results.mSuggestedWordInfoList.size)
        assertEquals(93, results.mSuggestedWordInfoList[1].mScore) // score is 93% of original

        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("a", 100),
            suggestion("next", 90),
        ))
        val results2 = getSuggestedWords(true, "", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("a", "next"), results2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `prefer next word suggestions`() {
        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("ab", 1000),
            suggestion("next", 935),
            suggestion("something", 925), // not used because less than 93%
        ))
        nextWordSuggestions = suggestionResults(listOf(
            suggestion("next", 180),
            suggestion("something", 175),
            suggestion("ab", 160), // not used because less than 170
        ))
        val results = getSuggestedWords(true, "", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("next", "ab", "something"), results.mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf(935, 1000, 925), results.mSuggestedWordInfoList.map { it.mScore })

        nextWordSuggestions = suggestionResults(listOf(
            suggestion("next", 180),
            suggestion("something", 175),
            suggestion("ab", 172), // now it's used
        ))
        assertEquals("ab", getSuggestedWords(true, "", WordComposer.CAPS_MODE_OFF).mSuggestedWordInfoList[0].mWord)
    }

    @Test fun `single quotes at end are attached to suggestions`() {
        // the single quote is a word connector, so it's part of the typed word if at the end (or in the middle, but we're not interested in that)
        // the suggestions are for the normal word without ', and then the ' is attached
        // this has weird effects, because there are suggestions where the ' does not fit
        //  e.g. for emoji suggestions and languages where ' isn't used in words
        // todo: improve the situation on this!
        //  disable for shortcuts?
        //  maybe also could be removed from word connectors for a bunch of languages

        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("ab", 1000),
            suggestion("next", 935),
            suggestion("something", 925),
            suggestion("🎄", 924),
        ))
        val results = getSuggestedWords(false, "someword'", WordComposer.CAPS_MODE_OFF)
        assertEquals(listOf("someword'", "ab'", "next'", "something'", "🎄'"), results.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `tap typing suggestions depend on typed word capitalization, ignoring automatic caps modes`() {
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        listOf(WordComposer.CAPS_MODE_OFF, WordComposer.CAPS_MODE_AUTO_SHIFTED, WordComposer.CAPS_MODE_AUTO_SHIFT_LOCKED).forEach { mode ->
            // not capitalized -> original suggestions
            assertEquals(listOf("but", "buy", "bit", "😢", "Butter"),
                getSuggestedWords(false, "but", mode).mSuggestedWordInfoList.map { it.mWord })

            // capitalized -> capitalized suggestions
            assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
                getSuggestedWords(false, "But", mode).mSuggestedWordInfoList.map { it.mWord })

            // more than one uppercase character, but not all -> original suggestions
            assertEquals(listOf("BuT", "but", "buy", "bit", "😢", "Butter"),
                getSuggestedWords(false, "BuT", mode).mSuggestedWordInfoList.map { it.mWord })

            // full uppercase typed word -> uppercase suggestions
            assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
                getSuggestedWords(false, "BUT", mode).mSuggestedWordInfoList.map { it.mWord })
        }
    }

    @Test fun `next word and glide typing suggestions use automatic caps modes`() {
        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))
        assertEquals(listOf("but", "buy", "bit", "😢", "Butter"),
            getSuggestedWords(true, "", WordComposer.CAPS_MODE_OFF).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
            getSuggestedWords(true, "", WordComposer.CAPS_MODE_AUTO_SHIFTED).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(true, "", WordComposer.CAPS_MODE_AUTO_SHIFT_LOCKED).mSuggestedWordInfoList.map { it.mWord })

        nextWordSuggestions = glideTypingSuggestions
        assertEquals(listOf("but", "buy", "bit", "😢", "Butter"),
            getSuggestedWords(false, "", WordComposer.CAPS_MODE_OFF).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
            getSuggestedWords(false, "", WordComposer.CAPS_MODE_AUTO_SHIFTED).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(false, "", WordComposer.CAPS_MODE_AUTO_SHIFT_LOCKED).mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `no autocorrect if more than one uppercase character in typed word, but not all uppercase`() {
        enableAutocorrect(confidenceVeryAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 650000),
            suggestion("bit", 620000),
            suggestion("buy", 500000),
            suggestion("😢", 100000),
            suggestion("Butter", 80000),
        ))

        val result = getSuggestedWords(false, "bur", WordComposer.CAPS_MODE_OFF)
        assert(result.mWillAutoCorrect)
        assertEquals(listOf("bur", "but", "bur", "bit", "buy", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        val result2 = getSuggestedWords(false, "Bur", WordComposer.CAPS_MODE_OFF)
        assert(result2.mWillAutoCorrect)
        assertEquals(listOf("Bur", "But", "Bur", "Bit", "Buy", "😢", "Butter"), result2.mSuggestedWordInfoList.map { it.mWord })

        val result3 = getSuggestedWords(false, "BUr", WordComposer.CAPS_MODE_OFF)
        assert(!result3.mWillAutoCorrect)
        assertEquals(listOf("BUr", "but", "bit", "buy", "😢", "Butter"), result3.mSuggestedWordInfoList.map { it.mWord })

        val result4 = getSuggestedWords(false, "BUR", WordComposer.CAPS_MODE_OFF)
        assert(result4.mWillAutoCorrect)
        assertEquals(listOf("BUR", "BUT", "BUR", "BIT", "BUY", "😢", "BUTTER"), result4.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `suggestions use manual caps modes`() {
        // added in https://github.com/HeliBorg/HeliBoard/pull/1807 / cb0eae695f0cbd061e5bbcc416d6e14d18d869d8
        // manual caps mode can be set any time, even if there are already suggestions
        // on the phone this will give different suggestions because there is different proximityInfo (coming from keyboard, which is different when shifted)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "but", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        assert(!result.mWillAutoCorrect)
        assertEquals(listOf("but", "But", "Buy", "Bit", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        assertEquals(listOf("but", "BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(false, "but", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED).mSuggestedWordInfoList.map { it.mWord })

        // same for glide typing
        glideTypingSuggestions = tapTypingSuggestions
        val result2 = getSuggestedWords(true, "", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        assert(!result2.mWillAutoCorrect)
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"), result2.mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(true, "", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED).mSuggestedWordInfoList.map { it.mWord })

        // same for next word suggestions
        nextWordSuggestions = tapTypingSuggestions
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
            getSuggestedWords(false, "", WordComposer.CAPS_MODE_MANUAL_SHIFTED).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(false, "", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED).mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `autocorrect will "correct" capitalization with manual caps modes`() {
        enableAutocorrect(confidenceAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "but", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        assert(result.mWillAutoCorrect)
        assertEquals(listOf("but", "But", "Buy", "Bit", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        val result2 = getSuggestedWords(false, "but", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED)
        assert(result2.mWillAutoCorrect)
        assertEquals(listOf("but", "BUT", "BUY", "BIT", "😢", "BUTTER"), result2.mSuggestedWordInfoList.map { it.mWord })

        glideTypingSuggestions = tapTypingSuggestions
        val result3 = getSuggestedWords(true, "", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        // mWillAutoCorrect is true on phone, because after glide typing is done, the result is set as typed word
        // when pressing shift to capitalize it, autocorrect will only work if the typed word changes when capitalized. this does not work without typed word
        //assert(result3.mWillAutoCorrect)
        assertEquals("but", result3.typedWordInfo.mWord)
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"), result3.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `no caps mode autocorrect if word has many uppercase letters`() {
        // bug report in https://github.com/HeliBorg/HeliBoard/issues/2162
        enableAutocorrect(confidenceAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "buT", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        assert(!result.mWillAutoCorrect)
        assertEquals(listOf("buT", "BuT", "But", "Buy", "Bit", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        val result2 = getSuggestedWords(false, "buT", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED)
        assert(!result2.mWillAutoCorrect)
        assertEquals(listOf("buT", "BUT", "BUY", "BIT", "😢", "BUTTER"), result2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `normal autocorrect works with manual caps modes`() {
        enableAutocorrect(confidenceAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 650000),
            suggestion("bit", 620000),
            suggestion("buy", 500000),
            suggestion("😢", 100000),
            suggestion("Butter", 80000),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "bur", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        assert(result.mWillAutoCorrect)
        assertEquals(listOf("bur", "But", "Bur", "Bit", "Buy", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        val result2 = getSuggestedWords(false, "bur", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED)
        assert(result2.mWillAutoCorrect)
        assertEquals(listOf("bur", "BUT", "BUR", "BIT", "BUY", "😢", "BUTTER"), result2.mSuggestedWordInfoList.map { it.mWord })

        // score too low will correct to capitalized typed word (except if using very aggressive threshold)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 6500),
            suggestion("bit", 6200),
            suggestion("buy", 5000),
            suggestion("😢", 1000),
            suggestion("Butter", 800),
        ))
        val result3 = getSuggestedWords(false, "bur", WordComposer.CAPS_MODE_MANUAL_SHIFTED)
        assert(result3.mWillAutoCorrect)
        assertEquals(listOf("bur", "Bur", "But", "Bit", "Buy", "😢", "Butter"), result3.mSuggestedWordInfoList.map { it.mWord })

        val result4 = getSuggestedWords(false, "bur", WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED)
        assert(result4.mWillAutoCorrect)
        assertEquals(listOf("bur", "BUR", "BUT", "BIT", "BUY", "😢", "BUTTER"), result4.mSuggestedWordInfoList.map { it.mWord })
    }

    private fun getSuggestedWords(gesture: Boolean, typedWord: String, shiftMode: Int): SuggestedWords {
        val wc = WordComposer()
        if (gesture) wc.setBatchInputPointers(InputPointers(1))
        if (typedWord.isNotEmpty()) {
            StringUtils.toCodePointArray(typedWord).forEach {
                val e = Event.createEventForCodePointFromAlreadyTypedText(it, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE)
                wc.applyProcessedEvent(wc.processEvent(e))
            }
        }
        // we ignore the special treatment of CAPS_MODE_AUTO_SHIFTED here (depends on settings, input type, and text before cursor)
        if (gesture)
            wc.setCapitalizedModeAtStartComposingTime(shiftMode) // done in inputlogic when entering batch mode
        wc.adviseCapitalizedModeBeforeFetchingSuggestions(shiftMode) // done in inputLogic before getSuggestedWords

        val params = KeyboardParams()
        val elementId = when (shiftMode) {
            WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED -> KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCKED
            WordComposer.CAPS_MODE_MANUAL_SHIFTED -> KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED
            WordComposer.CAPS_MODE_AUTO_SHIFTED -> KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED
            else -> KeyboardId.ELEMENT_ALPHABET // off
        }
        params.GRID_HEIGHT = 1
        params.GRID_WIDTH = 1
        params.mId = KeyboardLayoutSet.getFakeKeyboardId(elementId)

        suggest.clearNextWordSuggestionsCache()
        return suggest.getSuggestedWords(
            wc, NgramContext.EMPTY_PREV_WORDS_INFO, Keyboard(params), Settings.getValues().mSettingsValuesForSuggestion,
            Settings.getValues().mAutoCorrectEnabled, 0, 0
        )
    }

    private fun enableAutocorrect(autoCorrectThreshold: Float) {
        latinIME.prefs().edit {
            putBoolean(Settings.PREF_AUTO_CORRECTION, true)
            putFloat(Settings.PREF_AUTO_CORRECT_CONFIDENCE, autoCorrectThreshold)
            putBoolean(Settings.PREF_MORE_AUTO_CORRECTION, true)
        }
        suggest.setAutoCorrectionThreshold(Settings.getValues().mAutoCorrectionThreshold)
    }

    private fun shouldBeAutoCorrected(word: String, // typed word
                              suggestions: List<SuggestedWordInfo>, // suggestions ordered by score, including suggestion for typed word if in dictionary
                              firstSuggestionForEmpty: SuggestedWordInfo?, // first suggestion if typed word would be empty (null if none)
                              typedWordSuggestionForEmpty: SuggestedWordInfo?, // suggestion for actually typed word if typed word would be empty (null if none)
                              typingLocale: Locale, // used for checking whether suggestion locale is the same, relevant e.g. for English i -> I shortcut, but we want Polish i
                              autoCorrectThreshold: Float
    ): List<Boolean> {
        enableAutocorrect(autoCorrectThreshold)
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
private var tapTypingSuggestions = suggestionResults(emptyList())
private var glideTypingSuggestions = suggestionResults(emptyList())
private var nextWordSuggestions = suggestionResults(emptyList())

fun suggestion(word: String, score: Int, locale: Locale = currentTypingLocale, shortcut: Boolean = false) =
    SuggestedWordInfo(
        word,
        "", // irrelevant

        // typically IntMax for whitelisted, 1.5M for exact match, 600k for close match
        // when previous word context is empty, scores are usually 200+ if word is known and somewhat often used, 0 if unknown
        score,
        if (score == Int.MAX_VALUE) KIND_WHITELIST
            else if (shortcut) KIND_SHORTCUT // whitelist & shortcut only counts a whitelist
            else KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION, // shortcuts seem to never have this flag
        TestDict(locale),
        0, // irrelevant
        0 // irrelevant?
    )

fun suggestionResults(suggestions: List<SuggestedWordInfo>, isBeginningOfSentence: Boolean = false) =
    SuggestionResults(18, isBeginningOfSentence, false).apply { addAll(suggestions) }

@Implements(DictionaryFacilitatorImpl::class)
class ShadowFacilitator {
    @Implementation
    fun getCurrentLocale(): Locale = currentTypingLocale
    @Implementation
    fun getMainLocale(): Locale = currentTypingLocale
    @Implementation
    fun hasAtLeastOneInitializedMainDictionary() = true // otherwise no autocorrect
    @Implementation
    // what is relevant for facilitator?
    // only detecting whether we want next word suggestions (typed word empty and not batch mode)
    fun getSuggestionResults(composedData: ComposedData, ngramContext: NgramContext, keyboard: Keyboard,
                             settingsValuesForSuggestion: SettingsValuesForSuggestion, sessionId: Int, inputStyle: Int
    ): SuggestionResults = when {
        composedData.mIsBatchMode -> glideTypingSuggestions
        composedData.mTypedWord.isEmpty() -> nextWordSuggestions
        else -> tapTypingSuggestions
    }
}

private class TestDict(locale: Locale) : Dictionary(TYPE_MAIN, locale) {
    override fun getSuggestions(
        composedData: ComposedData?,
        ngramContext: NgramContext?,
        proximityInfoHandle: Long,
        settingsValuesForSuggestion: SettingsValuesForSuggestion?,
        sessionId: Int,
        weightForLocale: Float,
        inOutWeightOfLangModelVsSpatialModel: FloatArray?
    ): ArrayList<SuggestedWordInfo> {
        return ArrayList()
    }

    override fun isInDictionary(word: String?): Boolean = false
}
