package helium314.keyboard

import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.compat.CompatBreakIterator
import helium314.keyboard.latin.App
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.common.WordBreakIterator
import helium314.keyboard.latin.common.lastGraphemeLength
import helium314.keyboard.latin.common.stepsToCharCount
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
class CharacterIteratorTest {
    // not really a test, but helps finding what we want
    @Test fun compareIterators() {
        val c = CompatBreakIterator.characterInstance
        val g = CompatBreakIterator.graphemeInstance
        val w = CompatBreakIterator.getWordInstance(Locale.ROOT)
        val s = WordBreakIterator.getForSpacingAndPunctuations(SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT))
        fun setText(text: String) {
            c.text = text
            g.text = text
            w.text = text
            s.text = text
        }
        fun forAll(check: (CompatBreakIterator) -> Unit) {
            check(c)
            check(g)
            check(w)
            check(s)
        }
        setText("\uD83C\uDF83 ")
        forAll { assertEquals(2, it.following(0)) }
        setText("\uD83C\uDF83")
        forAll { assertEquals(2, it.following(1)) }
        setText("ℹ️ℹ️")
        forAll { assertEquals(0, it.preceding(2)) }
        assertEquals(2, c.following(0))
        assertEquals(2, g.following(0))
        assertEquals(4, w.following(0))
        assertEquals(2, s.following(0))
        assertEquals(2, c.preceding(4))
        assertEquals(2, g.preceding(4))
        assertEquals(0, w.preceding(4))
        assertEquals(2, s.preceding(4))
        setText("x\uD83D\uDE22")
        forAll {
            assertEquals(1, it.next())
            assertEquals(3, it.next())
        }
        setText("ok \uD83C\uDFF4\u200D☠️")
        assertTrue(c.isBoundary(1))
        assertTrue(g.isBoundary(1))
        assertFalse(w.isBoundary(1))
        assertFalse(s.isBoundary(1))
        forAll { assertEquals(3, it.preceding(6)) }
        setText("\uD83C\uDFF3️\u200D\uD83C\uDF08")
        forAll { assertEquals(6, it.following(0)) }
        forAll { assertEquals(0, it.preceding(4)) }
        setText("\uD83C\uDFF4\u200D☠️\uD83C\uDFF3️\u200D\uD83C\uDF08")
        forAll { assertEquals(5, it.following(3)) }
        forAll { assertEquals(0, it.preceding(3)) }
        forAll { assertEquals(11, it.following(5)) }
        forAll { assertEquals(0, it.preceding(5)) }
        forAll { assertEquals(11, it.following(7)) }
        forAll { assertEquals(5, it.preceding(8)) }
        forAll { assertEquals(11, it.following(8)) }
        forAll { assertEquals(5, it.preceding(8)) }
        forAll { assertEquals(11, it.following(9)) }
        forAll { assertEquals(5, it.preceding(9)) }
        setText("hello there🏳️‍⚧️")
        forAll { assertEquals("hello there🏳️‍⚧️".length, it.following(11)) }
        forAll { assertEquals(11, it.preceding("hello there🏳️‍⚧️".length)) }
        assertEquals(10, c.preceding(c.preceding(13)))
        assertEquals(10, g.preceding(g.preceding(13)))
        assertEquals(6, w.preceding(w.preceding(13)))
        assertEquals(6, s.preceding(s.preceding(13)))
        setText("🕵🏼")
        forAll { assertEquals(4, it.following(1)) }
        setText("\uD83C\uDFFC")
        forAll { assertEquals(0, it.preceding(1)) }
        setText("1\uFE0F⃣") // 1️⃣
        forAll { assertEquals(0, it.preceding(3)) }
        forAll { assertEquals(3, it.following(0)) }
        forAll { assertFalse(it.isBoundary(1)) }
        setText("©\uFE0F") // ©️
        forAll { assertEquals(0, it.preceding(2)) }
        forAll { assertEquals(2, it.following(0)) }
        forAll { assertFalse(it.isBoundary(1)) }
        setText(" \u200D\uD83D\uDE22") //  ‍?
        assertEquals(2, c.following(0))
        assertEquals(2, g.following(0)) // todo: should it space + u200D really be one grapheme?
        assertEquals(4, w.following(0))
        assertEquals(2, s.following(0))
        assertEquals(2, c.preceding(4))
        assertEquals(2, g.preceding(4))
        assertEquals(0, w.preceding(4))
        assertEquals(2, s.preceding(4))
        setText("\uD83C\uDF84\uD83C\uDFFC") // 🎄?
        assertEquals(4, c.following(0))
        assertEquals(2, g.following(0))
        assertEquals(4, w.following(0))
        assertEquals(2, s.following(0))
        assertEquals(0, c.preceding(4))
        assertEquals(2, g.preceding(4))
        assertEquals(0, w.preceding(4))
        assertEquals(2, s.preceding(4))
        setText("😢\u200D")
        forAll { assertEquals(3, it.following(0)) } // maybe u200D should not be included? but it's ZWJ...

        setText(" 🏼")
        assertEquals(3, c.following(0))
        assertEquals(1, g.following(0))
        assertEquals(3, w.following(0))
        assertEquals(1, s.following(0))
        assertEquals(0, c.preceding(3))
        assertEquals(1, g.preceding(3))
        assertEquals(0, w.preceding(3))
        assertEquals(1, s.preceding(3))
        setText(" 🇦") // interestingly that's different than single skin tone
        forAll { assertEquals(1, it.following(0)) }
        forAll { assertEquals(1, it.preceding(3)) }
        setText("🏼a")
        forAll { assertEquals(2, it.following(0)) }
        forAll { assertEquals(2, it.preceding(3)) }
        setText("🇦a")
        forAll { assertEquals(2, it.following(0)) }
        forAll { assertEquals(2, it.preceding(3)) }
        setText("🇦🇨🇧")
        forAll { assertEquals(4, it.following(0)) }
        forAll { assertEquals(4, it.preceding(6)) }
        setText("🇦🇧🇨")
        forAll { assertEquals(4, it.following(0)) }
        forAll { assertEquals(4, it.following(1)) }
        forAll { assertEquals(4, it.preceding(6)) }
        forAll { assertEquals(4, it.preceding(5)) }
        setText("\uD83C\uDF84\uD83C\uDFFC") // 🎄🏼
        assertEquals(4, c.following(0))
        assertEquals(2, g.following(0))
        assertEquals(4, w.following(0))
        assertEquals(2, s.following(0))
    }

    @Test fun characterStepsToCharCount() {
        val it = CompatBreakIterator.graphemeInstance
        assertEquals(3, stepsToCharCount("abcd", 3, it))
        assertEquals(-3, stepsToCharCount("abcd", -3, it))
        assertEquals(4, stepsToCharCount("abcd", 10, it))
        assertEquals(-4, stepsToCharCount("abcd", -10, it))
        assertEquals(8, stepsToCharCount("\uD83C\uDFF3️\u200D\uD83C\uDF08bcd", 3, it))
        assertEquals(-3, stepsToCharCount("\uD83C\uDFF3️\u200D\uD83C\uDF08bcd", -3, it))
        assertEquals(4, stepsToCharCount("aэ́cd", 3, it))
        assertEquals(-2, stepsToCharCount("abcэ́", -1, it))
        assertEquals(19, stepsToCharCount("H̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅэ́cd", 1, it))
        assertEquals(21, stepsToCharCount("H̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅэ́cd", 2, it))
        assertEquals(22, stepsToCharCount("H̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅэ́cd", 3, it))
        assertEquals(23, stepsToCharCount("H̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅэ́cd", 4, it))
        assertEquals(-2, stepsToCharCount("aH̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅcэ́", -1, it))
        assertEquals(-3, stepsToCharCount("aH̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅcэ́", -2, it))
        assertEquals(-23, stepsToCharCount("aH̵̛͕̞̦̰̜͍̰̥̟͆̏͂̌͑́ͅcэ́", -4, it))
        assertEquals(3, stepsToCharCount("abcd", 3, it))
        assertEquals(5, stepsToCharCount("\uD83C\uDFF4\u200D☠️\uD83C\uDFF3️\u200D\uD83C\uDF08", 1, it))
        assertEquals(-6, stepsToCharCount("\uD83C\uDFF4\u200D☠️\uD83C\uDFF3️\u200D\uD83C\uDF08", -1, it))
    }

    @Test fun lastGraphemeFails() {
        if (BuildConfig.BUILD_TYPE == "runTests") return
        assertEquals("🏼".length, "🏼🏼🏼".lastGraphemeLength)
    }

    @Test fun firstGraphemeFails() {
        if (BuildConfig.BUILD_TYPE == "runTests") return
        assertEquals("🏼", "🏼🏼🏼".firstGrapheme)
    }

    private val String.firstGrapheme: String get() {
        if (length <= 1) return this
        val iterator = CompatBreakIterator.graphemeInstance
        iterator.text = this
        return substring(0, iterator.following(0))
    }
}
