package helium314.keyboard

import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.compat.CompatBreakIterator
import helium314.keyboard.latin.App
import helium314.keyboard.latin.common.WordBreakIterator
import helium314.keyboard.latin.common.stepsToCharCount
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import helium314.keyboard.latin.utils.TextAtCursor
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
class WordIteratorTest {
    // for testing default behavior and comparing with SpacingAndPunctuationsWordIterator
    @Test fun breakIteratorBehavior() {
        val default = CompatBreakIterator.getWordInstance(Locale.ROOT)
        val text = "hello this is me"
        default.text = text
        assertEquals(0, default.current())
        assertEquals(0, default.first())
        assertEquals(5, default.next())
        assertEquals(5, default.current())
        assertEquals(6, default.next())
        assertEquals(13, default.following(11))
        assertEquals(13, default.current())
        assertEquals(16, default.next(2))
        assertEquals(0, default.first())
        assertEquals(16, default.last())
        assertEquals(16, default.current())
        assertEquals(14, default.previous())
        assertEquals(-1, default.next(2))
        assertEquals(-1, default.next())
        assertEquals(14, default.previous()) // why?
        assertEquals(0, default.first())
        assertEquals(0, default.current())
        assertEquals(-1, default.previous())
        assertEquals(-1, default.previous())
        assertEquals(-1, default.next()) // why? compare with the why above...
        assertEquals(true, default.isBoundary(14))
        assertEquals(14, default.current())
        assertEquals(false, default.isBoundary(8))
        assertEquals(10, default.current()) // this is weird but expected from documentation

        val wbi = WordBreakIterator.getForSpacingAndPunctuations(SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT))
        wbi.text = text
        assertEquals(0, wbi.current())
        assertEquals(0, wbi.first())
        assertEquals(5, wbi.next())
        assertEquals(5, wbi.current())
        assertEquals(6, wbi.next())
        assertEquals(13, wbi.following(11))
        assertEquals(13, wbi.current())
        assertEquals(16, wbi.next(2))
        assertEquals(0, wbi.first())
        assertEquals(16, wbi.last())
        assertEquals(16, wbi.current())
        assertEquals(14, wbi.previous())
        assertEquals(-1, wbi.next(2))
        assertEquals(-1, wbi.next())
//        assertEquals(14, wbi.previous()) // better
        assertEquals(0, wbi.first())
        assertEquals(0, wbi.current())
        assertEquals(-1, wbi.previous())
        assertEquals(-1, wbi.previous())
        assertEquals(-1, wbi.next())
        assertEquals(true, wbi.isBoundary(14))
        assertEquals(-1, wbi.current()) // does it matter?
        assertEquals(false, wbi.isBoundary(8))
        assertEquals(-1, wbi.current()) // does it matter?
    }

    @Test fun whatDoWeWant() {
        val default = CompatBreakIterator.getWordInstance(Locale.ROOT)
        val wbi = WordBreakIterator.getForSpacingAndPunctuations(SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT))
        default.text = "don't "
        wbi.text = default.text
        assertEquals(5, default.following(2))
        assertEquals(5, wbi.following(2))
        default.text = "don'"
        wbi.text = default.text
        assertEquals(3, default.following(2))
        assertEquals(4, wbi.following(2))
        default.text = "'don't"
        wbi.text = default.text
        assertEquals(6, default.following(2))
        assertEquals(6, wbi.following(2))
        assertEquals(1, default.preceding(2))
        assertEquals(1, wbi.preceding(2))
        default.text = "'don'"
        wbi.text = default.text
        assertEquals(4, default.following(2))
        assertEquals(4, wbi.following(2))
        default.text = "hell-bent"
        wbi.text = default.text
        assertEquals(4, default.following(2))
        assertEquals(9, wbi.following(2))
    }

    @Test fun wordBoundaries() {
        val wbi = WordBreakIterator.getForSpacingAndPunctuations(SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT))
        wbi.text = "no boundary in don't!"
        assertEquals(20, wbi.following(15))
        assertEquals(15 to 20, wbi.getTouchedWordBounds(16))
        wbi.text = "2 boundaries in 'don'!"
        assertEquals(17, wbi.following(16))
        assertEquals(16 to 16, wbi.getTouchedWordBounds(16))
        assertEquals(20, wbi.following(17))
        assertEquals(17 to 20, wbi.getTouchedWordBounds(17))
        assertEquals(21, wbi.following(20))
        assertEquals(17, wbi.preceding(20))
        assertEquals(17 to 20, wbi.getTouchedWordBounds(20))
        assertEquals(21 to 21, wbi.getTouchedWordBounds(21))
        assertEquals(17 to 20, wbi.getTouchedWordBounds(19))
        wbi.text = "no boundary in don'!"
        assertEquals(19, wbi.following(15))
        assertEquals(15, wbi.preceding(19))
        assertEquals(15 to 19, wbi.getTouchedWordBounds(15))
        assertEquals(15 to 19, wbi.getTouchedWordBounds(17))
        assertEquals(15 to 19, wbi.getTouchedWordBounds(19))
    }

    @Test fun wordBoundaryDifferences() { // between WordBreakIterators
        val sp = WordBreakIterator.getForSpacingAndPunctuations(SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT))
        val it = CompatBreakIterator.getWordInstance(Locale.ROOT)
        val text = "ab1"
        it.text = text
        sp.text = text
        assertTrue(sp.isBoundary(2))
        assertFalse(it.isBoundary(2))
        assertEquals(0 to 2, sp.getTouchedWordBounds(0))
        assertEquals(0 to 3, it.getTouchedWordBounds(0))
        assertEquals(3 to 3, sp.getTouchedWordBounds(3))
        assertEquals(0 to 3, it.getTouchedWordBounds(3))

    }

    // todo: default and SpacingAndPunctuations iterators behave differently, what do we want?
    //  (plus in the end it should be customizable anyway)
    @Test fun wordStepsToCharCount() {
//        val it = WordBreakIterator.getForSpacingAndPunctuations(SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT))
        val it = CompatBreakIterator.getWordInstance(Locale.ROOT)
        assertEquals(-5, stepsToCharCount("hello", -1, it))
        assertEquals(5, stepsToCharCount("hello", 1, it))
        assertEquals(-3, stepsToCharCount("hello yes", -1, it))
        assertEquals(5, stepsToCharCount("hello yes", 1, it))
        assertEquals(-4, stepsToCharCount("hello yes", -2, it))
        assertEquals(6, stepsToCharCount("hello yes", 2, it))
        assertEquals(-9, stepsToCharCount("hello yes", -3, it))
        assertEquals(9, stepsToCharCount("hello yes", 3, it))
        assertEquals(-9, stepsToCharCount("hello yes", -4, it))
        assertEquals(9, stepsToCharCount("hello yes", 4, it))
        assertEquals(0, stepsToCharCount("", -3, it))
        assertEquals(0, stepsToCharCount("", 3, it))
        assertEquals(2, stepsToCharCount("  ", 1, it))
        assertEquals(-3, stepsToCharCount("111", -1, it)) // do we want that?
        assertEquals(3, stepsToCharCount("111", 1, it)) // do we want that?
        assertEquals(-2, stepsToCharCount("31!xr", -1, it)) // better not
        assertEquals(2, stepsToCharCount("31!xr", 1, it)) // better not
        assertEquals(1, stepsToCharCount("....", 1, it)) // better not
    }

    @Test fun `get touched text range`() {
        val sp = SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT)
        val spUrl = SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, true, Locale.ROOT)
        checkTextRange("blabla this is v", "ery good", sp, 15, 19)
        checkTextRange(".hel", "lo...", sp, 1, 6)
        checkTextRange("(hi", ")", sp, 1, 3)
        checkTextRange("", "word", sp, 0, 4)

        checkTextRange("mail: blorb@", "florb.com or", sp, 12, 17)
        checkTextRange("mail: blorb@", "florb.com or", spUrl, 6, 21)
        checkTextRange("mail: blor", "b@florb.com or", sp, 6, 11)
        checkTextRange("mail: blor", "b@florb.com or", spUrl, 6, 21)
        checkTextRange("mail: blorb@f", "lorb.com or", sp, 12, 17)
        checkTextRange("mail: blorb@f", "lorb.com or", spUrl, 6, 21)

        checkTextRange("http://exam", "ple.com", sp, 7, 14)
        checkTextRange("http://exam", "ple.com", spUrl, 7, 18)
        checkTextRange("http://example.", "com", sp, 15, 18)
        checkTextRange("http://example.", "com", spUrl, 7, 18)
        checkTextRange("htt", "p://example.com", sp, 0, 4)
        checkTextRange("htt", "p://example.com", spUrl, 0, 18)
        checkTextRange("http:/", "/example.com", sp, 6, 6)
        checkTextRange("http:/", "/example.com", spUrl, 0, 18)

        checkTextRange("..", ".", spUrl, 2, 2)
        checkTextRange("...", "", spUrl, 3, 3)

        val spCyrl = SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources,
            false, Locale.forLanguageTag("ru"))
        checkTextRange("", "э́є", spCyrl,0, 3)
        checkTextRange("э́", "є", spCyrl, 0, 3)
        checkTextRange("э", "́", spCyrl, 0, 2)

        // todo: these are bad cases of url detection
        //  also: sometimesWordConnectors are for URL and should be named accordingly
        checkTextRange("@@@", "@@@", spUrl, 0, 6)
        checkTextRange("a...", "", spUrl, 0, 4)
        checkTextRange("@@@", "", spUrl, 0, 3)
    }

    private fun checkTextRange(before: String, after: String, sp: SpacingAndPunctuations, wantedWordStart: Int, wantedWordEnd: Int) {
        val bi = WordBreakIterator.getForSpacingAndPunctuations(sp)
        bi.text = before + after
        val (start, end) = bi.getTouchedWordBounds(before.length)
        assertEquals(wantedWordStart, start)
        assertEquals(wantedWordEnd, end)
        // todo: better use the check above (does the same thing)
        //  but actually the whole check needs to be more more understandable and the tests expanded and the URL stuff fixed
        val text = TextAtCursor(before, after)
        val got = text.getWordAtCursor(sp)
        assertEquals(wantedWordStart, got.start)
        assertEquals(wantedWordEnd, got.end)
    }
}
