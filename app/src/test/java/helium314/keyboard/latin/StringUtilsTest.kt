// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.ShadowInputMethodManager2
import helium314.keyboard.compat.CompatBreakIterator
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.common.codePointAt
import helium314.keyboard.latin.common.codePointBefore
import helium314.keyboard.latin.common.endsWithWordCodepoint
import helium314.keyboard.latin.common.isEmoji
import helium314.keyboard.latin.common.isSingleGrapheme
import helium314.keyboard.latin.common.nonWordCodePointAndNoSpaceBeforeCursor
import helium314.keyboard.latin.common.singleEmojiRegex
import helium314.keyboard.latin.common.splitOnWhitespace
import helium314.keyboard.latin.common.stripTrailingSeparatorsAndConnectors
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// many of those tests now actually test the iterators
@RunWith(RobolectricTestRunner::class)
@Config(shadows = [
    ShadowInputMethodManager2::class,
])
class StringUtilsTest {
    @Test fun `not inside double quotes without quotes`() {
        assertTrue(!StringUtils.isInsideDoubleQuoteOrAfterDigit("hello yes"))
    }

    @Test fun `inside double quotes after opening a quote`() {
        assertTrue(StringUtils.isInsideDoubleQuoteOrAfterDigit("hello \"yes"))
    }

    @Test fun `inside double quotes with quote at start`() {
        assertTrue(StringUtils.isInsideDoubleQuoteOrAfterDigit("\"hello yes"))
    }

    // maybe this is not that bad, should be correct after entering next text
    @Test fun `not inside double quotes directly after closing quote`() {
        assertTrue(!StringUtils.isInsideDoubleQuoteOrAfterDigit("hello \"yes\""))
    }

    @Test fun `not inside double quotes after closing quote`() {
        assertTrue(!StringUtils.isInsideDoubleQuoteOrAfterDigit("hello \"yes\" "))
    }

    @Test fun `not inside double quotes after closing quote followed by comma`() {
        assertTrue(!StringUtils.isInsideDoubleQuoteOrAfterDigit("hello \"yes\", "))
    }

    @Test fun `inside double quotes after opening another quote`() {
        assertTrue(StringUtils.isInsideDoubleQuoteOrAfterDigit("hello \"yes\" \"h"))
    }

    @Test fun `inside double quotes after opening another quote with closing quote followed by comma`() {
        assertTrue(StringUtils.isInsideDoubleQuoteOrAfterDigit("hello \"yes\", \"h"))
    }

    @Test fun `non-word codepoints and no space`() {
        val sp = SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT)
        assertTrue(!nonWordCodePointAndNoSpaceBeforeCursor("this is", sp))
        assertTrue(!nonWordCodePointAndNoSpaceBeforeCursor("this ", sp))
        assertTrue(!nonWordCodePointAndNoSpaceBeforeCursor("th.is ", sp))
        assertTrue(nonWordCodePointAndNoSpaceBeforeCursor("th.is", sp))
    }

    @Test fun `is word-like at end`() {
        val sp = SpacingAndPunctuations(ApplicationProvider.getApplicationContext<App>().resources, false, Locale.ROOT)
        assertTrue(!endsWithWordCodepoint("", sp))
        assertTrue(endsWithWordCodepoint("don'", sp))
        assertTrue(!endsWithWordCodepoint("hello!", sp))
        assertTrue(!endsWithWordCodepoint("when ", sp))
        assertTrue(!endsWithWordCodepoint("3-", sp))
        assertTrue(!endsWithWordCodepoint("5'", sp))
        assertTrue(!endsWithWordCodepoint("1", sp))
        assertTrue(endsWithWordCodepoint("a-", sp))
        assertTrue(!endsWithWordCodepoint("--", sp))
        assertTrue(!endsWithWordCodepoint("\uD83D\uDE42", sp))
    }

    @Test fun singleGrapheme() {
        assertTrue(!"".isSingleGrapheme)
        assertTrue(",".isSingleGrapheme)
        assertTrue(!"!!".isSingleGrapheme)
        assertTrue("э́".isSingleGrapheme)
        assertTrue(!"\uD83C\uDFF4\u200D☠️\uD83C\uDFF3️\u200D\uD83C\uDF08".isSingleGrapheme)
        assertTrue("\uD83C\uDFF3️\u200D\uD83C\uDF08".isSingleGrapheme)
        assertTrue("\uD83C\uDFF4\u200D☠️".isSingleGrapheme)
        assertTrue(!" 🏼".isSingleGrapheme)
        assertTrue(!"a🏼".isSingleGrapheme)
        assertTrue(!"🏼🏼".isSingleGrapheme)
        assertTrue("🏼".isSingleGrapheme)
        assertTrue("🇦".isSingleGrapheme) // 🇦 standalone regional indicator is a single grapheme
        assertTrue("🇩🇪".isSingleGrapheme) // 🇩🇪 flag (indicator pair) is still a single grapheme
        assertTrue(!"🇦🇧🇨".isSingleGrapheme) // 🇦🇧🇨 three indicators are not one grapheme
    }

    @Test fun detectEmojisAtEnd() {
        assertEquals("", getFullEmojiAtEnd("\uD83C\uDF83 "))
        assertEquals("", getFullEmojiAtEnd("a"))
        assertEquals("\uD83C\uDF83", getFullEmojiAtEnd("\uD83C\uDF83"))
        assertEquals("ℹ️", getFullEmojiAtEnd("ℹ️"))
        assertEquals("ℹ️", getFullEmojiAtEnd("ℹ️ℹ️"))
        assertEquals("\uD83D\uDE22", getFullEmojiAtEnd("x\uD83D\uDE22"))
        assertEquals("", getFullEmojiAtEnd("x\uD83D\uDE22 "))
        assertEquals("\uD83C\uDFF4\u200D☠️", getFullEmojiAtEnd("ok \uD83C\uDFF4\u200D☠️"))
        assertEquals("\uD83C\uDFF3️\u200D\uD83C\uDF08", getFullEmojiAtEnd("\uD83C\uDFF3️\u200D\uD83C\uDF08"))
        assertEquals("\uD83C\uDFF3️\u200D\uD83C\uDF08", getFullEmojiAtEnd("\uD83C\uDFF4\u200D☠️\uD83C\uDFF3️\u200D\uD83C\uDF08"))
        assertEquals("\uD83C\uDFF3️\u200D⚧️", getFullEmojiAtEnd("hello there🏳️‍⚧️"))
        assertEquals("\uD83D\uDD75\uD83C\uDFFC", getFullEmojiAtEnd(" 🕵🏼"))
        assertEquals("\uD83D\uDD75\uD83C\uDFFC", getFullEmojiAtEnd("🕵🏼"))
        assertEquals("\uD83C\uDFFC", getFullEmojiAtEnd(" \uD83C\uDFFC"))
        assertEquals("1\uFE0F⃣", getFullEmojiAtEnd("1\uFE0F⃣")) // 1️⃣
        assertEquals("©\uFE0F", getFullEmojiAtEnd("©\uFE0F")) // ©️
        assertEquals("", getFullEmojiAtEnd("\u200D"))
        assertEquals("", getFullEmojiAtEnd("a\u200D"))
        assertEquals("\uD83D\uDE22", getFullEmojiAtEnd(" \u200D\uD83D\uDE22"))
        assertEquals("🇨", getFullEmojiAtEnd("🇦🇧🇨"))
        assertEquals("🇦", getFullEmojiAtEnd(" 🇦"))
        assertEquals("🇦🇩", getFullEmojiAtEnd("🇦🇩"))
        assertEquals("\uD83C\uDFFC", getFullEmojiAtEnd("\uD83C\uDF84\uD83C\uDFFC")) // 🎄🏼
    }

    @Test fun detectEmojisAtStart() {
        assertEquals("\uD83C\uDF83", getFullEmojiAtStart("\uD83C\uDF83 "))
        assertEquals("", getFullEmojiAtStart("a"))
        assertEquals("\uD83C\uDF83", getFullEmojiAtStart("\uD83C\uDF83"))
        assertEquals("ℹ️", getFullEmojiAtStart("ℹ️"))
        assertEquals("ℹ️", getFullEmojiAtStart("ℹ️ℹ️"))
        assertEquals("\uD83D\uDE22", getFullEmojiAtStart("\uD83D\uDE22x"))
        assertEquals("", getFullEmojiAtStart("x\uD83D\uDE22 "))
        assertEquals("\uD83C\uDFF4\u200D☠️", getFullEmojiAtStart("\uD83C\uDFF4\u200D☠️ ok"))
        assertEquals("\uD83C\uDFF3️\u200D\uD83C\uDF08", getFullEmojiAtStart("\uD83C\uDFF3️\u200D\uD83C\uDF08"))
        assertEquals("\uD83C\uDFF4\u200D☠️", getFullEmojiAtStart("\uD83C\uDFF4\u200D☠️\uD83C\uDFF3️\u200D\uD83C\uDF08"))
        assertEquals("\uD83C\uDFF3️\u200D⚧️", getFullEmojiAtStart("🏳️‍⚧️hello there"))
        assertEquals("\uD83D\uDD75\uD83C\uDFFC", getFullEmojiAtStart("🕵🏼 "))
        assertEquals("\uD83D\uDD75\uD83C\uDFFC", getFullEmojiAtStart("🕵🏼"))
        assertEquals("\uD83C\uDFFC", getFullEmojiAtStart("\uD83C\uDFFC "))
        assertEquals("1\uFE0F⃣", getFullEmojiAtStart("1\uFE0F⃣")) // 1️⃣
        assertEquals("©\uFE0F", getFullEmojiAtStart("©\uFE0F")) // ©️
        assertEquals("", getFullEmojiAtStart("\u200D"))
        assertEquals("", getFullEmojiAtStart("\u200Da"))
        assertEquals("🇦🇧", getFullEmojiAtStart("🇦🇧🇨"))
        assertEquals("🇨", getFullEmojiAtStart("🇨\uD83C\uDF83"))
        assertEquals("🇦", getFullEmojiAtStart("🇦 "))
        assertEquals("🇦🇩", getFullEmojiAtStart("🇦🇩🇨🇩"))
        assertEquals("\uD83C\uDF84", getFullEmojiAtStart("\uD83C\uDF84\uD83C\uDFFC")) // 🎄🏼
    }

    @Test fun detectEmojiAtStartFails() {
        if (BuildConfig.BUILD_TYPE == "runTests") return
        // fails because first grapheme contains \u200D, which sort of makes sense... but we don't want that
        assertEquals("😢", getFullEmojiAtStart("😢\u200D "))
    }

    @Test fun isEmojiDetectsSingleEmojis() {
        assertTrue(isEmoji("🎄"))
        assertTrue(!isEmoji("🎄🎄"))
        assertTrue(!isEmoji("🎄🏼"))
        assertTrue(isEmoji("🏼")) // actually this is not a standalone emoji...
        assertTrue(!isEmoji("a🎄"))
        assertTrue(isEmoji("🖐️"))
        assertTrue(isEmoji("🖐🏾"))
        assertTrue(!isEmoji("🖐🏾🏼"))
        assertTrue(isEmoji("🇦🇧")) // needs special treatment because 🇦🇧 is not an emoji according to the regex (it's not valid, but it's displayed)
    }

    @Test fun isEmojiDetectsAllAvailableEmojis() {
        val ctx = ApplicationProvider.getApplicationContext<App>()
        val allEmojis = ctx.assets.list("emoji")!!.flatMap {
            if (it == "minApi.txt" || it == "EMOTICONS.txt") return@flatMap emptyList()
            ctx.assets.open("emoji/$it").reader().readLines()
        }.flatMap { it.splitOnWhitespace() }

        val brokenDetectionAtStart = listOf("〰️", "〽️", "©️", "®️", "#️⃣", "*️⃣", "0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "㊗️", "㊙️")
        allEmojis.forEach {
            assertTrue(isEmoji(it))
            assertTrue(StringUtils.mightBeEmoji(it.codePointBefore(it.length)))
            if (it !in brokenDetectionAtStart)
                assertTrue(StringUtils.mightBeEmoji(it.codePointAt(0)))
        }
    }

    @Test fun `strip trailing separators and connectors`() {
        val ctx = ApplicationProvider.getApplicationContext<App>()
        val svfs = SpacingAndPunctuations(ctx.resources, false, Locale.ROOT)
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word", svfs))
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word)", svfs))
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word\"", svfs))
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word:", svfs))
        assertEquals("wor:d", stripTrailingSeparatorsAndConnectors("wor:d:", svfs))
        assertEquals("", stripTrailingSeparatorsAndConnectors(":", svfs))
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word-", svfs))
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word'", svfs))
        assertEquals("word", stripTrailingSeparatorsAndConnectors("word'-'", svfs))
    }

    private fun getFullEmojiAtStart(text: String): String {
        val gi = CompatBreakIterator.graphemeInstance
        gi.text = text
        val firstGrapheme = text.substring(0, gi.following(0))
        return if (isEmoji(firstGrapheme)) firstGrapheme else ""
    }

    private fun getFullEmojiAtEnd(text: CharSequence): String {
        val gi = CompatBreakIterator.graphemeInstance
        gi.text = text
        val lastGrapheme = text.substring(gi.preceding(text.length))
        return if (isEmoji(lastGrapheme)) lastGrapheme else ""
    }
}
