// SPDX-License-Identifier: GPL-3.0-only AND APACHE 2.0

package helium314.keyboard.compat

import android.os.Build
import androidx.annotation.RequiresApi
import helium314.keyboard.latin.common.LocaleUtils
import helium314.keyboard.latin.common.WordBreakIterator
import helium314.keyboard.latin.common.codePointAt
import helium314.keyboard.latin.common.codePointBefore
import helium314.keyboard.latin.common.isRegionalIndicator
import helium314.keyboard.latin.common.singleEmojiRegex
import java.util.Locale
import kotlin.text.substring

/**
 *  BreakIterator that can use android.icu.text.BreakIterator, java.text.BreakIterator or custom implementations.
 *  Method names are related to boundaries, e.g. [first] gets the first boundary.
 */
interface CompatBreakIterator {
    /**
     *  Somewhat weird thing, gets the "current" boundary, which is determined by previous method calls that
     *  might be called unexpectedly internally. So don't count on this returning the same after calling e.g. [isBoundary].
     *  Further, the initial value before setting a text can be anything (because we reuse instances).
     */
    fun current(): Int
    fun first(): Int
    fun last(): Int
    fun following(offset: Int): Int
    fun preceding(offset: Int): Int
    fun next() = following(current())
    fun previous() = preceding(current())
    fun next(n: Int): Int {
        if (n > 0) repeat(n) { next() }
        else repeat(n) { previous() }
        return current()
    }
    fun isBoundary(offset: Int): Boolean
    var text: CharSequence

    companion object {
        // from https://github.com/chattymin/Pebble/blob/main/pebble/src/main/java/com/chattymin/pebble/LocalBreakIterator.kt, Apache-2.0 license
        // there is more potentially useful code like String.graphemeLength (should be graphemeCount though)
        // ThreadLocal stuff copied from above, maybe not actually necessary
        //  icu BreakIterator.GetInstance calls iterator.clone under the hood, is it too slow? but we have many short-lived instances otherwise
        private val characterIterator = ThreadLocal<CompatBreakIterator>() // maybe we want to use by locale later?

        /** gets an android.icu.text.BreakIterator if available, otherwise java.text.BreakIterator */
        // maybe also for locale if we have the need
        val characterInstance: CompatBreakIterator get() {
            characterIterator.get()?.let { return it }
            val bi = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) IcuBreakIterator(android.icu.text.BreakIterator.getCharacterInstance())
            else OldBreakIterator(java.text.BreakIterator.getCharacterInstance())
            characterIterator.set(bi)
            return bi
        }

        private val graphemeIterator = ThreadLocal<CompatBreakIterator>()
        /** similar to [characterInstance], but with special treatment of standalone regional indicators and skin tones */
        val graphemeInstance: CompatBreakIterator get() {
            graphemeIterator.get()?.let { return it }
            val bi = GraphemeIterator()
            graphemeIterator.set(bi)
            return bi
        }

        private val wordIterators = ThreadLocal<HashMap<Locale, WordBreakIterator>>()

        /** gets a WordInstance of android.icu.text.BreakIterator if available, otherwise java.text.BreakIterator */
        fun getWordInstance(locale: Locale): WordBreakIterator {
            wordIterators.get()?.get(locale)?.let { return it }
            val bi = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) IcuBreakIterator(android.icu.text.BreakIterator.getWordInstance(locale))
            else OldBreakIterator(java.text.BreakIterator.getWordInstance(locale))
            if (wordIterators.get() == null)
                wordIterators.set(hashMapOf(locale to bi))
            else wordIterators.get()!![locale] = bi
            return bi
        }

        fun getAvailableLocales() =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) android.icu.text.BreakIterator.getAvailableLocales()
            else java.text.BreakIterator.getAvailableLocales()

        private val availableLocales = mutableListOf<Locale>()
        fun isLocaleAvailable(locale: Locale) = locale in availableLocales || getAvailableLocales().any {
            if (LocaleUtils.getMatchLevel(locale, it) >= LocaleUtils.LOCALE_GOOD_MATCH) {
                availableLocales.add(it)
                return@any true
            }
            false
        }
    }
}

private class OldBreakIterator(val bi: java.text.BreakIterator) : WordBreakIterator {
    override fun current() = bi.current()
    override fun first() = bi.first()
    override fun following(offset: Int) = bi.following(offset)
    override fun last() = bi.last()
    override fun next() = bi.next()
    override fun next(n: Int) = bi.next(n)
    override fun previous() = bi.previous()
    override fun preceding(offset: Int) = bi.preceding(offset)
    override fun isBoundary(offset: Int) = bi.isBoundary(offset)
    override var text: CharSequence = ""
        set(value) {
            field = value
            bi.setText(value.toString())
        }
}

@RequiresApi(Build.VERSION_CODES.N)
private class IcuBreakIterator(val bi: android.icu.text.BreakIterator) : WordBreakIterator {
    override fun current() = bi.current()
    override fun first() = bi.first()
    override fun following(offset: Int) = bi.following(offset)
    override fun last() = bi.last()
    override fun next() = bi.next()
    override fun next(n: Int) = bi.next(n)
    override fun previous() = bi.previous()
    override fun preceding(offset: Int) = bi.preceding(offset)
    override fun isBoundary(offset: Int) = bi.isBoundary(offset)
    override var text: CharSequence = ""
        set(value) {
            field = value
            bi.setText(value.toString())
        }
}

private class GraphemeIterator : CompatBreakIterator {
    private val bi = CompatBreakIterator.characterInstance

    override var text: CharSequence = ""
        set(value) {
            field = value
            bi.text = text
            index = 0
        }

    private var index = 0

    override fun following(offset: Int): Int {
        val i = if (bi.current() == offset) bi.next() else bi.following(offset) // Looking at Android source, next is likely faster than following
        // e.g. " 🏼" is detected as single grapheme, but we don't want this
        val d83c = text.indexOfLastBefore(i, offset + 1) { it == '\uD83C' } // start of many emojis, including skin tones and regional indicators
        if (d83c == -1) {
            index = i
            return i
        }

        val offsetForEmojiCheck = if (bi.isBoundary(offset)) offset else bi.preceding(offset)
        if (i - offsetForEmojiCheck == 4 && text.codePointAt(offsetForEmojiCheck).isRegionalIndicator && text.codePointBefore(i).isRegionalIndicator) {
            index = i
            return i
        }
        index = if (singleEmojiRegex.matches(text.substring(offsetForEmojiCheck, i))) i
        else d83c.coerceAtLeast(offset + 1)
        return index
    }

    override fun preceding(offset: Int): Int {
        val i = if (bi.current() == offset) bi.previous() else bi.preceding(offset)
        val d83c = text.indexOfFirstAfter(i, offset - 1) { it == '\uD83C' }
        if (d83c == -1) {
            index = i
            return i
        }
        val offsetForEmojiCheck = if (bi.isBoundary(offset)) offset else bi.following(offset)
        if (offsetForEmojiCheck - i == 4 && text.codePointAt(i).isRegionalIndicator && text.codePointBefore(offsetForEmojiCheck).isRegionalIndicator) {
            index = i
            return i
        }
        index = if (singleEmojiRegex.matches(text.substring(i, offsetForEmojiCheck))) i
        else d83c
        return index
    }

    override fun isBoundary(offset: Int): Boolean {
        if (bi.isBoundary(offset)) return true
        // not a boundary -> do the same check as for following / preceding
        return preceding(offset + 1) == offset // if offset == length bi.isBoundary should have returned true, so we don't check
    }

    override fun current() = index // bi.current() might be different for those special case emojis
    override fun first() = bi.first().also { index = it }
    override fun last() = bi.last().also { index = it }
    override fun next() = following(index)
    override fun previous() = preceding(index)

    companion object {
        // both methods below are modified from kotlin source, APACHE 2.0
        // Copyright 2010-2026 JetBrains s.r.o. and Kotlin Programming Language contributors.
        private inline fun CharSequence.indexOfFirstAfter(i: Int, maxIdx: Int, predicate: (Char) -> Boolean): Int {
            for (index in i + 1..maxIdx) {
                if (predicate(this[index])) {
                    return index
                }
            }
            return -1
        }

        private inline fun CharSequence.indexOfLastBefore(i: Int, minIdx: Int, predicate: (Char) -> Boolean): Int {
            for (index in i - 1 downTo minIdx) {
                if (predicate(this[index])) {
                    return index
                }
            }
            return -1
        }
    }
}
