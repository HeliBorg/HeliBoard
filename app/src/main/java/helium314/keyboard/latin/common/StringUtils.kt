// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.common

import helium314.keyboard.latin.common.StringUtils.mightBeEmoji
import helium314.keyboard.latin.common.StringUtils.newSingleCodePointString
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import helium314.keyboard.latin.utils.ScriptUtils
import helium314.keyboard.latin.utils.SpacedTokens
import helium314.keyboard.latin.utils.SpannableStringUtils
import helium314.keyboard.latin.utils.TextRange
import java.math.BigInteger
import java.text.BreakIterator
import java.util.Locale
import kotlin.math.max
import kotlin.text.indexOfFirst

fun CharSequence.codePointAt(offset: Int) = Character.codePointAt(this, offset)
fun CharSequence.codePointBefore(offset: Int) = Character.codePointBefore(this, offset)

/** Loops over the codepoints in [text]. Exits when [loop] returns true */
inline fun loopOverCodePoints(text: CharSequence, loop: (cp: Int, charCount: Int) -> Boolean) {
    var offset = 0
    while (offset < text.length) {
        val cp = text.codePointAt(offset)
        val charCount = Character.charCount(cp)
        if (loop(cp, charCount)) return
        offset += charCount
    }
}

/** Loops backwards over the codepoints in [text]. Exits when [loop] returns true */
inline fun loopOverCodePointsBackwards(text: CharSequence, loop: (cp: Int, charCount: Int) -> Boolean) {
    var offset = text.length
    while (offset > 0) {
        val cp = text.codePointBefore(offset)
        val charCount = Character.charCount(cp)
        if (loop(cp, charCount)) return
        offset -= charCount
    }
}

fun nonWordCodePointAndNoSpaceBeforeCursor(text: CharSequence, spacingAndPunctuations: SpacingAndPunctuations): Boolean {
    var space = false
    var nonWordCodePoint = false
    loopOverCodePointsBackwards(text) { cp, _ ->
        if (!space && Character.isWhitespace(cp)) space = true
        // treat double quote like a word codepoint for this function (not great, maybe clarify name or extend list of chars?)
        if (!nonWordCodePoint && !spacingAndPunctuations.isWordCodePoint(cp) && cp != '"'.code) {
            nonWordCodePoint = true
        }
        space && nonWordCodePoint // stop if both are found
    }
    return nonWordCodePoint && !space // return true if a non-word codepoint and no space was found
}

fun hasLetterBeforeLastSpaceBeforeCursor(text: CharSequence): Boolean {
    loopOverCodePointsBackwards(text) { cp, _ ->
        if (Character.isWhitespace(cp)) return false
        else if (Character.isLetter(cp)) return true
        false // continue
    }
    return false
}

/** get the complete emoji at end of [text], considering that emojis can be joined with ZWJ resulting in different emojis */
// todo: this is now only used for tests, do we actually need it?
fun getFullEmojiAtEnd(text: CharSequence): String {
    val lastGrapheme = text.toString().lastGrapheme
    return if (isEmoji(lastGrapheme)) lastGrapheme else ""
}

/**
 *  Returns whether the [text] ends with word codepoint, ignoring all word connectors.
 *  If the [text] is empty (after ignoring word connectors), the method returns false.
 */
fun endsWithWordCodepoint(text: String, spacingAndPunctuations: SpacingAndPunctuations): Boolean {
    if (text.isEmpty()) return false
    var codePoint = Constants.NOT_A_CODE
    loopOverCodePointsBackwards(text) { cp, _ ->
        val isNotWordConnector = !spacingAndPunctuations.isWordConnector(cp)
        if (isNotWordConnector)
            codePoint = cp
        isNotWordConnector
    }
    return codePoint != Constants.NOT_A_CODE && spacingAndPunctuations.isWordCodePoint(codePoint)
}

// todo: simplify... maybe compare with original code?
// todo: this breaks at e.g. э́, but should not
fun getTouchedWordRange(before: CharSequence, after: CharSequence, script: String, spacingAndPunctuations: SpacingAndPunctuations): TextRange {
    // Going backward, find the first breaking point (separator)
    var startIndexInBefore = before.length
    var endIndexInAfter = -1 // todo: clarify why might we want to set it when checking before
    loopOverCodePointsBackwards(before) { codePoint, cpLength ->
        if (!isPartOfCompositionForScript(codePoint, spacingAndPunctuations, script)) {
            if (Character.isWhitespace(codePoint) || !spacingAndPunctuations.mCurrentLanguageHasSpaces)
                return@loopOverCodePointsBackwards true
            // continue to the next whitespace and see whether this contains a sometimesWordConnector
            for (i in startIndexInBefore - 1 downTo 0) {
                val c = before[i]
                if (spacingAndPunctuations.isSometimesWordConnector(c.code)) {
                    // if yes -> whitespace is the index
                    startIndexInBefore = max(StringUtils.charIndexOfLastWhitespace(before).toDouble(), 0.0).toInt()
                    val firstSpaceAfter = StringUtils.charIndexOfFirstWhitespace(after)
                    endIndexInAfter = if (firstSpaceAfter == -1) after.length else firstSpaceAfter - 1
                    return@loopOverCodePointsBackwards true
                } else if (Character.isWhitespace(c)) {
                    // if no, just break normally
                    return@loopOverCodePointsBackwards true
                }
            }
            return@loopOverCodePointsBackwards true
        }
        startIndexInBefore -= cpLength
        false
    }

    // Find last word separator after the cursor
    if (endIndexInAfter == -1) {
        endIndexInAfter = 0
        loopOverCodePoints(after) { codePoint, cpLength ->
            if (!isPartOfCompositionForScript(codePoint, spacingAndPunctuations, script)) {
                if (Character.isWhitespace(codePoint) || !spacingAndPunctuations.mCurrentLanguageHasSpaces)
                    return@loopOverCodePoints true
                // continue to the next whitespace and see whether this contains a sometimesWordConnector
                for (i in endIndexInAfter..<after.length) {
                    val c = after[i]
                    if (spacingAndPunctuations.isSometimesWordConnector(c.code)) {
                        // if yes -> whitespace is next to the index
                        startIndexInBefore = max(StringUtils.charIndexOfLastWhitespace(before), 0)
                        val firstSpaceAfter = StringUtils.charIndexOfFirstWhitespace(after)
                        endIndexInAfter = if (firstSpaceAfter == -1) after.length else firstSpaceAfter - 1
                        return@loopOverCodePoints true
                    } else if (Character.isWhitespace(c)) {
                        // if no, just break normally
                        return@loopOverCodePoints true
                    }
                }
                return@loopOverCodePoints true
            }
            endIndexInAfter += cpLength
            false
        }
    }

    // strip text before "//" (i.e. ignore http and other protocols)
    val beforeConsideringStart = before.substring(startIndexInBefore, before.length)
    val protocolEnd = beforeConsideringStart.lastIndexOf("//")
    if (protocolEnd != -1) startIndexInBefore += protocolEnd + 1

    // we don't want the end characters to be word separators
    while (endIndexInAfter > 0 && spacingAndPunctuations.isWordSeparator(after[endIndexInAfter - 1].code)) {
        --endIndexInAfter
    }
    while (startIndexInBefore < before.length && spacingAndPunctuations.isWordSeparator(before[startIndexInBefore].code)) {
        ++startIndexInBefore
    }

    val hasUrlSpans = SpannableStringUtils.hasUrlSpans(before, startIndexInBefore, before.length)
        || SpannableStringUtils.hasUrlSpans(after, 0, endIndexInAfter)

    // We don't use TextUtils#concat because it copies all spans without respect to their
    // nature. If the text includes a PARAGRAPH span and it has been split, then
    // TextUtils#concat will crash when it tries to concat both sides of it.
    return TextRange(
        SpannableStringUtils.concatWithNonParagraphSuggestionSpansOnly(before, after),
        startIndexInBefore, before.length + endIndexInAfter, before.length,
        hasUrlSpans
    )
}

// actually this should not be in STRING Utils, but only used for getTouchedWordRange
private fun isPartOfCompositionForScript(codePoint: Int, spacingAndPunctuations: SpacingAndPunctuations, script: String) =
    spacingAndPunctuations.isWordConnector(codePoint) // We always consider word connectors part of compositions.
        // Otherwise, it's part of composition if it's part of script and not a separator.
        || (!spacingAndPunctuations.isWordSeparator(codePoint) && ScriptUtils.isLetterPartOfScript(codePoint, script))

/** split the string on the first of consecutive space only, further consecutive spaces are added to the next split */
fun String.splitOnFirstSpacesOnly(): List<String> {
    val out = mutableListOf<String>()
    val sb = StringBuilder()
    var previousSpace = false
    for (c in this) {
        if (c != ' ') {
            sb.append(c)
            previousSpace = false
            continue
        }
        if (!previousSpace) {
            out.add(sb.toString())
            sb.clear()
            previousSpace = true
        } else {
            sb.append(c)
        }
    }
    if (sb.isNotBlank()) out.add(sb.toString())
    return out
}

fun CharSequence.isValidNumber(): Boolean {
    return this.toString().toDoubleOrNull() != null
}

fun String.decapitalize(locale: Locale): String {
    if (isEmpty() || !this[0].isUpperCase()) return this
    return replaceFirstChar { it.lowercase(locale) }
}

fun encodeBase36(string: String): String = BigInteger(string.toByteArray()).toString(36)

fun decodeBase36(string: String) = BigInteger(string, 36).toByteArray().decodeToString()

fun containsValueWhenSplit(string: String?, value: String, split: String): Boolean {
    if (string == null) return false
    return string.split(split).contains(value)
}

/** converts comma-separates character pairs into a list of codepoint-pair-arrays (sorted by first codepoint) */
fun toSortedCodepointArrays(string: String): List<IntArray> {
    val split = string.split(",")
    val list = split.mapTo(mutableListOf()) {
        intArrayOf(it[0].code, it[1].code)
    }
    list.sortBy { it[0] }
    return list
}

/** returns the paired codepoint in the sorted list of pairs, or Constants.NOT_A_CODE */
fun getSecondInSymbolPair(pairs: List<IntArray>, codePoint: Int): Int {
    val i = pairs.binarySearch { it[0].compareTo(codePoint) }
    if (i >= 0) return pairs[i][1]
    return Constants.NOT_A_CODE
}

/** returns whether the text contains a codepoint that might be an emoji */
fun mightBeEmoji(text: CharSequence): Boolean {
    loopOverCodePoints(text) { cp, _ ->
        if (mightBeEmoji(cp)) return true
        false
    }
    return false
}

fun isEmoji(c: Int): Boolean = mightBeEmoji(c) && isEmoji(newSingleCodePointString(c))

/** returns whether the text is a single emoji */
fun isEmoji(text: CharSequence): Boolean = text.toString().isSingleGrapheme && mightBeEmoji(text) && text.matches(singleEmojiRegex)

// from https://github.com/chattymin/Pebble/blob/main/pebble/src/main/java/com/chattymin/pebble/LocalBreakIterator.kt, Apache-2.0 license
// there is more potentially useful code like String.graphemeLength (should be graphemeCount though)
private val LocalBreakIterator = ThreadLocal<BreakIterator>().apply {
    // this is always Locale.ROOT, but (Android Studio) testing on all available key labels showed no difference by locale
    // maybe this works better with android.icu.text.BreakIterator, but that one requires API24
    set(BreakIterator.getCharacterInstance(Locale.ROOT))
}

private val localBreakIterator: BreakIterator = LocalBreakIterator.get() ?: initBreakIterator()

private fun initBreakIterator() = BreakIterator.getCharacterInstance(Locale.ROOT).also {
    LocalBreakIterator.set(it)
}

val String.isSingleGrapheme: Boolean get() {
    if (isEmpty()) return false
    if (length == 1) return true

    val iterator = localBreakIterator
    iterator.setText(this)
    iterator.next()
    if (iterator.next() != BreakIterator.DONE) return false
    // we have a single grapheme, but " 🏼" is detected as single grapheme which we don't want
    return if ('\uD83C' !in this) true // does not contain skin tone
    else singleEmojiRegex.matches(this) // single grapheme only if it's a single emoji
}

val String.lastGrapheme: String get() {
    if (length <= 1) return this

    val iterator = localBreakIterator
    iterator.setText(this)
    val res = substring(iterator.preceding(length))
    val tone = res.indexOfFirst { it == '\uD83C' }
    return if (tone == -1 || singleEmojiRegex.matches(res)) res
    else res.substring(tone) // " 🏼" is detected as single grapheme, but we don't want this
}

/** translates a move of [steps] graphemes in [text] to character count */
fun moveStepsToCharCount(text: CharSequence, steps: Int): Int {
    if (steps == 0) return 0
    val iterator = localBreakIterator
    iterator.setText(text.toString())
    if (steps > 0) {
        repeat(steps) { iterator.next() }
        return if (iterator.current() == BreakIterator.DONE) text.length
        else iterator.current()
    } else {
        iterator.last()
        repeat(-steps) { iterator.previous() }
        return if (iterator.current() == BreakIterator.DONE) -text.length
        else iterator.current() - text.length
    }
}

fun String.splitOnWhitespace() = SpacedTokens(this).toList()

fun stripTrailingSeparatorsAndConnectors(word: String, spacingAndPunctuations: SpacingAndPunctuations): String {
    var end = word.length
    loopOverCodePointsBackwards(word) { cp, l ->
        if (!spacingAndPunctuations.isWordSeparator(cp) && !spacingAndPunctuations.isWordConnector(cp))
            return@loopOverCodePointsBackwards true
        end -= l
        false
    }
    return if (end == word.length) word else word.substring(0, end)
}
