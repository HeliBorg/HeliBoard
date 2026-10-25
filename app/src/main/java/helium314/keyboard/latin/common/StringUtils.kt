// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.common

import helium314.keyboard.compat.CompatBreakIterator
import helium314.keyboard.latin.common.StringUtils.mightBeEmoji
import helium314.keyboard.latin.common.StringUtils.newSingleCodePointString
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import helium314.keyboard.latin.utils.SpacedTokens
import helium314.keyboard.latin.utils.SpannableStringUtils
import helium314.keyboard.latin.utils.TextRange
import java.math.BigInteger
import java.util.Locale
import kotlin.text.codePointAt

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

fun getTouchedWordRange(before: CharSequence, after: CharSequence, spacingAndPunctuations: SpacingAndPunctuations): TextRange {
    val iterator = WordBreakIterator.getForSpacingAndPunctuations(spacingAndPunctuations)
    iterator.text = before.toString() + after
    val (startIndexInBefore, endIndex) = iterator.getTouchedWordBounds(before.length)

    val hasUrlSpans = SpannableStringUtils.hasUrlSpans(before, startIndexInBefore, before.length)
        || SpannableStringUtils.hasUrlSpans(after, 0, endIndex - before.length)

    // We don't use TextUtils#concat because it copies all spans without respect to their
    // nature. If the text includes a PARAGRAPH span and it has been split, then
    // TextUtils#concat will crash when it tries to concat both sides of it.
    return TextRange(
        SpannableStringUtils.concatWithNonParagraphSuggestionSpansOnly(before, after),
        startIndexInBefore, endIndex, before.length,
        hasUrlSpans
    )
}

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
    val firstCp = codePointAt(0)
    if (isEmpty() || !Character.isUpperCase(firstCp)) return this
    if (firstCp < 0xffff)
        return replaceFirstChar { it.lowercase(locale) }
    // this doesn't consider locale, while String.lowercase would
    // but anyway were unlikely to arrive here...
    return String(Character.toChars(Character.toLowerCase(firstCp)))+
        substring(Character.charCount(firstCp))
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

/** returns whether the text is a single emoji or a single regional indicator */
fun isEmoji(text: CharSequence): Boolean = text.isSingleOrDoubleRegionalIndicator // not recognized by singleEmojiRegex
        || (text.toString().isSingleGrapheme && mightBeEmoji(text) && text.matches(singleEmojiRegex))

private inline val CharSequence.isSingleOrDoubleRegionalIndicator get() =
    (length == 2 && Character.codePointAt(this, 0).isRegionalIndicator)
        || (length == 4 && Character.codePointAt(this, 0).isRegionalIndicator
                && Character.codePointAt(this, 2).isRegionalIndicator
           )

inline val Int.isRegionalIndicator get() = this in 0x1F1E6..0x1F1FF
//inline val Int.isSkinTone get() = this in 0x1F3FB..0x1F3FF

inline val String.isSingleGrapheme: Boolean get() = length > 0 && lastGraphemeLength == length

val CharSequence.lastGraphemeLength: Int get() {
    if (length <= 1) return length
    val iterator = CompatBreakIterator.graphemeInstance
    iterator.text = this
    return length - iterator.preceding(length)
}

fun CharSequence.getLastWordLength(sp: SpacingAndPunctuations): Int {
    if (length <= 1) return length
    val iterator = WordBreakIterator.getForSpacingAndPunctuations(sp)
    iterator.text = this
    return length - iterator.preceding(length)
}

/** translates a move of [steps] in [text] as defined by [iterator] to character count */
fun stepsToCharCount(text: CharSequence, steps: Int, iterator: CompatBreakIterator): Int {
    if (steps == 0) return 0
    iterator.text = text
    if (steps > 0) {
        repeat(steps) { iterator.next() }
        return if (iterator.current() == -1) text.length
        else iterator.current()
    } else {
        iterator.last()
        repeat(-steps) { iterator.previous() }
        return if (iterator.current() == -1) -text.length
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
