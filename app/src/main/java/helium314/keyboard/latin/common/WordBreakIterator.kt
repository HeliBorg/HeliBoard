// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.common

import helium314.keyboard.compat.CompatBreakIterator
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import helium314.keyboard.latin.utils.ScriptUtils
import kotlin.math.max

interface WordBreakIterator : CompatBreakIterator {
    // very simple for now
    /**
     *  Returns the boundary indices of the word at [offset], or ([offset] to [offset]) if not touching a word.
     *  If [offset] is a boundary, it tries to determine on which side we have a word.
     *  What actually counts as word differs by implementation or underlying break iterator instance.
     */
    fun getTouchedWordBounds(offset: Int): Pair<Int, Int> {
        val preceding = preceding(offset).coerceAtLeast(0)
        var following = following(offset)
        if (following == -1) following = text.length
        if (!isBoundary(offset) || preceding == following)
            return preceding to following

        val originalText = text
        // start and end is always a boundary, so if we're at start or end, check whether we're still at a boundary when adding a letter
        // if between the word and the new letter there is a boundary, we were not touching a word
        if (offset == 0) {
            text = "a${text.substring(preceding, following)}"
            val isBoundary = isBoundary(1)
            text = originalText
            return if (isBoundary) preceding to offset
            else offset to following
        }
        if (offset == text.length) {
            text = "${text.substring(preceding, following)}a"
            val isBoundary = isBoundary(text.length - 1)
            text = originalText
            return if (isBoundary) offset to following
            else preceding to offset
        }

        // not touching a word
        return offset to offset
    }

    companion object {
        /** similar to [CompatBreakIterator.getWordInstance], but bit different behavior and planned to be somewhat customizable */
        fun getForSpacingAndPunctuations(spacingAndPunctuations: SpacingAndPunctuations) =
            if (!spacingAndPunctuations.mCurrentLanguageHasSpaces && spacingAndPunctuations.mCurrentLanguageHasSpacesOrWordIterator)
                // we don't have one for this case, so use on that ignores spacingAndPunctuations
                CompatBreakIterator.getWordInstance(spacingAndPunctuations.mLocale)
            else SpacingAndPunctuationsWordIterator(spacingAndPunctuations)
    }
}

// ideally this should be consistent with the other WordBreakIterators, except for when we want it
private class SpacingAndPunctuationsWordIterator(val spacingAndPunctuations: SpacingAndPunctuations) : WordBreakIterator {
    private val griter = CompatBreakIterator.graphemeInstance

    override var text: CharSequence = ""
        set(value) {
            field = value
            griter.text = value
            index = 0
        }
    private var index = 0
    override fun current() = index // initially we have no boundary, but apparently this is fine

    // is it really that useless?
    override fun first(): Int {
        index = 0
        return index
    }

    // we could relatively easily make following / preceding more performant, but do only if necessary
    override fun following(offset: Int): Int {
        index = offset
        if (index == -1) return index
        while (index < text.length) {
            index = griter.following(index)
            if (isBoundary(index)) return index
        }
        index = -1
        return index
    }

    // is it really that useless?
    override fun last(): Int {
        index = text.length
        return index
    }

    override fun preceding(offset: Int): Int {
        index = offset
        while (index > 0) {
            index = griter.preceding(index)
            if (isBoundary(index)) return index
        }
        index = -1
        return index
    }

    // todo: this should be consistent with getTouchedWordBounds, ideally with some testing
    override fun isBoundary(offset: Int): Boolean {
        if (offset == 0 || offset == text.length) return true // end is always a boundary
        if (!griter.isBoundary(offset)) return false

        val previousCp = text.codePointBefore(offset)
        val nextCp = text.codePointAt(offset)
        val prevIsLetter = previousCp.isLetter
        val nextIsLetter = nextCp.isLetter
        if (prevIsLetter && nextIsLetter) return false

        val prevIsWhitespace = Character.isWhitespace(previousCp)
        val nextIsWhitespace = Character.isWhitespace(nextCp)
        if (prevIsWhitespace != nextIsWhitespace || !spacingAndPunctuations.mCurrentLanguageHasSpaces)
            return true // whitespace / non-whitespace is always a boundary
        if (prevIsWhitespace && nextIsWhitespace)
            return false // consecutive whitespaces are not a boundary

        // we defer more complicated cases to the probably slower getTouchedWordBounds, so we get consistent behavior
        val (start, end) = getTouchedWordBounds(offset)
        return start == offset || end == offset
    }

    // this is what was previously in StringUtils.getTouchedWordRange
    //  for now we keep it the same, but it should definitely be improved
    // possible improvements
    //  more customizability via spacingAndPunctuations (which should then be somewhat configurable)
    //  understand character alternatives like ’ and ' (there should already be some replacement happening, not sure where though))
    //  better url detection
    //  split a word in some cases when it contains word connectors
    //   specifically thinking about cases like drop-in or step-by-step that are in dictionary
    //   but we can have much longer arbitrary combinations where it would be much better if every word was separate
    //   also we might want to treat ' and - differently (that's our only word connectors atm)
    //  better definition of what counts as word
    //   examples
    //    hello is a word
    //    123 is not a word
    //    4ever should probably be a word (then what about 4e? 4ev?)
    //    Eb8MX2Jl or s0;\zHagvOp should ideally not be words, but that's tricky
    //   also relevant in case a word (or shortcut) starts with a number!
    // use in InputLogic in connection with composing span, iirc there are already inconsistencies of writing a word vs touching a word
    //  though it might introduce unexpected behavior in special cases e.g. when a user wants to write 'don't come here' (including the ')
    //  and then it could happen that the composing span expands leftwards, e.g. while typing 4ever -> do we want that?
    // use in SentenceLevelAdapter.WordIterator
    override fun getTouchedWordBounds(offset: Int): Pair<Int, Int> {
        val before = text.substring(0, offset)
        val after = text.substring(offset)
        // Going backward, find the first breaking point (separator)
        var startIndexInBefore = before.length
        var endIndexInAfter = -1 // todo: clarify why might we want to set it when checking before
        loopOverCodePointsBackwards(before) { codePoint, cpLength ->
            if (isPartOfCompositionForScript(codePoint, spacingAndPunctuations) || codePoint in combiningRange) {
                startIndexInBefore -= cpLength
                return@loopOverCodePointsBackwards false
            }
            if (Character.isWhitespace(codePoint) || !spacingAndPunctuations.mCurrentLanguageHasSpaces)
                return@loopOverCodePointsBackwards true
            // continue to the next whitespace and see whether this contains a sometimesWordConnector
            for (i in startIndexInBefore - 1 downTo 0) {
                val c = before[i]
                if (spacingAndPunctuations.isSometimesWordConnector(c.code)) {
                    // if yes -> whitespace is the index
                    startIndexInBefore = max(StringUtils.charIndexOfLastWhitespace(before), 0)
                    val firstSpaceAfter = StringUtils.charIndexOfFirstWhitespace(after)
                    endIndexInAfter = if (firstSpaceAfter == -1) after.length else firstSpaceAfter - 1
                    return@loopOverCodePointsBackwards true
                } else if (Character.isWhitespace(c)) {
                    // if no, just break normally
                    return@loopOverCodePointsBackwards true
                }
            }
            true
        }

        // Find last word separator after the cursor
        if (endIndexInAfter == -1) {
            endIndexInAfter = 0
            loopOverCodePoints(after) { codePoint, cpLength ->
                if (isPartOfCompositionForScript(codePoint, spacingAndPunctuations) || codePoint in combiningRange) {
                    endIndexInAfter += cpLength
                    return@loopOverCodePoints false
                }
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
                true
            }
        }

        if (startIndexInBefore != text.length) {
            /*
             * deal with wordConnectors at start and end
             *  at start they should not be included
             *  at end they should be included
             *   unless the same code point is also at the start
             *   e.g. don' is a "word"
             *   e.g. for 'hello and 'hello' only hello is the word
             */
            val cpAtStart = Character.codePointAt(text, startIndexInBefore)
            if (spacingAndPunctuations.isWordConnector(cpAtStart)) {
                startIndexInBefore += Character.charCount(cpAtStart)

                if (startIndexInBefore > offset)
                    return offset to offset // outside the word

                // and if the same codepoint is at the end, we assume it should be removed too
                if (Character.codePointBefore(text, endIndexInAfter + offset) == cpAtStart)
                    endIndexInAfter -= Character.charCount(cpAtStart)

                if (endIndexInAfter + before.length < offset)
                    return offset to offset // outside the word
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
        return startIndexInBefore to (before.length + endIndexInAfter)
    }

    private val Int.isLetter get() =
        (!spacingAndPunctuations.isWordSeparator(this) && ScriptUtils.isLetterPartOfScript(this, spacingAndPunctuations.mScript))
            || this in combiningRange

    // need to be more careful here, because a wordConnector at the start should not be part of composition
    private fun isPartOfCompositionForScript(codePoint: Int, spacingAndPunctuations: SpacingAndPunctuations) =
        spacingAndPunctuations.isWordConnector(codePoint) // We always consider word connectors part of compositions.
            // Otherwise, it's part of composition if it's part of script and not a separator.
            || (!spacingAndPunctuations.isWordSeparator(codePoint) && ScriptUtils.isLetterPartOfScript(codePoint, spacingAndPunctuations.mScript))
}
