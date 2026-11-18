/*
 * Copyright (C) 2013 The Android Open Source Project
 * heavily modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.utils

import android.text.Spanned
import android.text.style.SuggestionSpan
import helium314.keyboard.latin.common.WordBreakIterator
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import kotlin.math.max
import kotlin.math.min

data class TextAtCursor(val text: CharSequence, val cursorPosition: Int) {
    constructor(before: CharSequence, after: CharSequence) : this(
        SpannableStringUtils.concatWithNonParagraphSuggestionSpansOnly(before, after),
        before.length
    )
    // todo: add caching if actually useful (needs equals & hashcode for S&P iterator, based on locale)
//    val cache = SimpleArrayMap<WordBreakIterator, WordAtCursor>(4)

    fun getWordAtCursor(spacingAndPunctuations: SpacingAndPunctuations) =
        getWordAtCursor(WordBreakIterator.getForSpacingAndPunctuations(spacingAndPunctuations))

    fun getWordAtCursor(iterator: WordBreakIterator): WordAtCursor {
        iterator.text = text
        val (startIndex, endIndex) = iterator.getTouchedWordBounds(cursorPosition)
        return WordAtCursor(startIndex, endIndex)
    }

    fun getAsWord() = WordAtCursor(0, text.length)

    inner class WordAtCursor(val start: Int, val end: Int) {
        val word = text.subSequence(start, end)
        val hasUrlSpans get() = SpannableStringUtils.hasUrlSpans(text, start, end)

        val numberOfCharsBeforeCursor get() = cursorPosition - start
        val numberOfCharsAfterCursor get() = end - cursorPosition

        fun getText() = this@TextAtCursor

        /**
         * Gets the suggestion spans that are put squarely on the word, with the exact start
         * and end of the span matching the boundaries of the word.
         * @return the array of non-null spans.
         */
        fun getSuggestionSpansAtWord(): Array<SuggestionSpan?> {
            if (!(text is Spanned && word is Spanned)) {
                return emptyArray()
            }
            // Note: it's fine to pass indices negative or greater than the length of the string
            // to the #getSpans() method. The reason we need to get from -1 to +1 is that, the
            // spans were cut at the cursor position, and #getSpans(start, end) does not return
            // spans that end at `start' or begin at `end'. Consider the following case:
            //              this| is          (The | symbolizes the cursor position
            //              ---- ---
            // In this case, the cursor is in position 4, so the 0~7 span has been split into
            // a 0~4 part and a 4~7 part.
            // If we called #getSpans(0, 4) in this case, we would only get the part from 0 to 4
            // of the span, and not the part from 4 to 7, so we would not realize the span actually
            // extends from 0 to 7. But if we call #getSpans(-1, 5) we'll get both the 0~4 and
            // the 4~7 spans and we can merge them accordingly.
            // Any span starting more than 1 char away from the word boundaries in any direction
            // does not touch the word, so we don't need to consider it. That's why requesting
            // -1 ~ +1 is enough.
            // Of course this is only relevant if the cursor is at one end of the word. If it's
            // in the middle, the -1 and +1 are not necessary, but they are harmless.
            val spans: Array<SuggestionSpan?> = text.getSpans(start - 1, end + 1, SuggestionSpan::class.java)
            var readIndex = 0
            var writeIndex = 0
            while (readIndex < spans.size) {
                val span = spans[readIndex]
                // The span may be null, as we null them when we find duplicates. Cf a few lines
                // down.
                if (span == null) {
                    ++readIndex
                    continue
                }
                // Tentative span start and end. This may be modified later if we realize the
                // same span is also applied to other parts of the string.
                var spanStart: Int = text.getSpanStart(span)
                var spanEnd: Int = text.getSpanEnd(span)
                for (i in readIndex + 1..<spans.size) {
                    if (span == spans[i]) {
                        // We found the same span somewhere else. Read the new extent of this
                        // span, and adjust our values accordingly.
                        spanStart = min(spanStart, text.getSpanStart(spans[i]))
                        spanEnd = max(spanEnd, text.getSpanEnd(spans[i]))
                        // ...and mark the span as processed.
                        spans[i] = null
                    }
                }
                if (spanStart == start && spanEnd == end) {
                    // If the span does not start and stop here, ignore it. It probably extends
                    // past the start or end of the word, as happens in missing space correction
                    // or EasyEditSpans put by voice input.
                    spans[writeIndex++] = spans[readIndex]
                }
                ++readIndex
            }
            return if (writeIndex == readIndex) spans else spans.copyOfRange(0, writeIndex)
        }

        override fun toString() = "$word in $text"
    }
}
