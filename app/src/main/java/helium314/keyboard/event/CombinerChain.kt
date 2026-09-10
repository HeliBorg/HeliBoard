/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.event

import android.text.SpannableStringBuilder
import android.text.TextUtils
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import java.util.*

/**
 * This class implements the logic chain between receiving events and generating code points.
 *
 * Event sources are multiple. It may be a hardware keyboard, a D-PAD, a software keyboard,
 * or any exotic input source.
 * This class will orchestrate the composing chain that starts with an event as its input. Each
 * composer will be given turns one after the other.
 * The output is composed of two sequences of code points: the first, representing the already
 * finished combining part, will be shown normally as the composing string, while the second is
 * feedback on the composing state and will typically be shown with different styling such as
 * a colored background.
 *
 * The combiner chain takes events as inputs and outputs code points and combining state.
 * For example, if the input language is Japanese, the combining chain will typically perform
 * kana conversion. This takes a string for initial text, taken to be present before the
 * cursor: we'll start after this.
 * @param initialText The text that has already been combined so far.
 */
class CombinerChain(initialText: String, combiningSpec: String) {
    // The already combined text, as described above
    private val mCombinedText = StringBuilder(initialText)
    // The feedback on the composing state, as described above
    private val mStateFeedback = SpannableStringBuilder()
    private val mCombiners = ArrayList<Combiner>()

    init {
        // The dead key combiner is always active, and always first
        mCombiners.add(DeadKeyCombiner())
        if (combiningSpec == "hangul")
            mCombiners.add(HangulCombiner())
        else if (combiningSpec == "bn_khipro")
            mCombiners.add(BnKhiproCombiner())
        else if (combiningSpec == "bn_avro")
            mCombiners.add(AvroPhoneticCombiner())
    }

    fun reset() {
        mCombinedText.setLength(0)
        mStateFeedback.clear()
        for (c in mCombiners) {
            c.reset()
        }
    }

    private fun updateStateFeedback() {
        mStateFeedback.clear()
        for (i in mCombiners.indices.reversed()) {
            mStateFeedback.append(mCombiners[i].combiningStateFeedback)
        }
    }

    /**
     * Process an event through the combining chain, and return a processed event to apply.
     * @param previousEvents the list of previous events in this composition
     * @param newEvent the new event to process
     * @return the processed event. It may be the same event, or a consumed event, or a completely
     * new event. However it may never be null.
     */
    fun processEvent(previousEvents: ArrayList<Event>?, newEvent: Event): Event {
        val modifiablePreviousEvents = ArrayList(previousEvents!!)
        var event = newEvent
        for (combiner in mCombiners) {
            // A combiner can never return more than one event; it can return several
            // code points, but they should be encapsulated within one event.
            event = combiner.processEvent(modifiablePreviousEvents, event)
            if (event.isConsumed) {
                // If the event is consumed, then we don't pass it to subsequent combiners:
                // they should not see it at all.
                break
            }
        }
        updateStateFeedback()
        return event
    }

    /**
     * Apply a processed event.
     * @param event the event to be applied
     */
    fun applyProcessedEvent(event: Event?) {
        if (null != event) { // TODO: figure out the generic way of doing this
            if (KeyCode.DELETE == event.keyCode) {
                val length = mCombinedText.length
                if (length > 0) {
                    val lastCodePoint = mCombinedText.codePointBefore(length)
                    mCombinedText.delete(length - Character.charCount(lastCodePoint), length)
                }
            } else {
                val textToCommit = event.textToCommit
                if (!TextUtils.isEmpty(textToCommit)) {
                    mCombinedText.append(textToCommit)
                }
            }
        }
        updateStateFeedback()
    }

    /**
     * Get the char sequence that should be displayed as the composing word. It may include
     * styling spans.
     */
    val composingWordWithCombiningFeedback: CharSequence
        get() {
            val s = SpannableStringBuilder(mCombinedText)
            return s.append(mStateFeedback)
        }

}
