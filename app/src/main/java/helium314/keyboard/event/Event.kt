/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.event

import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.StringUtils

/**
 * Class representing a generic input event as handled by Latin IME.
 *
 * This contains information about the origin of the event, but it is generalized and should
 * represent a software keypress, hardware keypress, or d-pad move alike.
 * Very importantly, this does not necessarily result in inputting one character, or even anything
 * at all - it may be a dead key, it may be a partial input, it may be a special key on the
 * keyboard, it may be a cancellation of a keypress (e.g. in a soft keyboard the finger of the
 * user has slid out of the key), etc. It may also be a batch input from a gesture or handwriting
 * for example.
 * The combiner should figure out what to do with this.
 */
class Event private constructor(
    // The type of event - one of the constants above
    private val eventType: Int,
    // If applicable, this contains the string that should be input.
    val text: CharSequence? = null,
    // The code point associated with the event, if relevant. This is a unicode code point, and
    // has nothing to do with other representations of the key. It is only relevant if this event
    // is of KEYPRESS type, but for a mode key like hankaku/zenkaku or ctrl, there is no code point
    // associated so this should be NOT_A_CODE_POINT to avoid unintentional use of its value when
    // it's not relevant.
    val codePoint: Int = NOT_A_CODE_POINT,
    // The key code associated with the event, if relevant. This is relevant whenever this event
    // has been triggered by a key press, but not for a gesture for example. This has conceptually
    // no link to the code point, although keys that enter a straight code point may often set
    // this to be equal to mCodePoint for convenience. If this is not a key, this must contain
    // NOT_A_KEY_CODE.
    val keyCode: Int = NOT_A_KEY_CODE,
    // State of meta keys (currently ctrl, alt, fn, meta)
    // same value as https://developer.android.com/reference/android/view/KeyEvent#getMetaState()
    val metaState: Int = 0,
    // Coordinates of the touch event, if relevant. If useful, we may want to replace this with
    // a MotionEvent or something in the future. This is only relevant when the keypress is from
    // a software keyboard obviously, unless there are touch-sensitive hardware keyboards in the
    // future or some other awesome sauce.
    val x: Int = Constants.NOT_A_COORDINATE,
    val y: Int = Constants.NOT_A_COORDINATE,
    // If this is of type EVENT_TYPE_SUGGESTION_PICKED, this must not be null (and must be null in
    // other cases).
    val suggestedWordInfo: SuggestedWordInfo? = null,
    // Some flags that can't go into the key code. It's a bit field of FLAG_*
    private val flags: Int = FLAG_NONE,
    // The next event, if any. Null if there is no next event yet.
    val nextEvent: Event? = null
    // This logic may need to be refined in the future
) {
    init {
        if ((EVENT_TYPE_SUGGESTION_PICKED == eventType) != (suggestedWordInfo != null))
            throw RuntimeException("Wrong event: SUGGESTION_PICKED event must have a non-null SuggestedWordInfo, other events may not")
    }

    // Returns whether this is a function key like backspace, ctrl, settings... as opposed to keys
    // that result in input like letters or space.
    val isFunctionalKeyEvent: Boolean
        get() = NOT_A_CODE_POINT == codePoint || metaState != 0 // This logic may need to be refined in the future

    // Returns whether this event is for a dead character. @see {@link #FLAG_DEAD}
    val isDead: Boolean get() = 0 != FLAG_DEAD and flags

    val isKeyRepeat: Boolean get() = 0 != FLAG_REPEAT and flags

    val isConsumed: Boolean get() = 0 != FLAG_CONSUMED and flags

    val isCombining: Boolean get() = 0 != FLAG_COMBINING and flags

    val isGesture: Boolean get() = EVENT_TYPE_GESTURE == eventType

    // Returns whether this is a fake key press from the suggestion strip. This happens with
    // punctuation signs selected from the suggestion strip.
    val isSuggestionStripPress: Boolean get() = EVENT_TYPE_SUGGESTION_PICKED == eventType

    val isHandled: Boolean get() = EVENT_TYPE_NOT_HANDLED != eventType

    // A consumed event should input no text.
    val textToCommit: CharSequence?
        get() {
            if (isConsumed) {
                return "" // A consumed event should input no text.
            }
            return when (eventType) {
                EVENT_TYPE_MODE_KEY, EVENT_TYPE_NOT_HANDLED, EVENT_TYPE_TOGGLE, EVENT_TYPE_CURSOR_MOVE -> ""
                EVENT_TYPE_INPUT_KEYPRESS -> StringUtils.newSingleCodePointString(codePoint)
                EVENT_TYPE_GESTURE, EVENT_TYPE_SOFTWARE_GENERATED_STRING, EVENT_TYPE_SUGGESTION_PICKED -> text
                else -> throw RuntimeException("Unknown event type: $eventType")
            }
        }

    companion object {
        // Should the types below be represented by separate classes instead? It would be cleaner
        // but probably a bit too much
        // An event we don't handle in Latin IME, for example pressing Ctrl on a hardware keyboard.
        const val EVENT_TYPE_NOT_HANDLED = 0
        // A key press that is part of input, for example pressing an alphabetic character on a
        // hardware qwerty keyboard. It may be part of a sequence that will be re-interpreted later
        // through combination.
        const val EVENT_TYPE_INPUT_KEYPRESS = 1
        // A toggle event is triggered by a key that affects the previous character. An example would
        // be a numeric key on a 10-key keyboard, which would toggle between 1 - a - b - c with
        // repeated presses.
        const val EVENT_TYPE_TOGGLE = 2
        // A mode event instructs the combiner to change modes. The canonical example would be the
        // hankaku/zenkaku key on a Japanese keyboard, or even the caps lock key on a qwerty keyboard
        // if handled at the combiner level.
        const val EVENT_TYPE_MODE_KEY = 3
        // An event corresponding to a gesture.
        const val EVENT_TYPE_GESTURE = 4
        // An event corresponding to the manual pick of a suggestion.
        const val EVENT_TYPE_SUGGESTION_PICKED = 5
        // An event corresponding to a string generated by some software process.
        const val EVENT_TYPE_SOFTWARE_GENERATED_STRING = 6
        // An event corresponding to a cursor move
        const val EVENT_TYPE_CURSOR_MOVE = 7

        // 0 is a valid code point, so we use -1 here.
        const val NOT_A_CODE_POINT = -1
        // -1 is a valid key code, so we use 0 here.
        const val NOT_A_KEY_CODE = 0

        private const val FLAG_NONE = 0
        // This event is a dead character, usually input by a dead key. Examples include dead-acute or dead-abovering.
        private const val FLAG_DEAD = 0x1
        // This event is coming from a key repeat, software or hardware.
        private const val FLAG_REPEAT = 0x2
        // This event has already been consumed.
        private const val FLAG_CONSUMED = 0x4
        // This event is a combining character, usually a hangul input.
        private const val FLAG_COMBINING = 0x8

        @JvmStatic
        fun createSoftwareKeypressEvent(codePoint: Int, keyCode: Int, metaState: Int, x: Int, y: Int, isKeyRepeat: Boolean) =
            Event(
                eventType = EVENT_TYPE_INPUT_KEYPRESS,
                codePoint = codePoint,
                keyCode = keyCode,
                metaState = metaState,
                x = x,
                y = y,
                flags = if (isKeyRepeat) FLAG_REPEAT else FLAG_NONE
            )

        // A helper method to split the code point and the key code.
        // todo: Ultimately, they should not be squashed into the same variable, and this method should be removed.
        @JvmStatic
        fun createSoftwareKeypressEvent(keyCodeOrCodePoint: Int, metaState: Int, keyX: Int, keyY: Int, isKeyRepeat: Boolean) =
            if (keyCodeOrCodePoint <= 0) {
                createSoftwareKeypressEvent(NOT_A_CODE_POINT, keyCodeOrCodePoint, metaState, keyX, keyY, isKeyRepeat)
            } else {
                createSoftwareKeypressEvent(keyCodeOrCodePoint, NOT_A_KEY_CODE, metaState, keyX, keyY, isKeyRepeat)
            }

        fun createHardwareKeypressEvent(codePoint: Int, keyCode: Int, metaState: Int, next: Event?, isKeyRepeat: Boolean) =
            Event(
                eventType = EVENT_TYPE_INPUT_KEYPRESS,
                codePoint = codePoint,
                keyCode = keyCode,
                metaState = metaState,
                x = Constants.EXTERNAL_KEYBOARD_COORDINATE,
                y = Constants.EXTERNAL_KEYBOARD_COORDINATE,
                flags = if (isKeyRepeat) FLAG_REPEAT else FLAG_NONE,
                nextEvent = next
            )

        // This creates an input event for a dead character. see FLAG_DEAD
        fun createDeadEvent(codePoint: Int, keyCode: Int, metaState: Int, next: Event?) =
            Event(
                eventType = EVENT_TYPE_INPUT_KEYPRESS,
                codePoint = codePoint,
                keyCode = keyCode,
                metaState = metaState,
                x = Constants.EXTERNAL_KEYBOARD_COORDINATE,
                y = Constants.EXTERNAL_KEYBOARD_COORDINATE,
                flags = FLAG_DEAD,
                nextEvent = next
            )

        // This creates an input event for a dead character. see FLAG_DEAD
        fun createSoftwareDeadEvent(codePoint: Int, keyCode: Int, metaState: Int, x: Int, y: Int, next: Event?) =
            Event(
                eventType = EVENT_TYPE_INPUT_KEYPRESS,
                codePoint = codePoint,
                keyCode = keyCode,
                metaState = metaState,
                x = x,
                y = y,
                flags = FLAG_DEAD,
                nextEvent = next
            )

        /**
         * Create an input event with nothing but a code point. This is the most basic possible input
         * event; it contains no information on many things the IME requires to function correctly,
         * so avoid using it unless really nothing is known about this input.
         * @param codePoint the code point.
         * @return an event for this code point.
         */
        @JvmStatic
        // TODO: should we have a different type of event for this? After all, it's not a key press.
        fun createEventForCodePointFromUnknownSource(codePoint: Int) = Event(eventType = EVENT_TYPE_INPUT_KEYPRESS, codePoint = codePoint)

        /**
         * Creates an input event with a code point and x, y coordinates. This is typically used when
         * resuming a previously-typed word, when the coordinates are still known.
         * @param codePoint the code point to input.
         * @param x the X coordinate.
         * @param y the Y coordinate.
         * @return an event for this code point and coordinates.
         */
        @JvmStatic
        // TODO: should we have a different type of event for this? After all, it's not a key press.
        fun createEventForCodePointFromAlreadyTypedText(codePoint: Int, x: Int, y: Int) =
            Event(eventType = EVENT_TYPE_INPUT_KEYPRESS, codePoint = codePoint, x = x, y = y)

        /**
         * Creates an input event representing the manual pick of a suggestion.
         * @return an event for this suggestion pick.
         */
        @JvmStatic
        fun createSuggestionPickedEvent(suggestedWordInfo: SuggestedWordInfo) =
            Event(
                eventType = EVENT_TYPE_SUGGESTION_PICKED,
                text = suggestedWordInfo.mWord,
                x = Constants.SUGGESTION_STRIP_COORDINATE,
                y = Constants.SUGGESTION_STRIP_COORDINATE,
                suggestedWordInfo = suggestedWordInfo
            )

        /**
         * Creates an input event with a CharSequence. This is used by some software processes whose
         * output is a string, possibly with styling. Examples include press on a multi-character key,
         * or combination that outputs a string.
         * @param text the CharSequence associated with this event.
         * @param keyCode the key code, or NOT_A_KEYCODE if not applicable.
         * @param nextEvent the next event, or null if not applicable.
         * @return an event for this text.
         */
        @JvmStatic
        fun createSoftwareTextEvent(text: CharSequence, keyCode: Int, nextEvent: Event? = null) =
            Event(eventType = EVENT_TYPE_SOFTWARE_GENERATED_STRING, text = text, keyCode = keyCode, nextEvent = nextEvent)

        /**
         * Creates an input event representing the manual pick of a punctuation suggestion.
         * @return an event for this suggestion pick.
         */
        @JvmStatic
        fun createPunctuationSuggestionPickedEvent(suggestedWordInfo: SuggestedWordInfo) =
            Event(
                eventType = EVENT_TYPE_SUGGESTION_PICKED,
                text = suggestedWordInfo.mWord,
                codePoint = suggestedWordInfo.mWord[0].code,
                x = Constants.SUGGESTION_STRIP_COORDINATE,
                y = Constants.SUGGESTION_STRIP_COORDINATE,
                suggestedWordInfo = suggestedWordInfo
            )

        /**
         * Creates an input event representing moving the cursor. The relative move amount is stored
         * in mX.
         * @param moveAmount the relative move amount.
         * @return an event for this cursor move.
         */
        @JvmStatic
        fun createCursorMovedEvent(moveAmount: Int) = Event(eventType = EVENT_TYPE_CURSOR_MOVE, x = moveAmount)

        /**
         * Creates an event identical to the passed event, but that has already been consumed.
         * @param source the event to copy the properties of.
         * @return an identical event marked as consumed.
         */
        // A consumed event should not input any text at all, so we pass the empty string as text.
        fun createConsumedEvent(source: Event) =
             Event(source.eventType, source.text, source.codePoint, source.keyCode, source.metaState,
                    source.x, source.y, source.suggestedWordInfo, source.flags or FLAG_CONSUMED, source.nextEvent)

        fun createCombiningEvent(source: Event) =
            Event(source.eventType, source.text, source.codePoint, source.keyCode, source.metaState,
                    source.x, source.y, source.suggestedWordInfo, source.flags or FLAG_COMBINING, source.nextEvent)

        val notHandledEvent = Event(eventType = EVENT_TYPE_NOT_HANDLED)
    }
}
