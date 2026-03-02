/*
 * Copyright (C) 2008 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import androidx.annotation.NonNull;

import helium314.keyboard.event.CombinerChain;
import helium314.keyboard.event.Event;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo;
import helium314.keyboard.latin.common.ComposedData;
import helium314.keyboard.latin.common.CoordinateUtils;
import helium314.keyboard.latin.common.InputPointers;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.define.DecoderSpecificConstants;

import java.util.ArrayList;
import java.util.Collections;

/**
 * A place to store the currently composing word with information such as adjacent key codes as well
 */
public final class WordComposer {
    private static final int MAX_WORD_LENGTH = DecoderSpecificConstants.DICTIONARY_MAX_WORD_LENGTH;

    public static final int CAPS_MODE_OFF = 0;
    // 1 is shift bit, 2 is caps bit, 4 is auto bit but this is just a convention as these bits
    // aren't used anywhere in the code
    public static final int CAPS_MODE_MANUAL_SHIFTED = 0x1;
    public static final int CAPS_MODE_MANUAL_SHIFT_LOCKED = 0x3;
    public static final int CAPS_MODE_AUTO_SHIFTED = 0x5;
    public static final int CAPS_MODE_AUTO_SHIFT_LOCKED = 0x7;

    private CombinerChain mCombinerChain;
    private String mCombiningSpec; // Memory so that we don't uselessly recreate the combiner chain

    // The list of events that served to compose this string.
    private final ArrayList<Event> mEvents;
    private final InputPointers mInputPointers = new InputPointers(MAX_WORD_LENGTH);
    private SuggestedWordInfo mAutoCorrection;
    private boolean mIsResumed;
    private boolean mIsBatchMode;
    // A memory of the last rejected batch mode suggestion, if any. This goes like this: the user
    // gestures a word, is displeased with the results and hits backspace, then gestures again.
    // At the very least we should avoid re-suggesting the same thing, and to do that we memorize
    // the rejected suggestion in this variable.
    // TODO: this should be done in a comprehensive way by the User History feature instead of
    // as an ad-hockery here.
    private String mRejectedBatchModeSuggestion;

    // Cache these values for performance
    private CharSequence mTypedWordCache;
    private int mCapsCount;
    private int mDigitsCount;
    private int mCapitalizedMode;
    // This is the number of code points entered so far. This is not limited to MAX_WORD_LENGTH.
    // In general, this contains the size of mPrimaryKeyCodes, except when this is greater than
    // MAX_WORD_LENGTH in which case mPrimaryKeyCodes only contain the first MAX_WORD_LENGTH
    // code points.
    private int mCodePointSize;
    private int mCursorPositionWithinWord;

    /**
     * Whether the composing word has the only first char capitalized.
     */
    private boolean mIsOnlyFirstCharCapitalized;

    public WordComposer() {
        mCombinerChain = new CombinerChain("", "");
        mEvents = new ArrayList<>();
        mAutoCorrection = null;
        mIsResumed = false;
        mIsBatchMode = false;
        mCursorPositionWithinWord = 0;
        mRejectedBatchModeSuggestion = null;
        refreshTypedWordCache();
    }

    public ComposedData getComposedDataSnapshot() {
        return new ComposedData(getInputPointers(), isBatchMode(), mTypedWordCache.toString());
    }

    /**
     * Restart the combiners, possibly with a new spec.
     * @param combiningSpec The spec string for combining. This is found in the extra value.
     */
    public void restartCombining(final String combiningSpec) {
        final String nonNullCombiningSpec = null == combiningSpec ? "" : combiningSpec;
        if (!nonNullCombiningSpec.equals(mCombiningSpec)) {
            mCombinerChain = new CombinerChain(mCombinerChain.getComposingWordWithCombiningFeedback().toString(), nonNullCombiningSpec);
            mCombiningSpec = nonNullCombiningSpec;
        }
    }

    /**
     * Clear out the keys registered so far.
     */
    public void reset() {
        mCombinerChain.reset();
        mEvents.clear();
        mAutoCorrection = null;
        mCapsCount = 0;
        mDigitsCount = 0;
        mIsOnlyFirstCharCapitalized = false;
        mIsResumed = false;
        mIsBatchMode = false;
        mCursorPositionWithinWord = 0;
        mRejectedBatchModeSuggestion = null;
        refreshTypedWordCache();
    }

    private void refreshTypedWordCache() {
        int oldSize = mCodePointSize;
        mTypedWordCache = mCombinerChain.getComposingWordWithCombiningFeedback();
        mCodePointSize = Character.codePointCount(mTypedWordCache, 0, mTypedWordCache.length());
        mCursorPositionWithinWord += mCodePointSize - oldSize;
    }

    /**
     * Number of keystrokes in the composing word.
     * @return the number of keystrokes
     */
    public int size() {
        return mCodePointSize;
    }

    public boolean isSingleLetter() {
        return size() == 1;
    }

    public boolean isComposingWord() {
        return size() > 0;
    }

    public InputPointers getInputPointers() {
        return mInputPointers;
    }

    /**
     * Process an event and return an event, and return a processed event to apply.
     * @param event the unprocessed event.
     * @return the processed event. Never null, but may be marked as consumed.
     */
    @NonNull
    public Event processEvent(@NonNull final Event event) {
        final Event processedEvent = mCombinerChain.processEvent(mEvents, event);
        // The retained state of the combiner chain may have changed while processing the event,
        // so we need to update our cache.
        refreshTypedWordCache();
        mEvents.add(event);
        return processedEvent;
    }

    /**
     * Apply a processed input event.
     * <p>
     * All input events should be supported, including software/hardware events, characters as well
     * as deletions, multiple inputs and gestures.
     *
     * @param event the event to apply. Must not be null.
     */
    public void applyProcessedEvent(final Event event) {
        applyProcessedEvent(event, false);
    }

    // specifically for that KeyCode.MULTIPLE_CODE_POINTS Hangul event: try keeping cursor position
    // because typically nothing changes, todo: if really nothing changes maybe there is a better way to do it
    public void applyProcessedEvent(final Event event, final boolean keepCursorPosition) {
        mCombinerChain.applyProcessedEvent(event);
        final int primaryCode = event.getCodePoint();
        final int keyX = event.getX();
        final int keyY = event.getY();
        final int newIndex = size();
        refreshTypedWordCache();
        if (!keepCursorPosition || newIndex == mCodePointSize)
            mCursorPositionWithinWord = mCodePointSize;
        // We may have deleted the last one.
        if (0 == mCodePointSize) {
            mIsOnlyFirstCharCapitalized = false;
        }
        if (KeyCode.DELETE != event.getKeyCode()) {
            if (newIndex < MAX_WORD_LENGTH) {
                // In the batch input mode, the {@code mInputPointers} holds batch input points and
                // shouldn't be overridden by the "typed key" coordinates
                // (See {@link #setBatchInputWord}).
                if (!mIsBatchMode) {
                    // TODO: Set correct pointer id and time
                    mInputPointers.addPointerAt(newIndex, keyX, keyY, 0, 0);
                }
            }
            if (0 == newIndex) {
                mIsOnlyFirstCharCapitalized = Character.isUpperCase(primaryCode);
            } else {
                mIsOnlyFirstCharCapitalized = mIsOnlyFirstCharCapitalized
                        && !Character.isUpperCase(primaryCode);
            }
            if (Character.isUpperCase(primaryCode)) mCapsCount++;
            if (Character.isDigit(primaryCode)) mDigitsCount++;
        }
        mAutoCorrection = null;
    }

    public void setCursorPositionWithinWord(final int posWithinWord) {
        mCursorPositionWithinWord = posWithinWord;
        // TODO: compute where that puts us inside the events
    }

    public boolean isCursorFrontOrMiddleOfComposingWord() {
        if (DebugFlags.DEBUG_ENABLED && mCursorPositionWithinWord > mCodePointSize) {
            throw new RuntimeException("Wrong cursor position : " + mCursorPositionWithinWord
                    + "in a word of size " + mCodePointSize);
        }
        return mCursorPositionWithinWord != mCodePointSize;
    }

    public boolean isCursorInFrontOfComposingWord() {
        return isComposingWord() && mCursorPositionWithinWord == 0;
    }

    /**
     * When the cursor is moved by the user, we need to update its position.
     * If it falls inside the currently composing word, we don't reset the composition, and
     * only update the cursor position.
     *
     * @param expectedMoveAmount How many java chars to move the cursor. Negative values move
     * the cursor backward, positive values move the cursor forward.
     * @return true if the cursor is still inside the composing word, false otherwise.
     */
    public boolean moveCursorByAndReturnIfInsideComposingWord(final int expectedMoveAmount) {
        int actualMoveAmount = 0;
        int cursorPos = mCursorPositionWithinWord;
        // TODO: Don't make that copy. We can do this directly from mTypedWordCache.
        final int[] codePoints = StringUtils.toCodePointArray(mTypedWordCache);
        if (expectedMoveAmount >= 0) {
            // Moving the cursor forward for the expected amount or until the end of the word has
            // been reached, whichever comes first.
            while (actualMoveAmount < expectedMoveAmount && cursorPos < codePoints.length) {
                actualMoveAmount += Character.charCount(codePoints[cursorPos]);
                ++cursorPos;
            }
        } else {
            // Moving the cursor backward for the expected amount or until the start of the word
            // has been reached, whichever comes first.
            while (actualMoveAmount > expectedMoveAmount && cursorPos > 0) {
                --cursorPos;
                actualMoveAmount -= Character.charCount(codePoints[cursorPos]);
            }
        }
        // If the actual and expected amounts differ, we crossed the start or the end of the word
        // so the result would not be inside the composing word.
        if (actualMoveAmount != expectedMoveAmount) {
            return false;
        }
        mCursorPositionWithinWord = cursorPos;
        mCombinerChain.applyProcessedEvent(mCombinerChain.processEvent(
                mEvents, Event.createCursorMovedEvent(cursorPos)));
        return true;
    }

    public void setBatchInputPointers(final InputPointers batchPointers) {
        mInputPointers.set(batchPointers);
        mIsBatchMode = true;
    }

    public void setBatchInputWord(final String word) {
        reset();
        mIsBatchMode = true;
        final int length = word.length();
        for (int i = 0; i < length; i = Character.offsetByCodePoints(word, i, 1)) {
            final int codePoint = Character.codePointAt(word, i);
            // We don't want to override the batch input points that are held in mInputPointers
            // (See {@link #add(int,int,int)}).
            final Event processedEvent = processEvent(Event.createEventForCodePointFromUnknownSource(codePoint));
            applyProcessedEvent(processedEvent);
        }
    }

    /**
     * Set the currently composing word to the one passed as an argument.
     * This will register NOT_A_COORDINATE for X and Ys, and use the passed keyboard for proximity.
     * @param codePoints the code points to set as the composing word.
     * @param coordinates the x, y coordinates of the key in the CoordinateUtils format
     */
    public void setComposingWord(final int[] codePoints, final int[] coordinates) {
        reset();
        final int length = codePoints.length;
        for (int i = 0; i < length; ++i) {
            final Event processedEvent =
                    processEvent(Event.createEventForCodePointFromAlreadyTypedText(codePoints[i],
                        CoordinateUtils.xFromArray(coordinates, i),
                        CoordinateUtils.yFromArray(coordinates, i))
                    );
            applyProcessedEvent(processedEvent);
        }
        mIsResumed = true;
    }

    /**
     * Returns the word as it was typed, without any correction applied.
     * @return the word that was typed so far. Never returns null.
     */
    public String getTypedWord() {
        return mTypedWordCache.toString();
    }

    /**
     * Whether this composer is composing or about to compose a word in which only the first letter
     * is a capital.
     * <p>
     * If we do have a composing word, we just return whether the word has indeed only its first
     * character capitalized. If we don't, then we return a value based on the capitalized mode,
     * which tell us what is likely to happen for the next composing word.
     *
     * @return capitalization preference
     */
    public boolean isOrWillBeOnlyFirstCharCapitalized() {
        return isComposingWord() ? mIsOnlyFirstCharCapitalized
                : (CAPS_MODE_OFF != mCapitalizedMode);
    }

    /**
     * Whether or not all of the user typed chars are upper case
     * @return true if all user typed chars are upper case, false otherwise
     */
    public boolean isAllUpperCase() {
        if (size() <= 1) {
            return mCapitalizedMode == CAPS_MODE_AUTO_SHIFT_LOCKED
                    || mCapitalizedMode == CAPS_MODE_MANUAL_SHIFT_LOCKED;
        }
        return mCapsCount == size();
    }

    public boolean wasShiftedNoLock() {
        return mCapitalizedMode == CAPS_MODE_AUTO_SHIFTED
                || mCapitalizedMode == CAPS_MODE_MANUAL_SHIFTED;
    }

    public char lastChar() {
        if (!isComposingWord()) return 0;
        return mTypedWordCache.charAt(mTypedWordCache.length() - 1);
    }

    /**
     * Returns true if more than one character is upper case, otherwise returns false.
     */
    public boolean isMostlyCaps() {
        return mCapsCount > 1;
    }

    /**
     * Returns true if we have digits in the composing word.
     */
    public boolean hasDigits() {
        return mDigitsCount > 0;
    }

    /**
     * Saves the caps mode at the start of composing.
     * <p>
     * WordComposer needs to know about the caps mode for several reasons. The first is, we need
     * to know after the fact what the reason was, to register the correct form into the user
     * history dictionary: if the word was automatically capitalized, we should insert it in
     * all-lower case but if it's a manual pressing of shift, then it should be inserted as is.
     * Also, batch input needs to know about the current caps mode to display correctly
     * capitalized suggestions.
     * @param mode the mode at the time of start
     */
    public void setCapitalizedModeAtStartComposingTime(final int mode) {
        mCapitalizedMode = mode;
    }

    /**
     * Before fetching suggestions, we don't necessarily know about the capitalized mode yet.
     * <p>
     * If we don't have a composing word yet, we take a note of this mode so that we can then
     * supply this information to the suggestion process. If we have a composing word, then
     * the previous mode has priority over this.
     * @param mode the mode just before fetching suggestions
     */
    public void adviseCapitalizedModeBeforeFetchingSuggestions(final int mode) {
        if (!isComposingWord()) {
            mCapitalizedMode = mode;
        }
    }

    /**
     * Returns whether the word was automatically capitalized.
     * @return whether the word was automatically capitalized
     */
    public boolean wasAutoCapitalized() {
        return mCapitalizedMode == CAPS_MODE_AUTO_SHIFT_LOCKED
                || mCapitalizedMode == CAPS_MODE_AUTO_SHIFTED;
    }

    /**
     * Sets the auto-correction for this word.
     */
    public void setAutoCorrection(final SuggestedWordInfo autoCorrection) {
        mAutoCorrection = autoCorrection;
    }

    /**
     * @return the auto-correction for this word, or null if none.
     */
    public SuggestedWordInfo getAutoCorrectionOrNull() {
        return mAutoCorrection;
    }

    /**
     * @return whether we started composing this word by resuming suggestion on an existing string
     */
    public boolean isResumed() {
        return mIsResumed;
    }

    // `type' should be one of the LastComposedWord.COMMIT_TYPE_* constants above.
    // committedWord should contain suggestion spans if applicable.
    public LastComposedWord commitWord(final int type, final CharSequence committedWord,
            final String separatorString, final NgramContext ngramContext) {
        // Note: currently, we come here whenever we commit a word. If it's a MANUAL_PICK
        // or a DECIDED_WORD we may cancel the commit later; otherwise, we should deactivate
        // the last composed word to ensure this does not happen.
        final LastComposedWord lastComposedWord = new LastComposedWord(mEvents,
                mInputPointers, mTypedWordCache.toString(), committedWord, separatorString,
                ngramContext, mCapitalizedMode);
        mInputPointers.reset();
        if (type != LastComposedWord.COMMIT_TYPE_DECIDED_WORD
                && type != LastComposedWord.COMMIT_TYPE_MANUAL_PICK) {
            lastComposedWord.deactivate();
        }
        mCapsCount = 0;
        mDigitsCount = 0;
        mIsBatchMode = false;
        mCombinerChain.reset();
        mEvents.clear();
        mCodePointSize = 0;
        mIsOnlyFirstCharCapitalized = false;
        mCapitalizedMode = CAPS_MODE_OFF;
        refreshTypedWordCache();
        mAutoCorrection = null;
        mCursorPositionWithinWord = 0;
        mIsResumed = false;
        mRejectedBatchModeSuggestion = null;
        return lastComposedWord;
    }

    public void resumeSuggestionOnLastComposedWord(final LastComposedWord lastComposedWord) {
        mEvents.clear();
        Collections.copy(mEvents, lastComposedWord.mEvents);
        mInputPointers.set(lastComposedWord.mInputPointers);
        mCombinerChain.reset();
        refreshTypedWordCache();
        mCapitalizedMode = lastComposedWord.mCapitalizedMode;
        mAutoCorrection = null; // This will be filled by the next call to updateSuggestion.
        mCursorPositionWithinWord = mCodePointSize;
        mRejectedBatchModeSuggestion = null;
        mIsResumed = true;
    }

    public boolean isBatchMode() {
        return mIsBatchMode;
    }

    public void unsetBatchMode() {
        mIsBatchMode = false;
    }

    public void setRejectedBatchModeSuggestion(final String rejectedSuggestion) {
        mRejectedBatchModeSuggestion = rejectedSuggestion;
    }

    public String getRejectedBatchModeSuggestion() {
        return mRejectedBatchModeSuggestion;
    }

    /**
     * Get the current combining spec.
     * @return the combining spec string, or null if none is set.
     */
    public String getCombiningSpec() {
        return mCombiningSpec;
    }

    void addInputPointerForTest(int index, int keyX, int keyY) {
        mInputPointers.addPointerAt(index, keyX, keyY, 0, 0);
    }

    void setTypedWordCacheForTests(String typedWordCacheForTests) {
        mTypedWordCache = typedWordCacheForTests;
    }

    static WordComposer getComposerForTest(boolean isEmpty) {
        return new WordComposer(isEmpty);
    }

    private WordComposer(boolean isEmpty) {
        mCodePointSize = isEmpty ? 0 : 1;
        mEvents = null;
    }

}
