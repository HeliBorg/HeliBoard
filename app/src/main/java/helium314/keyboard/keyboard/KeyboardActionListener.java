/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.view.KeyEvent;

import helium314.keyboard.event.HapticEvent;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.InputPointers;

public interface KeyboardActionListener {
    /**
     * Called when the user presses a key. This is sent before the {@link #onCodeInput} is called.
     * For keys that repeat, this is only called once.
     *
     * @param primaryCode the unicode of the key being pressed. If the touch is not on a valid key,
     *            the value will be zero.
     * @param repeatCount how many times the key was repeated. Zero if it is the first press.
     * @param isSinglePointer true if pressing has occurred while no other key is being pressed.
     * @param hapticEvent the type of haptic feedback to perform.
     */
    void onPressKey(int primaryCode, int repeatCount, boolean isSinglePointer, HapticEvent hapticEvent);

    void onLongPressKey(int primaryCode);

    /**
     * Called when the user releases a key. This is sent after the {@link #onCodeInput} is called.
     * For keys that repeat, this is only called once.
     *
     * @param primaryCode the code of the key that was released
     * @param withSliding true if releasing has occurred because the user slid finger from the key
     *             to other key without releasing the finger.
     */
    void onReleaseKey(int primaryCode, boolean withSliding);

    /** For handling hardware key presses. Returns whether the event was handled. */
    boolean onKeyDown(int keyCode, KeyEvent keyEvent);

    /** For handling hardware key presses. Returns whether the event was handled. */
    boolean onKeyUp(int keyCode, KeyEvent keyEvent);

    /**
     * Send a key code to the listener.
     *
     * @param primaryCode this is the code of the key that was pressed
     * @param x x-coordinate pixel of touched event. If onCodeInput is not called by
     *            {@link PointerTracker} or so, the value should be
     *            {@link Constants#NOT_A_COORDINATE}. If it's called on insertion from the
     *            suggestion strip, it should be {@link Constants#SUGGESTION_STRIP_COORDINATE}.
     * @param y y-coordinate pixel of touched event. If #onCodeInput is not called by
     *            {@link PointerTracker} or so, the value should be
     *            {@link Constants#NOT_A_COORDINATE}.If it's called on insertion from the
     *            suggestion strip, it should be {@link Constants#SUGGESTION_STRIP_COORDINATE}.
     * @param isKeyRepeat true if this is a key repeat, false otherwise
     */
    // TODO: change this to send an Event object instead
    void onCodeInput(int primaryCode, int x, int y, boolean isKeyRepeat);

    /**
     * Sends a string of characters to the listener.
     *
     * @param text the string of characters to be registered.
     */
    void onTextInput(String text);

    /**
     * Called when user started batch input.
     */
    void onStartBatchInput();

    /**
     * Sends the ongoing batch input points data.
     * @param batchPointers the batch input points representing the user input
     */
    void onUpdateBatchInput(InputPointers batchPointers);

    /**
     * Sends the final batch input points data.
     *
     * @param batchPointers the batch input points representing the user input
     */
    void onEndBatchInput(InputPointers batchPointers);

    void onCancelBatchInput();

    /**
     * Called when user released a finger outside any key.
     */
    void onCancelInput();

    /**
     * Called when user finished sliding key input.
     */
    void onFinishSlidingInput();

    /**
     * Send a non-"code input" custom request to the listener.
     * @return true if the request has been consumed, false otherwise.
     */
    boolean onCustomRequest(int requestCode);

    /**
     * Called when the user performs a horizontal or vertical swipe gesture
     * on the space bar.
     */
    boolean onHorizontalSpaceSwipe(int steps);
    boolean onVerticalSpaceSwipe(int steps);
    void onEndSpaceSwipe();
    boolean toggleNumpad(boolean withSliding, boolean forceReturnToAlpha);

    void onMoveDeletePointer(int steps);
    void onUpWithDeletePointerActive();
    void resetMetaState();

    KeyboardActionListener EMPTY_LISTENER = new Adapter();

    int SWIPE_NO_ACTION = 0;
    int SWIPE_MOVE_CURSOR = 1;
    int SWIPE_SWITCH_LANGUAGE = 2;
    int SWIPE_TOGGLE_NUMPAD = 3;
    int SWIPE_HIDE_KEYBOARD = 4;
    int SWIPE_TOUCHPAD_MODE = 5;

    class Adapter implements KeyboardActionListener {
        @Override
        public void onPressKey(int primaryCode, int repeatCount, boolean isSinglePointer, HapticEvent hapticEvent) {}
        @Override
        public void onLongPressKey(int primaryCode) {}
        @Override
        public void onReleaseKey(int primaryCode, boolean withSliding) {}
        @Override
        public boolean onKeyDown(int keyCode, KeyEvent keyEvent) { return false; }
        @Override
        public boolean onKeyUp(int keyCode, KeyEvent keyEvent) { return false; }
        @Override
        public void onCodeInput(int primaryCode, int x, int y, boolean isKeyRepeat) {}
        @Override
        public void onTextInput(String text) {}
        @Override
        public void onStartBatchInput() {}
        @Override
        public void onUpdateBatchInput(InputPointers batchPointers) {}
        @Override
        public void onEndBatchInput(InputPointers batchPointers) {}
        @Override
        public void onCancelBatchInput() {}
        @Override
        public void onCancelInput() {}
        @Override
        public void onFinishSlidingInput() {}
        @Override
        public boolean onCustomRequest(int requestCode) {
            return false;
        }
        @Override
        public boolean onHorizontalSpaceSwipe(int steps) {
            return false;
        }
        @Override
        public boolean onVerticalSpaceSwipe(int steps) {
            return false;
        }
        @Override
        public boolean toggleNumpad(boolean withSliding, boolean forceReturnToAlpha) {
            return false;
        }
        @Override
        public void onEndSpaceSwipe() {}
        @Override
        public void onMoveDeletePointer(int steps) {}
        @Override
        public void onUpWithDeletePointerActive() {}
        @Override
        public void resetMetaState() {}
    }
}
