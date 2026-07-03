/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.PointerTracker;

public interface TimerProxy {
    /**
     * Start a timer to detect if a user is typing keys.
     * @param typedKey the key that is typed.
     */
    void startTypingStateTimer(@NonNull Key typedKey);

    /**
     * Check if a user is key typing.
     * @return true if a user is in typing.
     */
    boolean isTypingState();

    /**
     * Start a timer to simulate repeated key presses while a user keep pressing a key.
     * @param tracker the {@link PointerTracker} that points the key to be repeated.
     * @param repeatCount the number of times that the key is repeating. Starting from 1.
     * @param delay the interval delay to the next key repeat, in millisecond.
     */
    void startKeyRepeatTimerOf(@NonNull PointerTracker tracker, int repeatCount, int delay);

    /**
     * Start a timer to detect a long pressed key.
     * If a key pointed by <code>tracker</code> is a shift key, start another timer to detect
     * long pressed shift key.
     * @param tracker the {@link PointerTracker} that starts long pressing.
     * @param delay the delay to fire the long press timer, in millisecond.
     */
    void startLongPressTimerOf(@NonNull PointerTracker tracker, int delay);

    /**
     * Cancel timers for detecting a long pressed key and a long press shift key.
     * @param tracker cancel long press timers of this {@link PointerTracker}.
     */
    void cancelLongPressTimersOf(@NonNull PointerTracker tracker);

    /**
     * Cancel a timer for detecting a long pressed shift key.
     */
    void cancelLongPressShiftKeyTimer();

    /**
     * Cancel timers for detecting repeated key press, long pressed key, and long pressed shift key.
     * @param tracker the {@link PointerTracker} that starts timers to be canceled.
     */
    void cancelKeyTimersOf(@NonNull PointerTracker tracker);

    /**
     * Start a timer to detect double tapped shift key.
     */
    void startDoubleTapShiftKeyTimer();

    /**
     * Cancel a timer of detecting double tapped shift key.
     */
    void cancelDoubleTapShiftKeyTimer();

    /**
     * Check if a timer of detecting double tapped shift key is running, then cancel it.
     * @return true if detecting double tapped shift key was ongoing.
     */
    boolean popDoubleTapShiftKeyTimer();

    /**
     * Start a timer to fire updating batch input while <code>tracker</code> is on hold.
     * @param tracker the {@link PointerTracker} that stops moving.
     */
    void startUpdateBatchInputTimer(@NonNull PointerTracker tracker);

    /**
     * Cancel a timer of firing updating batch input.
     * @param tracker the {@link PointerTracker} that resumes moving or ends gesture input.
     */
    void cancelUpdateBatchInputTimer(@NonNull PointerTracker tracker);

    /**
     * Cancel all timers of firing updating batch input.
     */
    void cancelAllUpdateBatchInputTimers();

    class Adapter implements TimerProxy {
        @Override
        public void startTypingStateTimer(@NonNull Key typedKey) {}
        @Override
        public boolean isTypingState() { return false; }
        @Override
        public void startKeyRepeatTimerOf(@NonNull PointerTracker tracker, int repeatCount,
                int delay) {}
        @Override
        public void startLongPressTimerOf(@NonNull PointerTracker tracker, int delay) {}
        @Override
        public void cancelLongPressTimersOf(@NonNull PointerTracker tracker) {}
        @Override
        public void cancelLongPressShiftKeyTimer() {}
        @Override
        public void cancelKeyTimersOf(@NonNull PointerTracker tracker) {}
        @Override
        public void startDoubleTapShiftKeyTimer() {}
        @Override
        public void cancelDoubleTapShiftKeyTimer() {}
        @Override
        public boolean popDoubleTapShiftKeyTimer() { return false; }
        @Override
        public void startUpdateBatchInputTimer(@NonNull PointerTracker tracker) {}
        @Override
        public void cancelUpdateBatchInputTimer(@NonNull PointerTracker tracker) {}
        @Override
        public void cancelAllUpdateBatchInputTimers() {}
    }
}
