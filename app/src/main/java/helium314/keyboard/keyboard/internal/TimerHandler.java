/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import android.os.Message;
import android.os.SystemClock;
import android.view.ViewConfiguration;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.PointerTracker;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.utils.LeakGuardHandlerWrapper;

public final class TimerHandler extends LeakGuardHandlerWrapper<DrawingProxy>
        implements TimerProxy {
    private static final int MSG_TYPING_STATE_EXPIRED = 0;
    private static final int MSG_REPEAT_KEY = 1;
    private static final int MSG_LONGPRESS_KEY = 2;
    private static final int MSG_LONGPRESS_SHIFT_KEY = 3;
    private static final int MSG_LONGPRESS_ALPHA_SYMBOL_KEY = 4;
    private static final int MSG_DOUBLE_TAP_SHIFT_KEY = 5;
    private static final int MSG_UPDATE_BATCH_INPUT = 6;
    private static final int MSG_DISMISS_KEY_PREVIEW = 7;
    private static final int MSG_DISMISS_GESTURE_FLOATING_PREVIEW_TEXT = 8;

    private final int mIgnoreAltCodeKeyTimeout;
    private final int mGestureRecognitionUpdateTime;

    public TimerHandler(@NonNull DrawingProxy ownerInstance,
            int ignoreAltCodeKeyTimeout, int gestureRecognitionUpdateTime) {
        super(ownerInstance);
        mIgnoreAltCodeKeyTimeout = ignoreAltCodeKeyTimeout;
        mGestureRecognitionUpdateTime = gestureRecognitionUpdateTime;
    }

    @Override
    public void handleMessage(Message msg) {
        DrawingProxy drawingProxy = getOwnerInstance();
        if (drawingProxy == null) {
            return;
        }
        switch (msg.what) {
        case MSG_TYPING_STATE_EXPIRED -> {
            drawingProxy.startWhileTypingAnimation(DrawingProxy.FADE_IN);
        }
        case MSG_REPEAT_KEY -> {
            PointerTracker tracker1 = (PointerTracker)msg.obj;
            tracker1.onKeyRepeat(msg.arg1 /* code */, msg.arg2 /* repeatCount */);
        }
        case MSG_LONGPRESS_KEY, MSG_LONGPRESS_SHIFT_KEY, MSG_LONGPRESS_ALPHA_SYMBOL_KEY -> {
            cancelLongPressTimers();
            PointerTracker tracker2 = (PointerTracker)msg.obj;
            tracker2.onLongPressed();
        }
        case MSG_UPDATE_BATCH_INPUT -> {
            PointerTracker tracker3 = (PointerTracker)msg.obj;
            tracker3.updateBatchInputByTimer(SystemClock.uptimeMillis());
            startUpdateBatchInputTimer(tracker3);
        }
        case MSG_DISMISS_KEY_PREVIEW -> {
            drawingProxy.onKeyReleased((Key)msg.obj, false /* withAnimation */);
        }
        case MSG_DISMISS_GESTURE_FLOATING_PREVIEW_TEXT -> {
            drawingProxy.dismissGestureFloatingPreviewTextWithoutDelay();
        }
        }
    }

    @Override
    public void startKeyRepeatTimerOf(@NonNull PointerTracker tracker,
            int repeatCount, int delay) {
        Key key = tracker.getKey();
        if (key == null || delay == 0) {
            return;
        }
        sendMessageDelayed(
                obtainMessage(MSG_REPEAT_KEY, key.getCode(), repeatCount, tracker), delay);
    }

    private void cancelKeyRepeatTimerOf(PointerTracker tracker) {
        removeMessages(MSG_REPEAT_KEY, tracker);
    }

    public void cancelKeyRepeatTimers() {
        removeMessages(MSG_REPEAT_KEY);
    }

    // TODO: Suppress layout changes in key repeat mode
    public boolean isInKeyRepeat() {
        return hasMessages(MSG_REPEAT_KEY);
    }

    @Override
    public void startLongPressTimerOf(@NonNull PointerTracker tracker, int delay) {
        Key key = tracker.getKey();
        if (key == null) {
            return;
        }
        // Use a separate message id for long pressing shift key, because long press shift key
        // timers should be canceled when other key is pressed.
        int messageId = switch (key.getCode()) {
            case KeyCode.SHIFT -> MSG_LONGPRESS_SHIFT_KEY;
            case KeyCode.SYMBOL_ALPHA -> MSG_LONGPRESS_ALPHA_SYMBOL_KEY;
            default -> MSG_LONGPRESS_KEY;
        };
        sendMessageDelayed(obtainMessage(messageId, tracker), delay);
    }

    @Override
    public void cancelLongPressTimersOf(@NonNull PointerTracker tracker) {
        removeMessages(MSG_LONGPRESS_KEY, tracker);
        removeMessages(MSG_LONGPRESS_SHIFT_KEY, tracker);
        removeMessages(MSG_LONGPRESS_ALPHA_SYMBOL_KEY, tracker);
    }

    @Override
    public void cancelLongPressShiftKeyTimer() {
        removeMessages(MSG_LONGPRESS_SHIFT_KEY);
    }

    @Override
    public void cancelLongPressAlphaSymbolKeyTimer() {
        removeMessages(MSG_LONGPRESS_ALPHA_SYMBOL_KEY);
    }

    public void cancelLongPressTimers() {
        removeMessages(MSG_LONGPRESS_KEY);
        removeMessages(MSG_LONGPRESS_SHIFT_KEY);
        removeMessages(MSG_LONGPRESS_ALPHA_SYMBOL_KEY);
    }

    @Override
    public void startTypingStateTimer(@NonNull Key typedKey) {
        if (typedKey.isModifier() || typedKey.altCodeWhileTyping()) {
            return;
        }

        boolean isTyping = isTypingState();
        removeMessages(MSG_TYPING_STATE_EXPIRED);
        DrawingProxy drawingProxy = getOwnerInstance();
        if (drawingProxy == null) {
            return;
        }

        // When user hits the space or the enter key, just cancel the while-typing timer.
        int typedCode = typedKey.getCode();
        if (typedCode == Constants.CODE_SPACE || typedCode == Constants.CODE_ENTER) {
            if (isTyping) {
                drawingProxy.startWhileTypingAnimation(DrawingProxy.FADE_IN);
            }
            return;
        }

        sendMessageDelayed(
                obtainMessage(MSG_TYPING_STATE_EXPIRED), mIgnoreAltCodeKeyTimeout);
        if (isTyping) {
            return;
        }
        drawingProxy.startWhileTypingAnimation(DrawingProxy.FADE_OUT);
    }

    @Override
    public boolean isTypingState() {
        return hasMessages(MSG_TYPING_STATE_EXPIRED);
    }

    @Override
    public void startDoubleTapShiftKeyTimer() {
        sendMessageDelayed(obtainMessage(MSG_DOUBLE_TAP_SHIFT_KEY),
                ViewConfiguration.getDoubleTapTimeout());
    }

    @Override
    public void cancelDoubleTapShiftKeyTimer() {
        removeMessages(MSG_DOUBLE_TAP_SHIFT_KEY);
    }

    @Override
    public boolean popDoubleTapShiftKeyTimer() {
        boolean isDoubleTap = hasMessages(MSG_DOUBLE_TAP_SHIFT_KEY);
        if (isDoubleTap) {
            removeMessages(MSG_DOUBLE_TAP_SHIFT_KEY);
        }
        return isDoubleTap;
    }

    @Override
    public void cancelKeyTimersOf(@NonNull PointerTracker tracker) {
        cancelKeyRepeatTimerOf(tracker);
        cancelLongPressTimersOf(tracker);
    }

    public void cancelAllKeyTimers() {
        cancelKeyRepeatTimers();
        cancelLongPressTimers();
    }

    @Override
    public void startUpdateBatchInputTimer(@NonNull PointerTracker tracker) {
        if (mGestureRecognitionUpdateTime <= 0) {
            return;
        }
        removeMessages(MSG_UPDATE_BATCH_INPUT, tracker);
        sendMessageDelayed(obtainMessage(MSG_UPDATE_BATCH_INPUT, tracker),
                mGestureRecognitionUpdateTime);
    }

    @Override
    public void cancelUpdateBatchInputTimer(@NonNull PointerTracker tracker) {
        removeMessages(MSG_UPDATE_BATCH_INPUT, tracker);
    }

    @Override
    public void cancelAllUpdateBatchInputTimers() {
        removeMessages(MSG_UPDATE_BATCH_INPUT);
    }

    public void postDismissKeyPreview(@NonNull Key key, long delay) {
        sendMessageDelayed(obtainMessage(MSG_DISMISS_KEY_PREVIEW, key), delay);
    }

    public void postDismissGestureFloatingPreviewText(long delay) {
        sendMessageDelayed(obtainMessage(MSG_DISMISS_GESTURE_FLOATING_PREVIEW_TEXT), delay);
    }

    public void cancelAllMessages() {
        cancelAllKeyTimers();
        cancelAllUpdateBatchInputTimers();
        removeMessages(MSG_DISMISS_KEY_PREVIEW);
        removeMessages(MSG_DISMISS_GESTURE_FLOATING_PREVIEW_TEXT);
    }
}
