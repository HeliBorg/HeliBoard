// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard;

import android.os.SystemClock;

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.settings.SettingsValues;

public class TouchpadHandler {
    private KeyboardActionListener mListener;
    private static boolean sTouchpadModeActive = false;
    private boolean mInTouchpadMode = false;
    private boolean mHasVibrated = false;
    private static final float TOUCHPAD_ACCELERATION_FACTOR = 50.0f; // Lower = more acceleration
    private long mTouchpadActivationTime;
    private int mTouchpadLastX, mTouchpadLastY;
    // Accumulators for fractional movement
    private int mTouchpadAccX = 0;
    private int mTouchpadAccY = 0;

    private final android.os.Handler mEdgeHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private boolean mIsScrolling = false;
    private static final int SCROLL_DELAY_MS = 100; // Edge scroll speed
    private static final int DIRECTION_UP = 1;
    private static final int DIRECTION_DOWN = 2;
    private static final int DIRECTION_LEFT = 3;
    private static final int DIRECTION_RIGHT = 4;
    private int mCurrentScrollDirection = 0;

    public static void setTouchpadModeActive(boolean active) {
        sTouchpadModeActive = active;
    }

    public void disableTouchpadMode() {
        if (!mInTouchpadMode) return;
        stopEdgeScrolling();
        mInTouchpadMode = false;
        sTouchpadModeActive = false;
        mListener.onCustomRequest(Constants.CODE_TOUCHPAD_OFF);
        stopEdgeScrolling();
        mListener = null;
    }

    public void enableTouchpadMove(int x, int y, KeyboardActionListener listener) {
        if (!sTouchpadModeActive) return;

        // Initialize
        if (!mInTouchpadMode) {
            mListener = listener;
            mInTouchpadMode = true;
            mHasVibrated = false;
            mTouchpadLastX = x;
            mTouchpadLastY = y;
            mTouchpadActivationTime = SystemClock.elapsedRealtime();
            mListener.onCustomRequest(Constants.CODE_TOUCHPAD_ON);
            return;
        }

        onMove(x, y);
    }

    private void onMove(int x, int y) {
        SettingsValues sv = Settings.getValues();

        // Debounce
        if (SystemClock.elapsedRealtime() - mTouchpadActivationTime < sv.mKeyLongpressTimeout) {
            mTouchpadLastX = x;
            mTouchpadLastY = y;
            return;
        }

        if (!mHasVibrated) {
            mListener.onCustomRequest(Constants.CODE_PERFORM_HAPTIC);
            mHasVibrated = true;
        }

        if (sv.mTouchpadEdgeScroll && handleEdgeScrolling(x, y)) {
            return;
        }

        // In touchpad mode - track both horizontal and vertical movement for 2D cursor control
        int deltaX = x - mTouchpadLastX;
        int deltaY = y - mTouchpadLastY;

        mTouchpadLastX = x;
        mTouchpadLastY = y;

        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            // Horizontal move, X only
            float accFactorX = 1.0f + (Math.abs(deltaX) / TOUCHPAD_ACCELERATION_FACTOR);
            mTouchpadAccX += (int) (deltaX * accFactorX);
            mTouchpadAccY = 0;
        } else {
            // Vertical move, Y only
            float accFactorY = 1.0f + (Math.abs(deltaY) / TOUCHPAD_ACCELERATION_FACTOR);
            mTouchpadAccY += (int) (deltaY * accFactorY);
            mTouchpadAccX = 0;
        }

        // Calculate dynamic threshold based on sensitivity setting (0-100)
        // Higher sensitivity = Lower threshold (faster cursor)
        // 0 -> 70px (Very Slow)
        // 50 -> 40px (Default)
        // 100 -> 10px (Very Fast)
        int sensitivity = Settings.getInstance().getCurrent().mTouchpadSensitivity;
        int moveThreshold = 70 - (int) (sensitivity * 0.6f);

        // Handle horizontal movement with accumulator
        while (Math.abs(mTouchpadAccX) >= moveThreshold) {
            boolean positive = mTouchpadAccX > 0;
            int direction = positive ? KeyCode.ARROW_RIGHT : KeyCode.ARROW_LEFT;
            mListener.onCodeInput(direction, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false);
            mTouchpadAccX -= (positive ? moveThreshold : -moveThreshold);
        }

        // Handle vertical movement with accumulator
        while (Math.abs(mTouchpadAccY) >= moveThreshold) {
            boolean positive = mTouchpadAccY > 0;
            int direction = positive ? KeyCode.ARROW_DOWN : KeyCode.ARROW_UP;
            mListener.onCodeInput(direction, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false);
            mTouchpadAccY -= (positive ? moveThreshold : -moveThreshold);
        }
    }

    private final Runnable mScrollRunnable = new Runnable() {
        @Override
        public void run() {
            if (mIsScrolling && mListener != null) {
                int keyCode = KeyCode.UNSPECIFIED;

                if (mCurrentScrollDirection == DIRECTION_UP) {
                    keyCode = KeyCode.ARROW_UP;
                } else if (mCurrentScrollDirection == DIRECTION_DOWN) {
                    keyCode = KeyCode.ARROW_DOWN;
                } else if (mCurrentScrollDirection == DIRECTION_LEFT) {
                    keyCode = KeyCode.ARROW_LEFT;
                } else if (mCurrentScrollDirection == DIRECTION_RIGHT) {
                    keyCode = KeyCode.ARROW_RIGHT;
                }
                mListener.onCodeInput(keyCode, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false);
                mEdgeHandler.postDelayed(this, SCROLL_DELAY_MS);
            }
        }
    };

    private boolean handleEdgeScrolling(int x, int y) {
        Keyboard currentKeyboard = KeyboardSwitcher.getInstance().getKeyboard();
        if (currentKeyboard == null) return false;

        int keyboardHeight = currentKeyboard.mBaseHeight;
        int keyboardWidth = currentKeyboard.mBaseWidth;
        int threshold = 50;

        if (y <= threshold) {
            mCurrentScrollDirection = DIRECTION_UP;
            startEdgeScrolling();
            return true;
        } else if (y >= (keyboardHeight - threshold)) {
            mCurrentScrollDirection = DIRECTION_DOWN;
            startEdgeScrolling();
            return true;
        } else if (x <= threshold * 2) {
            mCurrentScrollDirection = DIRECTION_LEFT;
            startEdgeScrolling();
            return true;
        } else if (x >= (keyboardWidth - threshold * 2)) {
            mCurrentScrollDirection = DIRECTION_RIGHT;
            startEdgeScrolling();
            return true;
        } else {
            stopEdgeScrolling();
            return false;
        }
    }

    private void startEdgeScrolling() {
        if (!mIsScrolling) {
            mIsScrolling = true;
            mEdgeHandler.removeCallbacks(mScrollRunnable);
            mEdgeHandler.post(mScrollRunnable);
        }
    }

    private void stopEdgeScrolling() {
        mIsScrolling = false;
        mEdgeHandler.removeCallbacks(mScrollRunnable);
    }
}
