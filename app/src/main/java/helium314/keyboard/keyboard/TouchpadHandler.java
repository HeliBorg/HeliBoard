package helium314.keyboard.keyboard;

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.settings.SettingsValues;

public class TouchpadHandler {
    private static boolean sTouchpadModeActive = false;
    private boolean mInTouchpadMode = false;
    private static final float TOUCHPAD_ACCELERATION_FACTOR = 50.0f; // Lower = more acceleration
    private long mTouchpadActivationTime;
    private int mTouchpadLastX, mTouchpadLastY;
    // Accumulators for fractional movement
    private int mTouchpadAccX = 0;
    private int mTouchpadAccY = 0;

    public static void setTouchpadModeActive(boolean active) {
        sTouchpadModeActive = active;
    }

    public void disableTouchpadMode(KeyboardActionListener listener) {
        if (!mInTouchpadMode) return;
        mInTouchpadMode = false;
        sTouchpadModeActive = false;
        listener.onCustomRequest(KeyboardActionListener.CODE_TOUCHPAD_OFF);
    }

    public void enableTouchpadMove(int x, int y, KeyboardActionListener listener) {
        if (!sTouchpadModeActive) return;

        // Initialize
        if (!mInTouchpadMode) {
            mInTouchpadMode = true;
            mTouchpadLastX = x;
            mTouchpadLastY = y;
            mTouchpadActivationTime = System.currentTimeMillis();
            listener.onCustomRequest(KeyboardActionListener.CODE_TOUCHPAD_ON);
            return;
        }

        onMove(x, y, listener);
    }

    private void onMove(int x, int y, KeyboardActionListener sListener) {
        SettingsValues sv = Settings.getValues();

        // Debounce
        if (System.currentTimeMillis() - mTouchpadActivationTime < sv.mKeyLongpressTimeout) {
            mTouchpadLastX = x;
            mTouchpadLastY = y;
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
            sListener.onCodeInput(direction, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false);
            mTouchpadAccX -= (positive ? moveThreshold : -moveThreshold);
        }

        // Handle vertical movement with accumulator
        while (Math.abs(mTouchpadAccY) >= moveThreshold) {
            boolean positive = mTouchpadAccY > 0;
            int direction = positive ? KeyCode.ARROW_DOWN : KeyCode.ARROW_UP;
            sListener.onCodeInput(direction, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false);
            mTouchpadAccY -= (positive ? moveThreshold : -moveThreshold);
        }
    }
}
