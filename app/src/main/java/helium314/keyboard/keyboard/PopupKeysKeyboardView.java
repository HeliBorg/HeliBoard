/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.NonNull;

import helium314.keyboard.accessibility.AccessibilityUtils;
import helium314.keyboard.accessibility.PopupKeysKeyboardAccessibilityDelegate;
import helium314.keyboard.keyboard.internal.KeyDrawParams;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.RichInputMethodManager;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.CoordinateUtils;
import kotlin.Unit;
import kotlin.jvm.functions.Function1;

/**
 * A view that renders a virtual {@link PopupKeysKeyboard}. It handles rendering of keys and
 * detecting key presses and touch movements.
 */
public class PopupKeysKeyboardView extends KeyboardView implements PopupKeysPanel {
    private final int[] mCoordinates = CoordinateUtils.newInstance();

    private final Drawable mDivider;
    protected final KeyDetector mKeyDetector;
    private Controller mController = EMPTY_CONTROLLER;
    protected KeyboardActionListener mListener;
    protected Function1<Key, Unit> mOnReleaseKey;
    private int mOriginX;
    private int mOriginY;
    private Key mCurrentKey;

    private int mActivePointerId;

    protected PopupKeysKeyboardAccessibilityDelegate mAccessibilityDelegate;

    public PopupKeysKeyboardView(final Context context, final AttributeSet attrs) {
        this(context, attrs, R.attr.popupKeysKeyboardViewStyle);
    }

    public PopupKeysKeyboardView(final Context context, final AttributeSet attrs,
                                 final int defStyle) {
        super(context, attrs, defStyle);
        final TypedArray popupKeysKeyboardViewAttr = context.obtainStyledAttributes(attrs,
                R.styleable.PopupKeysKeyboardView, defStyle, R.style.PopupKeysKeyboardView);
        mDivider = popupKeysKeyboardViewAttr.getDrawable(R.styleable.PopupKeysKeyboardView_divider);
        popupKeysKeyboardViewAttr.recycle();
        mKeyDetector = new PopupKeysDetector(getResources().getDimension(
                R.dimen.config_popup_keys_keyboard_slide_allowance));
    }

    @Override
    protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
        final Keyboard keyboard = getKeyboard();
        if (keyboard != null) {
            final int width = keyboard.mOccupiedWidth + getPaddingLeft() + getPaddingRight();
            final int height = keyboard.mOccupiedHeight + getPaddingTop() + getPaddingBottom();
            setMeasuredDimension(width, height);
        } else {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        }
    }

    @Override
    protected void onDrawKeyTopVisuals(@NonNull final Key key, @NonNull final Canvas canvas,
            @NonNull final Paint paint, @NonNull final KeyDrawParams params) {
        if (!key.isSpacer() || !(key instanceof PopupKeysKeyboard.PopupKeyDivider)
                || mDivider == null) {
            super.onDrawKeyTopVisuals(key, canvas, paint, params);
            return;
        }
        final int keyWidth = key.getDrawWidth();
        final int keyHeight = key.getHeight();
        final int iconWidth = Math.min(mDivider.getIntrinsicWidth(), keyWidth);
        final int iconHeight = mDivider.getIntrinsicHeight();
        final int iconX = (keyWidth - iconWidth) / 2; // Align horizontally center
        final int iconY = (keyHeight - iconHeight) / 2; // Align vertically center
        drawIcon(canvas, mDivider, iconX, iconY, iconWidth, iconHeight);
    }

    @Override
    public void setKeyboard(@NonNull final Keyboard keyboard) {
        super.setKeyboard(keyboard);
        mKeyDetector.setKeyboard(
                keyboard, -getPaddingLeft(), -getPaddingTop() + getVerticalCorrection());
        if (AccessibilityUtils.Companion.getInstance().isAccessibilityEnabled()) {
            if (mAccessibilityDelegate == null) {
                mAccessibilityDelegate = new PopupKeysKeyboardAccessibilityDelegate(
                        this, mKeyDetector);
                mAccessibilityDelegate.setOpenAnnounce(R.string.spoken_open_popup_keys_keyboard);
                mAccessibilityDelegate.setCloseAnnounce(R.string.spoken_close_popup_keys_keyboard);
            }
            mAccessibilityDelegate.setKeyboard(keyboard);
        } else {
            mAccessibilityDelegate = null;
        }
        final Key shortcutKey = keyboard.getKey(KeyCode.VOICE_INPUT);
        if (shortcutKey != null) {
            shortcutKey.setEnabled(RichInputMethodManager.getInstance().isShortcutImeReady());
            invalidateKey(shortcutKey);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void showPopupKeysPanel(View parentView, Controller controller,
            int pointX, int pointY, KeyboardActionListener listener) {
        mListener = listener;
        mOnReleaseKey = null;
        showPopupKeysPanelInternal(parentView, controller, pointX, pointY);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void showPopupKeysPanel(View parentView, Controller controller,
            int pointX, int pointY, Function1<Key, Unit> onReleaseKey) {
        mListener = null;
        mOnReleaseKey = onReleaseKey;
        showPopupKeysPanelInternal(parentView, controller, pointX, pointY);
    }

    @SuppressLint("RtlHardcoded") // a key on the left is on the left, independent of layout direction
    private void showPopupKeysPanelInternal(final View parentView, final Controller controller,
            final int pointX, final int pointY) {
        mController = controller;
        final View container = getContainerView();
        // The coordinates of panel's left-top corner in parentView's coordinate system.
        // We need to consider background drawable paddings.
        final int x = pointX - getDefaultCoordX() - container.getPaddingLeft() - getPaddingLeft();
        final int y = pointY - container.getMeasuredHeight() + container.getPaddingBottom()
                + getPaddingBottom();

        parentView.getLocationInWindow(mCoordinates);
        final int containerY = y + CoordinateUtils.y(mCoordinates);
        container.setY(containerY);

        // This is needed for cases where there's also a text popup above this keyboard
        final int panelMaxX = parentView.getMeasuredWidth() - getMeasuredWidth();
        var panelFinalX = Math.max(0, Math.min(panelMaxX, x));
        var center = panelFinalX + getMeasuredWidth() / 2;
        var layoutGravity = center < pointX - getKeyboard().mMostCommonKeyWidth / 2?
                        Gravity.RIGHT : center > pointX + getKeyboard().mMostCommonKeyWidth / 2? Gravity.LEFT : Gravity.CENTER_HORIZONTAL;

        int containerAdjustedX = x;
        if (getMeasuredWidth() < container.getMeasuredWidth()) {
            containerAdjustedX = layoutGravity == Gravity.LEFT? panelFinalX : layoutGravity == Gravity.RIGHT
                ? panelFinalX + getMeasuredWidth() - container.getMeasuredWidth()
                : panelFinalX + (getMeasuredWidth() - container.getMeasuredWidth()) / 2;
        }

        // Ensure the horizontal position of the panel does not extend past the parentView edges.
        int containerMaxX = parentView.getMeasuredWidth() - container.getMeasuredWidth();
        int containerFinalX = Math.max(0, Math.min(containerMaxX, containerAdjustedX));
        int containerX = containerFinalX + CoordinateUtils.x(mCoordinates);
        container.setX(containerX);
        setTranslationX(panelFinalX - containerFinalX);
        controller.setLayoutGravity(layoutGravity);

        mOriginX = panelFinalX;
        mOriginY = y + container.getPaddingTop() + (int) getY();
        controller.onShowPopupKeysPanel(this);
        final PopupKeysKeyboardAccessibilityDelegate accessibilityDelegate = mAccessibilityDelegate;
        if (accessibilityDelegate != null
                && AccessibilityUtils.Companion.getInstance().isAccessibilityEnabled()) {
            accessibilityDelegate.onShowPopupKeysKeyboard();
        }
    }

    /**
     * Returns the default x coordinate for showing this panel.
     */
    protected int getDefaultCoordX() {
        return ((PopupKeysKeyboard)getKeyboard()).getDefaultCoordX();
    }

    @Override
    public void onDownEvent(final int x, final int y, final int pointerId, final long eventTime) {
        mActivePointerId = pointerId;
        mCurrentKey = detectKey(x, y);
    }

    @Override
    public void onMoveEvent(final int x, final int y, final int pointerId, final long eventTime) {
        if (mActivePointerId != pointerId) {
            return;
        }
        final boolean hasOldKey = (mCurrentKey != null);
        mCurrentKey = detectKey(x, y);
        if (hasOldKey && mCurrentKey == null) {
            // A popup keys keyboard is canceled when detecting no key.
            mController.onCancelPopupKeysPanel();
        }
    }

    @Override
    public void onUpEvent(final int x, final int y, final int pointerId, final long eventTime) {
        if (mActivePointerId != pointerId) {
            return;
        }
        // Calling {@link #detectKey(int,int,int)} here is harmless because the last move event and
        // the following up event share the same coordinates.
        mCurrentKey = detectKey(x, y);
        if (mCurrentKey != null) {
            updateReleaseKeyGraphics(mCurrentKey);
            onKeyInput(mCurrentKey, x, y);
            mCurrentKey = null;
        }
    }

    /**
     * Performs the specific action for this panel when the user presses a key on the panel.
     */
    protected void onKeyInput(final Key key, final int x, final int y) {
        if (mListener != null) {
            final int code = key.getCode();
            if (code == KeyCode.MULTIPLE_CODE_POINTS) {
                mListener.onTextInput(mCurrentKey.getOutputText());
            } else if (code != KeyCode.NOT_SPECIFIED) {
                if (getKeyboard().hasProximityCharsCorrection(code)) {
                    mListener.onCodeInput(code, x, y, false /* isKeyRepeat */);
                } else {
                    mListener.onCodeInput(code, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE,
                            false /* isKeyRepeat */);
                }
            }
        } else if (mOnReleaseKey != null) {
            mOnReleaseKey.invoke(key);
        }
    }

    private Key detectKey(int x, int y) {
        final Key oldKey = mCurrentKey;
        final Key newKey = mKeyDetector.detectHitKey(x, y);
        if (newKey == oldKey) {
            return newKey;
        }
        // A new key is detected.
        if (oldKey != null) {
            updateReleaseKeyGraphics(oldKey);
            invalidateKey(oldKey);
        }
        if (newKey != null) {
            updatePressKeyGraphics(newKey);
            invalidateKey(newKey);
        }
        return newKey;
    }

    private void updateReleaseKeyGraphics(final Key key) {
        key.onReleased();
        invalidateKey(key);
    }

    private void updatePressKeyGraphics(final Key key) {
        key.onPressed();
        invalidateKey(key);
    }

    @Override
    public void dismissPopupKeysPanel() {
        if (!isShowingInParent()) {
            return;
        }
        final PopupKeysKeyboardAccessibilityDelegate accessibilityDelegate = mAccessibilityDelegate;
        if (accessibilityDelegate != null
                && AccessibilityUtils.Companion.getInstance().isAccessibilityEnabled()) {
            accessibilityDelegate.onDismissPopupKeysKeyboard();
        }
        mController.onDismissPopupKeysPanel();
    }

    @Override
    public int translateX(final int x) {
        return x - mOriginX;
    }

    @Override
    public int translateY(final int y) {
        return y - mOriginY;
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(final MotionEvent me) {
        final int action = me.getActionMasked();
        final long eventTime = me.getEventTime();
        final int index = me.getActionIndex();
        final int x = (int)me.getX(index);
        final int y = (int)me.getY(index);
        final int pointerId = me.getPointerId(index);
        switch (action) {
        case MotionEvent.ACTION_DOWN:
        case MotionEvent.ACTION_POINTER_DOWN:
            onDownEvent(x, y, pointerId, eventTime);
            break;
        case MotionEvent.ACTION_UP:
        case MotionEvent.ACTION_POINTER_UP:
            onUpEvent(x, y, pointerId, eventTime);
            break;
        case MotionEvent.ACTION_MOVE:
            onMoveEvent(x, y, pointerId, eventTime);
            break;
        }
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean onHoverEvent(final MotionEvent event) {
        final PopupKeysKeyboardAccessibilityDelegate accessibilityDelegate = mAccessibilityDelegate;
        if (accessibilityDelegate != null
                && AccessibilityUtils.Companion.getInstance().isTouchExplorationEnabled()) {
            return accessibilityDelegate.onHoverEvent(event);
        }
        return super.onHoverEvent(event);
    }
}
