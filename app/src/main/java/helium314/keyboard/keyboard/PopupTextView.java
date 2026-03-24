/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.content.Context;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.widget.TextView;
import helium314.keyboard.keyboard.emoji.EmojiViewCallback;
import helium314.keyboard.keyboard.internal.KeyDrawParams;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.common.ColorType;
import helium314.keyboard.latin.common.CoordinateUtils;
import helium314.keyboard.latin.settings.Settings;


/**
 * A view that displays popup text.
 */
public class PopupTextView extends TextView implements PopupKeysPanel {
    private final int[] mCoordinates = CoordinateUtils.newInstance();
    private Controller mController = EMPTY_CONTROLLER;
    private int mOriginX;
    private int mOriginY;
    private Key mKey;
    private EmojiViewCallback mEmojiViewCallback;

    public PopupTextView(final Context context, final AttributeSet attrs) {
        this(context, attrs, R.attr.popupKeysKeyboardViewStyle);
    }

    public PopupTextView(final Context context, final AttributeSet attrs,
                         final int defStyle) {
        super(context, attrs, defStyle);
    }

    public void setKeyDrawParams(Key key, KeyDrawParams drawParams) {
        mKey = key;
        Settings.getValues().mColors.setBackground(this, ColorType.KEY_PREVIEW_BACKGROUND);
        setTextColor(drawParams.mPreviewTextColor);
        setTextSize(TypedValue.COMPLEX_UNIT_PX, key.selectHintTextSize(drawParams) << 1);
        KeyboardTypeface.applyToTextView(this, this.getText(), key.selectTypeface(drawParams));
    }

    @Override
    public void showPopupKeysPanel(final View parentView, final Controller controller,
            final int pointX, final int pointY, final KeyboardActionListener listener) {
        showPopupKeysPanelInternal(parentView, controller, pointX, pointY);
    }

    @Override
    public void showPopupKeysPanel(final View parentView, final Controller controller,
            final int pointX, final int pointY, final EmojiViewCallback emojiViewCallback) {
        mEmojiViewCallback = emojiViewCallback;
        showPopupKeysPanelInternal(parentView, controller, pointX, pointY);
    }

    private void showPopupKeysPanelInternal(final View parentView, final Controller controller,
            final int pointX, final int pointY) {
        mController = controller;
        final View container = getContainerView();
        // The coordinates of panel's left-top corner in parentView's coordinate system.
        // We need to consider background drawable paddings.
        final int x = pointX - getMeasuredWidth() / 2 - container.getPaddingLeft() - getPaddingLeft();
        final int y = pointY - container.getMeasuredHeight() + container.getPaddingBottom()
                + getPaddingBottom();

        parentView.getLocationInWindow(mCoordinates);
        // Ensure the horizontal position of the panel does not extend past the parentView edges.
        final int maxX = parentView.getMeasuredWidth() - container.getMeasuredWidth();
        final int panelX = Math.max(0, Math.min(maxX, x)) + CoordinateUtils.x(mCoordinates);
        final int panelY = y + CoordinateUtils.y(mCoordinates);
        container.setX(panelX);
        container.setY(panelY);

        mOriginX = x + container.getPaddingLeft();
        mOriginY = y + container.getPaddingTop();
        controller.setLayoutGravity(Gravity.NO_GRAVITY);
        controller.onShowPopupKeysPanel(this);
    }

    @Override
    public void onDownEvent(final int x, final int y, final int pointerId, final long eventTime) {
    }

    @Override
    public void onMoveEvent(final int x, final int y, final int pointerId, final long eventTime) {
    }

    @Override
    public void onUpEvent(final int x, final int y, final int pointerId, final long eventTime) {
        mEmojiViewCallback.onReleaseKey(mKey);
    }

    @Override
    public void dismissPopupKeysPanel() {
        if (!isShowingInParent()) {
            return;
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
}
