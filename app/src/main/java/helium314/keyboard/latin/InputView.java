/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;

import androidx.core.view.ViewKt;

import helium314.keyboard.accessibility.AccessibilityUtils;
import helium314.keyboard.keyboard.MainKeyboardView;
import helium314.keyboard.latin.common.ColorType;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.suggestions.MoreSuggestionsView;
import helium314.keyboard.latin.suggestions.SuggestionStripView;
import kotlin.Unit;


public final class InputView extends FrameLayout {
    private final Rect mInputViewRect = new Rect();
    private MainKeyboardView mMainKeyboardView;
    private KeyboardTopPaddingForwarder mKeyboardTopPaddingForwarder;
    private MoreSuggestionsViewCanceler mMoreSuggestionsViewCanceler;
    private MotionEventForwarder<?, ?> mActiveForwarder;

    public InputView(final Context context, final AttributeSet attrs) {
        super(context, attrs, 0);
    }

    @Override
    protected void onFinishInflate() {
        super.onFinishInflate();
        final SuggestionStripView suggestionStripView =
                findViewById(R.id.suggestion_strip_view);
        mMainKeyboardView = findViewById(R.id.keyboard_view);
        mKeyboardTopPaddingForwarder = new KeyboardTopPaddingForwarder(
                mMainKeyboardView, suggestionStripView);
        mMoreSuggestionsViewCanceler = new MoreSuggestionsViewCanceler(
                mMainKeyboardView, suggestionStripView);
        ViewKt.doOnNextLayout(this, this::onNextLayout);
    }

    public void setKeyboardTopPadding(final int keyboardTopPadding) {
        mKeyboardTopPaddingForwarder.setKeyboardTopPadding(keyboardTopPadding);
    }

    @Override
    protected boolean dispatchHoverEvent(final MotionEvent event) {
        if (AccessibilityUtils.Companion.getInstance().isTouchExplorationEnabled()
                && mMainKeyboardView.isShowingPopupKeysPanel()) {
            // With accessibility mode on, discard hover events while a popup keys keyboard is shown.
            // The {@link PopupKeysKeyboard} receives hover events directly from the platform.
            return true;
        }
        return super.dispatchHoverEvent(event);
    }

    @Override
    public boolean onInterceptTouchEvent(final MotionEvent me) {
        final Rect rect = mInputViewRect;
        getGlobalVisibleRect(rect);
        final int index = me.getActionIndex();
        final int x = (int)me.getX(index) + rect.left;
        final int y = (int)me.getY(index) + rect.top;

        // The touch events that hit the top padding of keyboard should be forwarded to
        // {@link SuggestionStripView}.
        if (mKeyboardTopPaddingForwarder.onInterceptTouchEvent(x, y, me)) {
            mActiveForwarder = mKeyboardTopPaddingForwarder;
            return true;
        }

        // To cancel {@link MoreSuggestionsView}, we should intercept a touch event to
        // {@link MainKeyboardView} and dismiss the {@link MoreSuggestionsView}.
        if (mMoreSuggestionsViewCanceler.onInterceptTouchEvent(x, y, me)) {
            mActiveForwarder = mMoreSuggestionsViewCanceler;
            return true;
        }

        mActiveForwarder = null;
        return false;
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(final MotionEvent me) {
        if (mActiveForwarder == null) {
            return super.onTouchEvent(me);
        }

        final Rect rect = mInputViewRect;
        getGlobalVisibleRect(rect);
        final int index = me.getActionIndex();
        final int x = (int)me.getX(index) + rect.left;
        final int y = (int)me.getY(index) + rect.top;
        return mActiveForwarder.onTouchEvent(x, y, me);
    }

    private Unit onNextLayout(View v) {
        Settings.getValues().mColors.setBackground(findViewById(R.id.main_keyboard_frame), ColorType.MAIN_BACKGROUND);

        // Work around inset application being unreliable
        requestApplyInsets();
        return null;
    }

    /**
     * This class forwards series of {@link MotionEvent}s from <code>SenderView</code> to
     * <code>ReceiverView</code>.
     *
     * @param <SenderView> a {@link View} that may send a {@link MotionEvent} to <ReceiverView>.
     * @param <ReceiverView> a {@link View} that receives forwarded {@link MotionEvent} from
     *     <SenderView>.
     */
    private static abstract class
            MotionEventForwarder<SenderView extends View, ReceiverView extends View> {
        protected final SenderView mSenderView;
        protected final ReceiverView mReceiverView;

        protected final Rect mEventSendingRect = new Rect();
        protected final Rect mEventReceivingRect = new Rect();

        public MotionEventForwarder(final SenderView senderView, final ReceiverView receiverView) {
            mSenderView = senderView;
            mReceiverView = receiverView;
        }

        // Return true if a touch event of global coordinate x, y needs to be forwarded.
        protected abstract boolean needsToForward(final int x, final int y);

        // Translate global x-coordinate to <code>ReceiverView</code> local coordinate.
        protected int translateX(final int x) {
            return x - mEventReceivingRect.left;
        }

        // Translate global y-coordinate to <code>ReceiverView</code> local coordinate.
        protected int translateY(final int y) {
            return y - mEventReceivingRect.top;
        }

        /**
         * Callback when a {@link MotionEvent} is forwarded.
         * @param me the motion event to be forwarded.
         */
        protected void onForwardingEvent(final MotionEvent me) {}

        // Returns true if a {@link MotionEvent} is needed to be forwarded to
        // <code>ReceiverView</code>. Otherwise returns false.
        public boolean onInterceptTouchEvent(final int x, final int y, final MotionEvent me) {
            // Forwards a {link MotionEvent} only if both <code>SenderView</code> and
            // <code>ReceiverView</code> are visible.
            if (mSenderView.getVisibility() != View.VISIBLE ||
                    mReceiverView.getVisibility() != View.VISIBLE) {
                return false;
            }
            mSenderView.getGlobalVisibleRect(mEventSendingRect);
            if (!mEventSendingRect.contains(x, y)) {
                return false;
            }

            if (me.getActionMasked() == MotionEvent.ACTION_DOWN) {
                // If the down event happens in the forwarding area, successive
                // {@link MotionEvent}s should be forwarded to <code>ReceiverView</code>.
                return needsToForward(x, y);
            }

            return false;
        }

        // Returns true if a {@link MotionEvent} is forwarded to <code>ReceiverView</code>.
        // Otherwise returns false.
        public boolean onTouchEvent(final int x, final int y, final MotionEvent me) {
            mReceiverView.getGlobalVisibleRect(mEventReceivingRect);
            // Translate global coordinates to <code>ReceiverView</code> local coordinates.
            me.setLocation(translateX(x), translateY(y));
            mReceiverView.dispatchTouchEvent(me);
            onForwardingEvent(me);
            return true;
        }
    }

    /**
     * This class forwards {@link MotionEvent}s happened in the top padding of
     * {@link MainKeyboardView} to {@link SuggestionStripView}.
     */
    private static class KeyboardTopPaddingForwarder
            extends MotionEventForwarder<MainKeyboardView, SuggestionStripView> {
        private int mKeyboardTopPadding;

        public KeyboardTopPaddingForwarder(final MainKeyboardView mainKeyboardView,
                final SuggestionStripView suggestionStripView) {
            super(mainKeyboardView, suggestionStripView);
        }

        public void setKeyboardTopPadding(final int keyboardTopPadding) {
            mKeyboardTopPadding = keyboardTopPadding;
        }

        private boolean isInKeyboardTopPadding(final int y) {
            return y < mEventSendingRect.top + mKeyboardTopPadding;
        }

        @Override
        protected boolean needsToForward(final int x, final int y) {
            // Forwarding an event only when {@link MainKeyboardView} is visible.
            // Because the visibility of {@link MainKeyboardView} is controlled by its parent
            // view in {@link KeyboardSwitcher#setMainKeyboardFrame()}, we should check the
            // visibility of the parent view.
            final View mainKeyboardFrame = (View)mSenderView.getParent();
            return mainKeyboardFrame.getVisibility() == View.VISIBLE && isInKeyboardTopPadding(y);
        }

        @Override
        protected int translateY(final int y) {
            final int translatedY = super.translateY(y);
            if (isInKeyboardTopPadding(y)) {
                // The forwarded event should have coordinates that are inside of the target.
                return Math.min(translatedY, mEventReceivingRect.height() - 1);
            }
            return translatedY;
        }
    }

    /**
     * This class forwards {@link MotionEvent}s happened in the {@link MainKeyboardView} to
     * {@link SuggestionStripView} when the {@link MoreSuggestionsView} is showing.
     * {@link SuggestionStripView} dismisses {@link MoreSuggestionsView} when it receives any event
     * outside of it.
     */
    private static class MoreSuggestionsViewCanceler
            extends MotionEventForwarder<View, SuggestionStripView> {
        public MoreSuggestionsViewCanceler(final MainKeyboardView mainKeyboardView,
                final SuggestionStripView suggestionStripView) {
            super(mainKeyboardView.getRootView(), suggestionStripView);
        }

        @Override
        protected boolean needsToForward(final int x, final int y) {
            return mReceiverView.isShowingMoreSuggestionPanel() && mEventSendingRect.contains(x, y);
        }

        @Override
        protected void onForwardingEvent(final MotionEvent me) {
            if (me.getActionMasked() == MotionEvent.ACTION_DOWN) {
                mReceiverView.dismissMoreSuggestionsPanel();
            }
        }
    }
}
