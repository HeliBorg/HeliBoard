/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.view.View;
import android.view.ViewGroup;
import helium314.keyboard.keyboard.emoji.EmojiViewCallback;
import kotlin.Unit;
import kotlin.jvm.functions.Function1;

public interface PopupKeysPanel {
    interface Controller {
        /**
         * Set the layout gravity.
         * @param layoutGravity requested by the popup
         */
        default void setLayoutGravity(int layoutGravity) {
        }

        /**
         * Add the {@link PopupKeysPanel} to the target view.
         * @param panel the panel to be shown.
         */
        void onShowPopupKeysPanel(final PopupKeysPanel panel);

        /**
         * Remove the current {@link PopupKeysPanel} from the target view.
         */
        void onDismissPopupKeysPanel();

        /**
         * Instructs the parent to cancel the panel (e.g., when entering a different input mode).
         */
        void onCancelPopupKeysPanel();
    }

    Controller EMPTY_CONTROLLER = new Controller() {
        @Override
        public void onShowPopupKeysPanel(final PopupKeysPanel panel) {}
        @Override
        public void onDismissPopupKeysPanel() {}
        @Override
        public void onCancelPopupKeysPanel() {}
    };

    /**
     * Initializes the layout and event handling of this {@link PopupKeysPanel} and calls the
     * controller's onShowPopupKeysPanel to add the panel's container view.
     *
     * @param parentView the parent view of this {@link PopupKeysPanel}
     * @param controller the controller that can dismiss this {@link PopupKeysPanel}
     * @param pointX x coordinate of this {@link PopupKeysPanel}
     * @param pointY y coordinate of this {@link PopupKeysPanel}
     * @param listener the listener that will receive keyboard action from this
     * {@link PopupKeysPanel}.
     */
    // TODO: Currently the PopupKeysPanel is inside a container view that is added to the parent.
    // Consider the simpler approach of placing the PopupKeysPanel itself into the parent view.
    void showPopupKeysPanel(View parentView, Controller controller, int pointX,
                           int pointY, KeyboardActionListener listener);

    /**
     *
     * Initializes the layout and event handling of this {@link PopupKeysPanel} and calls the
     * controller's onShowPopupKeysPanel to add the panel's container view.
     * Same as {@link PopupKeysPanel#showPopupKeysPanel(View, Controller, int, int, KeyboardActionListener)},
     * but with a {@link EmojiViewCallback}.
     *
     * @param parentView the parent view of this {@link PopupKeysPanel}
     * @param controller the controller that can dismiss this {@link PopupKeysPanel}
     * @param pointX x coordinate of this {@link PopupKeysPanel}
     * @param pointY y coordinate of this {@link PopupKeysPanel}
     * @param onReleaseKey to receive keyboard actions from this {@link PopupKeysPanel}.
     */
    // TODO: Currently the PopupKeysPanel is inside a container view that is added to the parent.
    // Consider the simpler approach of placing the PopupKeysPanel itself into the parent view.
    void showPopupKeysPanel(View parentView, Controller controller, int pointX, int pointY, Function1<Key, Unit> onReleaseKey);

    /**
     * Dismisses the popup keys panel and calls the controller's onDismissPopupKeysPanel to remove
     * the panel's container view.
     */
    void dismissPopupKeysPanel();

    /**
     * Process a move event on the popup keys panel.
     *
     * @param x translated x coordinate of the touch point
     * @param y translated y coordinate of the touch point
     * @param pointerId pointer id touch point
     * @param eventTime timestamp of touch point
     */
    void onMoveEvent(final int x, final int y, final int pointerId, final long eventTime);

    /**
     * Process a down event on the popup keys panel.
     *
     * @param x translated x coordinate of the touch point
     * @param y translated y coordinate of the touch point
     * @param pointerId pointer id touch point
     * @param eventTime timestamp of touch point
     */
    void onDownEvent(final int x, final int y, final int pointerId, final long eventTime);

    /**
     * Process an up event on the popup keys panel.
     *
     * @param x translated x coordinate of the touch point
     * @param y translated y coordinate of the touch point
     * @param pointerId pointer id touch point
     * @param eventTime timestamp of touch point
     */
    void onUpEvent(final int x, final int y, final int pointerId, final long eventTime);

    /**
     * Translate X-coordinate of touch event to the local X-coordinate of this
     * {@link PopupKeysPanel}.
     *
     * @param x the global X-coordinate
     * @return the local X-coordinate to this {@link PopupKeysPanel}
     */
    int translateX(int x);

    /**
     * Translate Y-coordinate of touch event to the local Y-coordinate of this
     * {@link PopupKeysPanel}.
     *
     * @param y the global Y-coordinate
     * @return the local Y-coordinate to this {@link PopupKeysPanel}
     */
    int translateY(int y);

    default View getContainerView() {
        return (View) ((View) this).getParent();
    }

    /**
     * Show this {@link PopupKeysPanel} in the parent view.
     *
     * @param parentView the {@link ViewGroup} that hosts this {@link PopupKeysPanel}.
     */
    default void showInParent(ViewGroup parentView) {
        removeFromParent();
        parentView.addView(getContainerView());
    }

    /**
     * Remove this {@link PopupKeysPanel} from the parent view.
     */
    default void removeFromParent() {
        final View containerView = getContainerView();
        final ViewGroup currentParent = (ViewGroup)containerView.getParent();
        if (currentParent != null) {
            currentParent.removeView(containerView);
        }
    }

    /**
     * Return whether the panel is currently being shown.
     */
    default boolean isShowingInParent() {
        return getContainerView().getParent() != null;
    }
}
