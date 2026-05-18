/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.PopupKeysPanel;
import helium314.keyboard.keyboard.PointerTracker;

public interface DrawingProxy {
    /**
     * Called when a key is being pressed.
     * @param key the {@link Key} that is being pressed.
     * @param withPreview true if key popup preview should be displayed.
     */
    void onKeyPressed(@NonNull Key key, boolean withPreview);

    /**
     * Called when a key is being released.
     * @param key the {@link Key} that is being released.
     * @param withAnimation when true, key popup preview should be dismissed with animation.
     */
    void onKeyReleased(@NonNull Key key, boolean withAnimation);

    /**
     * Start showing popup keys keyboard of a key that is being long pressed.
     * @param key the {@link Key} that is being long pressed and showing popup keys keyboard.
     * @param tracker the {@link PointerTracker} that detects this long pressing.
     * @return {@link PopupKeysPanel} that is being shown. null if there is no need to show popup keys keyboard.
     */
    @Nullable
    PopupKeysPanel showPopupKeysKeyboard(@NonNull Key key, @NonNull PointerTracker tracker);

    /**
     * Start a while-typing-animation.
     * @param fadeInOrOut {@link #FADE_IN} starts while-typing-fade-in animation.
     * {@link #FADE_OUT} starts while-typing-fade-out animation.
     */
    void startWhileTypingAnimation(int fadeInOrOut);
    int FADE_IN = 0;
    int FADE_OUT = 1;

    /**
     * Show sliding-key input preview.
     * @param tracker the {@link PointerTracker} that is currently doing the sliding-key input.
     * null to dismiss the sliding-key input preview.
     */
    void showSlidingKeyInputPreview(@Nullable PointerTracker tracker);

    /**
     * Show gesture trails.
     * @param tracker the {@link PointerTracker} whose gesture trail will be shown.
     * @param showsFloatingPreviewText when true, a gesture floating preview text will be shown
     * with this <code>tracker</code>'s trail.
     */
    void showGestureTrail(@NonNull PointerTracker tracker, boolean showsFloatingPreviewText);

    /**
     * Dismiss a gesture floating preview text without delay.
     */
    void dismissGestureFloatingPreviewTextWithoutDelay();

    /**
     * Update the debug-points overlay (feature #2.1, gated by
     * {@code PREF_GESTURE_DEBUG_DRAW_POINTS}). Implementations may no-op when the pref is off.
     *
     * @param raw the unprocessed batch pointers as seen by the keyboard layer; must not be
     *     {@code null}.
     * @param synthetic only the points added by {@link DualThumbHinter} on top of {@code raw}.
     *     Pass an empty (zero-size) {@link helium314.keyboard.latin.common.InputPointers} when
     *     no hinting was applied — the overlay then draws only the raw stream.
     */
    void setGestureDebugPoints(@NonNull helium314.keyboard.latin.common.InputPointers raw,
            @NonNull helium314.keyboard.latin.common.InputPointers synthetic);

    /** Clear the debug-points overlay (e.g. on gesture start or cancel). */
    void clearGestureDebugPoints();
}
