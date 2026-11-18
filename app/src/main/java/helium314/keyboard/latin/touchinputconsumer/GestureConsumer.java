/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.touchinputconsumer;

import android.view.inputmethod.EditorInfo;

import helium314.keyboard.keyboard.Keyboard;
import helium314.keyboard.latin.DictionaryFacilitator;
import helium314.keyboard.latin.common.InputPointers;
import helium314.keyboard.latin.inputlogic.PrivateCommandPerformer;
import helium314.keyboard.latin.suggestions.SuggestedWords;

import java.util.Locale;

/**
 * Stub for GestureConsumer.
 * <br>
 * The methods of this class should only be called from a single thread, e.g.,
 * the UI Thread.
 */
@SuppressWarnings("unused")
public class GestureConsumer {
    public static final GestureConsumer NULL_GESTURE_CONSUMER =
            new GestureConsumer();

    public static GestureConsumer newInstance(
            final EditorInfo editorInfo, final PrivateCommandPerformer commandPerformer,
            final Locale locale, final Keyboard keyboard) {
        return GestureConsumer.NULL_GESTURE_CONSUMER;
    }

    private GestureConsumer() {
    }

    public boolean willConsume() {
        return false;
    }

    public void onInit(final Locale locale, final Keyboard keyboard) {
    }

    public void onGestureStarted(final Locale locale, final Keyboard keyboard) {
    }

    public void onGestureCanceled() {
    }

    public void onGestureCompleted(final InputPointers inputPointers) {
    }

    public void onImeSuggestionsProcessed(final SuggestedWords suggestedWords,
            final int composingStart, final int composingLength,
            final DictionaryFacilitator dictionaryFacilitator) {
    }
}
