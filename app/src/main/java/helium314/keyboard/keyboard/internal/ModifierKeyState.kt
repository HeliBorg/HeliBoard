/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import helium314.keyboard.latin.utils.Log;

import androidx.annotation.NonNull;

/* package */ class ModifierKeyState {
    protected static final String TAG = ModifierKeyState.class.getSimpleName();
    protected static final boolean DEBUG = false;

    protected static final int RELEASING = 0;
    protected static final int PRESSING = 1;
    protected static final int CHORDING = 2;

    protected final String mName;
    protected int mState = RELEASING;

    public ModifierKeyState(String name) {
        mName = name;
    }

    public void onPress() {
        final int oldState = mState;
        mState = PRESSING;
        if (DEBUG)
            Log.d(TAG, mName + ".onPress: " + toString(oldState) + " > " + this);
    }

    public void onRelease() {
        final int oldState = mState;
        mState = RELEASING;
        if (DEBUG)
            Log.d(TAG, mName + ".onRelease: " + toString(oldState) + " > " + this);
    }

    public void onOtherKeyPressed() {
        final int oldState = mState;
        if (oldState == PRESSING)
            mState = CHORDING;
        if (DEBUG)
            Log.d(TAG, mName + ".onOtherKeyPressed: " + toString(oldState) + " > " + this);
    }

    public boolean isPressing() {
        return mState == PRESSING;
    }

    public boolean isReleasing() {
        return mState == RELEASING;
    }

    public boolean isChording() {
        return mState == CHORDING;
    }

    @NonNull
    @Override
    public String toString() {
        return toString(mState);
    }

    protected String toString(int state) {
        return switch (state) {
            case RELEASING -> "RELEASING";
            case PRESSING -> "PRESSING";
            case CHORDING -> "CHORDING";
            default -> "UNKNOWN";
        };
    }
}
