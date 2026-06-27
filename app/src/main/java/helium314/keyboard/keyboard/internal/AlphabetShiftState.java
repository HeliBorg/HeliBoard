/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import helium314.keyboard.latin.utils.Log;

public enum AlphabetShiftState {
    UNSHIFTED,
    MANUAL_SHIFTED,
    MANUAL_SHIFTED_FROM_AUTO,
    AUTOMATIC_SHIFTED,
    SHIFT_LOCKED,
    SHIFT_LOCK_SHIFTED,
;
    private static final String TAG = AlphabetShiftState.class.getSimpleName();
    private static final boolean DEBUG = false;

    public AlphabetShiftState shift(boolean newShiftState) {
        AlphabetShiftState shift = newShiftState
            ? switch (this) {
                case UNSHIFTED -> MANUAL_SHIFTED;
                case AUTOMATIC_SHIFTED -> MANUAL_SHIFTED_FROM_AUTO;
                case SHIFT_LOCKED -> SHIFT_LOCK_SHIFTED;
                default -> this;
            } : switch (this) {
                case MANUAL_SHIFTED, MANUAL_SHIFTED_FROM_AUTO, AUTOMATIC_SHIFTED -> UNSHIFTED;
                case SHIFT_LOCK_SHIFTED -> SHIFT_LOCKED;
                default -> this;
            }
        ;
        if (DEBUG) {
            Log.d(TAG, "shift(" + newShiftState + "): " + this + " > " + shift);
        }
        return shift;
    }

    public AlphabetShiftState shiftLock() {
        AlphabetShiftState shiftLock = switch (this) {
            case UNSHIFTED, MANUAL_SHIFTED, MANUAL_SHIFTED_FROM_AUTO, AUTOMATIC_SHIFTED -> SHIFT_LOCKED;
            default -> this;
        };
        if (DEBUG) {
            Log.d(TAG, "shiftLock(): " + this + " > " + shiftLock);
        }
        return shiftLock;
    }

    public boolean isShiftedOrShiftLocked() {
        return this != UNSHIFTED;
    }

    public boolean isShiftLocked() {
        return this == SHIFT_LOCKED || this == SHIFT_LOCK_SHIFTED;
    }

    public boolean isManualShifted() {
        return switch (this) {
            case MANUAL_SHIFTED, MANUAL_SHIFTED_FROM_AUTO, SHIFT_LOCK_SHIFTED -> true;
            default -> false;
        };
    }
}
