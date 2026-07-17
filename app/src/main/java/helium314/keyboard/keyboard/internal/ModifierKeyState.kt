/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard.internal

internal enum class ModifierKeyState {
    RELEASED,
    PRESSING,
    CHORDING,
;
    fun chordIfPressing(): ModifierKeyState {
        return if (this == PRESSING) {
            CHORDING
        } else {
            this
        }
    }
}
