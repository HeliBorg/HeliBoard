/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.define

object ProductionFlags {
    // supporting hardware keyboard still has a bunch of issues
    // crash https://github.com/HeliBorg/HeliBoard/issues/2047 (possibly fixed with b7cb95fc9da213c99d82e8833fb5f950f39d232e)
    // different crash https://github.com/HeliBorg/HeliBoard/issues/2001
    //  LatinIME.isInputViewShown() returns true when there is no input view, thus crashing in onUpdateSelection
    // physical layout ignored https://github.com/HeliBorg/HeliBoard/issues/1957, https://github.com/HeliBorg/HeliBoard/issues/1949
    // physical layout ignored for uppercase letters only (?) https://github.com/HeliBorg/HeliBoard/issues/2030
    const val IS_HARDWARE_KEYBOARD_SUPPORTED = false

    /**
     * Include all suggestions from all dictionaries in
     * [helium314.keyboard.latin.SuggestedWords.mRawSuggestions].
     */
    const val INCLUDE_RAW_SUGGESTIONS = false
}
