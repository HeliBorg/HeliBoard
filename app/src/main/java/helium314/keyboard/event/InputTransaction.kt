/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.event

import helium314.keyboard.latin.CapsMode
import helium314.keyboard.latin.settings.SettingsValues

/** An object encapsulating a single transaction for input. */
class InputTransaction(
    // Initial conditions
    val settingsValues: SettingsValues,
    val event: Event,
    val timestamp: Long,
    val spaceState: Int,
    val shiftState: CapsMode
) {
    /** Gets what type of shift update this transaction requires. */
    // Outputs
    var requiredShiftUpdate = SHIFT_NO_UPDATE
        private set
    private var requiresUpdateSuggestions = false
    private var didAffectContents = false
    private var didAutoCorrect = false
    /**
     * Indicate that this transaction requires some type of shift update.
     * @param updateType What type of shift update this requires.
     */
    fun requireShiftUpdate(updateType: Int) {
        requiredShiftUpdate = requiredShiftUpdate.coerceAtLeast(updateType)
    }

    /** Indicate that this transaction requires updating the suggestions.*/
    fun setRequiresUpdateSuggestions() {
        requiresUpdateSuggestions = true
    }

    /** Whether this transaction requires updating the suggestions. */
    fun requiresUpdateSuggestions() = requiresUpdateSuggestions

    /** Indicate that this transaction affected the contents of the editor. */
    fun setDidAffectContents() {
        didAffectContents = true
    }

    /** Whether this transaction affected contents of the editor. */
    fun didAffectContents() = didAffectContents

    /** Indicate that this transaction performed an auto-correction. */
    fun setDidAutoCorrect() {
        didAutoCorrect = true
    }

    /** Whether this transaction performed an auto-correction. */
    fun didAutoCorrect() = didAutoCorrect

    companion object {
        // UPDATE_LATER is stronger than UPDATE_NOW. The reason for this is, if we have to update later,
        // it's because something will change that we can't evaluate now, which means that even if we
        // re-evaluate now we'll have to do it again later. The only case where that wouldn't apply
        // would be if we needed to update now to find out the new state right away, but then we
        // can't do it with this deferred mechanism anyway.
        const val SHIFT_NO_UPDATE = 0
        const val SHIFT_UPDATE_NOW = 1
        const val SHIFT_UPDATE_LATER = 2
    }
}
