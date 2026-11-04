/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.common

import android.graphics.Outline
import android.inputmethodservice.InputMethodService
import android.view.View
import android.view.ViewOutlineProvider

class InsetsOutlineProvider : ViewOutlineProvider() {
    private var mLastVisibleTopInsets = NO_DATA

    fun setInsets(insets: InputMethodService.Insets, view: View) {
        val visibleTopInsets = insets.visibleTopInsets
        if (mLastVisibleTopInsets != visibleTopInsets) {
            mLastVisibleTopInsets = visibleTopInsets
            view.invalidateOutline()
        }
    }

    override fun getOutline(view: View, outline: Outline) {
        if (mLastVisibleTopInsets == NO_DATA) { // Call default implementation.
            BACKGROUND.getOutline(view, outline)
            return
        }
        outline.setRect(view.left, mLastVisibleTopInsets, view.right, view.bottom)
    }

    companion object {
        private const val NO_DATA = -1
    }
}
