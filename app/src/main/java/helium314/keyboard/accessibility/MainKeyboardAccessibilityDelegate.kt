/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.accessibility

import android.graphics.Rect
import android.os.SystemClock
import helium314.keyboard.latin.utils.Log
import android.util.SparseIntArray
import android.view.MotionEvent
import helium314.keyboard.accessibility.AccessibilityLongPressTimer.LongPressTimerCallback
import helium314.keyboard.keyboard.*
import helium314.keyboard.latin.R
import helium314.keyboard.latin.utils.SubtypeLocaleUtils.displayName

/**
 * This class represents a delegate that can be registered in [MainKeyboardView] to enhance
 * accessibility support via composition rather via inheritance.
 */
class MainKeyboardAccessibilityDelegate(
    mainKeyboardView: MainKeyboardView,
    keyDetector: KeyDetector
) : KeyboardAccessibilityDelegate<MainKeyboardView>(mainKeyboardView, keyDetector), LongPressTimerCallback {
    /** The most recently set keyboard mode.  */
    private var mLastKeyboardMode: KeyboardMode? = null
    // The rectangle region to ignore hover events.
    private val mBoundsToIgnoreHoverEvent = Rect()
    private val mAccessibilityLongPressTimer = AccessibilityLongPressTimer(this /* callback */, mainKeyboardView.context)

    // Since this method is called even when accessibility is off, make sure
    // to check the state before announcing anything.
    // Announce the language name only when the language is changed.
    // Announce the mode only when the mode is changed.
    // Announce the keyboard type only when the type is changed.
    /**
     * {@inheritDoc}
     */
    override var keyboard: Keyboard?
        get() = super.keyboard
        set(keyboard) {
            if (keyboard == null) {
                return
            }
            val lastKeyboard = super.keyboard
            super.keyboard = keyboard
            val lastKeyboardMode = mLastKeyboardMode
            mLastKeyboardMode = keyboard.mId.mode
            // Since this method is called even when accessibility is off, make sure
            // to check the state before announcing anything.
            if (!AccessibilityUtils.instance.isAccessibilityEnabled) {
                return
            }
            // Announce the language name only when the language is changed.
            if (lastKeyboard == null || keyboard.mId.subtype != lastKeyboard.mId.subtype) {
                announceKeyboardLanguage(keyboard)
                return
            }
            // Announce the mode only when the mode is changed.
            if (keyboard.mId.mode != lastKeyboardMode) {
                announceKeyboardMode(keyboard)
                return
            }
            // Announce the keyboard type only when the type is changed.
            if (keyboard.mId.element != lastKeyboard.mId.element) {
                announceKeyboardType(keyboard, lastKeyboard)
                return
            }
        }

    /**
     * Called when the keyboard is hidden and accessibility is enabled.
     */
    fun onHideWindow() {
        if (mLastKeyboardMode != null) {
            announceKeyboardHidden()
        }
        mLastKeyboardMode = null
    }

    /**
     * Announces which language of keyboard is being displayed.
     *
     * @param keyboard The new keyboard.
     */
    private fun announceKeyboardLanguage(keyboard: Keyboard) {
        sendWindowStateChanged(keyboard.mId.subtype.rawSubtype.displayName())
    }

    /**
     * Announces which type of keyboard is being displayed.
     * If the keyboard type is unknown, no announcement is made.
     *
     * @param keyboard The new keyboard.
     */
    private fun announceKeyboardMode(keyboard: Keyboard) {
        val context = mKeyboardView.context
        val modeTextResId = KEYBOARD_MODE_RES_IDS[keyboard.mId.mode]
        if (modeTextResId == 0 || modeTextResId == null) {
            return
        }
        val modeText = context.getString(modeTextResId)
        val text = context.getString(R.string.announce_keyboard_mode, modeText)
        sendWindowStateChanged(text)
    }

    /**
     * Announces which type of keyboard is being displayed.
     *
     * @param keyboard The new keyboard.
     * @param lastKeyboard The last keyboard.
     */
    private fun announceKeyboardType(keyboard: Keyboard, lastKeyboard: Keyboard) {
        val lastElementId = lastKeyboard.mId.element
        val resId = when (keyboard.mId.element) {
            KeyboardElement.ALPHABET_AUTOMATIC_SHIFTED, KeyboardElement.ALPHABET -> {
                if (lastElementId == KeyboardElement.ALPHABET
                        || lastElementId == KeyboardElement.ALPHABET_AUTOMATIC_SHIFTED) {
                    // Transition between alphabet mode and automatic shifted mode should be silently
                    // ignored because it can be determined by each key's talk back announce.
                    return
                }
                R.string.spoken_description_mode_alpha
            }
            KeyboardElement.ALPHABET_MANUAL_SHIFTED -> {
                if (lastElementId == KeyboardElement.ALPHABET_AUTOMATIC_SHIFTED) {
                    // Resetting automatic shifted mode by pressing the shift key causes the transition
                    // from automatic shifted to manual shifted that should be silently ignored.
                    return
                }
                R.string.spoken_description_shiftmode_on
            }
            KeyboardElement.ALPHABET_SHIFT_LOCKED -> R.string.spoken_description_shiftmode_locked
            KeyboardElement.SYMBOLS -> R.string.spoken_description_mode_symbol
            KeyboardElement.SYMBOLS_SHIFTED -> R.string.spoken_description_mode_symbol_shift
            KeyboardElement.PHONE -> R.string.spoken_description_mode_phone
            KeyboardElement.PHONE_SYMBOLS -> R.string.spoken_description_mode_phone_shift
            else -> return
        }
        sendWindowStateChanged(resId)
    }

    /**
     * Announces that the keyboard has been hidden.
     */
    private fun announceKeyboardHidden() {
        sendWindowStateChanged(R.string.announce_keyboard_hidden)
    }

    override fun performClickOn(key: Key) {
        val x = key.hitBox.centerX()
        val y = key.hitBox.centerY()
        if (DEBUG_HOVER) {
            Log.d(TAG, "performClickOn: key=" + key
                    + " inIgnoreBounds=" + mBoundsToIgnoreHoverEvent.contains(x, y))
        }
        if (mBoundsToIgnoreHoverEvent.contains(x, y)) {
            // This hover exit event points to the key that should be ignored.
            // Clear the ignoring region to handle further hover events.
            mBoundsToIgnoreHoverEvent.setEmpty()
            return
        }
        super.performClickOn(key)
    }

    override fun onHoverEnterTo(key: Key) {
        val x = key.hitBox.centerX()
        val y = key.hitBox.centerY()
        if (DEBUG_HOVER) {
            Log.d(TAG, "onHoverEnterTo: key=" + key
                    + " inIgnoreBounds=" + mBoundsToIgnoreHoverEvent.contains(x, y))
        }
        mAccessibilityLongPressTimer.cancelLongPress()
        if (mBoundsToIgnoreHoverEvent.contains(x, y)) {
            return
        }
        // This hover enter event points to the key that isn't in the ignoring region.
        // Further hover events should be handled.
        mBoundsToIgnoreHoverEvent.setEmpty()
        super.onHoverEnterTo(key)
        if (key.isLongPressEnabled) {
            mAccessibilityLongPressTimer.startLongPress(key)
        }
    }

    override fun onHoverExitFrom(key: Key) {
        val x = key.hitBox.centerX()
        val y = key.hitBox.centerY()
        if (DEBUG_HOVER) {
            Log.d(TAG, "onHoverExitFrom: key=" + key
                    + " inIgnoreBounds=" + mBoundsToIgnoreHoverEvent.contains(x, y))
        }
        mAccessibilityLongPressTimer.cancelLongPress()
        super.onHoverExitFrom(key)
    }

    override fun performLongClickOn(key: Key) {
        if (DEBUG_HOVER) {
            Log.d(TAG, "performLongClickOn: key=$key")
        }
        val tracker = PointerTracker.getPointerTracker(HOVER_EVENT_POINTER_ID)
        val eventTime = SystemClock.uptimeMillis()
        val x = key.hitBox.centerX()
        val y = key.hitBox.centerY()
        val downEvent = MotionEvent.obtain(eventTime, eventTime, MotionEvent.ACTION_DOWN, x.toFloat(), y.toFloat(), 0)
        // Inject a fake down event to {@link PointerTracker} to handle a long press correctly.
        tracker.processMotionEvent(downEvent, mKeyDetector)
        downEvent.recycle()
        // Invoke {@link PointerTracker#onLongPressed()} as if a long press timeout has passed.
        tracker.onLongPressed()
        // If {@link Key#hasNoPanelAutoPopupKeys()} is true (such as "0 +" key on the phone layout)
        // or a key invokes IME switcher dialog, we should just ignore the next
        // {@link #onRegisterHoverKey(Key,MotionEvent)}. It can be determined by whether
        // {@link PointerTracker} is in operation or not.
        if (tracker.isInOperation) {
            // This long press shows a popup keys keyboard and further hover events should be
            // handled.
            mBoundsToIgnoreHoverEvent.setEmpty()
            return
        }
        // This long press has handled at {@link MainKeyboardView#onLongPress(PointerTracker)}.
        // We should ignore further hover events on this key.
        mBoundsToIgnoreHoverEvent.set(key.hitBox)
        if (key.hasNoPanelAutoPopupKey()) {
            // This long press has registered a code point without showing a popup keys keyboard.
            // We should talk back the code point if possible.
            val codePointOfNoPanelAutoPopupKey = key.popupKeys?.get(0)?.mCode ?: return
            val text: String = KeyCodeDescriptionMapper.instance.getDescriptionForCodePoint(
                    mKeyboardView.context, codePointOfNoPanelAutoPopupKey) ?: return
            sendWindowStateChanged(text)
        }
    }

    companion object {
        private val TAG = MainKeyboardAccessibilityDelegate::class.java.simpleName
        /** Map of keyboard modes to resource IDs.  */
        private val KEYBOARD_MODE_RES_IDS = hashMapOf(
            KeyboardMode.DATE to R.string.keyboard_mode_date,
            KeyboardMode.DATETIME to R.string.keyboard_mode_date_time,
            KeyboardMode.EMAIL to R.string.keyboard_mode_email,
            KeyboardMode.IM to R.string.keyboard_mode_im,
            KeyboardMode.NUMBER to R.string.keyboard_mode_number,
            KeyboardMode.PHONE to R.string.keyboard_mode_phone,
            KeyboardMode.TEXT to R.string.keyboard_mode_text,
            KeyboardMode.TIME to R.string.keyboard_mode_time,
            KeyboardMode.URL to R.string.keyboard_mode_url,
        )
    }
}
