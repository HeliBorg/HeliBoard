/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard.internal

import android.text.TextUtils
import helium314.keyboard.event.Event
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.PointerTracker
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.CapsModeUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.RecapitalizeMode

/**
 * Keyboard state machine.
 * This class contains all keyboard state transition logic.
 *
 * The input events are [onLoadKeyboard], [onSaveKeyboardState],
 * [onPressKey], [onReleaseKey],
 * [onEvent], [onFinishSlidingInput],
 * [onUpdateShiftState], [onResetKeyboardStateToAlphabet].
 *
 * The actions are [SwitchActions]'s methods.
 */
class KeyboardState(private val switchActions: SwitchActions) {
    interface SwitchActions {
        fun setAlphabetKeyboard(shiftMode: ShiftMode)
        fun setEmojiKeyboard()
        fun setClipboardKeyboard()
        fun setNumpadKeyboard()
        fun toggleNumpad(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?, forceReturnToAlpha: Boolean)
        fun setSymbolsKeyboard()
        fun setSymbolsShiftedKeyboard()

        /** Request to call back [KeyboardState.onUpdateShiftState]. */
        fun requestUpdatingShiftState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?)

        fun startDoubleTapShiftKeyTimer()
        fun popDoubleTapShiftKeyTimer(): Boolean
        fun cancelDoubleTapShiftKeyTimer()

        fun setOneHandedModeEnabled(enabled: Boolean)
        fun switchOneHandedMode()
        fun setFloatingKeyboardEnabled(enabled: Boolean)

        companion object {
            const val DEBUG_ACTION = false
            const val DEBUG_TIMER_ACTION = false
        }
    }

    private var shiftKeyState = ModifierKeyState.RELEASED
    private var symbolKeyState = ModifierKeyState.RELEASED
    private var shiftMode = ShiftMode.UNSHIFT

    private var switchState = SwitchState.ALPHA

    private var mode = Mode.ALPHABET
    private var modeBeforeNumpad = Mode.ALPHABET // todo: replace with stack logic
    private var prevShiftMode: ShiftMode? = null
    private var prevSymbolsKeyboardWasShifted = false // todo: replace with stack logic
    private var recapitalizeMode: RecapitalizeMode? = null

    // For handling double tap.
    private var isInDoubleTapShiftKey = false

    private val savedKeyboardState = SavedKeyboardState()

    private class SavedKeyboardState {
        var isValid = false
        var mode = Mode.ALPHABET
        var shiftMode = ShiftMode.UNSHIFT

        override fun toString(): String {
            if (!isValid) return "INVALID"
            return when (mode) {
                Mode.ALPHABET -> "ALPHABET_$shiftMode"
                else -> mode.toString()
            }
        }
    }

    fun onLoadKeyboard(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?, onHandedModeEnabled: Boolean) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onLoadKeyboard: " + stateToString(autoCapsFlags, recapitalizeMode))
        }
        // Reset alphabet shift state.
        shiftMode = ShiftMode.UNSHIFT
        prevShiftMode = null
        prevSymbolsKeyboardWasShifted = false
        shiftKeyState = ModifierKeyState.RELEASED
        symbolKeyState = ModifierKeyState.RELEASED
        if (savedKeyboardState.isValid) {
            onRestoreKeyboardState(autoCapsFlags, recapitalizeMode)
            savedKeyboardState.isValid = false
        } else {
            // Reset keyboard to alphabet mode.
            setAlphabetKeyboard(ShiftMode.UNSHIFT, autoCapsFlags, recapitalizeMode)
        }
        switchActions.setOneHandedModeEnabled(onHandedModeEnabled)
        switchActions.setFloatingKeyboardEnabled(Settings.getValues().mIsFloatingKeyboard)
    }

    fun onSaveKeyboardState() {
        savedKeyboardState.mode = mode
        savedKeyboardState.shiftMode = shiftMode
        savedKeyboardState.isValid = true
        if (DEBUG_EVENT) {
            Log.d(TAG, "onSaveKeyboardState: saved=$savedKeyboardState $this")
        }
    }

    private fun onRestoreKeyboardState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onRestoreKeyboardState: saved=$savedKeyboardState ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        when (savedKeyboardState.mode) {
            Mode.ALPHABET -> setAlphabetKeyboard(savedKeyboardState.shiftMode, autoCapsFlags, recapitalizeMode)
            Mode.SYMBOLS -> setSymbolsKeyboard()
            Mode.SYMBOLS_SHIFTED -> setSymbolsShiftedKeyboard()
            Mode.EMOJI -> setEmojiKeyboard()
            Mode.CLIPBOARD -> setClipboardKeyboard()
            // don't overwrite toggle state if reloading from orientation change, etc.
            Mode.NUMPAD -> setNumpadKeyboard(false, false, false)
        }
    }

    private fun setShifted(shiftMode: ShiftMode) {
        if (mode != Mode.ALPHABET) return
        if (this.shiftMode != shiftMode) {
            if (DebugFlags.DEBUG_ENABLED) {
                Log.d(TAG, "setShifted: shiftMode=$shiftMode $this")
            }
            this.shiftMode = shiftMode
            switchActions.setAlphabetKeyboard(shiftMode)
        }
    }

    private fun toggleAlphabetAndSymbols(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "toggleAlphabetAndSymbols: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        if (mode == Mode.ALPHABET) {
            if (prevSymbolsKeyboardWasShifted) setSymbolsShiftedKeyboard() else setSymbolsKeyboard()
            prevSymbolsKeyboardWasShifted = false
        } else {
            prevSymbolsKeyboardWasShifted = mode == Mode.SYMBOLS_SHIFTED
            setAlphabetKeyboard(shiftMode, autoCapsFlags, recapitalizeMode)
        }
    }

    // TODO: Remove this method. Come up with a more comprehensive way to reset the keyboard layout
    //  when a keyboard layout set doesn't get reloaded in LatinIME.onStartInputViewInternal().
    private fun resetKeyboardStateToAlphabet(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "resetKeyboardStateToAlphabet: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        if (mode == Mode.ALPHABET) return

        prevSymbolsKeyboardWasShifted = mode == Mode.SYMBOLS_SHIFTED
        setAlphabetKeyboard(shiftMode, autoCapsFlags, recapitalizeMode)
    }

    private fun toggleShiftInSymbols() {
        if (mode == Mode.SYMBOLS_SHIFTED) {
            setSymbolsKeyboard()
        } else {
            setSymbolsShiftedKeyboard()
        }
    }

    private fun setAlphabetKeyboard(shiftMode: ShiftMode, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setAlphabetKeyboard: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }

        switchActions.setAlphabetKeyboard(shiftMode)
        mode = Mode.ALPHABET
        this.recapitalizeMode = null
        switchState = SwitchState.ALPHA
        if (shiftMode == ShiftMode.UNSHIFT || shiftMode == ShiftMode.AUTOMATIC) {
            switchActions.requestUpdatingShiftState(autoCapsFlags, recapitalizeMode)
        }
    }

    private fun setSymbolsKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setSymbolsKeyboard")
        }
        switchActions.setSymbolsKeyboard()
        mode = Mode.SYMBOLS
        recapitalizeMode = null
        switchState = SwitchState.SYMBOL_BEGIN
    }

    private fun setSymbolsShiftedKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setSymbolsShiftedKeyboard")
        }
        switchActions.setSymbolsShiftedKeyboard()
        mode = Mode.SYMBOLS_SHIFTED
        recapitalizeMode = null
        switchState = SwitchState.SYMBOL_BEGIN
    }

    private fun setEmojiKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setEmojiKeyboard")
        }
        switchActions.setEmojiKeyboard()
        mode = Mode.EMOJI
        recapitalizeMode = null
    }

    private fun setClipboardKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setClipboardKeyboard")
        }
        switchActions.setClipboardKeyboard()
        mode = Mode.CLIPBOARD
        recapitalizeMode = null
    }

    private fun setNumpadKeyboard(withSliding: Boolean, forceReturnToAlpha: Boolean, rememberState: Boolean) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setNumpadKeyboard")
        }
        if (rememberState) {
            // When d-pad is added, "selection mode" may need to be remembered if not a global state
            modeBeforeNumpad = if (forceReturnToAlpha) Mode.ALPHABET else mode
        }
        mode = Mode.NUMPAD
        recapitalizeMode = null
        switchActions.setNumpadKeyboard()
        switchState = if (withSliding) SwitchState.MOMENTARY_TO_NUMPAD else SwitchState.NUMPAD_BEGIN
    }

    fun toggleNumpad(
        withSliding: Boolean,
        autoCapsFlags: Int,
        recapitalizeMode: RecapitalizeMode?,
        forceReturnToAlpha: Boolean,
        rememberState: Boolean
    ) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "toggleNumpad")
        }
        if (mode != Mode.NUMPAD) {
            setNumpadKeyboard(withSliding, forceReturnToAlpha, rememberState)
            return
        }
        if (modeBeforeNumpad == Mode.ALPHABET || forceReturnToAlpha) {
            setAlphabetKeyboard(shiftMode, autoCapsFlags, recapitalizeMode)
        } else when (modeBeforeNumpad) {
            Mode.ALPHABET -> {}
            Mode.SYMBOLS -> setSymbolsKeyboard()
            Mode.SYMBOLS_SHIFTED -> setSymbolsShiftedKeyboard()
            Mode.EMOJI -> setEmojiKeyboard()
            Mode.CLIPBOARD -> setClipboardKeyboard()
            Mode.NUMPAD -> {}
        }
        if (withSliding) switchState = SwitchState.MOMENTARY_FROM_NUMPAD
    }

    fun onPressKey(code: Int, pointerCount: Int, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, ("onPressKey: code=${Constants.printableCode(code)} pointerCount=$pointerCount ${stateToString(autoCapsFlags, recapitalizeMode)}"))
        }
        if (code != KeyCode.SHIFT) {
            // Because the double tap shift key timer is to detect two consecutive shift key press,
            // it should be canceled when a non-shift key is pressed.
            switchActions.cancelDoubleTapShiftKeyTimer()
        }
        when (code) {
            KeyCode.SHIFT -> onPressShift()
            KeyCode.CAPS_LOCK -> {} // Nothing to do here. See onReleaseKey.
            KeyCode.SYMBOL_ALPHA -> onPressAlphaSymbol(autoCapsFlags, recapitalizeMode)
            KeyCode.SYMBOL, KeyCode.ALPHA, KeyCode.NUMPAD -> {} // don't start sliding, causes issues with fully customizable layouts (also does not allow chording, but can be fixed later)
            else -> {
                shiftKeyState = shiftKeyState.chordIfPressing()
                symbolKeyState = symbolKeyState.chordIfPressing()
                // We need to unshift when all the following conditions are met:
                // 1) we're in the temporary-shifted alphabet layout
                // 2) the shift key is not being actively pressed
                // 3) two or more fingers are in action, not including a chording layout key
                // 4) we're not in all characters caps mode
                if (mode == Mode.ALPHABET && (shiftMode == ShiftMode.MANUAL || shiftMode == ShiftMode.AUTOMATIC)
                    && shiftKeyState == ModifierKeyState.RELEASED
                    && (pointerCount > 2 || (pointerCount == 2 && symbolKeyState != ModifierKeyState.CHORDING))
                    && autoCapsFlags != TextUtils.CAP_MODE_CHARACTERS
                ) {
                    switchActions.setAlphabetKeyboard(ShiftMode.UNSHIFT)
                }
            }
        }
    }

    fun onReleaseKey(code: Int, withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onReleaseKey: code=${Constants.printableCode(code)} sliding=$withSliding ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        when (code) {
            KeyCode.SHIFT        -> onReleaseShift(withSliding, autoCapsFlags, recapitalizeMode)
            KeyCode.CAPS_LOCK    -> setShifted(
                if (shiftMode == ShiftMode.LOCKED) {
                    ShiftMode.UNSHIFT
                } else {
                    ShiftMode.LOCKED
                }
            )
            KeyCode.SYMBOL_ALPHA -> onReleaseAlphaSymbol(withSliding, autoCapsFlags, recapitalizeMode)
            // if no sliding, switching is instead handled by onEvent()
            // to accommodate toolbar keys and prevent double-loads.
            KeyCode.SYMBOL       -> if (withSliding) slideIntoSymbol()
            KeyCode.ALPHA        -> if (withSliding) slideIntoAlpha(autoCapsFlags, recapitalizeMode)
            KeyCode.NUMPAD       -> if (withSliding) setNumpadKeyboard(true, modeBeforeNumpad == Mode.CLIPBOARD, true)
        }
    }

    private fun onPressAlphaSymbol(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
        symbolKeyState = ModifierKeyState.PRESSING
        switchState = SwitchState.MOMENTARY_ALPHA_AND_SYMBOL
    }

    private fun onReleaseAlphaSymbol(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (symbolKeyState == ModifierKeyState.CHORDING) {
            // Switch back to the previous keyboard mode if the user chords the mode change key and
            // another key, then releases the mode change key.
            toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
        } else if (!withSliding) {
            // If the mode change key is being released without sliding, we should forget the
            // previous symbols keyboard shift state and simply switch back to symbols layout
            // (never symbols shifted) next time the mode gets changed to symbols layout.
            prevSymbolsKeyboardWasShifted = false
        }
        symbolKeyState = ModifierKeyState.RELEASED
    }

    private fun slideIntoSymbol() {
        val oldMode = mode
        setSymbolsKeyboard()
        if (oldMode == Mode.NUMPAD) {
            switchState = SwitchState.MOMENTARY_FROM_NUMPAD
        }
    }

    private fun slideIntoAlpha(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        val oldMode = mode
        setAlphabetKeyboard(shiftMode, autoCapsFlags, recapitalizeMode)
        if (oldMode == Mode.NUMPAD) {
            switchState = SwitchState.MOMENTARY_FROM_NUMPAD
        }
    }

    fun onUpdateShiftState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onUpdateShiftState: " + stateToString(autoCapsFlags, recapitalizeMode))
        }
        this.recapitalizeMode = recapitalizeMode
        updateAlphabetShiftState(autoCapsFlags, recapitalizeMode)
    }

    // TODO: Remove this method. Come up with a more comprehensive way to reset the keyboard layout
    //  when a keyboard layout set doesn't get reloaded in LatinIME.onStartInputViewInternal().
    fun onResetKeyboardStateToAlphabet(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onResetKeyboardStateToAlphabet: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        resetKeyboardStateToAlphabet(autoCapsFlags, recapitalizeMode)
    }

    private fun updateAlphabetShiftState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (mode != Mode.ALPHABET) return
        if (recapitalizeMode != null) {
            setShifted(
                when (recapitalizeMode) {
                    RecapitalizeMode.ORIGINAL_MIXED_CASE -> ShiftMode.AUTOMATIC
                    RecapitalizeMode.ALL_LOWER -> ShiftMode.UNSHIFT
                    RecapitalizeMode.FIRST_WORD_UPPER -> ShiftMode.MANUAL
                    RecapitalizeMode.ALL_UPPER -> ShiftMode.LOCKED
                }
            )
            return
        }
        if (shiftKeyState != ModifierKeyState.RELEASED) {
            // Don't update when pressing or chording
            return
        }
        if (shiftMode != ShiftMode.LOCKED) {
            // todo: it's annoying when this happens in manual shift, but in order to block it we would
            //  need to know whether this is an update for typing a letter in shift or moving the caret
            setShifted(
                if (autoCapsFlags != Constants.TextUtils.CAP_MODE_OFF) {
                    ShiftMode.AUTOMATIC
                } else {
                    ShiftMode.UNSHIFT
                }
            )
        }
    }

    private fun onPressShift() {
        fun inner() {
            if (recapitalizeMode != null) {
                PointerTracker.suppressShiftLongPress()
                // As we are recapitalizing, we don't do any of the normal
                // processing, including importantly the double tap timer.
                // todo: this isn't detected before the first recapitalization
                return
            }
            if (mode == Mode.SYMBOLS || mode == Mode.SYMBOLS_SHIFTED) {
                // In symbol mode, just toggle symbol and symbol popup keyboard.
                toggleShiftInSymbols()
                switchState = SwitchState.MOMENTARY_SYMBOL_AND_MORE
                return
            }
            if (mode != Mode.ALPHABET) {
                return
            }
            isInDoubleTapShiftKey = switchActions.popDoubleTapShiftKeyTimer()
            if (isInDoubleTapShiftKey) {
                setShifted(ShiftMode.LOCKED)
                PointerTracker.suppressShiftLongPress()
                return
            }
            prevShiftMode = shiftMode
            if (shiftMode == ShiftMode.LOCKED) {
                PointerTracker.suppressShiftLongPress()
                // Don't start the timer again if we're already shift-locked
            } else {
                switchActions.startDoubleTapShiftKeyTimer()
            }
            setShifted(
                if (shiftMode == ShiftMode.UNSHIFT) {
                    ShiftMode.MANUAL
                } else {
                    ShiftMode.UNSHIFT
                }
            )
        }
        inner()
        shiftKeyState = ModifierKeyState.PRESSING
    }

    private fun onReleaseShift(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        fun inner() {
            if (this.recapitalizeMode != null) {
                // Note: recapitalization is handled in InputLogic.handleFunctionalEvent
                return
            }
            if (mode == Mode.SYMBOLS || mode == Mode.SYMBOLS_SHIFTED) {
                if (shiftKeyState == ModifierKeyState.CHORDING) {
                    toggleShiftInSymbols()
                }
                return
            }
            if (mode != Mode.ALPHABET) {
                return
            }
            if (isInDoubleTapShiftKey) {
                // Double tap shift key has been handled in {@link #onPressShift}
                isInDoubleTapShiftKey = false
                return
            }
            if (shiftKeyState == ModifierKeyState.CHORDING) {
                // After chording input
                when (shiftMode) {
                    ShiftMode.UNSHIFT -> restorePreviousShiftMode()
                    ShiftMode.MANUAL, ShiftMode.AUTOMATIC -> setShifted(ShiftMode.UNSHIFT)
                    ShiftMode.LOCKED -> {}
                }
                // Automatic shift state may have been changed depending on what characters were input.
                switchActions.requestUpdatingShiftState(autoCapsFlags, recapitalizeMode)
                return
            }
            if (withSliding) {
                switchState = SwitchState.MOMENTARY_ALPHA_SHIFT
            }
        }
        inner()
        shiftKeyState = ModifierKeyState.RELEASED
        if (!withSliding) {
            prevShiftMode = null
        }
    }

    private fun restorePreviousShiftMode() {
        when (val prevShiftMode = this.prevShiftMode) {
            null, ShiftMode.AUTOMATIC -> {}
            ShiftMode.UNSHIFT, ShiftMode.MANUAL, ShiftMode.LOCKED -> setShifted(prevShiftMode)
        }
    }

    fun onFinishSlidingInput(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onFinishSlidingInput: " + stateToString(autoCapsFlags, recapitalizeMode))
        }
        // Switch back to the previous keyboard mode if the user didn't enter the numpad.
        if (mode != Mode.NUMPAD) when (switchState) {
            SwitchState.MOMENTARY_ALPHA_AND_SYMBOL -> toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
            SwitchState.MOMENTARY_SYMBOL_AND_MORE  -> toggleShiftInSymbols()
            SwitchState.MOMENTARY_ALPHA_SHIFT      -> {
                restorePreviousShiftMode()
                prevShiftMode = null
            }
            SwitchState.MOMENTARY_FROM_NUMPAD      -> setNumpadKeyboard(false, false, false)
            else                                   -> {}
        } else if (switchState == SwitchState.MOMENTARY_TO_NUMPAD) {
            toggleNumpad(false, autoCapsFlags, recapitalizeMode, false, false)
        }
    }

    fun onEvent(event: Event, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        val code = if (event.isFunctionalKeyEvent) event.keyCode else event.codePoint
        if (DEBUG_EVENT) {
            Log.d(TAG, "onEvent: code=${Constants.printableCode(code)} ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }

        when (switchState) {
            SwitchState.SYMBOL ->
                // Switch back to alpha keyboard mode if user types one or more non-space/enter
                // characters followed by a space/enter.
                if (isSpaceOrEnter(code) && Settings.getValues().mAlphaAfterSymbolAndSpace) {
                    toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
                    prevSymbolsKeyboardWasShifted = false
                }
            SwitchState.SYMBOL_BEGIN ->
                if (mode == Mode.EMOJI || mode == Mode.CLIPBOARD) {
                    // When in the Emoji keyboard or clipboard one, we don't want to switch back to the main layout even
                    // after the user hits an emoji letter followed by an enter or a space.
                } else if (!isSpaceOrEnter(code) && (Constants.isLetterCode(code) || code == KeyCode.MULTIPLE_CODE_POINTS)) {
                    switchState = SwitchState.SYMBOL
                }
            SwitchState.NUMPAD ->
                // Switch back to alpha keyboard mode if user types one or more non-space/enter
                // characters followed by a space/enter.
                if (isSpaceOrEnter(code) && Settings.getValues().mAlphaAfterNumpadAndSpace) {
                    toggleNumpad(false, autoCapsFlags, recapitalizeMode, true, false)
                }
            SwitchState.NUMPAD_BEGIN ->
                if (!isSpaceOrEnter(code)) {
                    switchState = SwitchState.NUMPAD
                }
            SwitchState.MOMENTARY_ALPHA_AND_SYMBOL ->
                if (code == KeyCode.SYMBOL_ALPHA) {
                    // Detected only the mode change key has been pressed, and then released.
                    switchState = if (mode == Mode.ALPHABET) SwitchState.ALPHA else SwitchState.SYMBOL_BEGIN
                }
            SwitchState.MOMENTARY_SYMBOL_AND_MORE ->
                if (code == KeyCode.SHIFT) {
                    // Detected only the shift key has been pressed on symbol layout, and then released.
                    switchState = SwitchState.SYMBOL_BEGIN
                } else if (isSpaceOrEnter(code)) {
                    // Switch back to alpha keyboard mode if user types one or more non-space/enter characters followed by a space/enter.
                    toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
                    prevSymbolsKeyboardWasShifted = false
                }
            else -> {}
        }

        if (Constants.isLetterCode(code)) {
            // If the code is a letter, update keyboard shift state.
            updateAlphabetShiftState(autoCapsFlags, recapitalizeMode)
        } else when (code) {
            KeyCode.EMOJI -> setEmojiKeyboard()
            KeyCode.ALPHA -> setAlphabetKeyboard(shiftMode, autoCapsFlags, recapitalizeMode)
            // Note: Printing clipboard content is handled in InputLogic.handleFunctionalEvent
            KeyCode.CLIPBOARD -> if (Settings.getValues().mClipboardHistoryEnabled) setClipboardKeyboard()
            KeyCode.NUMPAD -> toggleNumpad(false, autoCapsFlags, recapitalizeMode, false, true)
            KeyCode.SYMBOL -> setSymbolsKeyboard()
            KeyCode.TOGGLE_ONE_HANDED_MODE -> switchActions.setOneHandedModeEnabled(!Settings.getValues().mOneHandedModeEnabled)
            KeyCode.SWITCH_ONE_HANDED_MODE -> switchActions.switchOneHandedMode()
            KeyCode.TOGGLE_FLOATING_WINDOW -> {
                switchActions.setFloatingKeyboardEnabled(!Settings.getValues().mIsFloatingKeyboard)
                KeyboardSwitcher.getInstance().reloadKeyboard()
            }
        }
    }

    override fun toString(): String {
        val keyboard = when (mode) {
            Mode.ALPHABET -> shiftMode.toString()
            else -> mode.toString()
        }
        return "[keyboard=$keyboard shift=$shiftKeyState symbol=$symbolKeyState switch=$switchState]"
    }

    private fun stateToString(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) =
        "$this autoCapsFlags=${CapsModeUtils.flagsToString(autoCapsFlags)} recapitalizeMode=$recapitalizeMode"

    private enum class SwitchState {
        ALPHA,
        SYMBOL,
        SYMBOL_BEGIN,
        NUMPAD,
        NUMPAD_BEGIN,
        MOMENTARY_ALPHA_AND_SYMBOL,
        MOMENTARY_SYMBOL_AND_MORE,
        MOMENTARY_ALPHA_SHIFT,
        MOMENTARY_TO_NUMPAD,
        MOMENTARY_FROM_NUMPAD,
    }

    private enum class Mode {
        ALPHABET,
        SYMBOLS,
        SYMBOLS_SHIFTED,
        EMOJI,
        CLIPBOARD,
        NUMPAD,
    }

    companion object {
        private val TAG = KeyboardState::class.java.simpleName
        private const val DEBUG_EVENT = false

        private fun isSpaceOrEnter(c: Int) = c == Constants.CODE_SPACE || c == Constants.CODE_ENTER
    }
}
