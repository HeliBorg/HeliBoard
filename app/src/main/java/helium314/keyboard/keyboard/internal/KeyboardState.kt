/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard.internal

import android.text.TextUtils
import helium314.keyboard.event.Event
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
        fun setAlphabetKeyboard()
        fun setAlphabetManualShiftedKeyboard()
        fun setAlphabetAutomaticShiftedKeyboard()
        fun setAlphabetShiftLockedKeyboard()
        fun setAlphabetShiftLockShiftedKeyboard()
        fun setEmojiKeyboard()
        fun setClipboardKeyboard()
        fun setNumpadKeyboard()
        fun toggleNumpad(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?, forceReturnToAlpha: Boolean)
        fun setSymbolsKeyboard()
        fun setSymbolsShiftedKeyboard()

        /** Request to call back [KeyboardState.onUpdateShiftState]. */
        fun requestUpdatingShiftState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?)

        fun startDoubleTapShiftKeyTimer()
        val isInDoubleTapShiftKeyTimeout: Boolean
        fun cancelDoubleTapShiftKeyTimer()

        fun setOneHandedModeEnabled(enabled: Boolean)
        fun switchOneHandedMode()

        companion object {
            const val DEBUG_ACTION = false
            const val DEBUG_TIMER_ACTION = false
        }
    }

    private val shiftKeyState = ShiftKeyState("Shift")
    private val symbolKeyState = ModifierKeyState("Symbol")
    private val alphabetShiftState = AlphabetShiftState()

    private var switchState = SwitchState.ALPHA

    private var mode = Mode.ALPHABET
    private var modeBeforeNumpad = Mode.ALPHABET
    private var isSymbolShifted = false
    private var prevMainKeyboardWasShiftLocked = false
    private var prevSymbolsKeyboardWasShifted = false
    private var recapitalizeMode: RecapitalizeMode? = null

    // For handling double tap.
    private var isInAlphabetUnshiftedFromShifted = false
    private var isInDoubleTapShiftKey = false

    private val savedKeyboardState = SavedKeyboardState()

    private class SavedKeyboardState {
        var isValid = false
        var isAlphabetShiftLocked = false
        var mode = Mode.ALPHABET
        var shiftMode = ShiftMode.UNSHIFT

        override fun toString(): String {
            if (!isValid) return "INVALID"
            return when (mode) {
                Mode.ALPHABET -> "${mode}_${if (isAlphabetShiftLocked) ShiftMode.SHIFT_LOCKED else shiftMode}"
                Mode.SYMBOLS -> "${mode}_$shiftMode"
                else -> mode.toString()
            }
        }
    }

    fun onLoadKeyboard(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?, onHandedModeEnabled: Boolean) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onLoadKeyboard: " + stateToString(autoCapsFlags, recapitalizeMode))
        }
        // Reset alphabet shift state.
        alphabetShiftState.setShiftLocked(false)
        prevMainKeyboardWasShiftLocked = false
        prevSymbolsKeyboardWasShifted = false
        shiftKeyState.onRelease()
        symbolKeyState.onRelease()
        if (savedKeyboardState.isValid) {
            onRestoreKeyboardState(autoCapsFlags, recapitalizeMode)
            savedKeyboardState.isValid = false
        } else {
            // Reset keyboard to alphabet mode.
            setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
        }
        switchActions.setOneHandedModeEnabled(onHandedModeEnabled)
    }

    fun onSaveKeyboardState() {
        savedKeyboardState.mode = mode
        if (mode == Mode.ALPHABET) {
            savedKeyboardState.isAlphabetShiftLocked = alphabetShiftState.isShiftLocked
            savedKeyboardState.shiftMode = when {
                alphabetShiftState.isAutomaticShifted -> ShiftMode.AUTOMATIC
                alphabetShiftState.isShiftedOrShiftLocked -> ShiftMode.MANUAL
                else -> ShiftMode.UNSHIFT
            }
        } else {
            savedKeyboardState.isAlphabetShiftLocked = prevMainKeyboardWasShiftLocked
            savedKeyboardState.shiftMode = if (isSymbolShifted) ShiftMode.MANUAL else ShiftMode.UNSHIFT
        }
        savedKeyboardState.isValid = true
        if (DEBUG_EVENT) {
            Log.d(TAG, "onSaveKeyboardState: saved=$savedKeyboardState $this")
        }
    }

    private fun onRestoreKeyboardState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onRestoreKeyboardState: saved=$savedKeyboardState ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        prevMainKeyboardWasShiftLocked = savedKeyboardState.isAlphabetShiftLocked
        when (savedKeyboardState.mode) {
            Mode.ALPHABET -> {
                setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
                setShiftLocked(savedKeyboardState.isAlphabetShiftLocked)
                if (!savedKeyboardState.isAlphabetShiftLocked) {
                    setShifted(savedKeyboardState.shiftMode)
                }
            }
            Mode.SYMBOLS -> if (savedKeyboardState.shiftMode == ShiftMode.MANUAL) setSymbolsShiftedKeyboard() else setSymbolsKeyboard()
            Mode.EMOJI -> setEmojiKeyboard()
            Mode.CLIPBOARD -> setClipboardKeyboard()
            // don't overwrite toggle state if reloading from orientation change, etc.
            Mode.NUMPAD -> setNumpadKeyboard(false, false, false)
        }
    }

    private fun setShifted(shiftMode: ShiftMode) {
        if (mode != Mode.ALPHABET) return
        val prevShiftMode = when {
            alphabetShiftState.isAutomaticShifted -> ShiftMode.AUTOMATIC
            alphabetShiftState.isManualShifted -> ShiftMode.MANUAL
            else -> ShiftMode.UNSHIFT
        }
        if (DebugFlags.DEBUG_ENABLED && shiftMode != prevShiftMode) {
            Log.d(TAG, "setShifted: shiftMode=$shiftMode $this")
        }
        when (shiftMode) {
            ShiftMode.UNSHIFT -> {
                alphabetShiftState.setShifted(false)
                if (shiftMode != prevShiftMode)
                    switchActions.setAlphabetKeyboard()
            }
            ShiftMode.MANUAL -> {
                alphabetShiftState.setShifted(true)
                if (shiftMode != prevShiftMode)
                    switchActions.setAlphabetManualShiftedKeyboard()
            }
            ShiftMode.AUTOMATIC -> {
                alphabetShiftState.setAutomaticShifted()
                if (shiftMode != prevShiftMode)
                    switchActions.setAlphabetAutomaticShiftedKeyboard()
            }
            ShiftMode.SHIFT_LOCKED -> {
                alphabetShiftState.setShifted(true)
                switchActions.setAlphabetShiftLockShiftedKeyboard()
            }
        }
    }

    private fun setShiftLocked(shiftLocked: Boolean) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setShiftLocked: shiftLocked=$shiftLocked $this")
        }
        if (mode != Mode.ALPHABET) return
        if (shiftLocked && (!alphabetShiftState.isShiftLocked || alphabetShiftState.isShiftLockShifted)) {
            switchActions.setAlphabetShiftLockedKeyboard()
        }
        if (!shiftLocked && alphabetShiftState.isShiftLocked) {
            switchActions.setAlphabetKeyboard()
        }
        alphabetShiftState.setShiftLocked(shiftLocked)
    }

    private fun toggleAlphabetAndSymbols(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "toggleAlphabetAndSymbols: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        if (mode == Mode.ALPHABET) {
            prevMainKeyboardWasShiftLocked = alphabetShiftState.isShiftLocked
            if (prevSymbolsKeyboardWasShifted) setSymbolsShiftedKeyboard() else setSymbolsKeyboard()
            prevSymbolsKeyboardWasShifted = false
        } else {
            prevSymbolsKeyboardWasShifted = isSymbolShifted
            setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
            if (prevMainKeyboardWasShiftLocked) setShiftLocked(true)
            prevMainKeyboardWasShiftLocked = false
        }
    }

    // TODO: Remove this method. Come up with a more comprehensive way to reset the keyboard layout
    //  when a keyboard layout set doesn't get reloaded in LatinIME.onStartInputViewInternal().
    private fun resetKeyboardStateToAlphabet(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "resetKeyboardStateToAlphabet: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        if (mode == Mode.ALPHABET) return

        prevSymbolsKeyboardWasShifted = isSymbolShifted
        setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
        if (prevMainKeyboardWasShiftLocked) {
            setShiftLocked(true)
        }
        prevMainKeyboardWasShiftLocked = false
    }

    private fun toggleShiftInSymbols() {
        if (isSymbolShifted) {
            setSymbolsKeyboard()
        } else {
            setSymbolsShiftedKeyboard()
        }
    }

    private fun setAlphabetKeyboard(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setAlphabetKeyboard: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }

        switchActions.setAlphabetKeyboard()
        mode = Mode.ALPHABET
        isSymbolShifted = false
        this.recapitalizeMode = null
        switchState = SwitchState.ALPHA
        switchActions.requestUpdatingShiftState(autoCapsFlags, recapitalizeMode)
    }

    private fun setSymbolsKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setSymbolsKeyboard")
        }
        switchActions.setSymbolsKeyboard()
        mode = Mode.SYMBOLS
        isSymbolShifted = false
        recapitalizeMode = null
        // Reset alphabet shift state.
        alphabetShiftState.setShiftLocked(false)
        switchState = SwitchState.SYMBOL_BEGIN
    }

    private fun setSymbolsShiftedKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setSymbolsShiftedKeyboard")
        }
        switchActions.setSymbolsShiftedKeyboard()
        mode = Mode.SYMBOLS
        isSymbolShifted = true
        recapitalizeMode = null
        // Reset alphabet shift state.
        alphabetShiftState.setShiftLocked(false)
        switchState = SwitchState.SYMBOL_BEGIN
    }

    private fun setEmojiKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setEmojiKeyboard")
        }
        mode = Mode.EMOJI
        recapitalizeMode = null
        // Remember caps lock mode and reset alphabet shift state.
        prevMainKeyboardWasShiftLocked = alphabetShiftState.isShiftLocked
        alphabetShiftState.setShiftLocked(false)
        switchActions.setEmojiKeyboard()
    }

    private fun setClipboardKeyboard() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setClipboardKeyboard")
        }
        mode = Mode.CLIPBOARD
        recapitalizeMode = null
        // Remember caps lock mode and reset alphabet shift state.
        prevMainKeyboardWasShiftLocked = alphabetShiftState.isShiftLocked
        alphabetShiftState.setShiftLocked(false)
        switchActions.setClipboardKeyboard()
    }

    private fun setNumpadKeyboard(withSliding: Boolean, forceReturnToAlpha: Boolean, rememberState: Boolean) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setNumpadKeyboard")
        }
        if (rememberState) {
            if (mode == Mode.ALPHABET) {
                // Remember caps lock mode and reset alphabet shift state.
                prevMainKeyboardWasShiftLocked = alphabetShiftState.isShiftLocked
                alphabetShiftState.setShiftLocked(false)
            } else if (mode == Mode.SYMBOLS) {
                // Remember symbols shifted state
                prevSymbolsKeyboardWasShifted = isSymbolShifted
            }
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
            setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
            if (prevMainKeyboardWasShiftLocked) {
                setShiftLocked(true)
            }
            prevMainKeyboardWasShiftLocked = false
        } else when (modeBeforeNumpad) {
            Mode.ALPHABET -> {}
            Mode.SYMBOLS -> {
                if (prevSymbolsKeyboardWasShifted) setSymbolsShiftedKeyboard() else setSymbolsKeyboard()
                prevSymbolsKeyboardWasShifted = false
            }
            Mode.EMOJI -> setEmojiKeyboard()
            Mode.CLIPBOARD -> setClipboardKeyboard()
            Mode.NUMPAD -> {}
        }
        if (withSliding) switchState = SwitchState.MOMENTARY_FROM_NUMPAD
    }

    private fun setOneHandedModeEnabled(enabled: Boolean) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "setOneHandedModeEnabled")
        }
        switchActions.setOneHandedModeEnabled(enabled)
    }

    private fun switchOneHandedMode() {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "switchOneHandedMode")
        }
        switchActions.switchOneHandedMode()
    }

    fun onPressKey(code: Int, isSinglePointer: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, ("onPressKey: code=${Constants.printableCode(code)} single=$isSinglePointer ${stateToString(autoCapsFlags, recapitalizeMode)}"))
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
                shiftKeyState.onOtherKeyPressed()
                symbolKeyState.onOtherKeyPressed()
                // It is required to reset the auto caps state when all of the following conditions
                // are met:
                // 1) two or more fingers are in action
                // 2) in alphabet layout
                // 3) not in all characters caps mode
                // As for #3, please note that it's required to check even when the auto caps mode is
                // off because, for example, we may be in the #1 state within the manual temporary
                // shifted mode.
                if (!isSinglePointer
                    && mode == Mode.ALPHABET
                    && autoCapsFlags != TextUtils.CAP_MODE_CHARACTERS
                    && (alphabetShiftState.isAutomaticShifted || (alphabetShiftState.isManualShifted && shiftKeyState.isReleasing))
                ) {
                    switchActions.setAlphabetKeyboard()
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
            KeyCode.CAPS_LOCK    -> setShiftLocked(!alphabetShiftState.isShiftLocked)
            KeyCode.SYMBOL_ALPHA -> onReleaseAlphaSymbol(withSliding, autoCapsFlags, recapitalizeMode)
            KeyCode.SYMBOL       -> onReleaseSymbol(withSliding, autoCapsFlags, recapitalizeMode)
            KeyCode.ALPHA        -> onReleaseAlpha(withSliding, autoCapsFlags, recapitalizeMode)
            KeyCode.NUMPAD       -> {
                // if no sliding, toggling is instead handled by onEvent to accommodate toolbar key.
                // also prevent sliding to clipboard layout, which isn't supported yet.
                if (withSliding) setNumpadKeyboard(true, modeBeforeNumpad == Mode.CLIPBOARD, true)
            }
        }
    }

    private fun onPressAlphaSymbol(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
        symbolKeyState.onPress()
        switchState = SwitchState.MOMENTARY_ALPHA_AND_SYMBOL
    }

    private fun onReleaseAlphaSymbol(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (symbolKeyState.isChording) {
            // Switch back to the previous keyboard mode if the user chords the mode change key and
            // another key, then releases the mode change key.
            toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
        } else if (!withSliding) {
            // If the mode change key is being released without sliding, we should forget the
            // previous symbols keyboard shift state and simply switch back to symbols layout
            // (never symbols shifted) next time the mode gets changed to symbols layout.
            prevSymbolsKeyboardWasShifted = false
        }
        symbolKeyState.onRelease()
    }

    private fun onReleaseSymbol(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        val oldMode = mode
        setSymbolsKeyboard()
        if (withSliding && oldMode == Mode.NUMPAD) switchState = SwitchState.MOMENTARY_FROM_NUMPAD
    }

    private fun onReleaseAlpha(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        val oldMode = mode
        setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
        if (withSliding && oldMode == Mode.NUMPAD) switchState = SwitchState.MOMENTARY_FROM_NUMPAD
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

    private fun updateShiftStateForRecapitalize(recapitalizeMode: RecapitalizeMode?) {
        val shiftMode = when (recapitalizeMode) {
            null                                 -> ShiftMode.UNSHIFT
            RecapitalizeMode.ORIGINAL_MIXED_CASE -> ShiftMode.UNSHIFT
            RecapitalizeMode.ALL_LOWER           -> ShiftMode.UNSHIFT
            RecapitalizeMode.FIRST_WORD_UPPER    -> ShiftMode.AUTOMATIC
            RecapitalizeMode.ALL_UPPER           -> ShiftMode.SHIFT_LOCKED
        }
        setShifted(shiftMode)
    }

    private fun updateAlphabetShiftState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (mode != Mode.ALPHABET) return
        if (recapitalizeMode != null) {
            // We are recapitalizing. Match the keyboard to the current recapitalize state.
            updateShiftStateForRecapitalize(recapitalizeMode)
        } else if (!shiftKeyState.isReleasing) {
            // Ignore update shift state event while the shift key is being pressed (including chording).
        } else if (!alphabetShiftState.isShiftLocked && !shiftKeyState.isIgnoring) {
            val shifted = when {
                // Only when shift key is releasing, automatic temporary upper case will be set.
                shiftKeyState.isReleasing && autoCapsFlags != Constants.TextUtils.CAP_MODE_OFF -> ShiftMode.AUTOMATIC
                shiftKeyState.isChording -> ShiftMode.MANUAL
                else -> ShiftMode.UNSHIFT
            }
            setShifted(shifted)
        }
    }

    private fun onPressShift() {
        // If we are recapitalizing, we don't do any of the normal processing, including importantly the double tap timer.
        if (recapitalizeMode != null) {
            return
        }
        if (mode != Mode.ALPHABET) {
            // In symbol mode, just toggle symbol and symbol popup keyboard.
            toggleShiftInSymbols()
            switchState = SwitchState.MOMENTARY_SYMBOL_AND_MORE
            shiftKeyState.onPress()
            return
        }
        isInDoubleTapShiftKey = switchActions.isInDoubleTapShiftKeyTimeout
        if (isInDoubleTapShiftKey) {
            if (alphabetShiftState.isManualShifted || isInAlphabetUnshiftedFromShifted) {
                // Shift key has been double tapped while in manual shifted or automatic shifted state.
                setShiftLocked(true)
            }
            // Else shift key has been double tapped while in normal state.
            // This is the second tap to disable shift locked state, so just ignore this.
        } else {
            // This is first tap.
            switchActions.startDoubleTapShiftKeyTimer()
            if (alphabetShiftState.isShiftLocked) {
                // Shift key is pressed while shift locked state, we will treat this state as
                // shift lock shifted state and mark as if shift key pressed while normal state.
                setShifted(ShiftMode.SHIFT_LOCKED)
                shiftKeyState.onPress()
            } else if (alphabetShiftState.isAutomaticShifted) {
                // Shift key is pressed while automatic shifted, we have to move to manual shifted.
                setShifted(ShiftMode.MANUAL)
                shiftKeyState.onPress()
            } else if (alphabetShiftState.isShiftedOrShiftLocked) {
                // In manual shifted state, we just record shift key has been pressing while shifted state.
                shiftKeyState.onPressOnShifted()
            } else {
                // In base layout, chording or manual shifted mode is started.
                setShifted(ShiftMode.MANUAL)
                shiftKeyState.onPress()
            }
        }
    }

    private fun onReleaseShift(withSliding: Boolean, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (this.recapitalizeMode != null) {
            // We are recapitalizing. We should match the keyboard state to the recapitalize state in priority.
            updateShiftStateForRecapitalize(this.recapitalizeMode)
        } else if (mode != Mode.ALPHABET) {
            // In symbol mode, switch back to the previous keyboard mode if the user chords the
            // shift key and another key, then releases the shift key.
            if (shiftKeyState.isChording) {
                toggleShiftInSymbols()
            }
        } else {
            val isShiftLocked = alphabetShiftState.isShiftLocked
            isInAlphabetUnshiftedFromShifted = false
            when {
                // Double tap shift key has been handled in {@link #onPressShift}, so that just ignore this release shift key here.
                isInDoubleTapShiftKey -> isInDoubleTapShiftKey = false
                // After chording input
                shiftKeyState.isChording -> {
                    if (alphabetShiftState.isShiftLockShifted) setShiftLocked(true) else setShifted(ShiftMode.UNSHIFT)
                    // Automatic shift state may have been changed depending on what characters were input.
                    shiftKeyState.onRelease()
                    switchActions.requestUpdatingShiftState(autoCapsFlags, recapitalizeMode)
                    return
                }
                // In shift locked state, shift has been pressed and slid out to other key.
                withSliding && alphabetShiftState.isShiftLockShifted -> setShiftLocked(true)
                // Shift has been pressed and slid out to other key.
                withSliding && alphabetShiftState.isManualShifted -> switchState = SwitchState.MOMENTARY_ALPHA_SHIFT
                // Shift has been long pressed, ignore this release.
                isShiftLocked && !withSliding && !alphabetShiftState.isShiftLockShifted
                    && (shiftKeyState.isPressing || shiftKeyState.isPressingOnShifted) -> {}
                // Shift has been pressed without chording while shift locked state.
                isShiftLocked && !shiftKeyState.isIgnoring && !withSliding -> setShiftLocked(false)
                // Shift has been pressed without chording while shifted state.
                !withSliding && ((alphabetShiftState.isShiftedOrShiftLocked && shiftKeyState.isPressingOnShifted)
                    // Shift has been pressed without chording while manual shifted transited from automatic shifted
                    || (alphabetShiftState.isManualShiftedFromAutomaticShifted && shiftKeyState.isPressing)) -> {
                        setShifted(ShiftMode.UNSHIFT)
                        isInAlphabetUnshiftedFromShifted = true
                    }
            }
        }
        shiftKeyState.onRelease()
    }

    fun onFinishSlidingInput(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DEBUG_EVENT) {
            Log.d(TAG, "onFinishSlidingInput: " + stateToString(autoCapsFlags, recapitalizeMode))
        }
        // Switch back to the previous keyboard mode if the user didn't enter the numpad.
        if (mode != Mode.NUMPAD) when (switchState) {
            SwitchState.MOMENTARY_ALPHA_AND_SYMBOL -> toggleAlphabetAndSymbols(autoCapsFlags, recapitalizeMode)
            SwitchState.MOMENTARY_SYMBOL_AND_MORE  -> toggleShiftInSymbols()
            SwitchState.MOMENTARY_ALPHA_SHIFT      -> setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
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
            KeyCode.ALPHA -> setAlphabetKeyboard(autoCapsFlags, recapitalizeMode)
            // Note: Printing clipboard content is handled in InputLogic.handleFunctionalEvent
            KeyCode.CLIPBOARD -> if (Settings.getValues().mClipboardHistoryEnabled) setClipboardKeyboard()
            KeyCode.NUMPAD -> toggleNumpad(false, autoCapsFlags, recapitalizeMode, false, true)
            KeyCode.SYMBOL -> setSymbolsKeyboard()
            KeyCode.TOGGLE_ONE_HANDED_MODE -> setOneHandedModeEnabled(!Settings.getValues().mOneHandedModeEnabled)
            KeyCode.SWITCH_ONE_HANDED_MODE -> switchOneHandedMode()
        }
    }

    override fun toString(): String {
        val keyboard = when {
            mode == Mode.ALPHABET -> alphabetShiftState.toString()
            isSymbolShifted -> "SYMBOLS_SHIFTED"
            else -> "SYMBOLS"
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
        EMOJI,
        CLIPBOARD,
        NUMPAD,
    }

    private enum class ShiftMode {
        UNSHIFT,
        MANUAL,
        AUTOMATIC,
        SHIFT_LOCKED,
    }

    companion object {
        private val TAG = KeyboardState::class.java.simpleName
        private const val DEBUG_EVENT = false

        private fun isSpaceOrEnter(c: Int) = c == Constants.CODE_SPACE || c == Constants.CODE_ENTER
    }
}
