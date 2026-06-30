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
import helium314.keyboard.keyboard.internal.LayoutDirective.Alphabet
import helium314.keyboard.keyboard.internal.LayoutDirective.Utility
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
        fun setSymbolsKeyboard()
        fun setSymbolsShiftedKeyboard()
        fun toggleLayout(layout: Utility, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?)
        fun onLongPressAlphaSymbolForNumpad()

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
    private var prevShiftMode: ShiftMode? = null
    private var isInShiftSlide = false
    private var isInLayoutSlide = false

    private var mode = Mode.ALPHABET
    private val prevLayouts = WeakStack(Mode.entries)
    private var isInSpaceToAlpha = false
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
        isInShiftSlide = false
        isInLayoutSlide = false
        prevLayouts.wipe()
        isInSpaceToAlpha = false
        shiftKeyState = ModifierKeyState.RELEASED
        symbolKeyState = ModifierKeyState.RELEASED
        if (savedKeyboardState.isValid) {
            onRestoreKeyboardState(autoCapsFlags, recapitalizeMode)
            savedKeyboardState.isValid = false
        } else {
            // Reset keyboard to alphabet mode.
            loadLayout(Alphabet(ShiftMode.UNSHIFT, autoCapsFlags, recapitalizeMode))
            switchActions.requestUpdatingShiftState(autoCapsFlags, recapitalizeMode)
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
        // don't save previous layout if reloading from orientation change, etc.
        prevLayouts.wipe()
        loadLayout(savedKeyboardState.mode.directive(savedKeyboardState.shiftMode, autoCapsFlags, recapitalizeMode))
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

    private fun resetToAlpha(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "resetToAlpha: ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }
        prevLayouts.wipe()
        if (mode == Mode.ALPHABET) {
            return
        }
        loadLayout(Alphabet(shiftMode, autoCapsFlags, recapitalizeMode))
    }

    private fun slideInto(layout: LayoutDirective) {
        isInLayoutSlide = true
        setLayout(layout)
    }

    private fun setLayout(layout: LayoutDirective) {
        prevLayouts.push(mode)
        loadLayout(layout)
    }

    private fun loadLayout(layout: LayoutDirective) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "loadLayout($layout)")
        }
        when (layout) {
            is Alphabet -> switchActions.setAlphabetKeyboard(layout.shiftMode)
            Utility.SYMBOLS -> switchActions.setSymbolsKeyboard()
            Utility.SYMBOLS_SHIFTED -> switchActions.setSymbolsShiftedKeyboard()
            Utility.EMOJI -> switchActions.setEmojiKeyboard()
            Utility.CLIPBOARD -> switchActions.setClipboardKeyboard()
            Utility.NUMPAD -> switchActions.setNumpadKeyboard()
        }
        mode = layout.mode()
        recapitalizeMode = null
        isInSpaceToAlpha = false
        if (layout is Alphabet && layout.shiftMode == ShiftMode.AUTOMATIC) {
            switchActions.requestUpdatingShiftState(layout.autoCapsFlags, layout.recapitalizeMode)
        }
    }

    fun onLongPressAlphaSymbolForNumpad() {
        // We want sliding input to return to the original layout, so
        // don't remember the layout shown momentarily when holding
        loadLayout(Utility.NUMPAD)
    }

    fun toggleLayout(layout: Utility, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "toggleLayout(layout=$layout, autoCapsFlags=${CapsModeUtils.flagsToString(autoCapsFlags)}, recapitalizeMode=$recapitalizeMode)")
        }
        if (mode == layout.mode()) {
            loadPreviousLayout(autoCapsFlags, recapitalizeMode)
        } else {
            setLayout(layout)
        }
        if (isInLayoutSlide) {
            prevLayouts.pop()
            isInLayoutSlide = false
        }
    }

    private fun loadPreviousLayout(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        loadLayout(prevLayouts.pop().directive(shiftMode, autoCapsFlags, recapitalizeMode))
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
            KeyCode.SYMBOL_ALPHA -> {
                if (symbolKeyState == ModifierKeyState.CHORDING) {
                    // Switch back to the previous keyboard mode if the user chords the mode change key and
                    // another key, then releases the mode change key.
                    loadPreviousLayout(autoCapsFlags, recapitalizeMode)
                }
                if (withSliding) {
                    isInLayoutSlide = true
                }
                symbolKeyState = ModifierKeyState.RELEASED
            }
            // if no sliding, switching is instead handled by onEvent()
            // to accommodate toolbar keys and prevent double-loads.
            KeyCode.SYMBOL       -> if (withSliding) slideInto(Utility.SYMBOLS)
            KeyCode.ALPHA        -> if (withSliding) slideInto(Alphabet(shiftMode, autoCapsFlags, recapitalizeMode))
            KeyCode.NUMPAD       -> if (withSliding) slideInto(Utility.NUMPAD)
        }
    }

    private fun onPressAlphaSymbol(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        setLayout(
            if (mode == Mode.ALPHABET) {
                Utility.SYMBOLS
            } else {
                Alphabet(shiftMode, autoCapsFlags, recapitalizeMode)
            }
        )
        symbolKeyState = ModifierKeyState.PRESSING
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
        resetToAlpha(autoCapsFlags, recapitalizeMode)
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
            //  need to know whether this is an update for typing a letter that pops manual shift or
            //  moving the caret or whatever other cases. would also be nice for the space bar (and
            //  maybe other non-letter characters?) to not pop manual shift, which is especially
            //  confusing when space-to-alpha activates on the symbol or numpad layouts.
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
            // In symbols mode, the shift key is actually the more-symbols toggle
            when (mode) {
                Mode.ALPHABET -> {}
                Mode.SYMBOLS -> {
                    setLayout(Utility.SYMBOLS_SHIFTED)
                    return
                }
                Mode.SYMBOLS_SHIFTED -> {
                    setLayout(Utility.SYMBOLS)
                    return
                }
                else -> return
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
                // In symbols mode, the shift key is actually the more-symbols toggle
                if (shiftKeyState == ModifierKeyState.CHORDING) {
                    loadPreviousLayout(autoCapsFlags, recapitalizeMode)
                }
                if (withSliding) {
                    isInLayoutSlide = true
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
                isInShiftSlide = true
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
        if (isInShiftSlide) {
            isInShiftSlide = false
            restorePreviousShiftMode()
            prevShiftMode = null
        } else if (isInLayoutSlide) {
            isInLayoutSlide = false
            loadPreviousLayout(autoCapsFlags, recapitalizeMode)
        }
    }

    fun onEvent(event: Event, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        val code = if (event.isFunctionalKeyEvent) event.keyCode else event.codePoint
        if (DEBUG_EVENT) {
            Log.d(TAG, "onEvent: code=${Constants.printableCode(code)} ${stateToString(autoCapsFlags, recapitalizeMode)}")
        }

        if (mode in Settings.getValues().mAlphaAfterSpace) {
            if (isSpaceOrEnter(code)) {
                if (isInSpaceToAlpha) {
                    resetToAlpha(autoCapsFlags, recapitalizeMode)
                }
            } else if (Constants.isLetterCode(code) || code == KeyCode.MULTIPLE_CODE_POINTS) {
                isInSpaceToAlpha = true
            }
        }

        if (Constants.isLetterCode(code)) {
            // If the code is a letter, update keyboard shift state.
            updateAlphabetShiftState(autoCapsFlags, recapitalizeMode)
        } else when (code) {
            KeyCode.EMOJI -> toggleLayout(Utility.EMOJI, autoCapsFlags, recapitalizeMode)
            KeyCode.ALPHA -> resetToAlpha(autoCapsFlags, recapitalizeMode)
            // Note: Printing clipboard content is handled in InputLogic.handleFunctionalEvent
            KeyCode.CLIPBOARD -> if (Settings.getValues().mClipboardHistoryEnabled) {
                toggleLayout(Utility.CLIPBOARD, autoCapsFlags, recapitalizeMode)
            }
            KeyCode.NUMPAD -> toggleLayout(Utility.NUMPAD, autoCapsFlags, recapitalizeMode)
            KeyCode.SYMBOL -> toggleLayout(Utility.SYMBOLS, autoCapsFlags, recapitalizeMode)
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
        return "[keyboard=$keyboard shift=$shiftKeyState symbol=$symbolKeyState]"
    }

    private fun stateToString(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) =
        "$this autoCapsFlags=${CapsModeUtils.flagsToString(autoCapsFlags)} recapitalizeMode=$recapitalizeMode"

    enum class Mode {
        ALPHABET,
        SYMBOLS,
        SYMBOLS_SHIFTED,
        EMOJI,
        CLIPBOARD,
        NUMPAD,
    ;
        fun directive(shiftMode: ShiftMode, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?): LayoutDirective {
            return when (this) {
                ALPHABET -> Alphabet(shiftMode, autoCapsFlags, recapitalizeMode)
                SYMBOLS -> Utility.SYMBOLS
                SYMBOLS_SHIFTED -> Utility.SYMBOLS_SHIFTED
                EMOJI -> Utility.EMOJI
                CLIPBOARD -> Utility.CLIPBOARD
                NUMPAD -> Utility.NUMPAD
            }
        }
    }

    companion object {
        private val TAG = KeyboardState::class.java.simpleName
        private const val DEBUG_EVENT = false

        private fun isSpaceOrEnter(c: Int) = c == Constants.CODE_SPACE || c == Constants.CODE_ENTER
    }
}
