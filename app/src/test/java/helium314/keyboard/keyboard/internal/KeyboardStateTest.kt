package helium314.keyboard.keyboard.internal

import helium314.keyboard.keyboard.internal.LayoutDirective.Utility
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.utils.RecapitalizeMode
import kotlin.test.Test
import kotlin.test.assertEquals

class KeyboardStateTest {
    @Test
    fun deliberateNumpadToDpadSwitchClosesToAlphabet() {
        val actions = RecordingSwitchActions()
        val state = KeyboardState(actions)

        state.toggleLayout(Utility.NUMPAD, AUTO_CAPS_FLAGS, null)
        state.toggleLayout(Utility.DPAD, AUTO_CAPS_FLAGS, null)
        state.toggleLayout(Utility.DPAD, AUTO_CAPS_FLAGS, null)

        assertEquals(
            listOf(KeyboardState.Mode.NUMPAD, KeyboardState.Mode.DPAD, KeyboardState.Mode.ALPHABET),
            actions.loadedLayouts,
        )
    }

    @Test
    fun deliberateDpadToNumpadSwitchClosesToAlphabet() {
        val actions = RecordingSwitchActions()
        val state = KeyboardState(actions)

        state.toggleLayout(Utility.DPAD, AUTO_CAPS_FLAGS, null)
        state.toggleLayout(Utility.NUMPAD, AUTO_CAPS_FLAGS, null)
        state.toggleLayout(Utility.NUMPAD, AUTO_CAPS_FLAGS, null)

        assertEquals(
            listOf(KeyboardState.Mode.DPAD, KeyboardState.Mode.NUMPAD, KeyboardState.Mode.ALPHABET),
            actions.loadedLayouts,
        )
    }

    @Test
    fun slidingFromNumpadThroughSymbolsToDpadStillClosesToNumpad() {
        val actions = RecordingSwitchActions()
        val state = KeyboardState(actions)

        state.toggleLayout(Utility.NUMPAD, AUTO_CAPS_FLAGS, null)
        slideFromCurrentLayoutThroughSymbolsTo(state, Utility.DPAD)
        state.toggleLayout(Utility.DPAD, AUTO_CAPS_FLAGS, null)

        assertEquals(
            listOf(
                KeyboardState.Mode.NUMPAD,
                KeyboardState.Mode.SYMBOLS,
                KeyboardState.Mode.DPAD,
                KeyboardState.Mode.NUMPAD,
            ),
            actions.loadedLayouts,
        )
    }

    @Test
    fun slidingFromDpadThroughSymbolsToNumpadStillClosesToDpad() {
        val actions = RecordingSwitchActions()
        val state = KeyboardState(actions)

        state.toggleLayout(Utility.DPAD, AUTO_CAPS_FLAGS, null)
        slideFromCurrentLayoutThroughSymbolsTo(state, Utility.NUMPAD)
        state.toggleLayout(Utility.NUMPAD, AUTO_CAPS_FLAGS, null)

        assertEquals(
            listOf(
                KeyboardState.Mode.DPAD,
                KeyboardState.Mode.SYMBOLS,
                KeyboardState.Mode.NUMPAD,
                KeyboardState.Mode.DPAD,
            ),
            actions.loadedLayouts,
        )
    }

    private fun slideFromCurrentLayoutThroughSymbolsTo(state: KeyboardState, target: Utility) {
        state.onPressKey(KeyCode.SYMBOL, 1, AUTO_CAPS_FLAGS, null)
        state.onReleaseKey(KeyCode.SYMBOL, true, AUTO_CAPS_FLAGS, null)
        state.toggleLayout(target, AUTO_CAPS_FLAGS, null)
        state.onReleaseKey(
            if (target == Utility.DPAD) KeyCode.DPAD else KeyCode.NUMPAD,
            false,
            AUTO_CAPS_FLAGS,
            null,
        )
        state.onFinishSlidingInput(AUTO_CAPS_FLAGS, null)
    }

    private class RecordingSwitchActions : KeyboardState.SwitchActions {
        val loadedLayouts = mutableListOf<KeyboardState.Mode>()

        override fun setAlphabetKeyboard(shiftMode: ShiftMode) {
            loadedLayouts += KeyboardState.Mode.ALPHABET
        }

        override fun setEmojiKeyboard() {
            loadedLayouts += KeyboardState.Mode.EMOJI
        }

        override fun setClipboardKeyboard() {
            loadedLayouts += KeyboardState.Mode.CLIPBOARD
        }

        override fun setNumpadKeyboard() {
            loadedLayouts += KeyboardState.Mode.NUMPAD
        }

        override fun setDpadKeyboard() {
            loadedLayouts += KeyboardState.Mode.DPAD
        }

        override fun setSymbolsKeyboard() {
            loadedLayouts += KeyboardState.Mode.SYMBOLS
        }

        override fun setSymbolsShiftedKeyboard() {
            loadedLayouts += KeyboardState.Mode.SYMBOLS_SHIFTED
        }

        override fun startDoubleTapShiftKeyTimer() = Unit
        override fun popDoubleTapShiftKeyTimer() = false
        override fun cancelDoubleTapShiftKeyTimer() = Unit
        override fun setOneHandedModeEnabled(enabled: Boolean) = Unit
        override fun switchOneHandedMode() = Unit
        override fun setFloatingKeyboardEnabled(enabled: Boolean) = Unit
    }

    companion object {
        private const val AUTO_CAPS_FLAGS = 0
    }
}
