// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.internal;

import androidx.annotation.Nullable;

import helium314.keyboard.latin.utils.CapsModeUtils;
import helium314.keyboard.latin.utils.RecapitalizeMode;

public sealed interface LayoutDirective {
    KeyboardState.Mode mode();

    record Alphabet(
        ShiftMode shiftMode,
        int autoCapsFlags,
        @Nullable RecapitalizeMode recapitalizeMode
    ) implements LayoutDirective {
        @Override
        public KeyboardState.Mode mode() {
            return KeyboardState.Mode.ALPHABET;
        }

        @Override
        public String toString() {
            return "Alphabet{shiftMode=" + shiftMode
                 + ", autoCapsFlags=" + CapsModeUtils.flagsToString(autoCapsFlags)
                 + ", recapitalizeMode=" + recapitalizeMode
                 + '}'
            ;
        }
    }

    enum Utility implements LayoutDirective {
        SYMBOLS(KeyboardState.Mode.SYMBOLS),
        SYMBOLS_SHIFTED(KeyboardState.Mode.SYMBOLS_SHIFTED),
        EMOJI(KeyboardState.Mode.EMOJI),
        CLIPBOARD(KeyboardState.Mode.CLIPBOARD),
        NUMPAD(KeyboardState.Mode.NUMPAD),
    ;
        private final KeyboardState.Mode mMode;

        Utility(KeyboardState.Mode mode) {
            mMode = mode;
        }

        @Override
        public KeyboardState.Mode mode() {
            return mMode;
        }
    }
}
