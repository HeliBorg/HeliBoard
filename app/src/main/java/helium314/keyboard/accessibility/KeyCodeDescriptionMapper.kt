/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.accessibility

import android.content.Context
import android.text.TextUtils
import helium314.keyboard.latin.utils.Log
import android.util.SparseIntArray
import android.view.inputmethod.EditorInfo
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.StringUtils

internal class KeyCodeDescriptionMapper private constructor() {
    // Sparse array of spoken description resource IDs indexed by key codes
    private val mKeyCodeMap = SparseIntArray().apply {
        // Special non-character codes defined in Keyboard
        put(Constants.CODE_SPACE, R.string.spoken_description_space)
        put(KeyCode.DELETE, R.string.spoken_description_delete)
        put(Constants.CODE_ENTER, R.string.spoken_description_return)
        put(KeyCode.SETTINGS, R.string.spoken_description_settings)
        put(KeyCode.SHIFT, R.string.spoken_description_shift)
        put(KeyCode.VOICE_INPUT, R.string.spoken_description_mic)
        put(KeyCode.SYMBOL_ALPHA, R.string.spoken_description_to_symbol)
        put(Constants.CODE_TAB, R.string.spoken_description_tab)
        put(KeyCode.LANGUAGE_SWITCH, R.string.spoken_description_language_switch)
        put(KeyCode.ACTION_NEXT, R.string.spoken_description_action_next)
        put(KeyCode.ACTION_PREVIOUS, R.string.spoken_description_action_previous)
        put(KeyCode.EMOJI, R.string.spoken_description_emoji)
        put(KeyCode.EMOJI_SEARCH, R.string.spoken_description_search)
        // Because the upper-case and lower-case mappings of the following letters is depending on
        // the locale, the upper case descriptions should be defined here. The lower case
        // descriptions are handled in {@link #getSpokenLetterDescriptionId(Context,int)}.
        // U+0049: "I" LATIN CAPITAL LETTER I
        // U+0069: "i" LATIN SMALL LETTER I
        // U+0130: "İ" LATIN CAPITAL LETTER I WITH DOT ABOVE
        // U+0131: "ı" LATIN SMALL LETTER DOTLESS I
        put(0x0049, R.string.spoken_letter_0049)
        put(0x0130, R.string.spoken_letter_0130)
    }

    /**
     * Returns the localized description of the action performed by a specified
     * key based on the current keyboard state.
     *
     * @param context The package's context.
     * @param keyboard The keyboard on which the key resides.
     * @param key The key from which to obtain a description.
     * @param shouldObscure {@true} if text (e.g. non-control) characters should be obscured.
     * @return a character sequence describing the action performed by pressing the key
     */
    fun getDescriptionForKey(context: Context, keyboard: Keyboard?, key: Key, shouldObscure: Boolean): String? {
        val code = key.code
        if (code == KeyCode.SYMBOL_ALPHA || code == KeyCode.SYMBOL || code == KeyCode.ALPHA) {
            val description = getDescriptionForSwitchAlphaSymbol(context, keyboard)
            if (description != null) {
                return description
            }
        }
        if (code == KeyCode.SHIFT) {
            return getDescriptionForShiftKey(context, keyboard)
        }
        if (code == Constants.CODE_ENTER) {
            // The following function returns the correct description in all action and
            // regular enter cases, taking care of all modes.
            return getDescriptionForActionKey(context, keyboard, key)
        }
        if (code == KeyCode.MULTIPLE_CODE_POINTS) {
            return key.outputText ?: context.getString(R.string.spoken_description_unknown)
        }
        // Just attempt to speak the description.
        if (code != KeyCode.NOT_SPECIFIED) {
            // If the key description should be obscured, now is the time to do it.
            val isDefinedNonCtrl = (Character.isDefined(code)
                    && !Character.isISOControl(code))
            if (shouldObscure && isDefinedNonCtrl) {
                return context.getString(OBSCURED_KEY_RES_ID)
            }
            val description = getDescriptionForCodePoint(context, code)
            if (description != null) {
                return description
            }
            return if (!TextUtils.isEmpty(key.label)) {
                key.label
            } else context.getString(R.string.spoken_description_unknown)
        }
        return null
    }

    /**
     * Returns a localized character sequence describing what will happen when
     * the specified key is pressed based on its key code point.
     *
     * @param context The package's context.
     * @param codePoint The code point from which to obtain a description.
     * @return a character sequence describing the code point.
     */
    fun getDescriptionForCodePoint(context: Context, codePoint: Int): String? {
        // If the key description should be obscured, now is the time to do it.
        val index = mKeyCodeMap.indexOfKey(codePoint)
        if (index >= 0) {
            return context.getString(mKeyCodeMap.valueAt(index))
        }
        return if (Character.isDefined(codePoint) && !Character.isISOControl(codePoint)) {
            StringUtils.newSingleCodePointString(codePoint)
        } else null
    }

    companion object {
        private val TAG = KeyCodeDescriptionMapper::class.java.simpleName
        // The resource ID of the string spoken for obscured keys
        private val OBSCURED_KEY_RES_ID = R.string.spoken_description_dot
        val instance = KeyCodeDescriptionMapper()

        /**
         * Returns a context-specific description for the CODE_SWITCH_ALPHA_SYMBOL
         * key or `null` if there is not a description provided for the
         * current keyboard context.
         *
         * @param context The package's context.
         * @param keyboard The keyboard on which the key resides.
         * @return a character sequence describing the action performed by pressing the key
         */
        private fun getDescriptionForSwitchAlphaSymbol(context: Context, keyboard: Keyboard?): String? {
            val resId = when (val elementId = keyboard?.mId?.mElementId) {
                KeyboardId.ELEMENT_ALPHABET, KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED, KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED, KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED, KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCKED -> R.string.spoken_description_to_symbol
                KeyboardId.ELEMENT_SYMBOLS, KeyboardId.ELEMENT_SYMBOLS_SHIFTED -> R.string.spoken_description_to_alpha
                KeyboardId.ELEMENT_PHONE -> R.string.spoken_description_to_symbol
                KeyboardId.ELEMENT_PHONE_SYMBOLS -> R.string.spoken_description_to_numeric
                else -> {
                    Log.e(TAG, "Missing description for keyboard element ID:$elementId")
                    return null
                }
            }
            return context.getString(resId)
        }

        /**
         * Returns a context-sensitive description of the "Shift" key.
         *
         * @param context The package's context.
         * @param keyboard The keyboard on which the key resides.
         * @return A context-sensitive description of the "Shift" key.
         */
        private fun getDescriptionForShiftKey(context: Context, keyboard: Keyboard?): String {
            val resId: Int = when (keyboard?.mId?.mElementId) {
                KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED, KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCKED -> R.string.spoken_description_caps_lock
                KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED, KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED -> R.string.spoken_description_shift_shifted
                KeyboardId.ELEMENT_SYMBOLS -> R.string.spoken_description_symbols_shift
                KeyboardId.ELEMENT_SYMBOLS_SHIFTED -> R.string.spoken_description_symbols_shift_shifted
                else -> R.string.spoken_description_shift
            }
            return context.getString(resId)
        }

        /**
         * Returns a context-sensitive description of the "Enter" action key.
         *
         * @param context The package's context.
         * @param keyboard The keyboard on which the key resides.
         * @param key The key to describe.
         * @return Returns a context-sensitive description of the "Enter" action key.
         */
        private fun getDescriptionForActionKey(context: Context, keyboard: Keyboard?, key: Key): String {
            // Always use the label, if available.
            if (!TextUtils.isEmpty(key.label)) {
                return key.label!!.trim { it <= ' ' }
            }
            val resId = when (keyboard?.mId?.imeAction()) {
                EditorInfo.IME_ACTION_SEARCH -> R.string.label_search_key
                EditorInfo.IME_ACTION_GO -> R.string.label_go_key
                EditorInfo.IME_ACTION_SEND -> R.string.label_send_key
                EditorInfo.IME_ACTION_NEXT -> R.string.label_next_key
                EditorInfo.IME_ACTION_DONE -> R.string.label_done_key
                EditorInfo.IME_ACTION_PREVIOUS -> R.string.label_previous_key
                else -> R.string.spoken_description_return
            }
            return context.getString(resId)
        }
    }

}
