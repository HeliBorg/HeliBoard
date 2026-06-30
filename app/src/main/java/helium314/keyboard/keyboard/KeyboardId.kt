/*
 * Copyright (C) 2015 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard

import android.text.InputType
import android.text.TextUtils
import android.view.inputmethod.EditorInfo
import helium314.keyboard.compat.EditorInfoCompatUtils.imeActionName
import helium314.keyboard.latin.CapsMode
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodSubtype
import helium314.keyboard.latin.utils.InputTypeUtils

/**
 * Unique identifier for each keyboard type.
 */
data class KeyboardId(
    val element: KeyboardElement,
    val subtype: RichInputMethodSubtype,
    val width: Int,
    val height: Int,
    val mode: KeyboardMode,
    val inputType: Int,
    val imeOptions: Int,
    val imeAction: Int,
    val deviceLocked: Boolean,
    val numberRowEnabled: Boolean,
    val numberRowInSymbols: Boolean,
    val languageSwitchKeyEnabled: Boolean,
    val emojiKeyEnabled: Boolean,
    val customActionLabel: String?,
    val hasShortcutKey: Boolean,
    val isSplitLayout: Boolean,
    val oneHandedModeEnabled: Boolean,
    val internalAction: KeyboardLayoutSet.InternalAction?,
    val emojiSearchAvailable: Boolean
) {
    lateinit var editorInfo: EditorInfo // we don't want it in the data class constructor

    constructor(element: KeyboardElement, params: KeyboardLayoutSet.Params) : this(
        element,
        params.subtype,
        params.keyboardWidth,
        params.keyboardHeight,
        params.mode,
        params.editorInfo.inputType,
        params.editorInfo.imeOptions,
        InputTypeUtils.getImeOptionsActionIdFromEditorInfo(params.editorInfo),
        params.deviceLocked,
        params.numberRowEnabled,
        params.numberRowInSymbols,
        params.languageSwitchKeyEnabled,
        params.emojiKeyEnabled,
        params.editorInfo.actionLabel?.toString(),
        params.voiceInputKeyEnabled,
        params.isSplitLayoutEnabled,
        params.oneHandedModeEnabled,
        params.internalAction,
        params.emojiSearchAvailable,
    ) {
        editorInfo = params.editorInfo
    }

    fun navigateNext(): Boolean {
        return (imeOptions and EditorInfo.IME_FLAG_NAVIGATE_NEXT) != 0
            || imeAction == EditorInfo.IME_ACTION_NEXT
    }

    fun navigatePrevious(): Boolean {
        return (imeOptions and EditorInfo.IME_FLAG_NAVIGATE_PREVIOUS) != 0
            || imeAction == EditorInfo.IME_ACTION_PREVIOUS
    }

    val isPasswordInput get() = InputTypeUtils.isAnyPasswordInputType(inputType)

    val isMultiLine get() = (inputType and InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0

    val locale get() = subtype.locale

    companion object {
        fun equivalentEditorInfoForKeyboard(a: EditorInfo?, b: EditorInfo?): Boolean {
            if (a == null && b == null) return true
            if (a == null || b == null) return false
            return a.inputType == b.inputType && a.imeOptions == b.imeOptions && TextUtils.equals(a.privateImeOptions, b.privateImeOptions)
        }

        fun actionName(actionId: Int) = if (actionId == InputTypeUtils.IME_ACTION_CUSTOM_LABEL) "actionCustomLabel"
            else imeActionName(actionId)
    }
}

enum class KeyboardElement(val descriptionResId: Int) {
    ALPHABET(R.string.spoken_description_mode_alpha),
    ALPHABET_AUTOMATIC_SHIFTED(R.string.spoken_description_mode_alpha),
    ALPHABET_MANUAL_SHIFTED(R.string.spoken_description_shiftmode_on),
    ALPHABET_SHIFT_LOCKED(R.string.spoken_description_shiftmode_locked),
    SYMBOLS(R.string.spoken_description_mode_symbol),
    SYMBOLS_SHIFTED(R.string.spoken_description_mode_symbol_shift),
    DPAD(R.string.spoken_description_mode_dpad),
    NUMPAD(R.string.spoken_description_mode_numpad),
    NUMBER(R.string.spoken_description_mode_number),
    PHONE(R.string.spoken_description_mode_phone),
    PHONE_SYMBOLS(R.string.spoken_description_mode_phone_shift),
    EMOJI_RECENTS(R.string.spoken_description_emoji_category_recents),
    EMOJI_SMILEYS(R.string.spoken_description_emoji_category_eight_smiley),
    EMOJI_PEOPLE(R.string.spoken_description_emoji_category_eight_smiley_people),
    EMOJI_NATURE(R.string.spoken_description_emoji_category_eight_animals_nature),
    EMOJI_FOOD(R.string.spoken_description_emoji_category_eight_food_drink),
    EMOJI_TRAVEL_PLACES(R.string.spoken_description_emoji_category_eight_travel_places),
    EMOJI_ACTIVITIES(R.string.spoken_description_emoji_category_eight_activity),
    EMOJI_OBJECTS(R.string.spoken_description_emoji_category_objects),
    EMOJI_SYMBOLS(R.string.spoken_description_emoji_category_symbols),
    EMOJI_FLAGS(R.string.spoken_description_emoji_category_flags),
    EMOJI_EMOTICONS(R.string.spoken_description_emoji_category_emoticons),
    EMOJI_BOTTOM_ROW(R.string.spoken_description_emoji),
    CLIPBOARD(R.string.spoken_description_mode_clipboard),
    CLIPBOARD_BOTTOM_ROW(R.string.spoken_description_mode_clipboard);

    val isAlphabet get() = this < SYMBOLS
    val isAlphaOrSymbol get() = this <= SYMBOLS_SHIFTED
    val isAlphabetShifted get() = isAlphabet && this != ALPHABET
    val isAlphabetShiftedManually get() = this == ALPHABET_MANUAL_SHIFTED || this == ALPHABET_SHIFT_LOCKED
    val isNumberLayout get() = this in NUMPAD..PHONE_SYMBOLS
    val takesFunctionalKeys get() = this <= DPAD
    val isEmojiLayout get() = this in EMOJI_RECENTS..EMOJI_EMOTICONS
    val isBottomRow get() = this == EMOJI_BOTTOM_ROW || this == CLIPBOARD_BOTTOM_ROW
    val capsMode get() = when (this) {
        ALPHABET_AUTOMATIC_SHIFTED -> CapsMode.AUTO
        ALPHABET_MANUAL_SHIFTED -> CapsMode.MANUAL
        ALPHABET_SHIFT_LOCKED -> CapsMode.MANUAL_LOCKED
        else -> CapsMode.OFF
    }
}

enum class KeyboardMode { TEXT, URL, EMAIL, IM, PHONE, NUMBER, DATE, TIME, DATETIME }
