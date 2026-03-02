/*
 * Copyright (C) 2015 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.text.InputType;
import android.text.TextUtils;
import android.view.inputmethod.EditorInfo;

import helium314.keyboard.compat.EditorInfoCompatUtils;
import helium314.keyboard.latin.RichInputMethodSubtype;
import helium314.keyboard.latin.WordComposer;
import helium314.keyboard.latin.utils.InputTypeUtils;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

import static helium314.keyboard.latin.common.Constants.Subtype.ExtraValue.KEYBOARD_LAYOUT_SET;

/**
 * Unique identifier for each keyboard type.
 */
public final class KeyboardId {
    public static final int MODE_TEXT = 0;
    public static final int MODE_URL = 1;
    public static final int MODE_EMAIL = 2;
    public static final int MODE_IM = 3;
    public static final int MODE_PHONE = 4;
    public static final int MODE_NUMBER = 5;
    public static final int MODE_DATE = 6;
    public static final int MODE_TIME = 7;
    public static final int MODE_DATETIME = 8;
    public static final int MODE_NUMPAD = 9;

    public static final int ELEMENT_ALPHABET = 0;
    public static final int ELEMENT_ALPHABET_MANUAL_SHIFTED = 1;
    public static final int ELEMENT_ALPHABET_AUTOMATIC_SHIFTED = 2;
    public static final int ELEMENT_ALPHABET_SHIFT_LOCKED = 3;
    public static final int ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED = 4;
    public static final int ELEMENT_SYMBOLS = 5;
    public static final int ELEMENT_SYMBOLS_SHIFTED = 6;
    public static final int ELEMENT_PHONE = 7;
    public static final int ELEMENT_PHONE_SYMBOLS = 8;
    public static final int ELEMENT_NUMBER = 9;
    public static final int ELEMENT_EMOJI_RECENTS = 10;
    public static final int ELEMENT_EMOJI_CATEGORY1 = 11;
    public static final int ELEMENT_EMOJI_CATEGORY2 = 12;
    public static final int ELEMENT_EMOJI_CATEGORY3 = 13;
    public static final int ELEMENT_EMOJI_CATEGORY4 = 14;
    public static final int ELEMENT_EMOJI_CATEGORY5 = 15;
    public static final int ELEMENT_EMOJI_CATEGORY6 = 16;
    public static final int ELEMENT_EMOJI_CATEGORY7 = 17;
    public static final int ELEMENT_EMOJI_CATEGORY8 = 18;
    public static final int ELEMENT_EMOJI_CATEGORY9 = 19;
    public static final int ELEMENT_EMOJI_CATEGORY10 = 20;
    public static final int ELEMENT_EMOJI_CATEGORY11 = 21;
    public static final int ELEMENT_EMOJI_CATEGORY12 = 22;
    public static final int ELEMENT_EMOJI_CATEGORY13 = 23;
    public static final int ELEMENT_EMOJI_CATEGORY14 = 24;
    public static final int ELEMENT_EMOJI_CATEGORY15 = 25;
    public static final int ELEMENT_EMOJI_CATEGORY16 = 26;  // Emoji search
    public static final int ELEMENT_CLIPBOARD = 27;
    public static final int ELEMENT_NUMPAD = 28;
    public static final int ELEMENT_EMOJI_BOTTOM_ROW = 29;
    public static final int ELEMENT_CLIPBOARD_BOTTOM_ROW = 30;

    public final RichInputMethodSubtype mSubtype;
    public final int mWidth;
    public final int mHeight;
    public final int mMode;
    public final int mElementId;
    public final EditorInfo mEditorInfo;
    public final boolean mDeviceLocked;
    public final boolean mNumberRowEnabled;
    public final boolean mNumberRowInSymbols;
    public final boolean mLanguageSwitchKeyEnabled;
    public final boolean mEmojiKeyEnabled;
    public final String mCustomActionLabel;
    public final boolean mHasShortcutKey;
    public final boolean mIsSplitLayout;
    public final boolean mOneHandedModeEnabled;
    public final KeyboardLayoutSet.InternalAction mInternalAction;
    public final boolean mEmojiSearchAvailable;

    private final int mHashCode;

    public KeyboardId(final int elementId, final KeyboardLayoutSet.Params params) {
        mSubtype = params.mSubtype;
        mWidth = params.mKeyboardWidth;
        mHeight = params.mKeyboardHeight;
        mMode = params.mMode;
        mElementId = elementId;
        mEditorInfo = params.mEditorInfo;
        mDeviceLocked = params.mDeviceLocked;
        mNumberRowEnabled = params.mNumberRowEnabled;
        mNumberRowInSymbols = params.mNumberRowInSymbols;
        mLanguageSwitchKeyEnabled = params.mLanguageSwitchKeyEnabled;
        mEmojiKeyEnabled = params.mEmojiKeyEnabled;
        mCustomActionLabel = (mEditorInfo.actionLabel != null)
                ? mEditorInfo.actionLabel.toString() : null;
        mHasShortcutKey = params.mVoiceInputKeyEnabled;
        mIsSplitLayout = params.mIsSplitLayoutEnabled;
        mOneHandedModeEnabled = params.mOneHandedModeEnabled;
        mInternalAction = params.mInternalAction;
        mEmojiSearchAvailable = params.mEmojiSearchAvailable;

        mHashCode = computeHashCode(this);
    }

    private static int computeHashCode(final KeyboardId id) {
        return Arrays.hashCode(new Object[] {
                id.mElementId,
                id.mMode,
                id.mWidth,
                id.mHeight,
                id.passwordInput(),
                id.mDeviceLocked,
                id.mHasShortcutKey,
                id.mNumberRowEnabled,
                id.mLanguageSwitchKeyEnabled,
                id.mEmojiKeyEnabled,
                id.isMultiLine(),
                id.imeAction(),
                id.mCustomActionLabel,
                id.navigateNext(),
                id.navigatePrevious(),
                id.mSubtype,
                id.mIsSplitLayout,
                id.mInternalAction
        });
    }

    private boolean equals(final KeyboardId other) {
        if (other == this)
            return true;
        return other.mElementId == mElementId
                && other.mMode == mMode
                && other.mWidth == mWidth
                && other.mHeight == mHeight
                && other.passwordInput() == passwordInput()
                && other.mDeviceLocked == mDeviceLocked
                && other.mHasShortcutKey == mHasShortcutKey
                && other.mNumberRowEnabled == mNumberRowEnabled
                && other.mLanguageSwitchKeyEnabled == mLanguageSwitchKeyEnabled
                && other.mEmojiKeyEnabled == mEmojiKeyEnabled
                && other.isMultiLine() == isMultiLine()
                && other.imeAction() == imeAction()
                && TextUtils.equals(other.mCustomActionLabel, mCustomActionLabel)
                && other.navigateNext() == navigateNext()
                && other.navigatePrevious() == navigatePrevious()
                && other.mSubtype.equals(mSubtype)
                && other.mIsSplitLayout == mIsSplitLayout
                && Objects.equals(other.mInternalAction, mInternalAction);
    }

    private static boolean isAlphabetKeyboard(final int elementId) {
        return elementId < ELEMENT_SYMBOLS;
    }

    public boolean isAlphaOrSymbolKeyboard() {
        return mElementId <= ELEMENT_SYMBOLS_SHIFTED;
    }

    public boolean isAlphabetKeyboard() {
        return isAlphabetKeyboard(mElementId);
    }

    public boolean navigateNext() {
        return (mEditorInfo.imeOptions & EditorInfo.IME_FLAG_NAVIGATE_NEXT) != 0
                || imeAction() == EditorInfo.IME_ACTION_NEXT;
    }

    public boolean navigatePrevious() {
        return (mEditorInfo.imeOptions & EditorInfo.IME_FLAG_NAVIGATE_PREVIOUS) != 0
                || imeAction() == EditorInfo.IME_ACTION_PREVIOUS;
    }

    public boolean passwordInput() {
        final int inputType = mEditorInfo.inputType;
        return InputTypeUtils.isAnyPasswordInputType(inputType);
    }

    public boolean isMultiLine() {
        return (mEditorInfo.inputType & InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0;
    }

    public boolean isAlphabetShifted() {
        return mElementId == ELEMENT_ALPHABET_SHIFT_LOCKED || mElementId == ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED
                || mElementId == ELEMENT_ALPHABET_AUTOMATIC_SHIFTED || mElementId == ELEMENT_ALPHABET_MANUAL_SHIFTED;
    }

    public boolean isAlphabetShiftedManually() {
        return mElementId == ELEMENT_ALPHABET_SHIFT_LOCKED || mElementId == ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED
            || mElementId == ELEMENT_ALPHABET_MANUAL_SHIFTED;
    }

    public boolean isNumberLayout() {
        return mElementId == ELEMENT_NUMBER || mElementId == ELEMENT_NUMPAD
                || mElementId == ELEMENT_PHONE || mElementId == ELEMENT_PHONE_SYMBOLS;
    }

    public boolean isEmojiKeyboard() {
        return mElementId >= ELEMENT_EMOJI_RECENTS && mElementId <= ELEMENT_EMOJI_CATEGORY16;
    }

    public boolean isEmojiClipBottomRow() {
        return mElementId == ELEMENT_CLIPBOARD_BOTTOM_ROW || mElementId == ELEMENT_EMOJI_BOTTOM_ROW;
    }

    public int imeAction() {
        return InputTypeUtils.getImeOptionsActionIdFromEditorInfo(mEditorInfo);
    }

    public Locale getLocale() {
        return mSubtype.getLocale();
    }

    @Override
    public boolean equals(final Object other) {
        return other instanceof KeyboardId && equals((KeyboardId) other);
    }

    @Override
    public int hashCode() {
        return mHashCode;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "[%s %s:%s %dx%d %s %s%s%s%s%s%s%s%s%s%s%s%s%s]",
                elementIdToName(mElementId),
                mSubtype.getLocale(),
                mSubtype.getExtraValueOf(KEYBOARD_LAYOUT_SET),
                mWidth, mHeight,
                modeName(mMode),
                actionName(imeAction()),
                (navigateNext() ? " navigateNext" : ""),
                (navigatePrevious() ? " navigatePrevious" : ""),
                (mDeviceLocked ? " deviceLocked" : ""),
                (passwordInput() ? " passwordInput" : ""),
                (mHasShortcutKey ? " hasShortcutKey" : ""),
                (mNumberRowEnabled ? " numberRowEnabled" : ""),
                (mLanguageSwitchKeyEnabled ? " languageSwitchKeyEnabled" : ""),
                (mEmojiKeyEnabled ? " emojiKeyEnabled" : ""),
                (isMultiLine() ? " isMultiLine" : ""),
                (mIsSplitLayout ? " isSplitLayout" : ""),
                (mInternalAction != null ? " internalAction=" + mInternalAction : ""),
                (mEmojiSearchAvailable ? " emojiSearchAvailable" : "")
        );
    }

    public static boolean equivalentEditorInfoForKeyboard(final EditorInfo a, final EditorInfo b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.inputType == b.inputType
                && a.imeOptions == b.imeOptions
                && TextUtils.equals(a.privateImeOptions, b.privateImeOptions);
    }

    public static String elementIdToName(final int elementId) {
        return switch (elementId) {
            case ELEMENT_ALPHABET -> "alphabet";
            case ELEMENT_ALPHABET_MANUAL_SHIFTED -> "alphabetManualShifted";
            case ELEMENT_ALPHABET_AUTOMATIC_SHIFTED -> "alphabetAutomaticShifted";
            case ELEMENT_ALPHABET_SHIFT_LOCKED -> "alphabetShiftLocked";
            case ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED -> "alphabetShiftLockShifted";
            case ELEMENT_SYMBOLS -> "symbols";
            case ELEMENT_SYMBOLS_SHIFTED -> "symbolsShifted";
            case ELEMENT_PHONE -> "phone";
            case ELEMENT_PHONE_SYMBOLS -> "phoneSymbols";
            case ELEMENT_NUMBER -> "number";
            case ELEMENT_EMOJI_RECENTS -> "emojiRecents";
            case ELEMENT_EMOJI_CATEGORY1 -> "emojiCategory1";
            case ELEMENT_EMOJI_CATEGORY2 -> "emojiCategory2";
            case ELEMENT_EMOJI_CATEGORY3 -> "emojiCategory3";
            case ELEMENT_EMOJI_CATEGORY4 -> "emojiCategory4";
            case ELEMENT_EMOJI_CATEGORY5 -> "emojiCategory5";
            case ELEMENT_EMOJI_CATEGORY6 -> "emojiCategory6";
            case ELEMENT_EMOJI_CATEGORY7 -> "emojiCategory7";
            case ELEMENT_EMOJI_CATEGORY8 -> "emojiCategory8";
            case ELEMENT_EMOJI_CATEGORY9 -> "emojiCategory9";
            case ELEMENT_EMOJI_CATEGORY10 -> "emojiCategory10";
            case ELEMENT_EMOJI_CATEGORY11 -> "emojiCategory11";
            case ELEMENT_EMOJI_CATEGORY12 -> "emojiCategory12";
            case ELEMENT_EMOJI_CATEGORY13 -> "emojiCategory13";
            case ELEMENT_EMOJI_CATEGORY14 -> "emojiCategory14";
            case ELEMENT_EMOJI_CATEGORY15 -> "emojiCategory15";
            case ELEMENT_EMOJI_CATEGORY16 -> "emojiCategory16";
            case ELEMENT_CLIPBOARD -> "clipboard";
            case ELEMENT_NUMPAD -> "numpad";
            default -> null;
        };
    }

    public static String modeName(final int mode) {
        return switch (mode) {
            case MODE_TEXT -> "text";
            case MODE_URL -> "url";
            case MODE_EMAIL -> "email";
            case MODE_IM -> "im";
            case MODE_PHONE -> "phone";
            case MODE_NUMBER -> "number";
            case MODE_DATE -> "date";
            case MODE_TIME -> "time";
            case MODE_DATETIME -> "datetime";
            case MODE_NUMPAD -> "numpad";
            default -> null;
        };
    }

    public static String actionName(final int actionId) {
        return (actionId == InputTypeUtils.IME_ACTION_CUSTOM_LABEL) ? "actionCustomLabel"
                : EditorInfoCompatUtils.imeActionName(actionId);
    }

    public int getKeyboardCapsMode() {
        return switch (mElementId) {
            case KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCKED, KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED ->
                WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED;
            case KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED -> WordComposer.CAPS_MODE_MANUAL_SHIFTED;
            case KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED -> WordComposer.CAPS_MODE_AUTO_SHIFTED;
            default -> WordComposer.CAPS_MODE_OFF;
        };
    }
}
