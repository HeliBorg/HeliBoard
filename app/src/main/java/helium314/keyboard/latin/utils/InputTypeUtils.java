/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import android.text.InputType;
import android.view.inputmethod.EditorInfo;
import helium314.keyboard.compat.AppWorkarounds;

public final class InputTypeUtils implements InputType {
    private static final int WEB_TEXT_PASSWORD_INPUT_TYPE = TYPE_CLASS_TEXT | TYPE_TEXT_VARIATION_WEB_PASSWORD;
    private static final int NUMBER_PASSWORD_INPUT_TYPE = TYPE_CLASS_NUMBER | TYPE_NUMBER_VARIATION_PASSWORD;
    private static final int TEXT_PASSWORD_INPUT_TYPE = TYPE_CLASS_TEXT | TYPE_TEXT_VARIATION_PASSWORD;
    private static final int TEXT_VISIBLE_PASSWORD_INPUT_TYPE = TYPE_CLASS_TEXT | TYPE_TEXT_VARIATION_VISIBLE_PASSWORD;
    private static final int[] SUPPRESSING_AUTO_SPACES_FIELD_VARIATION = {
        InputType.TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS,
        InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS,
        InputType.TYPE_TEXT_VARIATION_PASSWORD,
        InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
        InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD };
    public static final int IME_ACTION_CUSTOM_LABEL = EditorInfo.IME_MASK_ACTION + 1;

    private InputTypeUtils() {
        // This utility class is not publicly instantiable.
    }

    public static boolean isNumberInputType(final int inputType) {
        // flags like TYPE_NUMBER_FLAG_DECIMAL are only meaningful within their class, and
        // TYPE_NUMBER_FLAG_DECIMAL == TYPE_TEXT_FLAG_CAP_WORDS, so the class must be checked
        return (inputType & TYPE_MASK_CLASS) == TYPE_CLASS_NUMBER;
    }

    public static boolean isEmailVariation(final int variation) {
        return variation == TYPE_TEXT_VARIATION_EMAIL_ADDRESS || variation == TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS;
    }

    public static boolean isUriOrEmailType(final int inputType) {
        if ((inputType & TYPE_MASK_CLASS) != TYPE_CLASS_TEXT) return false;
        final int maskedInputType = inputType & TYPE_MASK_VARIATION;
        return maskedInputType == TYPE_TEXT_VARIATION_URI || isEmailVariation(maskedInputType);
    }

    // Please refer to TextView.isPasswordInputType
    public static boolean isPasswordInputType(final int inputType) {
        final int maskedInputType = inputType & (TYPE_MASK_CLASS | TYPE_MASK_VARIATION);
        return maskedInputType == WEB_TEXT_PASSWORD_INPUT_TYPE || maskedInputType == TEXT_PASSWORD_INPUT_TYPE
                || maskedInputType == NUMBER_PASSWORD_INPUT_TYPE;
    }

    // Please refer to TextView.isVisiblePasswordInputType
    public static boolean isVisiblePasswordInputType(final int inputType) {
        final int maskedInputType = inputType & (TYPE_MASK_CLASS | TYPE_MASK_VARIATION);
        return maskedInputType == TEXT_VISIBLE_PASSWORD_INPUT_TYPE;
    }

    public static boolean isAnyPasswordInputType(final int inputType) {
        return isPasswordInputType(inputType) || isVisiblePasswordInputType(inputType);
    }

    public static boolean isAutoSpaceFriendlyType(final int inputType) {
        if (TYPE_CLASS_TEXT != (TYPE_MASK_CLASS & inputType)) return false;
        final int variation = TYPE_MASK_VARIATION & inputType;
        for (final int fieldVariation : SUPPRESSING_AUTO_SPACES_FIELD_VARIATION) {
            if (variation == fieldVariation) return false;
        }
        return true;
    }

    public static int getImeOptionsActionIdFromEditorInfo(final EditorInfo editorInfo) {
        final int imeOptions = AppWorkarounds.INSTANCE.adjustImeOptions(editorInfo.imeOptions, editorInfo.packageName);
        if ((imeOptions & EditorInfo.IME_FLAG_NO_ENTER_ACTION) != 0) {
            return EditorInfo.IME_ACTION_NONE;
        } else if (editorInfo.actionLabel != null) {
            return IME_ACTION_CUSTOM_LABEL;
        } else {
            // Note: this is different from editorInfo.actionId, hence "ImeOptionsActionId"
            return imeOptions & EditorInfo.IME_MASK_ACTION;
        }
    }
}
