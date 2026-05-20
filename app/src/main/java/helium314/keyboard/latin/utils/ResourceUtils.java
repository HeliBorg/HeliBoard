/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import android.content.Context;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Insets;
import android.graphics.Rect;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.view.WindowMetrics;

import helium314.keyboard.latin.R;
import helium314.keyboard.latin.settings.SettingsValues;

public final class ResourceUtils {

    public static final float UNDEFINED_RATIO = -1.0f;
    public static final int UNDEFINED_DIMENSION = -1;

    private ResourceUtils() {
        // This utility class is not publicly instantiable.
    }

    public static int getKeyboardWidth(final Context ctx, final SettingsValues settingsValues) {
        // Floating keyboard width takes priority
        if (settingsValues.mIsFloatingKeyboard) {
            return settingsValues.mFloatingWidth;
        }
        int defaultKeyboardWidth = getDefaultKeyboardWidth(ctx);
        if (settingsValues.mOneHandedModeEnabled) {
            return (int) (settingsValues.mOneHandedModeScale * defaultKeyboardWidth);
        }
        return defaultKeyboardWidth;
    }

    public static int getDefaultKeyboardWidth(final Context ctx) {
        if (Build.VERSION.SDK_INT < 35) {
            final DisplayMetrics dm = ctx.getResources().getDisplayMetrics();
            return dm.widthPixels;
        }
        // Since Android 15, insets aren't subtracted from DisplayMetrics.widthPixels, despite
        // targetSdk remaining set to 30.
        WindowManager wm = ctx.getSystemService(WindowManager.class);
        WindowMetrics windowMetrics = wm.getCurrentWindowMetrics();
        Rect windowBounds = windowMetrics.getBounds();
        WindowInsets windowInsets = windowMetrics.getWindowInsets();
        int insetTypes = WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout();
        Insets insets = windowInsets.getInsetsIgnoringVisibility(insetTypes);
        return windowBounds.width() - insets.left - insets.right;
    }

    public static int getSecondaryKeyboardHeight(final Resources res, final SettingsValues settingsValues) {
        final int keyboardHeight = getKeyboardHeight(res, settingsValues);
        if (settingsValues.mToolbarMode == ToolbarMode.HIDDEN && ! settingsValues.mToolbarHidingGlobal) {
            // Small adjustment to match the height of the main keyboard which has a hidden strip container.
            return keyboardHeight - (int) res.getDimension(R.dimen.config_suggestions_strip_height);
        }
        return keyboardHeight;
    }

    public static int getKeyboardHeight(Resources res, SettingsValues settingsValues) {
        if (settingsValues.mIsFloatingKeyboard) {
            return settingsValues.mFloatingHeight;
        }
        int defaultKeyboardHeight = getDefaultKeyboardHeight(res, settingsValues.mShowsNumberRow);
        return (int)(defaultKeyboardHeight * settingsValues.mKeyboardHeightScale);
    }

    public static int getDefaultKeyboardHeight(final Resources res, final boolean showsNumberRow) {
        final DisplayMetrics dm = res.getDisplayMetrics();
        final float keyboardHeight = res.getDimension(R.dimen.config_default_keyboard_height) * (showsNumberRow ? 1.33f : 1f);
        final float maxKeyboardHeight = res.getFraction(
                R.fraction.config_max_keyboard_height, dm.heightPixels, dm.heightPixels);
        float minKeyboardHeight = res.getFraction(
                R.fraction.config_min_keyboard_height, dm.heightPixels, dm.heightPixels);
        if (minKeyboardHeight < 0.0f) {
            // Specified fraction was negative, so it should be calculated against display
            // width.
            minKeyboardHeight = -res.getFraction(
                    R.fraction.config_min_keyboard_height, dm.widthPixels, dm.widthPixels);
        }
        // Keyboard height will not exceed maxKeyboardHeight and will not be less than
        // minKeyboardHeight.
        return (int)Math.max(Math.min(keyboardHeight, maxKeyboardHeight), minKeyboardHeight);
    }

    public static boolean isValidFraction(final float fraction) {
        return fraction >= 0.0f;
    }

    // {@link Resources#getDimensionPixelSize(int)} returns at least one pixel size.
    public static boolean isValidDimensionPixelSize(final int dimension) {
        return dimension > 0;
    }

    public static float getFraction(final TypedArray a, final int index, final float defValue) {
        final TypedValue value = a.peekValue(index);
        if (value == null || !isFractionValue(value)) {
            return defValue;
        }
        return a.getFraction(index, 1, 1, defValue);
    }

    public static float getFraction(final TypedArray a, final int index) {
        return getFraction(a, index, UNDEFINED_RATIO);
    }

    public static int getDimensionPixelSize(final TypedArray a, final int index) {
        final TypedValue value = a.peekValue(index);
        if (value == null || !isDimensionValue(value)) {
            return ResourceUtils.UNDEFINED_DIMENSION;
        }
        return a.getDimensionPixelSize(index, ResourceUtils.UNDEFINED_DIMENSION);
    }

    public static float getDimensionOrFraction(final TypedArray a, final int index, final int base,
            final float defValue) {
        final TypedValue value = a.peekValue(index);
        if (value == null) {
            return defValue;
        }
        if (isFractionValue(value)) {
            return a.getFraction(index, base, base, defValue);
        } else if (isDimensionValue(value)) {
            return a.getDimension(index, defValue);
        }
        return defValue;
    }

    public static boolean isFractionValue(final TypedValue v) {
        return v.type == TypedValue.TYPE_FRACTION;
    }

    public static boolean isDimensionValue(final TypedValue v) {
        return v.type == TypedValue.TYPE_DIMENSION;
    }

    public static boolean isNight(final Resources res) {
        return (res.getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
    }
}
