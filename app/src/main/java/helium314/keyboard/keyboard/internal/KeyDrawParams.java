/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import android.graphics.Typeface;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.latin.utils.ResourceUtils;

public final class KeyDrawParams {
    @NonNull
    public Typeface mTypeface = Typeface.DEFAULT;

    public int mLetterSize;
    public int mLabelSize;
    public int mLargeLetterSize;
    public int mHintLetterSize;
    public int mShiftedLetterHintSize;
    public int mHintLabelSize;
    public int mPreviewTextSize;

    public int mTextColor;
    public int mTextInactivatedColor;
    public int mTextShadowColor;
    public int mFunctionalTextColor;
    public int mHintLetterColor;
    public int mHintLabelColor;
    public int mShiftedLetterHintInactivatedColor;
    public int mShiftedLetterHintActivatedColor;
    public int mPreviewTextColor;

    public float mHintLabelVerticalAdjustment;
    public float mLabelOffCenterRatio;
    public float mHintLabelOffCenterRatio;

    public int mAnimAlpha;

    public KeyDrawParams() {}

    private KeyDrawParams(@NonNull final KeyDrawParams copyFrom) {
        mTypeface = copyFrom.mTypeface;

        mLetterSize = copyFrom.mLetterSize;
        mLabelSize = copyFrom.mLabelSize;
        mLargeLetterSize = copyFrom.mLargeLetterSize;
        mHintLetterSize = copyFrom.mHintLetterSize;
        mShiftedLetterHintSize = copyFrom.mShiftedLetterHintSize;
        mHintLabelSize = copyFrom.mHintLabelSize;
        mPreviewTextSize = copyFrom.mPreviewTextSize;

        mTextColor = copyFrom.mTextColor;
        mTextInactivatedColor = copyFrom.mTextInactivatedColor;
        mTextShadowColor = copyFrom.mTextShadowColor;
        mFunctionalTextColor = copyFrom.mFunctionalTextColor;
        mHintLetterColor = copyFrom.mHintLetterColor;
        mHintLabelColor = copyFrom.mHintLabelColor;
        mShiftedLetterHintInactivatedColor = copyFrom.mShiftedLetterHintInactivatedColor;
        mShiftedLetterHintActivatedColor = copyFrom.mShiftedLetterHintActivatedColor;
        mPreviewTextColor = copyFrom.mPreviewTextColor;

        mHintLabelVerticalAdjustment = copyFrom.mHintLabelVerticalAdjustment;
        mLabelOffCenterRatio = copyFrom.mLabelOffCenterRatio;
        mHintLabelOffCenterRatio = copyFrom.mHintLabelOffCenterRatio;

        mAnimAlpha = copyFrom.mAnimAlpha;
    }

    public void updateParams(int keySize, @Nullable KeyVisualAttributes attr) {
        if (attr == null) {
            return;
        }

        if (attr.mTypeface != null) {
            mTypeface = attr.mTypeface;
        }

        mLetterSize = selectTextSizeFromDimensionOrRatio(keySize,
                attr.mLetterSize, attr.mLetterRatio, mLetterSize);
        mLabelSize = selectTextSizeFromDimensionOrRatio(keySize,
                attr.mLabelSize, attr.mLabelRatio, mLabelSize);
        mLargeLetterSize = selectTextSize(keySize, attr.mLargeLetterRatio, mLargeLetterSize);
        mHintLetterSize = selectTextSize(keySize, attr.mHintLetterRatio, mHintLetterSize);
        mShiftedLetterHintSize = selectTextSize(keySize,
                attr.mShiftedLetterHintRatio, mShiftedLetterHintSize);
        mHintLabelSize = selectTextSize(keySize, attr.mHintLabelRatio, mHintLabelSize);
        mPreviewTextSize = selectTextSize(keySize, attr.mPreviewTextRatio, mPreviewTextSize);

        mTextColor = selectColor(attr.mTextColor, mTextColor);
        mTextInactivatedColor = selectColor(attr.mTextInactivatedColor, mTextInactivatedColor);
        mTextShadowColor = selectColor(attr.mTextShadowColor, mTextShadowColor);
        mFunctionalTextColor = selectColor(attr.mFunctionalTextColor, mFunctionalTextColor);
        mHintLetterColor = selectColor(attr.mHintLetterColor, mHintLetterColor);
        mHintLabelColor = selectColor(attr.mHintLabelColor, mHintLabelColor);
        mShiftedLetterHintInactivatedColor = selectColor(
                attr.mShiftedLetterHintInactivatedColor, mShiftedLetterHintInactivatedColor);
        mShiftedLetterHintActivatedColor = selectColor(
                attr.mShiftedLetterHintActivatedColor, mShiftedLetterHintActivatedColor);
        mPreviewTextColor = selectColor(attr.mPreviewTextColor, mPreviewTextColor);

        mHintLabelVerticalAdjustment = selectFloatIfNonZero(
                attr.mHintLabelVerticalAdjustment, mHintLabelVerticalAdjustment);
        mLabelOffCenterRatio = selectFloatIfNonZero(
                attr.mLabelOffCenterRatio, mLabelOffCenterRatio);
        mHintLabelOffCenterRatio = selectFloatIfNonZero(
                attr.mHintLabelOffCenterRatio, mHintLabelOffCenterRatio);
    }

    @NonNull
    public KeyDrawParams mayCloneAndUpdateParams(int keySize, @Nullable KeyVisualAttributes attr) {
        if (attr == null) {
            return this;
        }
        KeyDrawParams newParams = new KeyDrawParams(this);
        newParams.updateParams(keySize, attr);
        return newParams;
    }

    private static int selectTextSizeFromDimensionOrRatio(int keySize, int dimens, float ratio, int defaultDimens) {
        if (ResourceUtils.isValidDimensionPixelSize(dimens)) {
            return dimens;
        }
        if (ResourceUtils.isValidFraction(ratio)) {
            return (int)(keySize * ratio);
        }
        return defaultDimens;
    }

    private static int selectTextSize(int keySize, float ratio, int defaultSize) {
        if (ResourceUtils.isValidFraction(ratio)) {
            return (int)(keySize * ratio);
        }
        return defaultSize;
    }

    private static int selectColor(final int attrColor, final int defaultColor) {
        if (attrColor != 0) {
            return attrColor;
        }
        return defaultColor;
    }

    private static float selectFloatIfNonZero(final float attrFloat, final float defaultFloat) {
        if (attrFloat != 0) {
            return attrFloat;
        }
        return defaultFloat;
    }
}
