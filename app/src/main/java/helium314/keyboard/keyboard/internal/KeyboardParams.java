/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.util.SparseIntArray;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.KeyboardId;
import helium314.keyboard.keyboard.internal.keyboard_parser.LocaleKeyboardInfos;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.settings.SettingsValues;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.ResourceUtils;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.SortedSet;
import java.util.TreeSet;

public class KeyboardParams {
    private static final int DEFAULT_KEYBOARD_COLUMNS = 10;
    public static final int DEFAULT_KEYBOARD_ROWS = 4;

    public KeyboardId mId;
    public int mThemeId;

    /** Total height and width of the keyboard, including the paddings and keys */
    public int mOccupiedHeight;
    public int mOccupiedWidth;

    /** Base height and width of the keyboard used to calculate rows' or keys' heights and
     *  widths
     */
    public int mBaseHeight;
    public int mBaseWidth;

    public int mTopPadding;
    public int mBottomPadding;
    public int mLeftPadding;
    public int mRightPadding;

    @Nullable
    public KeyVisualAttributes mKeyVisualAttributes;

    public float mDefaultRowHeight;
    public float mDefaultKeyWidth;
    public float mRelativeHorizontalGap;
    public float mRelativeVerticalGap;
    // relative values multiplied with baseHeight / baseWidth
    public int mDefaultAbsoluteRowHeight;
    public int mDefaultAbsoluteKeyWidth;
    public int mHorizontalGap;
    public int mVerticalGap;

    public int mPopupKeysTemplate;
    public int mMaxPopupKeysKeyboardColumn;
    // popup key width is separate from mDefaultAbsoluteKeyWidth because it should not depend on alpha or number layout
    public int mAbsolutePopupKeyWidth;

    public int GRID_WIDTH;
    public int GRID_HEIGHT;

    // Keys are sorted from top-left to bottom-right order.
    @NonNull
    public final SortedSet<Key> mSortedKeys = new TreeSet<>(ROW_COLUMN_COMPARATOR);
    @NonNull
    public final ArrayList<Key> mShiftKeys = new ArrayList<>();
    @NonNull
    public final ArrayList<Key> mAltCodeKeysWhileTyping = new ArrayList<>();
    @NonNull
    public final KeyboardIconsSet mIconsSet = KeyboardIconsSet.Companion.getInstance();
    @NonNull // todo: not good, this only works because params are currently always created for the active subtype
    public final List<Locale> mSecondaryLocales = Settings.getValues().mSecondaryLocales;
    public final ArrayList<String> mPopupKeyOrder = new ArrayList<>();
    public final ArrayList<String> mPopupKeyHintOrder = new ArrayList<>();

    @NonNull
    private final UniqueKeysCache mUniqueKeysCache;
    public boolean mAllowRedundantPopupKeys;
    @NonNull
    public LocaleKeyboardInfos mLocaleKeyboardInfos;
    public boolean setTabletExtraKeys;

    public int mMostCommonKeyHeight = 0;
    public int mMostCommonKeyWidth = 0;

    // should be enabled for all alphabet layouts, except for specific layouts when shifted
    public boolean mProximityCharsCorrectionEnabled;

    // only for removing redundant popup keys
    public List<Key.KeyParams> baseKeys;

    @NonNull
    public final TouchPositionCorrection mTouchPositionCorrection = new TouchPositionCorrection();

    // Comparator to sort {@link Key}s from top-left to bottom-right order.
    private static final Comparator<Key> ROW_COLUMN_COMPARATOR = (lhs, rhs) -> {
        if (lhs.getY() < rhs.getY()) return -1;
        if (lhs.getY() > rhs.getY()) return 1;
        if (lhs.getX() < rhs.getX()) return -1;
        if (lhs.getX() > rhs.getX()) return 1;
        return 0;
    };

    public KeyboardParams() {
        this(UniqueKeysCache.NO_CACHE);
    }

    public KeyboardParams(@NonNull final UniqueKeysCache keysCache) {
        mUniqueKeysCache = keysCache;
    }

    protected void clearKeys() {
        mSortedKeys.clear();
        mShiftKeys.clear();
        clearHistogram();
    }

    public void onAddKey(@NonNull final Key newKey) {
        final Key key = mUniqueKeysCache.getUniqueKey(newKey);
        final boolean isSpacer = key.isSpacer();
        if (isSpacer && key.getWidth() == 0) {
            // Ignore zero width {@link Spacer}.
            return;
        }
        mSortedKeys.add(key);
        if (isSpacer) {
            return;
        }
        updateHistogram(key);
        if (key.getCode() == KeyCode.SHIFT) {
            mShiftKeys.add(key);
        }
        if (key.altCodeWhileTyping()) {
            mAltCodeKeysWhileTyping.add(key);
        }
    }

    public void removeRedundantPopupKeys() {
        if (mAllowRedundantPopupKeys || baseKeys == null) {
            return;
        }
        final PopupKeySpec.LettersOnBaseLayout lettersOnBaseLayout =
                new PopupKeySpec.LettersOnBaseLayout();
        for (final Key.KeyParams key : baseKeys) {
            lettersOnBaseLayout.addLetter(key);
        }
        final ArrayList<Key> allKeys = new ArrayList<>(mSortedKeys);
        mSortedKeys.clear();
        for (final Key key : allKeys) {
            final Key filteredKey = Key.removeRedundantPopupKeys(key, lettersOnBaseLayout);
            mSortedKeys.add(mUniqueKeysCache.getUniqueKey(filteredKey));
        }
        baseKeys = null;
    }

    private int mMaxHeightCount = 0;
    private int mMaxWidthCount = 0;
    private final SparseIntArray mHeightHistogram = new SparseIntArray();
    private final SparseIntArray mWidthHistogram = new SparseIntArray();

    private void clearHistogram() {
        mMostCommonKeyHeight = 0;
        mMaxHeightCount = 0;
        mHeightHistogram.clear();

        mMaxWidthCount = 0;
        mMostCommonKeyWidth = 0;
        mWidthHistogram.clear();
    }

    private static int updateHistogramCounter(final SparseIntArray histogram, final int key) {
        final int index = histogram.indexOfKey(key);
        final int count = (index >= 0 ? histogram.get(key) : 0) + 1;
        histogram.put(key, count);
        return count;
    }

    private void updateHistogram(final Key key) {
        final int height = key.getHeight() + mVerticalGap;
        final int heightCount = updateHistogramCounter(mHeightHistogram, height);
        if (heightCount > mMaxHeightCount) {
            mMaxHeightCount = heightCount;
            mMostCommonKeyHeight = height;
        }

        final int width = key.getWidth() + mHorizontalGap;
        final int widthCount = updateHistogramCounter(mWidthHistogram, width);
        if (widthCount > mMaxWidthCount) {
            mMaxWidthCount = widthCount;
            mMostCommonKeyWidth = width;
        }
    }

    // when attr is null, default attributes will be loaded
    //  these are good for basic keyboards already, but have wrong/unsuitable sizes e.g. for emojis,
    //  popupKeys and moreSuggestions
    public void readAttributes(final Context context, @Nullable final AttributeSet attr) {
        final TypedArray keyboardAttr = context.obtainStyledAttributes(
                attr, R.styleable.Keyboard, R.attr.keyboardStyle, R.style.Keyboard);
        TypedArray keyAttr = attr == null
            ? context.obtainStyledAttributes(attr, R.styleable.Keyboard_Key)
            : context.getResources().obtainAttributes(attr, R.styleable.Keyboard_Key);
        SettingsValues sv = Settings.getValues();
        try {
            final int height = mId.mHeight;
            final int width = mId.mWidth;
            mOccupiedHeight = height;
            mOccupiedWidth = width;
            mTopPadding = (int) keyboardAttr.getFraction(
                    R.styleable.Keyboard_keyboardTopPadding, height, height, 0);
            mBottomPadding = (int) (keyboardAttr.getFraction(
                    R.styleable.Keyboard_keyboardBottomPadding, height, height, 0)
                    * sv.mBottomPaddingScale);
            mLeftPadding = (int) (keyboardAttr.getFraction(
                    R.styleable.Keyboard_keyboardLeftPadding, width, width, 0)
                    * sv.mSidePaddingScale);
            mRightPadding = (int) (keyboardAttr.getFraction(
                    R.styleable.Keyboard_keyboardRightPadding, width, width, 0)
                    * sv.mSidePaddingScale);

            mBaseWidth = mOccupiedWidth - mLeftPadding - mRightPadding;
            final float defaultKeyWidthFactor = context.getResources().getInteger(R.integer.config_screen_metrics) > 2 ? 0.9f : 1f;
            final float alphaSymbolKeyWidth = keyAttr.getFraction(R.styleable.Keyboard_Key_keyWidth,
                    1, 1, defaultKeyWidthFactor / DEFAULT_KEYBOARD_COLUMNS);
            mDefaultKeyWidth = mId.isNumberLayout() ? 0.17f : alphaSymbolKeyWidth;
            mDefaultAbsoluteKeyWidth = (int) (mDefaultKeyWidth * mBaseWidth);
            mAbsolutePopupKeyWidth = (int) (alphaSymbolKeyWidth * mBaseWidth);

            mRelativeHorizontalGap = keyboardAttr.getFraction(
                    R.styleable.Keyboard_horizontalGap, 1, 1, 0) * sv.mKeyGapScale;
            mRelativeVerticalGap = keyboardAttr.getFraction(
                    R.styleable.Keyboard_verticalGap, 1, 1, 0) * sv.mKeyGapScale;
            mHorizontalGap = (int) (mRelativeHorizontalGap * width);
            mVerticalGap = (int) (mRelativeVerticalGap * height);

            mBaseHeight = mOccupiedHeight - mTopPadding - mBottomPadding + mVerticalGap;
            mDefaultRowHeight = ResourceUtils.getDimensionOrFraction(keyboardAttr,
                    R.styleable.Keyboard_rowHeight, 1, 1f / DEFAULT_KEYBOARD_ROWS);
            if (mDefaultRowHeight > 1) { // can be absolute size, in that case will be > 1
                mDefaultAbsoluteRowHeight = (int) mDefaultRowHeight;
                mDefaultRowHeight *= -1; // make it negative when it's absolute
            } else {
                mDefaultAbsoluteRowHeight = (int) (mDefaultRowHeight * mBaseHeight);
            }

            mKeyVisualAttributes = KeyVisualAttributes.newInstance(keyAttr);

            mPopupKeysTemplate = keyboardAttr.getResourceId(R.styleable.Keyboard_popupKeysTemplate, 0);
            mMaxPopupKeysKeyboardColumn = keyAttr.getInt(R.styleable.Keyboard_Key_maxPopupKeysColumn, 5);

            mThemeId = keyboardAttr.getInt(R.styleable.Keyboard_themeId, 0);
            mIconsSet.loadIcons(context);

            // touchPositionResId currently is 0 for popups, and touch_position_correction_data_holo for others
            final int touchPositionResId = keyboardAttr.getResourceId(R.styleable.Keyboard_touchPositionCorrectionData, 0);
            if (touchPositionResId != 0) {
                final int actualId = mId.isAlphabetKeyboard() ? touchPositionResId : R.array.touch_position_correction_data_default;
                final String[] data = context.getResources().getStringArray(actualId);
                mTouchPositionCorrection.load(data);
            }
        } finally {
            keyAttr.recycle();
            keyboardAttr.recycle();
        }
        setTabletExtraKeys = Settings.getInstance().isTablet() && !mId.mSubtype.isCustom();
    }
}
