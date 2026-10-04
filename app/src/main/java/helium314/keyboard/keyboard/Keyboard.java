/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.util.SparseArray;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.android.inputmethod.keyboard.ProximityInfo;

import helium314.keyboard.keyboard.internal.KeyVisualAttributes;
import helium314.keyboard.keyboard.internal.KeyboardIconsSet;
import helium314.keyboard.keyboard.internal.KeyboardParams;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.CoordinateUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Loads an XML description of a keyboard and stores the attributes of the keys. A keyboard
 * consists of rows of keys.
 * <p>The layout file for a keyboard contains XML that looks like the following snippet:</p>
 * <pre>
 * &lt;Keyboard
 *         latin:keyWidth="10%p"
 *         latin:rowHeight="50px"
 *         latin:horizontalGap="2%p"
 *         latin:verticalGap="2%p" &gt;
 *     &lt;Row latin:keyWidth="10%p" &gt;
 *         &lt;Key latin:keyLabel="A" /&gt;
 *         ...
 *     &lt;/Row&gt;
 *     ...
 * &lt;/Keyboard&gt;
 * </pre>
 */
public class Keyboard {
    @NonNull
    public final KeyboardId mId;
    public final int mThemeId;

    /** Total height of the keyboard, including the padding and keys */
    public final int mOccupiedHeight;
    /** Total width of the keyboard, including the padding and keys */
    public int mOccupiedWidth;

    /** Base height of the keyboard, used to calculate rows' height */
    public final int mBaseHeight;
    /** Base width of the keyboard, used to calculate keys' width */
    public int mBaseWidth;

    /** The padding above the keyboard */
    public final int mTopPadding;
    /** Default gap between rows */
    public final int mVerticalGap;

    /** Per keyboard key visual parameters */
    public final KeyVisualAttributes mKeyVisualAttributes;

    public final int mMostCommonKeyHeight;
    public final int mMostCommonKeyWidth;

    /** Popup keys keyboard template */
    public final int mPopupKeysTemplate;

    /** Maximum column for popup keys keyboard */
    public final int mMaxPopupKeysKeyboardColumn;

    /** List of keys in this keyboard */
    @NonNull
    private final List<Key> mSortedKeys;
    @NonNull
    public final List<Key> mShiftKeys;
    @NonNull
    public final List<Key> mAltCodeKeysWhileTyping;
    @NonNull
    public final KeyboardIconsSet mIconsSet;

    private final SparseArray<Key> mKeyCache = new SparseArray<>();

    @NonNull
    private final ProximityInfo mProximityInfo;

    private final boolean mProximityCharsCorrectionEnabled;

    public Keyboard(@NonNull final KeyboardParams params) {
        mId = params.mId;
        mThemeId = params.mThemeId;
        mOccupiedHeight = params.mOccupiedHeight;
        mOccupiedWidth = params.mOccupiedWidth;
        mBaseHeight = params.mBaseHeight;
        mBaseWidth = params.mBaseWidth;
        mMostCommonKeyHeight = params.mMostCommonKeyHeight;
        mMostCommonKeyWidth = params.mMostCommonKeyWidth;
        mPopupKeysTemplate = params.mPopupKeysTemplate;
        mMaxPopupKeysKeyboardColumn = params.mMaxPopupKeysKeyboardColumn;
        mKeyVisualAttributes = params.mKeyVisualAttributes;
        mTopPadding = params.mTopPadding;
        mVerticalGap = params.mVerticalGap;

        mSortedKeys = Collections.unmodifiableList(new ArrayList<>(params.mSortedKeys));
        mShiftKeys = Collections.unmodifiableList(params.mShiftKeys);
        mAltCodeKeysWhileTyping = Collections.unmodifiableList(params.mAltCodeKeysWhileTyping);
        mIconsSet = params.mIconsSet;

        mProximityInfo = new ProximityInfo(params.GRID_WIDTH, params.GRID_HEIGHT,
                mOccupiedWidth, mOccupiedHeight, mMostCommonKeyWidth, mMostCommonKeyHeight,
                mSortedKeys, params.mTouchPositionCorrection);
        mProximityCharsCorrectionEnabled = params.mProximityCharsCorrectionEnabled;
    }

    protected Keyboard(@NonNull final Keyboard keyboard) {
        mId = keyboard.mId;
        mThemeId = keyboard.mThemeId;
        mOccupiedHeight = keyboard.mOccupiedHeight;
        mOccupiedWidth = keyboard.mOccupiedWidth;
        mBaseHeight = keyboard.mBaseHeight;
        mBaseWidth = keyboard.mBaseWidth;
        mMostCommonKeyHeight = keyboard.mMostCommonKeyHeight;
        mMostCommonKeyWidth = keyboard.mMostCommonKeyWidth;
        mPopupKeysTemplate = keyboard.mPopupKeysTemplate;
        mMaxPopupKeysKeyboardColumn = keyboard.mMaxPopupKeysKeyboardColumn;
        mKeyVisualAttributes = keyboard.mKeyVisualAttributes;
        mTopPadding = keyboard.mTopPadding;
        mVerticalGap = keyboard.mVerticalGap;

        mSortedKeys = keyboard.mSortedKeys;
        mShiftKeys = keyboard.mShiftKeys;
        mAltCodeKeysWhileTyping = keyboard.mAltCodeKeysWhileTyping;
        mIconsSet = keyboard.mIconsSet;

        mProximityInfo = keyboard.mProximityInfo;
        mProximityCharsCorrectionEnabled = keyboard.mProximityCharsCorrectionEnabled;
    }

    public boolean hasProximityCharsCorrection(final int code) {
        if (!mProximityCharsCorrectionEnabled) {
            return false;
        }
        // Note: The native code has the main keyboard layout only at this moment.
        // TODO: Figure out how to handle proximity characters information of all layouts.
        final boolean canAssumeNativeHasProximityCharsInfoOfAllKeys = (
                mId.getElement() == KeyboardElement.ALPHABET
                || mId.getElement() == KeyboardElement.ALPHABET_AUTOMATIC_SHIFTED);
        return canAssumeNativeHasProximityCharsInfoOfAllKeys || Character.isLetter(code);
    }

    @NonNull
    public ProximityInfo getProximityInfo() {
        return mProximityInfo;
    }

    /**
     * Return the sorted list of keys of this keyboard.
     * The keys are sorted from top-left to bottom-right order.
     * The list may contain {@link Key.Spacer} object as well.
     * @return the sorted unmodifiable list of {@link Key}s of this keyboard.
     */
    @NonNull
    public List<Key> getSortedKeys() {
        return mSortedKeys;
    }

    @Nullable
    public Key getKey(final int code) {
        if (code == KeyCode.NOT_SPECIFIED) {
            return null;
        }
        synchronized (mKeyCache) {
            final int index = mKeyCache.indexOfKey(code);
            if (index >= 0) {
                return mKeyCache.valueAt(index);
            }

            for (final Key key : getSortedKeys()) {
                if (key.getCode() == code) {
                    mKeyCache.put(code, key);
                    return key;
                }
            }
            mKeyCache.put(code, null);
            return null;
        }
    }

    public boolean hasKey(@NonNull final Key aKey) {
        if (mKeyCache.indexOfValue(aKey) >= 0) {
            return true;
        }

        for (final Key key : getSortedKeys()) {
            if (key == aKey) {
                mKeyCache.put(key.getCode(), key);
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return mId.toString();
    }

    /**
     * Returns the array of the keys that are closest to the given point.
     * @param x the x-coordinate of the point
     * @param y the y-coordinate of the point
     * @return the list of the nearest keys to the given point. If the given
     * point is out of range, then an array of size zero is returned.
     */
    @NonNull
    public List<Key> getNearestKeys(final int x, final int y) {
        // Avoid dead pixels at edges of the keyboard
        final int adjustedX = Math.max(0, Math.min(x, mOccupiedWidth - 1));
        final int adjustedY = Math.max(0, Math.min(y, mOccupiedHeight - 1));
        return mProximityInfo.getNearestKeys(adjustedX, adjustedY);
    }

    @NonNull
    public int[] getCoordinates(@NonNull final int[] codePoints) {
        final int length = codePoints.length;
        final int[] coordinates = CoordinateUtils.newCoordinateArray(length);
        for (int i = 0; i < length; ++i) {
            final Key key = getKey(codePoints[i]);
            if (null != key) {
                CoordinateUtils.setXYInArray(coordinates, i,
                        key.getX() + key.getWidth() / 2, key.getY() + key.getHeight() / 2);
            } else {
                CoordinateUtils.setXYInArray(coordinates, i,
                        Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE);
            }
        }
        return coordinates;
    }
}
