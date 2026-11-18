/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.content.Context;
import android.graphics.Paint;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.internal.KeyboardBuilder;
import helium314.keyboard.keyboard.internal.KeyboardParams;
import helium314.keyboard.keyboard.internal.PopupKeySpec;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.utils.TypefaceUtils;

public final class PopupKeysKeyboard extends Keyboard {
    private final int mDefaultKeyCoordX;

    PopupKeysKeyboard(final PopupKeysKeyboardParams params) {
        super(params);
        mDefaultKeyCoordX = params.getDefaultKeyCoordX() + params.mAbsolutePopupKeyWidth / 2;
    }

    public int getDefaultCoordX() {
        return mDefaultKeyCoordX;
    }

    static class PopupKeysKeyboardParams extends KeyboardParams {
        public boolean mIsPopupKeysFixedOrder;
        /* package */int mTopRowAdjustment;
        public int mNumRows;
        public int mNumColumns;
        public int mTopKeys;
        public int mLeftKeys;
        public int mRightKeys; // includes default key.
        public int mDividerWidth;
        public int mColumnWidth;

        public PopupKeysKeyboardParams() {
            super();
        }

        /**
         * Set keyboard parameters of popup keys keyboard.
         *
         * @param numKeys number of keys in this popup keys keyboard.
         * @param numColumn number of columns of this popup keys keyboard.
         * @param keyWidth popup keys keyboard key width in pixel, including horizontal gap.
         * @param rowHeight popup keys keyboard row height in pixel, including vertical gap.
         * @param coordXInParent coordinate x of the key preview in parent keyboard.
         * @param parentKeyboardWidth parent keyboard width in pixel.
         * @param isPopupKeysFixedColumn true if popup keys keyboard should have
         *   <code>numColumn</code> columns. Otherwise popup keys keyboard should have
         *   <code>numColumn</code> columns at most.
         * @param isPopupKeysFixedOrder true if the order of popup keys is determined by the order in
         *   the popup keys' specification. Otherwise the order of popup keys is automatically
         *   determined.
         * @param dividerWidth width of divider, zero for no dividers.
         */
        public void setParameters(final int numKeys, final int numColumn, final int keyWidth,
                final int rowHeight, final int coordXInParent, final int parentKeyboardWidth,
                final boolean isPopupKeysFixedColumn, final boolean isPopupKeysFixedOrder,
                final int dividerWidth) {
            mIsPopupKeysFixedOrder = isPopupKeysFixedOrder;
            if (parentKeyboardWidth / keyWidth < Math.min(numKeys, numColumn)) {
                throw new IllegalArgumentException("Keyboard is too small to hold popup keys: "
                        + parentKeyboardWidth + " " + keyWidth + " " + numKeys + " " + numColumn);
            }
            mDefaultAbsoluteKeyWidth = keyWidth;
            mDefaultAbsoluteRowHeight = rowHeight;

            mNumRows = (numKeys + numColumn - 1) / numColumn;
            final int numColumns = isPopupKeysFixedColumn ? Math.min(numKeys, numColumn)
                    : getOptimizedColumns(numKeys, numColumn);
            mNumColumns = numColumns;
            final int topKeys = numKeys % numColumns;
            mTopKeys = topKeys == 0 ? numColumns : topKeys;

            final int numLeftKeys = (numColumns - 1) / 2;
            final int numRightKeys = numColumns - numLeftKeys; // including default key.
            // Maximum number of keys we can layout both side of the parent key
            final int maxLeftKeys = coordXInParent / keyWidth;
            final int maxRightKeys = (parentKeyboardWidth - coordXInParent) / keyWidth;
            int leftKeys, rightKeys;
            if (numLeftKeys > maxLeftKeys) {
                leftKeys = maxLeftKeys;
                rightKeys = numColumns - leftKeys;
            } else if (numRightKeys > maxRightKeys + 1) {
                rightKeys = maxRightKeys + 1; // include default key
                leftKeys = numColumns - rightKeys;
            } else {
                leftKeys = numLeftKeys;
                rightKeys = numRightKeys;
            }
            // If the left keys fill the left side of the parent key, entire popup keys keyboard
            // should be shifted to the right unless the parent key is on the left edge.
            if (maxLeftKeys == leftKeys && leftKeys > 0) {
                leftKeys--;
                rightKeys++;
            }
            // If the right keys fill the right side of the parent key, entire popup keys
            // should be shifted to the left unless the parent key is on the right edge.
            if (maxRightKeys == rightKeys - 1 && rightKeys > 1) {
                leftKeys++;
                rightKeys--;
            }
            mLeftKeys = leftKeys;
            mRightKeys = rightKeys;

            // Adjustment of the top row.
            mTopRowAdjustment = isPopupKeysFixedOrder ? getFixedOrderTopRowAdjustment()
                    : getAutoOrderTopRowAdjustment();
            mDividerWidth = dividerWidth;
            mColumnWidth = mDefaultAbsoluteKeyWidth + mDividerWidth;
            mBaseWidth = mOccupiedWidth = mNumColumns * mColumnWidth - mDividerWidth;
            // Need to subtract the bottom row's gutter only.
            mBaseHeight = mOccupiedHeight = mNumRows * mDefaultAbsoluteRowHeight - mVerticalGap
                    + mTopPadding + mBottomPadding;
        }

        private int getFixedOrderTopRowAdjustment() {
            if (mNumRows == 1 || mTopKeys % 2 == 1 || mTopKeys == mNumColumns
                    || mLeftKeys == 0  || mRightKeys == 1) {
                return 0;
            }
            return -1;
        }

        private int getAutoOrderTopRowAdjustment() {
            if (mNumRows == 1 || mTopKeys == 1 || mNumColumns % 2 == mTopKeys % 2
                    || mLeftKeys == 0 || mRightKeys == 1) {
                return 0;
            }
            return -1;
        }

        // Return key position according to column count (0 is default).
        /* package */int getColumnPos(final int n) {
            return mIsPopupKeysFixedOrder ? getFixedOrderColumnPos(n) : getAutomaticColumnPos(n);
        }

        private int getFixedOrderColumnPos(final int n) {
            final int col = n % mNumColumns;
            final int row = n / mNumColumns;
            if (!isTopRow(row)) {
                return col - mLeftKeys;
            }
            final int rightSideKeys = mTopKeys / 2;
            final int leftSideKeys = mTopKeys - (rightSideKeys + 1);
            final int pos = col - leftSideKeys;
            final int numLeftKeys = mLeftKeys + mTopRowAdjustment;
            final int numRightKeys = mRightKeys - 1;
            if (numRightKeys >= rightSideKeys && numLeftKeys >= leftSideKeys) {
                return pos;
            } else if (numRightKeys < rightSideKeys) {
                return pos - (rightSideKeys - numRightKeys);
            } else { // numLeftKeys < leftSideKeys
                return pos + (leftSideKeys - numLeftKeys);
            }
        }

        private int getAutomaticColumnPos(final int n) {
            final int col = n % mNumColumns;
            final int row = n / mNumColumns;
            int leftKeys = mLeftKeys;
            if (isTopRow(row)) {
                leftKeys += mTopRowAdjustment;
            }
            if (col == 0) {
                // default position.
                return 0;
            }

            int pos = 0;
            int right = 1; // include default position key.
            int left = 0;
            int i = 0;
            while (true) {
                // Assign right key if available.
                if (right < mRightKeys) {
                    pos = right;
                    right++;
                    i++;
                }
                if (i >= col)
                    break;
                // Assign left key if available.
                if (left < leftKeys) {
                    left++;
                    pos = -left;
                    i++;
                }
                if (i >= col)
                    break;
            }
            return pos;
        }

        private static int getTopRowEmptySlots(final int numKeys, final int numColumns) {
            final int remainings = numKeys % numColumns;
            return remainings == 0 ? 0 : numColumns - remainings;
        }

        private int getOptimizedColumns(final int numKeys, final int maxColumns) {
            int numColumns = Math.min(numKeys, maxColumns);
            while (getTopRowEmptySlots(numKeys, numColumns) >= mNumRows) {
                numColumns--;
            }
            return numColumns;
        }

        public int getDefaultKeyCoordX() {
            return mLeftKeys * mColumnWidth + mLeftPadding;
        }

        public int getX(final int n, final int row) {
            final int x = getColumnPos(n) * mColumnWidth + getDefaultKeyCoordX();
            if (isTopRow(row)) {
                return x + mTopRowAdjustment * (mColumnWidth / 2);
            }
            return x;
        }

        public int getY(final int row) {
            return (mNumRows - 1 - row) * mDefaultAbsoluteRowHeight + mTopPadding;
        }

        public void markAsEdgeKey(final Key key, final int row) {
            if (row == 0)
                key.markAsTopEdge(this);
            if (isTopRow(row))
                key.markAsBottomEdge(this);
        }

        private boolean isTopRow(final int rowCount) {
            return mNumRows > 1 && rowCount == mNumRows - 1;
        }
    }

    public static class Builder extends KeyboardBuilder<PopupKeysKeyboardParams> {
        private final Key mParentKey;

        private static final float LABEL_PADDING_RATIO = 0.2f;
        private static final float DIVIDER_RATIO = 0.2f;

        /**
         * The builder of PopupKeysKeyboard.
         * @param context the context of {@link PopupKeysKeyboardView}.
         * @param key the {@link Key} that invokes popup keys keyboard.
         * @param keyboard the {@link Keyboard} that contains the parentKey.
         * @param isSinglePopupKeyWithPreview true if the <code>key</code> has just a single
         *        "popup key" and its key popup preview is enabled.
         * @param keyPreviewVisibleWidth the width of visible part of key popup preview.
         * @param keyPreviewVisibleHeight the height of visible part of key popup preview
         * @param paintToMeasure the {@link Paint} object to measure a "popup key" width
         */
        public Builder(final Context context, final Key key, final Keyboard keyboard,
                final boolean isSinglePopupKeyWithPreview, final int keyPreviewVisibleWidth,
                final int keyPreviewVisibleHeight, final Paint paintToMeasure) {
            super(context, new PopupKeysKeyboardParams());
            mParams.mId = keyboard.mId;
            readAttributes(keyboard.mPopupKeysTemplate);

            // TODO: Popup keys keyboard's vertical gap is currently calculated heuristically.
            // Should revise the algorithm.
            mParams.mVerticalGap = keyboard.mVerticalGap / 2;
            // This {@link PopupKeysKeyboard} is invoked from the <code>key</code>.
            mParentKey = key;

            final int keyWidth, rowHeight;
            if (isSinglePopupKeyWithPreview) {
                // Use pre-computed width and height if this popup keys keyboard has only one key to
                // mitigate visual flicker between key preview and popup keys keyboard.
                // Caveats for the visual assets: To achieve this effect, both the key preview
                // backgrounds and the popup keys keyboard panel background have the exact same
                // left/right/top paddings. The bottom paddings of both backgrounds don't need to
                // be considered because the vertical positions of both backgrounds were already
                // adjusted with their bottom paddings deducted.
                keyWidth = keyPreviewVisibleWidth;
                rowHeight = keyPreviewVisibleHeight + mParams.mVerticalGap;
            } else {
                final float padding = context.getResources().getDimension(
                        R.dimen.config_popup_keys_keyboard_key_horizontal_padding)
                        + (key.hasLabelsInPopupKeys()
                                ? mParams.mAbsolutePopupKeyWidth * LABEL_PADDING_RATIO : 0.0f);
                keyWidth = getMaxKeyWidth(key, mParams.mAbsolutePopupKeyWidth, padding, paintToMeasure);
                rowHeight = keyboard.mMostCommonKeyHeight;
            }
            final int dividerWidth;
            if (key.needsDividersInPopupKeys()) {
                dividerWidth = (int)(keyWidth * DIVIDER_RATIO);
            } else {
                dividerWidth = 0;
            }
            final PopupKeySpec[] popupKeys = key.getPopupKeys();
            final int defaultColumns = key.getPopupKeysColumnNumber();
            final int spaceForKeys = keyboard.mId.getWidth() / keyWidth;
            int finalNumColumns = spaceForKeys >= Math.min(popupKeys.length, defaultColumns)
                    ? defaultColumns
                    : (spaceForKeys > 0 ? spaceForKeys : defaultColumns);
            if (finalNumColumns < 1) finalNumColumns = 1; // avoid exception in setParameters
            mParams.setParameters(popupKeys.length, finalNumColumns, keyWidth,
                    rowHeight, key.getX() + key.getWidth() / 2, keyboard.mId.getWidth(),
                    key.isPopupKeysFixedColumn(), key.isPopupKeysFixedOrder(), dividerWidth);
        }

        private static int getMaxKeyWidth(final Key parentKey, final int minKeyWidth,
                final float padding, final Paint paint) {
            int maxWidth = minKeyWidth;
            for (final PopupKeySpec spec : parentKey.getPopupKeys()) {
                final String label = spec.mLabel;
                // If the label is single letter, minKeyWidth is enough to hold the label.
                if (label != null && StringUtils.codePointCount(label) > 1) {
                    maxWidth = Math.max(maxWidth,
                            (int)(TypefaceUtils.getStringWidth(label, paint) + padding));
                }
            }
            return maxWidth;
        }

        @Override
        @NonNull
        public PopupKeysKeyboard build() {
            final PopupKeysKeyboardParams params = mParams;
            final int popupKeyFlags = mParentKey.getPopupKeyLabelFlags();
            final PopupKeySpec[] popupKeys = mParentKey.getPopupKeys();
            final int background = mParentKey.hasActionKeyPopups() ? Key.BACKGROUND_TYPE_ACTION : Key.BACKGROUND_TYPE_NORMAL;
            for (int n = 0; n < popupKeys.length; n++) {
                final PopupKeySpec popupKeySpec = popupKeys[n];
                final int row = n / params.mNumColumns;
                final int x = params.getX(n, row);
                final int y = params.getY(row);
                final Key key = popupKeySpec.buildKey(x, y, popupKeyFlags, background, params);
                params.markAsEdgeKey(key, row);
                params.onAddKey(key);

                final int pos = params.getColumnPos(n);
                // The "pos" value represents the offset from the default position. Negative means
                // left of the default position.
                if (params.mDividerWidth > 0 && pos != 0) {
                    final int dividerX = (pos > 0) ? x - params.mDividerWidth
                            : x + params.mAbsolutePopupKeyWidth;
                    final Key divider = new PopupKeyDivider(
                            params, dividerX, y, params.mDividerWidth, params.mDefaultAbsoluteRowHeight);
                    params.onAddKey(divider);
                }
            }
            return new PopupKeysKeyboard(params);
        }
    }

    // Used as a divider maker. A divider is drawn by {@link PopupKeysKeyboardView}.
    public static class PopupKeyDivider extends Key.Spacer {
        public PopupKeyDivider(final KeyboardParams params, final int x, final int y,
                final int width, final int height) {
            super(params, x, y, width, height);
        }
    }
}
