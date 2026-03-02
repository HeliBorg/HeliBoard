/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.emoji;

import static helium314.keyboard.keyboard.internal.keyboard_parser.EmojiParserKt.EMOJI_HINT_LABEL;

import android.content.SharedPreferences;
import android.text.TextUtils;

import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.settings.Defaults;
import helium314.keyboard.latin.utils.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.Keyboard;
import helium314.keyboard.keyboard.internal.PopupKeySpec;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.JsonUtils;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * This is a Keyboard class where you can add keys dynamically shown in a grid layout
 */
final class DynamicGridKeyboard extends Keyboard {
    private static final String TAG = DynamicGridKeyboard.class.getSimpleName();
    private final Object mLock = new Object();

    private final SharedPreferences mPrefs;
    private final int mHorizontalStep;
    private final int mHorizontalGap;
    private final int mVerticalStep;
    private final int mColumnsNum;
    private final int mMaxKeyCount;
    private final boolean mFixedRowCount;
    private final boolean mIsRecents;
    private final ArrayDeque<GridKey> mGridKeys = new ArrayDeque<>();
    private final ArrayDeque<Key> mPendingKeys = new ArrayDeque<>();

    private List<Key> mCachedGridKeys;
    private final ArrayList<Integer> mEmptyColumnIndices = new ArrayList<>(4);

    public static DynamicGridKeyboard ofKeyCount(final SharedPreferences prefs, final Keyboard templateKeyboard,
            final int maxKeyCount, final int categoryId, final int width) {
        return new DynamicGridKeyboard(prefs, templateKeyboard, maxKeyCount, categoryId, width, false);
    }

    public static DynamicGridKeyboard ofRowCount(final SharedPreferences prefs, final Keyboard templateKeyboard,
            final int maxRowCount, final int categoryId, final int width) {
        return new DynamicGridKeyboard(prefs, templateKeyboard, maxRowCount, categoryId, width, true);
    }

    private DynamicGridKeyboard(final SharedPreferences prefs, final Keyboard templateKeyboard,
            final int maxCount, final int categoryId, final int width, boolean fixedRowCount) {
        super(templateKeyboard);
        // todo: would be better to keep them final and not require width, but how to properly set width of the template keyboard?
        //  an alternative would be to always create the templateKeyboard with full width
        final int paddingWidth = mOccupiedWidth - mBaseWidth;
        mBaseWidth = width - paddingWidth;
        mOccupiedWidth = width;
        final float spacerWidth = Settings.getValues().mSplitKeyboardSpacerRelativeWidth * mBaseWidth;
        final Key key0 = getTemplateKey(Constants.RECENTS_TEMPLATE_KEY_CODE_0);
        final Key key1 = getTemplateKey(Constants.RECENTS_TEMPLATE_KEY_CODE_1);
        final int horizontalGap = Math.abs(key1.getX() - key0.getX()) - key0.getWidth();
        final float widthScale = determineWidthScale(key0.getWidth() + horizontalGap);
        mHorizontalGap = (int) (horizontalGap * widthScale);
        mHorizontalStep = (int) ((key0.getWidth() + horizontalGap) * widthScale);
        mVerticalStep = (int) ((key0.getHeight() + mVerticalGap) / Math.sqrt(Settings.getValues().mKeyboardHeightScale));
        mColumnsNum = mBaseWidth / mHorizontalStep;
        if (spacerWidth > 0)
            setSpacerColumns(spacerWidth);
        mMaxKeyCount = fixedRowCount? maxCount * getOccupiedColumnCount() : maxCount;
        mFixedRowCount = fixedRowCount;
        mIsRecents = categoryId == EmojiCategory.ID_RECENTS;
        mPrefs = prefs;
    }

    private void setSpacerColumns(final float spacerWidth) {
        int spacerColumnsWidth = (int) (spacerWidth / mHorizontalStep);
        if (spacerColumnsWidth == 0) return;
        if (mColumnsNum % 2 != spacerColumnsWidth % 2)
            spacerColumnsWidth++;
        final int leftmost;
        final int rightmost;
        if (spacerColumnsWidth % 2 == 0) {
            int center = mColumnsNum / 2;
            leftmost = center - (spacerColumnsWidth / 2 - 1);
            rightmost = center + spacerColumnsWidth / 2;
        } else {
            int center = mColumnsNum / 2 + 1;
            leftmost = center - spacerColumnsWidth / 2;
            rightmost = center + spacerColumnsWidth / 2;
        }
        for (int i = leftmost; i <= rightmost; i++) {
            mEmptyColumnIndices.add(i - 1);
        }
    }

    // determine a width scale so emojis evenly fill the entire width
    private float determineWidthScale(final float horizontalStep) {
        final float columnsNumRaw = mBaseWidth / horizontalStep;
        final float columnsNum = Math.round(columnsNumRaw);
        return columnsNumRaw / columnsNum;
    }

    private Key getTemplateKey(final int code) {
        for (final Key key : super.getSortedKeys()) {
            if (key.getCode() == code) {
                return key;
            }
        }
        throw new RuntimeException("Can't find template key: code=" + code);
    }

    // height is dynamic if we don't have a fixed row count
    int getOccupiedHeight() {
        final int count = mFixedRowCount ? mMaxKeyCount : mGridKeys.size();
        final int row = (count - 1) / getOccupiedColumnCount() + 1;
        return row * mVerticalStep;
    }

    public int getOccupiedColumnCount() {
        return mColumnsNum - mEmptyColumnIndices.size();
    }

    public void addPendingKey(final Key usedKey) {
        synchronized (mLock) {
            mPendingKeys.addLast(usedKey);
        }
    }

    public void flushPendingRecentKeys() {
        synchronized (mLock) {
            while (!mPendingKeys.isEmpty()) {
                addKey(mPendingKeys.pollFirst(), true);
            }
            saveRecentKeys();
        }
    }

    public void addKeyFirst(final Key usedKey) {
        addKey(usedKey, true);
        if (mIsRecents) {
            saveRecentKeys();
        }
    }

    public void addKeyLast(final Key usedKey) {
        addKey(usedKey, false);
    }

    public void removeAllKeys() {
        synchronized (mLock) {
            mGridKeys.clear();
            mCachedGridKeys = null;
        }
    }

    private void addKey(final Key usedKey, final boolean addFirst) {
        if (usedKey == null) {
            return;
        }
        synchronized (mLock) {
            mCachedGridKeys = null;
            // When a key is added to recents keyboard, we don't want to keep its popup keys
            // neither its hint label. Also, we make sure its background type is matching our keyboard
            // if key comes from another keyboard (ie. a {@link PopupKeysKeyboard}).
            final boolean dropPopupKeys = mIsRecents;
            // Check if hint was a more emoji indicator and prevent its copy if popup keys aren't copied
            final boolean dropHintLabel = dropPopupKeys && EMOJI_HINT_LABEL.equals(usedKey.getHintLabel());
            final GridKey key = new GridKey(usedKey,
                    dropPopupKeys ? null : usedKey.getPopupKeys(),
                    dropHintLabel ? null : usedKey.getHintLabel(),
                    mIsRecents ? Key.BACKGROUND_TYPE_EMPTY : usedKey.getBackgroundType());
            while (mGridKeys.remove(key)) {
                // Remove duplicate keys.
            }
            if (addFirst) {
                mGridKeys.addFirst(key);
            } else {
                mGridKeys.addLast(key);
            }
            while (mGridKeys.size() > mMaxKeyCount) {
                mGridKeys.removeLast();
            }
            int index = 0;
            for (final GridKey gridKey : mGridKeys) {
                while (mEmptyColumnIndices.contains(index % mColumnsNum)) {
                    index++;
                }
                final int keyX0 = getKeyX0(index);
                final int keyY0 = getKeyY0(index);
                final int keyX1 = getKeyX1(index);
                final int keyY1 = getKeyY1(index);
                gridKey.updateCoordinates(keyX0, keyY0, keyX1, keyY1);
                index++;
            }
        }
    }

    private void saveRecentKeys() {
        final ArrayList<Object> keys = new ArrayList<>();
        for (final Key key : mGridKeys) {
            if (key.getOutputText() != null) {
                keys.add(key.getOutputText());
            } else {
                keys.add(key.getCode());
            }
        }
        final String jsonStr = JsonUtils.listToJsonStr(keys);
        mPrefs.edit().putString(Settings.PREF_EMOJI_RECENT_KEYS, jsonStr).apply();
    }

    private Key getKeyByCode(final Collection<DynamicGridKeyboard> keyboards,
            final int code) {
        for (final DynamicGridKeyboard keyboard : keyboards) {
            for (final Key key : keyboard.getSortedKeys()) {
                if (key.getCode() == code) {
                    return key;
                }
            }
        }

        // fall back to creating the key
        return new Key(getTemplateKey(Constants.RECENTS_TEMPLATE_KEY_CODE_0), null, null, Key.BACKGROUND_TYPE_EMPTY, code, null);
    }

    private Key getKeyByOutputText(final Collection<DynamicGridKeyboard> keyboards,
            final String outputText) {
        for (final DynamicGridKeyboard keyboard : keyboards) {
            for (final Key key : keyboard.getSortedKeys()) {
                if (outputText.equals(key.getOutputText())) {
                    return key;
                }
            }
        }

        // fall back to creating the key
        return new Key(getTemplateKey(Constants.RECENTS_TEMPLATE_KEY_CODE_0), null, null, Key.BACKGROUND_TYPE_EMPTY, 0, outputText);
    }

    public void loadRecentKeys(final Collection<DynamicGridKeyboard> keyboards) {
        final String str = mPrefs.getString(Settings.PREF_EMOJI_RECENT_KEYS, Defaults.PREF_EMOJI_RECENT_KEYS);
        final List<Object> keys = JsonUtils.jsonStrToList(str);
        for (final Object o : keys) {
            final Key key;
            if (o instanceof Integer) {
                final int code = (Integer)o;
                key = getKeyByCode(keyboards, code);
            } else if (o instanceof final String outputText) {
                key = getKeyByOutputText(keyboards, outputText);
            } else {
                Log.w(TAG, "Invalid object: " + o);
                continue;
            }
            addKeyLast(key);
        }
    }

    private int getKeyX0(final int index) {
        final int column = index % mColumnsNum;
        return column * mHorizontalStep + mHorizontalGap / 2;
    }

    private int getKeyX1(final int index) {
        final int column = index % mColumnsNum + 1;
        return column * mHorizontalStep + mHorizontalGap / 2;
    }

    private int getKeyY0(final int index) {
        final int row = index / mColumnsNum;
        return row * mVerticalStep + mVerticalGap / 2;
    }

    private int getKeyY1(final int index) {
        final int row = index / mColumnsNum + 1;
        return row * mVerticalStep + mVerticalGap / 2;
    }

    @NonNull
    @Override
    public List<Key> getSortedKeys() {
        synchronized (mLock) {
            if (mCachedGridKeys != null) {
                return mCachedGridKeys;
            }
            final ArrayList<Key> cachedKeys = new ArrayList<>(mGridKeys);
            mCachedGridKeys = Collections.unmodifiableList(cachedKeys);
            return mCachedGridKeys;
        }
    }

    @NonNull
    @Override
    public List<Key> getNearestKeys(final int x, final int y) {
        // TODO: Calculate the nearest key index in mGridKeys from x and y.
        return getSortedKeys();
    }

    static final class GridKey extends Key {
        private int mCurrentX;
        private int mCurrentY;

        public GridKey(@NonNull final Key originalKey, @Nullable final PopupKeySpec[] popupKeys,
             @Nullable final String labelHint, final int backgroundType) {
            super(originalKey, popupKeys, labelHint, backgroundType);
        }

        public void updateCoordinates(final int x0, final int y0, final int x1, final int y1) {
            mCurrentX = x0;
            mCurrentY = y0;
            getHitBox().set(x0, y0, x1, y1);
        }

        @Override
        public int getX() {
            return mCurrentX;
        }

        @Override
        public int getY() {
            return mCurrentY;
        }

        @Override
        public boolean equals(final Object o) {
            if (!(o instanceof final Key key)) return false;
            if (getCode() != key.getCode()) return false;
            if (!TextUtils.equals(getLabel(), key.getLabel())) return false;
            return TextUtils.equals(getOutputText(), key.getOutputText());
        }

        @NonNull
        @Override
        public String toString() {
            return "GridKey: " + super.toString();
        }
    }
}
