/*
 * Copyright (C) 2015 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.emoji;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Paint;
import android.graphics.Rect;

import helium314.keyboard.latin.settings.Defaults;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.Log;

import androidx.core.graphics.PaintCompat;
import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.Keyboard;
import helium314.keyboard.keyboard.KeyboardId;
import helium314.keyboard.keyboard.KeyboardLayoutSet;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.ResourceUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

final class EmojiCategory {
    private final String TAG = EmojiCategory.class.getSimpleName();

    private static final int ID_UNSPECIFIED = -1;
    public static final int ID_RECENTS = 0;
    private static final int ID_SMILEYS_EMOTION = 1;
    private static final int ID_PEOPLE_BODY = 2;
    private static final int ID_ANIMALS_NATURE = 3;
    private static final int ID_FOOD_DRINK = 4;
    private static final int ID_TRAVEL_PLACES = 5;
    private static final int ID_ACTIVITIES = 6;
    private static final int ID_OBJECTS = 7;
    private static final int ID_SYMBOLS = 8;
    private static final int ID_FLAGS = 9;
    private static final int ID_EMOTICONS = 10;

    private static final int MAX_LINE_COUNT_PER_PAGE = 3;

    public final class CategoryProperties {
        public final int mCategoryId;
        private int mPageCount = -1;
        public CategoryProperties(final int categoryId) {
            mCategoryId = categoryId;
        }
        public int getPageCount() {
            if (mPageCount < 0)
                mPageCount = computeCategoryPageCount(mCategoryId);
            return mPageCount;
        }
    }

    private static final String[] sCategoryName = {
            "recents",
            "smileys & emotion",
            "people & body",
            "animals & nature",
            "food & drink",
            "travel & places",
            "activities",
            "objects",
            "symbols",
            "flags",
            "emoticons" };

    private static final int[] sCategoryTabIconAttr = {
            R.styleable.EmojiPalettesView_iconEmojiRecentsTab,
            R.styleable.EmojiPalettesView_iconEmojiCategory1Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory2Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory3Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory4Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory5Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory6Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory7Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory8Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory9Tab,
            R.styleable.EmojiPalettesView_iconEmojiCategory10Tab };

    private static final int[] sAccessibilityDescriptionResourceIdsForCategories = {
            R.string.spoken_description_emoji_category_recents,
            R.string.spoken_description_emoji_category_eight_smiley,
            R.string.spoken_description_emoji_category_eight_smiley_people,
            R.string.spoken_description_emoji_category_eight_animals_nature,
            R.string.spoken_description_emoji_category_eight_food_drink,
            R.string.spoken_description_emoji_category_eight_travel_places,
            R.string.spoken_description_emoji_category_eight_activity,
            R.string.spoken_description_emoji_category_objects,
            R.string.spoken_description_emoji_category_symbols,
            R.string.spoken_description_emoji_category_flags,
            R.string.spoken_description_emoji_category_emoticons};

    private static final int[] sCategoryElementId = {
            KeyboardId.ELEMENT_EMOJI_RECENTS,
            KeyboardId.ELEMENT_EMOJI_CATEGORY1,
            KeyboardId.ELEMENT_EMOJI_CATEGORY2,
            KeyboardId.ELEMENT_EMOJI_CATEGORY3,
            KeyboardId.ELEMENT_EMOJI_CATEGORY4,
            KeyboardId.ELEMENT_EMOJI_CATEGORY5,
            KeyboardId.ELEMENT_EMOJI_CATEGORY6,
            KeyboardId.ELEMENT_EMOJI_CATEGORY7,
            KeyboardId.ELEMENT_EMOJI_CATEGORY8,
            KeyboardId.ELEMENT_EMOJI_CATEGORY9,
            KeyboardId.ELEMENT_EMOJI_CATEGORY10 };

    private final SharedPreferences mPrefs;
    private final Resources mRes;
    private final Context mContext;
    private final int mMaxRecentsKeyCount;
    private final KeyboardLayoutSet mLayoutSet;
    private final HashMap<String, Integer> mCategoryNameToIdMap = new HashMap<>();
    private final int[] mCategoryTabIconId = new int[sCategoryName.length];
    private final ArrayList<CategoryProperties> mShownCategories = new ArrayList<>();
    private final ConcurrentHashMap<Long, DynamicGridKeyboard> mCategoryKeyboardMap = new ConcurrentHashMap<>();

    private int mCurrentCategoryId = EmojiCategory.ID_UNSPECIFIED;
    private int mCurrentCategoryPageId = 0;

    public EmojiCategory(final Context ctx, final KeyboardLayoutSet layoutSet, final TypedArray emojiPaletteViewAttr) {
        mPrefs = KtxKt.prefs(ctx);
        mRes = ctx.getResources();
        mContext = ctx;
        mMaxRecentsKeyCount = mRes.getInteger(R.integer.config_emoji_keyboard_max_recents_key_count);
        mLayoutSet = layoutSet;
        for (int i = 0; i < sCategoryName.length; ++i) {
            mCategoryNameToIdMap.put(sCategoryName[i], i);
            mCategoryTabIconId[i] = emojiPaletteViewAttr.getResourceId(sCategoryTabIconAttr[i], 0);
        }
    }

    public void initialize() {
        int defaultCategoryId = EmojiCategory.ID_SMILEYS_EMOTION;
        addShownCategoryId(EmojiCategory.ID_RECENTS);
        addShownCategoryId(EmojiCategory.ID_SMILEYS_EMOTION);
        addShownCategoryId(EmojiCategory.ID_PEOPLE_BODY);
        addShownCategoryId(EmojiCategory.ID_ANIMALS_NATURE);
        addShownCategoryId(EmojiCategory.ID_FOOD_DRINK);
        addShownCategoryId(EmojiCategory.ID_TRAVEL_PLACES);
        addShownCategoryId(EmojiCategory.ID_ACTIVITIES);
        addShownCategoryId(EmojiCategory.ID_OBJECTS);
        addShownCategoryId(EmojiCategory.ID_SYMBOLS);
        if (canShowFlagEmoji()) {
            addShownCategoryId(EmojiCategory.ID_FLAGS);
        }
        addShownCategoryId(EmojiCategory.ID_EMOTICONS);

        DynamicGridKeyboard recentsKbd = getKeyboard(EmojiCategory.ID_RECENTS, 0);
        mCurrentCategoryId = mPrefs.getInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_ID, defaultCategoryId);
        mCurrentCategoryPageId = mPrefs.getInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID, Defaults.PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID);
        if (!isShownCategoryId(mCurrentCategoryId)) {
            mCurrentCategoryId = defaultCategoryId;
        } else if (mCurrentCategoryId == EmojiCategory.ID_RECENTS &&
                recentsKbd.getSortedKeys().isEmpty()) {
            mCurrentCategoryId = defaultCategoryId;
        }

        if (mCurrentCategoryPageId >= computeCategoryPageCount(mCurrentCategoryId)) {
            mCurrentCategoryPageId = 0;
        }
    }

    public void clearKeyboardCache() {
        mCategoryKeyboardMap.clear();
        for (CategoryProperties props: mShownCategories)
            props.mPageCount = -1; // reset page count in case size (number of keys per row) changed
    }

    private void addShownCategoryId(final int categoryId) {
        // Load a keyboard of categoryId
        final CategoryProperties properties = new CategoryProperties(categoryId);
        mShownCategories.add(properties);
    }

    private boolean isShownCategoryId(final int categoryId) {
        for (final CategoryProperties prop : mShownCategories) {
            if (prop.mCategoryId == categoryId) {
                return true;
            }
        }
        return false;
    }

    public static String getCategoryName(final int categoryId, final int categoryPageId) {
        return sCategoryName[categoryId] + "-" + categoryPageId;
    }

    public int getCategoryId(final String name) {
        final String[] strings = name.split("-");
        return mCategoryNameToIdMap.get(strings[0]);
    }

    public int getCategoryTabIcon(final int categoryId) {
        return mCategoryTabIconId[categoryId];
    }

    public String getAccessibilityDescription(final int categoryId) {
        return mRes.getString(sAccessibilityDescriptionResourceIdsForCategories[categoryId]);
    }

    public ArrayList<CategoryProperties> getShownCategories() {
        return mShownCategories;
    }

    public int getCurrentCategoryId() {
        return mCurrentCategoryId;
    }

    public int getCurrentCategoryPageCount() {
        return getCategoryPageCount(mCurrentCategoryId);
    }

    public int getCategoryPageCount(final int categoryId) {
        for (final CategoryProperties prop : mShownCategories) {
            if (prop.mCategoryId == categoryId) {
                return prop.getPageCount();
            }
        }
        Log.w(TAG, "Invalid category id: " + categoryId);
        // Should not reach here.
        return 0;
    }

    public void setCurrentCategoryId(final int categoryId) {
        mCurrentCategoryId = categoryId;
        mPrefs.edit().putInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_ID, categoryId).apply();
    }

    public void setCurrentCategoryPageId(final int id) {
        mCurrentCategoryPageId = id;
        mPrefs.edit().putInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID, id).apply();
    }

    public int getCurrentCategoryPageId() {
        return mCurrentCategoryPageId;
    }

    public boolean isInRecentTab() {
        return mCurrentCategoryId == EmojiCategory.ID_RECENTS;
    }

    public int getTabIdFromCategoryId(final int categoryId) {
        for (int i = 0; i < mShownCategories.size(); ++i) {
            if (mShownCategories.get(i).mCategoryId == categoryId) {
                return i;
            }
        }
        Log.w(TAG, "categoryId not found: " + categoryId);
        return 0;
    }

    public int getRecentTabId() {
        return getTabIdFromCategoryId(EmojiCategory.ID_RECENTS);
    }

    private int computeCategoryPageCount(final int categoryId) {
        final Keyboard keyboard = mLayoutSet.getKeyboard(sCategoryElementId[categoryId]);
        return (keyboard.getSortedKeys().size() - 1) / computeMaxKeyCountPerPage() + 1;
    }

    // Returns a keyboard from the recycler view's adapter position.
    public DynamicGridKeyboard getKeyboardFromAdapterPosition(int categoryId, final int position) {
        if (position >= 0 && position < getCategoryPageCount(categoryId)) {
            return getKeyboard(categoryId, position);
        }
        Log.w(TAG, "invalid position for categoryId : " + categoryId);
        return null;
    }

    private static Long getCategoryKeyboardMapKey(final int categoryId, final int id) {
        return (((long) categoryId) << Integer.SIZE) | id;
    }

    public DynamicGridKeyboard getKeyboard(final int categoryId, final int id) {
        synchronized (mCategoryKeyboardMap) {
            final Long categoryKeyboardMapKey = getCategoryKeyboardMapKey(categoryId, id);
            if (mCategoryKeyboardMap.containsKey(categoryKeyboardMapKey)) {
                return mCategoryKeyboardMap.get(categoryKeyboardMapKey);
            }

            final int currentWidth = ResourceUtils.getKeyboardWidth(mContext, Settings.getValues());
            if (categoryId == EmojiCategory.ID_RECENTS) {
                final DynamicGridKeyboard kbd = DynamicGridKeyboard.ofKeyCount(mPrefs,
                        mLayoutSet.getKeyboard(KeyboardId.ELEMENT_EMOJI_RECENTS),
                        mMaxRecentsKeyCount, categoryId, currentWidth);
                mCategoryKeyboardMap.put(categoryKeyboardMapKey, kbd);
                kbd.loadRecentKeys(mCategoryKeyboardMap.values());
                return kbd;
            }

            final Keyboard keyboard = mLayoutSet.getKeyboard(sCategoryElementId[categoryId]);
            final int keyCountPerPage = computeMaxKeyCountPerPage();
            final Key[][] sortedKeysPages = sortKeysGrouped(
                    keyboard.getSortedKeys(), keyCountPerPage);
            for (int pageId = 0; pageId < sortedKeysPages.length; ++pageId) {
                final DynamicGridKeyboard tempKeyboard = DynamicGridKeyboard.ofKeyCount(mPrefs,
                        mLayoutSet.getKeyboard(KeyboardId.ELEMENT_EMOJI_RECENTS),
                        keyCountPerPage, categoryId, currentWidth);
                for (final Key emojiKey : sortedKeysPages[pageId]) {
                    if (emojiKey == null) {
                        break;
                    }
                    tempKeyboard.addKeyLast(emojiKey);
                }
                mCategoryKeyboardMap.put(getCategoryKeyboardMapKey(categoryId, pageId), tempKeyboard);
            }
            return mCategoryKeyboardMap.get(categoryKeyboardMapKey);
        }
    }

    private int computeMaxKeyCountPerPage() {
        final DynamicGridKeyboard tempKeyboard = DynamicGridKeyboard.ofKeyCount(mPrefs,
                mLayoutSet.getKeyboard(KeyboardId.ELEMENT_EMOJI_RECENTS),
                0, 0, ResourceUtils.getKeyboardWidth(mContext, Settings.getValues()));
        return MAX_LINE_COUNT_PER_PAGE * tempKeyboard.getOccupiedColumnCount();
    }

    private static final Comparator<Key> EMOJI_KEY_COMPARATOR = (lhs, rhs) -> {
        final Rect lHitBox = lhs.getHitBox();
        final Rect rHitBox = rhs.getHitBox();
        if (lHitBox.top < rHitBox.top) {
            return -1;
        } else if (lHitBox.top > rHitBox.top) {
            return 1;
        }
        if (lHitBox.left < rHitBox.left) {
            return -1;
        } else if (lHitBox.left > rHitBox.left) {
            return 1;
        }
        if (lhs.getCode() == rhs.getCode()) {
            return 0;
        }
        return lhs.getCode() < rhs.getCode() ? -1 : 1;
    };

    private static Key[][] sortKeysGrouped(final List<Key> inKeys, final int maxPageCount) {
        final ArrayList<Key> keys = new ArrayList<>(inKeys);
        Collections.sort(keys, EMOJI_KEY_COMPARATOR);
        final int pageCount = (keys.size() - 1) / maxPageCount + 1;
        final Key[][] retval = new Key[pageCount][maxPageCount];
        for (int i = 0; i < keys.size(); ++i) {
            retval[i / maxPageCount][i % maxPageCount] = keys.get(i);
        }
        return retval;
    }

    private static boolean canShowFlagEmoji() {
        Paint paint = new Paint();
        String switzerland = "\uD83C\uDDE8\uD83C\uDDED"; //  U+1F1E8 U+1F1ED Flag for Switzerland
        return PaintCompat.hasGlyph(paint, switzerland);
    }

}
