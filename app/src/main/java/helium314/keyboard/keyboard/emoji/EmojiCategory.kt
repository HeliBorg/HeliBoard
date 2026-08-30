/*
 * Copyright (C) 2015 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard.emoji

import android.content.Context
import android.content.res.TypedArray
import android.graphics.Paint
import androidx.core.graphics.PaintCompat
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.KeyboardElement
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.latin.R
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.ResourceUtils
import helium314.keyboard.latin.utils.prefs
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap
import androidx.core.content.edit

internal class EmojiCategory(private val context: Context, private val layoutSet: KeyboardLayoutSet, emojiPaletteViewAttr: TypedArray) {
    inner class CategoryProperties(val category: Category) {
        var mPageCount = -1

        val pageCount: Int
            get() {
                if (mPageCount < 0) mPageCount = computeCategoryPageCount(category)
                return mPageCount
            }
    }

    private val prefs = context.prefs()
    private val maxRecentsKeyCount = context.resources.getInteger(R.integer.config_emoji_keyboard_max_recents_key_count)

    private val categoryTabIconId = IntArray(Category.entries.size) { i ->
        emojiPaletteViewAttr.getResourceId(Category.entries[i].iconAttr, 0)
    }

    val shownCategories = listOfNotNull(
        CategoryProperties(Category.RECENTS),
        CategoryProperties(Category.SMILEYS),
        CategoryProperties(Category.PEOPLE),
        CategoryProperties(Category.NATURE),
        CategoryProperties(Category.FOOD),
        CategoryProperties(Category.TRAVEL_PLACES),
        CategoryProperties(Category.ACTIVITIES),
        CategoryProperties(Category.OBJECTS),
        CategoryProperties(Category.SYMBOLS),
        if (canShowFlagEmoji()) CategoryProperties(Category.FLAGS) else null,
        CategoryProperties(Category.EMOTICONS)
    )

    private val categoryKeyboardMap = ConcurrentHashMap<Long, DynamicGridKeyboard>()

    var currentCategory: Category = defaultCategory
        set(value) {
            field = value
            prefs.edit { putInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_ID, value.ordinal) }
        }

    var currentCategoryPageId = 0
        set(value) {
            field = value
            prefs.edit { putInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID, value) }
        }

    fun initialize() {
        val recentsKbd = getKeyboard(Category.RECENTS, 0)
        currentCategory = Category.entries[prefs.getInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_ID, defaultCategory.ordinal)]
        currentCategoryPageId =
            prefs.getInt(Settings.PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID, Defaults.PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID)
        if (!isShownCategory(currentCategory)) {
            currentCategory = defaultCategory
        } else if (currentCategory == Category.RECENTS && recentsKbd.sortedKeys.isEmpty()) {
            currentCategory = defaultCategory
        }

        if (currentCategoryPageId >= computeCategoryPageCount(currentCategory)) {
            currentCategoryPageId = 0
        }
    }

    fun clearKeyboardCache() {
        categoryKeyboardMap.clear()
        for (props in shownCategories) props.mPageCount = -1 // reset page count in case size (number of keys per row) changed
    }

    private fun isShownCategory(category: Category): Boolean {
        for (prop in shownCategories) {
            if (prop.category == category) {
                return true
            }
        }
        return false
    }

    fun getCategoryTabIcon(category: Category) = categoryTabIconId[category.ordinal]

    fun getAccessibilityDescription(category: Category) = context.getString(category.element.descriptionResId)

    val currentCategoryPageCount get() = getCategoryPageCount(currentCategory)

    fun getCategoryPageCount(category: Category): Int {
        for (prop in shownCategories) {
            if (prop.category == category) {
                return prop.pageCount
            }
        }
        Log.w(TAG, "Invalid category: $category")
        // Should not reach here.
        return 0
    }

    val isInRecentTab get() = currentCategory == Category.RECENTS

    fun getTabIdFromCategoryId(category: Category): Int {
        val i = shownCategories.indexOfFirst { it.category == category }
        return if (i == -1) 0 else i
    }

    val recentTabId get() = getTabIdFromCategoryId(Category.RECENTS)

    private fun computeCategoryPageCount(category: Category): Int {
        val keyboard = layoutSet.getKeyboard(category.element)
        return (keyboard.sortedKeys.size - 1) / computeMaxKeyCountPerPage() + 1
    }

    // Returns a keyboard from the recycler view's adapter position.
    fun getKeyboardFromAdapterPosition(category: Category, position: Int): DynamicGridKeyboard? {
        if (position >= 0 && position < getCategoryPageCount(category)) {
            return getKeyboard(category, position)
        }
        Log.w(TAG, "invalid position for category : $category")
        return null
    }

    fun getKeyboard(category: Category, id: Int): DynamicGridKeyboard {
        synchronized(categoryKeyboardMap) {
            val categoryKeyboardMapKey = getCategoryKeyboardMapKey(category, id)
            categoryKeyboardMap[categoryKeyboardMapKey]?.let { return it }

            val currentWidth = ResourceUtils.getKeyboardWidth(context, Settings.getValues())
            if (category == Category.RECENTS) {
                val kbd = DynamicGridKeyboard.ofKeyCount(
                    prefs,
                    layoutSet.getKeyboard(KeyboardElement.EMOJI_RECENTS),
                    maxRecentsKeyCount, category == Category.RECENTS, currentWidth
                )
                categoryKeyboardMap[categoryKeyboardMapKey] = kbd
                kbd.loadRecentKeys(getKeyboards())
                return kbd
            }

            val keyboard = layoutSet.getKeyboard(category.element)
            val keyCountPerPage = computeMaxKeyCountPerPage()
            val sortedKeysPages = sortKeysGrouped(keyboard.sortedKeys, keyCountPerPage)
            for (pageId in sortedKeysPages.indices) {
                val tempKeyboard = DynamicGridKeyboard.ofKeyCount(prefs,
                    layoutSet.getKeyboard(KeyboardElement.EMOJI_RECENTS),
                    keyCountPerPage, category == Category.RECENTS, currentWidth)
                for (emojiKey in sortedKeysPages[pageId]) {
                    if (emojiKey == null) {
                        break
                    }
                    tempKeyboard.addKeyLast(emojiKey)
                }
                categoryKeyboardMap[getCategoryKeyboardMapKey(category, pageId)] = tempKeyboard
            }
            return categoryKeyboardMap[categoryKeyboardMapKey]!!
        }
    }

    fun getKeyboards(): Collection<DynamicGridKeyboard> = categoryKeyboardMap.values

    private fun computeMaxKeyCountPerPage(): Int {
        val tempKeyboard = DynamicGridKeyboard.ofKeyCount(prefs,
            layoutSet.getKeyboard(KeyboardElement.EMOJI_RECENTS),
            0, true, ResourceUtils.getKeyboardWidth(context, Settings.getValues())
        )
        return MAX_LINE_COUNT_PER_PAGE * tempKeyboard.occupiedColumnCount
    }

    enum class Category(val element: KeyboardElement, val iconAttr: Int) {
        RECENTS(KeyboardElement.EMOJI_RECENTS, R.styleable.EmojiPalettesView_iconEmojiRecentsTab),
        SMILEYS(KeyboardElement.EMOJI_SMILEYS, R.styleable.EmojiPalettesView_iconEmojiSmileysTab),
        PEOPLE(KeyboardElement.EMOJI_PEOPLE, R.styleable.EmojiPalettesView_iconEmojiPeopleTab),
        NATURE(KeyboardElement.EMOJI_NATURE, R.styleable.EmojiPalettesView_iconEmojiNatureTab),
        FOOD(KeyboardElement.EMOJI_FOOD, R.styleable.EmojiPalettesView_iconEmojiFoodTab),
        TRAVEL_PLACES(KeyboardElement.EMOJI_TRAVEL_PLACES, R.styleable.EmojiPalettesView_iconEmojiTravelPlacesTab),
        ACTIVITIES(KeyboardElement.EMOJI_ACTIVITIES, R.styleable.EmojiPalettesView_iconEmojiActivitiesTab),
        OBJECTS(KeyboardElement.EMOJI_OBJECTS, R.styleable.EmojiPalettesView_iconEmojiObjectsTab),
        SYMBOLS(KeyboardElement.EMOJI_SYMBOLS, R.styleable.EmojiPalettesView_iconEmojiSymbolsTab),
        FLAGS(KeyboardElement.EMOJI_FLAGS, R.styleable.EmojiPalettesView_iconEmojiFlagsTab),
        EMOTICONS(KeyboardElement.EMOJI_EMOTICONS, R.styleable.EmojiPalettesView_iconEmojiEmoticonsTab)
    }

    companion object {
        private val TAG = EmojiCategory::class.java.simpleName
        private val defaultCategory = Category.SMILEYS

        private const val MAX_LINE_COUNT_PER_PAGE = 3

        private fun getCategoryKeyboardMapKey(category: Category, id: Int) =
            ((category.ordinal.toLong()) shl Integer.SIZE) or id.toLong()

        private val EMOJI_KEY_COMPARATOR = Comparator { lhs: Key, rhs: Key ->
            val lHitBox = lhs.hitBox
            val rHitBox = rhs.hitBox
            if (lHitBox.top < rHitBox.top) {
                return@Comparator -1
            } else if (lHitBox.top > rHitBox.top) {
                return@Comparator 1
            }
            if (lHitBox.left < rHitBox.left) {
                return@Comparator -1
            } else if (lHitBox.left > rHitBox.left) {
                return@Comparator 1
            }
            if (lhs.code == rhs.code) {
                return@Comparator 0
            }
            if (lhs.code < rhs.code) -1 else 1
        }

        private fun sortKeysGrouped(inKeys: MutableList<Key>, maxPageCount: Int): Array<Array<Key?>> {
            val keys = ArrayList(inKeys)
            Collections.sort(keys, EMOJI_KEY_COMPARATOR)
            val pageCount = (keys.size - 1) / maxPageCount + 1
            val retval = Array<Array<Key?>>(pageCount) { arrayOfNulls(maxPageCount) }
            for (i in keys.indices) {
                retval[i / maxPageCount][i % maxPageCount] = keys[i]
            }
            return retval
        }

        private fun canShowFlagEmoji(): Boolean {
            val paint = Paint()
            val switzerland = "\uD83C\uDDE8\uD83C\uDDED" //  U+1F1E8 U+1F1ED Flag for Switzerland
            return PaintCompat.hasGlyph(paint, switzerland)
        }
    }
}
