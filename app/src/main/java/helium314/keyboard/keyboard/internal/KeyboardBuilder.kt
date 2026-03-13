/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard.internal

import android.content.Context
import android.content.res.Resources
import android.util.Xml
import androidx.annotation.XmlRes
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.Key.KeyParams
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.internal.keyboard_parser.EmojiParser
import helium314.keyboard.keyboard.internal.keyboard_parser.KeyboardParser
import helium314.keyboard.keyboard.internal.keyboard_parser.addLocaleKeyTextsToParams
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.sumOf
import org.xmlpull.v1.XmlPullParser

// TODO: Write unit tests for this class.
open class KeyboardBuilder<KP : KeyboardParams>(protected val mContext: Context, @JvmField val mParams: KP) {
    @JvmField
    protected val mResources: Resources
    private var mCurrentY = 0

    private var mLeftEdge = false
    private var mTopEdge = false
    private var mRightEdgeKey: Key? = null
    private lateinit var keysInRows: ArrayList<ArrayList<KeyParams>>

    init {
        val res = mContext.resources
        mResources = res
        mParams.GRID_WIDTH = res.getInteger(R.integer.config_keyboard_grid_width)
        mParams.GRID_HEIGHT = res.getInteger(R.integer.config_keyboard_grid_height)
    }

    fun load(id: KeyboardId): KeyboardBuilder<KP> {
        mParams.mId = id
        if (id.isEmojiKeyboard) {
            mParams.mAllowRedundantPopupKeys = true
            readAttributes(R.xml.kbd_emoji)
            keysInRows = EmojiParser(mParams, mContext).parse()
        } else {
            try {
                setupParams()
                keysInRows = KeyboardParser(mParams, mContext).parseLayout()
                if (keysInRows.size != 4) // that was effectively the default for OpenBoard
                    mParams.mTouchPositionCorrection.load(mContext.resources.getStringArray(R.array.touch_position_correction_data_default))
                determineAbsoluteValues()
            } catch (e: Exception) {
                Log.e(TAG, "error parsing layout $id ${id.mElementId}", e)
                throw e
            }
        }
        return this
    }

    private fun setupParams() {
        val sv = Settings.getValues()
        mParams.mAllowRedundantPopupKeys = !sv.mRemoveRedundantPopups
        mParams.mProximityCharsCorrectionEnabled = mParams.mId.mElementId == KeyboardId.ELEMENT_ALPHABET
                || (mParams.mId.isAlphabetKeyboard && !mParams.mId.mSubtype.hasExtraValue(Constants.Subtype.ExtraValue.NO_SHIFT_PROXIMITY_CORRECTION))

        addLocaleKeyTextsToParams(mContext, mParams, sv.mShowMorePopupKeys)
        mParams.mPopupKeyTypes.addAll(sv.mPopupKeyTypes)
        // add label source only if popup key type enabled
        sv.mPopupKeyLabelSources.forEach { if (it in sv.mPopupKeyTypes) mParams.mPopupKeyLabelSources.add(it) }
    }

    // todo: remnant of old parser, replace it if reasonably simple
    protected fun readAttributes(@XmlRes xmlId: Int) {
        val parser = mResources.getXml(xmlId)
        while (parser.eventType != XmlPullParser.END_DOCUMENT) {
            val event = parser.next()
            if (event == XmlPullParser.START_TAG) {
                val tag = parser.name
                if ("Keyboard" == tag) {
                    mParams.readAttributes(mContext, Xml.asAttributeSet(parser))
                    return
                }
            }
        }
        mParams.readAttributes(mContext, null)
    }

    fun disableTouchPositionCorrectionDataForTest() {
        mParams.mTouchPositionCorrection.setEnabled(false)
    }

    fun setProximityCharsCorrectionEnabled(enabled: Boolean) {
        mParams.mProximityCharsCorrectionEnabled = enabled
    }

    open fun build(): Keyboard {
        if (mParams.mId.mIsSplitLayout
                && mParams.mId.mElementId in KeyboardId.ELEMENT_ALPHABET..KeyboardId.ELEMENT_SYMBOLS_SHIFTED) {
            addSplit()
        }
        addKeysToParams()
        return Keyboard(mParams)
    }

    // determine key size and positions using relative width and height
    private fun determineAbsoluteValues() {
        var currentY = mParams.mTopPadding.toFloat()
        for (row in keysInRows) {
            if (row.isEmpty()) continue
            var currentX = mParams.mLeftPadding.toFloat()
            row.forEach {
                it.setAbsoluteDimensions(currentX, currentY)
                currentX += it.mAbsoluteWidth
            }
            currentY += row.first().mAbsoluteHeight
        }
    }

    private fun addSplit() {
        val spacerRelativeWidth = Settings.getValues().mSplitKeyboardSpacerRelativeWidth
        // adjust gaps for the whole keyboard, so it's the same for all rows
        mParams.mRelativeHorizontalGap *= 1f / (1f + spacerRelativeWidth)
        mParams.mHorizontalGap = (mParams.mRelativeHorizontalGap * mParams.mId.mWidth).toInt()
        var maxWidthBeforeSpacer = 0f
        var maxWidthAfterSpacer = 0f
        for (row in keysInRows) {
            val y = row.first().yPos // all have the same y, so this is fine
            val relativeWidthSum = row.sumOf { it.mWidth } // sum up relative widths
            val spacer = KeyParams.newSpacer(mParams, spacerRelativeWidth)
            // insert spacer before first key that starts right of the center (also consider gap)
            var insertIndex = row.indexOfFirst { it.xPos + it.mAbsoluteWidth / 3 > mParams.mOccupiedWidth / 2 }
                .takeIf { it > -1 } ?: (row.size / 2) // fallback should never be needed, but better than having an error
            val indexOfProperSpace = row.indexOfFirst { key ->
                // should work reasonably with customizable layouts, where space key might be completely different:
                // "normal" width space keys are ignored, and the possibility of space being first in row is considered
                key.mCode == Constants.CODE_SPACE && key.mWidth > row.first { !it.isSpacer && it.mCode != Constants.CODE_SPACE }.mWidth * 1.5f
            }
            if (indexOfProperSpace >= 0) {
                val spaceLeft = row[indexOfProperSpace]
                reduceSymbolAndActionKeyWidth(row)
                insertIndex = row.indexOf(spaceLeft) + 1
                val widthBeforeSpace = row.subList(0, insertIndex - 1).sumOf { it.mWidth }
                val widthAfterSpace = row.subList(insertIndex, row.size).sumOf { it.mWidth }
                val spaceLeftWidth = (maxWidthBeforeSpacer - widthBeforeSpace).coerceAtLeast(mParams.mDefaultKeyWidth)
                val spaceRightWidth = (maxWidthAfterSpacer - widthAfterSpace).coerceAtLeast(mParams.mDefaultKeyWidth)
                val spacerWidth = spaceLeft.mWidth + spacerRelativeWidth - spaceLeftWidth - spaceRightWidth
                if (spacerWidth > 0.05f) {
                    // only insert if the spacer has a reasonable width
                    val spaceRight = KeyParams(spaceLeft)
                    spaceLeft.mWidth = spaceLeftWidth
                    spaceRight.mWidth = spaceRightWidth
                    spacer.mWidth = spacerWidth
                    row.add(insertIndex, spaceRight)
                    row.add(insertIndex, spacer)
                } else {
                    // otherwise increase space width, so other keys are resized properly
                    spaceLeft.mWidth += spacerWidth
                }
            } else {
                val widthBeforeSpacer = row.subList(0, insertIndex).sumOf { it.mWidth }
                val widthAfterSpacer = row.subList(insertIndex, row.size).sumOf { it.mWidth }
                maxWidthBeforeSpacer = maxWidthBeforeSpacer.coerceAtLeast(widthBeforeSpacer)
                maxWidthAfterSpacer = maxWidthAfterSpacer.coerceAtLeast(widthAfterSpacer)
                row.add(insertIndex, spacer)
            }
            // re-calculate relative widths
            val relativeWidthSumNew = row.sumOf { it.mWidth }
            val widthFactor = relativeWidthSum / relativeWidthSumNew
            // re-calculate absolute sizes and positions
            var currentX = mParams.mLeftPadding.toFloat()
            row.forEach {
                it.mWidth *= widthFactor
                it.setAbsoluteDimensions(currentX, y)
                currentX += it.mAbsoluteWidth
            }
        }
    }

    // reduce width of symbol and action key if in the row, and add this width to space to keep other key size constant
    // todo: this assumes fixed layout for symbols keys, which will change soon!
    private fun reduceSymbolAndActionKeyWidth(row: ArrayList<KeyParams>) {
        val spaceKey = row.first { it.mCode == Constants.CODE_SPACE }
        val symbolKey = row.firstOrNull { it.mCode == KeyCode.SYMBOL_ALPHA }
        val symbolKeyWidth = symbolKey?.mWidth ?: 0f
        if (symbolKeyWidth > mParams.mDefaultKeyWidth) {
            val widthToChange = symbolKey!!.mWidth - mParams.mDefaultKeyWidth
            symbolKey.mWidth -= widthToChange
            spaceKey.mWidth += widthToChange
        }
        val actionKey = row.firstOrNull { it.mBackgroundType == Key.BACKGROUND_TYPE_ACTION }
        val actionKeyWidth = actionKey?.mWidth ?: 0f
        if (actionKeyWidth > mParams.mDefaultKeyWidth * 1.1f) { // allow it to stay a little wider
            val widthToChange = actionKey!!.mWidth - mParams.mDefaultKeyWidth * 1.1f
            actionKey.mWidth -= widthToChange
            spaceKey.mWidth += widthToChange
        }
    }

    private fun startKeyboard() {
        mCurrentY += mParams.mTopPadding
        mTopEdge = true
    }

    private fun startRow() {
        mLeftEdge = true
        mRightEdgeKey = null
    }

    private fun endRow() {
        val rightEdgeKey = mRightEdgeKey
        if (rightEdgeKey != null) {
            rightEdgeKey.markAsRightEdge(mParams)
            mCurrentY += rightEdgeKey.height + rightEdgeKey.verticalGap
            mRightEdgeKey = null
        }
        mLeftEdge = false
        mTopEdge = false
    }

    private fun endKey(key: Key) {
        mParams.onAddKey(key)
        if (mLeftEdge && !key.isSpacer) {
            key.markAsLeftEdge(mParams)
            mLeftEdge = false
        }
        if (mTopEdge) {
            key.markAsTopEdge(mParams)
        }
        if (!key.isSpacer) {
            mRightEdgeKey = key
        }
    }

    private fun endKeyboard() {
        mParams.removeRedundantPopupKeys()
        // {@link #parseGridRows(XmlPullParser,boolean)} may populate keyboard rows higher than
        // previously expected.
        // todo (low priority): mCurrentY may end up too high with the new parser and 4 row keyboards in landscape mode
        //  -> why is this happening?
        // but anyway, since the height is resized correctly already, we don't need to adjust the
        // occupied height, except for the scrollable emoji keyoards
        if (!mParams.mId.isEmojiKeyboard) return
        val actualHeight = mCurrentY - mParams.mVerticalGap + mParams.mBottomPadding
        mParams.mOccupiedHeight = mParams.mOccupiedHeight.coerceAtLeast(actualHeight)
    }

    private fun addKeysToParams() {
        // need to reset it, we need to sum it up to get the height nicely
        // (though in the end we could just not touch it at all, final used value is the same as the one before resetting)
        mCurrentY = 0
        startKeyboard()
        for (row in keysInRows) {
            startRow()
            for (keyParams in row) {
                endKey(keyParams.createKey())
            }
            endRow()
        }
        endKeyboard()
    }

    companion object {
        private const val TAG = "Keyboard.Builder"
    }
}
