package helium314.keyboard.keyboard

import android.content.Context
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.PopupKeySpec
import helium314.keyboard.keyboard.internal.keyboard_parser.LayoutParser
import helium314.keyboard.keyboard.internal.keyboard_parser.addLocaleKeyTextsToParams
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.LayoutType

object ShortcutRowKeys {
    @JvmStatic
    fun createPopupParentKey(
        context: Context,
        sourceKey: Key,
        parentKeyboard: Keyboard,
        layoutType: LayoutType,
    ): Key? {
        val params = KeyboardParams()
        params.mId = parentKeyboard.mId
        params.mThemeId = parentKeyboard.mThemeId
        params.mOccupiedWidth = parentKeyboard.mOccupiedWidth
        params.mOccupiedHeight = parentKeyboard.mOccupiedHeight
        params.mBaseWidth = parentKeyboard.mBaseWidth
        params.mBaseHeight = parentKeyboard.mBaseHeight
        params.mTopPadding = parentKeyboard.mTopPadding
        params.mVerticalGap = parentKeyboard.mVerticalGap / 2
        params.mDefaultKeyWidth = parentKeyboard.mMostCommonKeyWidth.toFloat() / parentKeyboard.mBaseWidth
        params.mDefaultRowHeight = parentKeyboard.mMostCommonKeyHeight.toFloat() / parentKeyboard.mBaseHeight
        params.mDefaultAbsoluteKeyWidth = parentKeyboard.mMostCommonKeyWidth
        params.mAbsolutePopupKeyWidth = parentKeyboard.mMostCommonKeyWidth
        params.mDefaultAbsoluteRowHeight = parentKeyboard.mMostCommonKeyHeight
        params.mMaxPopupKeysKeyboardColumn = 12
        addLocaleKeyTextsToParams(context, params, Settings.getValues().mShowMorePopupKeys)

        val row = LayoutParser.parseLayout(layoutType, params, context)
            .firstOrNull { it.isNotEmpty() }
            ?: return null
        val specs = row.mapNotNull { keyData ->
            keyData.compute(params)?.getPopupLabel(params)
        }.map { popupLabel ->
            PopupKeySpec(popupLabel, false, params.mId.locale)
        }.toTypedArray()
        if (specs.isEmpty()) return null
        return Key.copyWithShortcutPopupKeys(sourceKey, specs)
    }
}
