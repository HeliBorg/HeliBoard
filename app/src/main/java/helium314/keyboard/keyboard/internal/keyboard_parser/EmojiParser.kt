// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.internal.keyboard_parser

import android.content.Context
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.Key.KeyParams
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.emoji.SupportedEmojis
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.common.splitOnWhitespace
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ResourceUtils
import helium314.keyboard.latin.utils.prefs
import java.util.Collections
import kotlin.let
import kotlin.math.sqrt

class EmojiParser(private val params: KeyboardParams, private val context: Context) {

    fun parse(): ArrayList<ArrayList<KeyParams>> {
        val emojiFileName = getEmojiFileName(params.mId.mElementId)
        val emojiLines = if (emojiFileName == null) {
            listOf( // special template keys for recents category
                StringUtils.newSingleCodePointString(Constants.RECENTS_TEMPLATE_KEY_CODE_0),
                StringUtils.newSingleCodePointString(Constants.RECENTS_TEMPLATE_KEY_CODE_1),
            )
        } else {
            loadEmojiFile(emojiFileName, context)
        }
        if (params.mId.mElementId == KeyboardId.ELEMENT_EMOJI_CATEGORY2) {
            loadEmojiDefaultVersionsAndPopupSpecs(context, emojiLines)
            return parseEmojis(emojiLines.map { line -> getEmojiDefaultVersion(line.splitOnWhitespace().first()) })
        }
        return parseEmojis(emojiLines)
    }

    private fun parseEmojis(emojis: List<String>): ArrayList<ArrayList<KeyParams>> {
        val row = ArrayList<KeyParams>(emojis.size)
        var currentX = params.mLeftPadding.toFloat()
        val currentY = params.mTopPadding.toFloat() // no need to ever change, assignment to rows into rows is done in DynamicGridKeyboard

        val (keyWidth, keyHeight) = getEmojiKeyDimensions(params, context)

        emojis.forEach { emoji ->
            val keyParams = parseEmojiKeyNew(emoji) ?: return@forEach
            keyParams.xPos = currentX
            keyParams.yPos = currentY
            keyParams.mAbsoluteWidth = keyWidth
            keyParams.mAbsoluteHeight = keyHeight
            currentX += keyParams.mAbsoluteWidth
            row.add(keyParams)
        }
        return arrayListOf(row)
    }

    private fun parseEmojiKeyNew(emoji: String): KeyParams? {
        if (SupportedEmojis.isUnsupported(emoji)) return null
        return KeyParams(
            emoji,
            emoji.getCode(),
            if (emojiPopupSpecs[emoji] != null) EMOJI_HINT_LABEL else null,
            emojiPopupSpecs[emoji],
            Key.LABEL_FLAGS_FONT_NORMAL,
            params
        )
    }
}

fun getEmojiKeyDimensions(params: KeyboardParams, context: Context): Pair<Float, Float> {
    // determine key width for default settings (no number row, no one-handed mode, 100% height and bottom padding scale)
    // this is a bit long, but ensures that emoji size stays the same, independent of these settings
    // we also ignore side padding for key width, and prefer fewer keys per row over narrower keys
    val defaultKeyWidth = ResourceUtils.getDefaultKeyboardWidth(context) * params.mDefaultKeyWidth
    var keyWidth = defaultKeyWidth * sqrt(Settings.getValues().mKeyboardHeightScale)
    val defaultKeyboardHeight = ResourceUtils.getDefaultKeyboardHeight(context.resources, false)
    val defaultBottomPadding = context.resources.getFraction(
        R.fraction.config_keyboard_bottom_padding_holo, defaultKeyboardHeight, defaultKeyboardHeight
    )
    val emojiKeyboardHeight = defaultKeyboardHeight * 0.75f + params.mVerticalGap - defaultBottomPadding -
        context.resources.getDimensionPixelSize(R.dimen.config_emoji_category_page_id_height)
    var keyHeight =
        emojiKeyboardHeight * params.mDefaultRowHeight * Settings.getValues().mKeyboardHeightScale // still apply height scale to key

    if (Settings.getValues().mEmojiKeyFit) {
        keyWidth *= Settings.getValues().mFontSizeMultiplierEmoji
        keyHeight *= Settings.getValues().mFontSizeMultiplierEmoji
    }
    return keyWidth to keyHeight
}

fun String.getCode(): Int =
    if (StringUtils.codePointCount(this) != 1) KeyCode.MULTIPLE_CODE_POINTS
    else Character.codePointAt(this, 0)

fun loadEmojiDefaultVersionsAndPopupSpecs(context: Context) {
    loadEmojiDefaultVersionsAndPopupSpecs(context, null)
}

private fun loadEmojiDefaultVersionsAndPopupSpecs(context: Context, category2EmojiLines: List<String>?) {
    val defaultTone = context.prefs().getString(Settings.PREF_EMOJI_SKIN_TONE, Defaults.PREF_EMOJI_SKIN_TONE)
    if (defaultSkinTone == defaultTone) {
        return
    }

    defaultSkinTone = defaultTone
    emojiDefaultVersions.clear()
    emojiNeutralVersions.clear()
    emojiPopupSpecs.clear()
    (category2EmojiLines ?: loadEmojiFile(getEmojiFileName(KeyboardId.ELEMENT_EMOJI_CATEGORY2)!!, context)).forEach { line ->
        var split = line.splitOnWhitespace()
        if (defaultSkinTone != "") {
            // adjust PEOPLE_AND_BODY if we have a non-yellow default skin tone
            // find the line containing the skin tone, and swap with first
            val foundIndex = split.indexOfFirst { it.contains(defaultSkinTone!!) }
            if (foundIndex > 0) {
                emojiDefaultVersions[split[0]] = split[foundIndex]
                emojiNeutralVersions[split[foundIndex]] = split[0]
                split = split.toMutableList()
                Collections.swap(split, 0, foundIndex)
            }
        }
        split.drop(1).filterNot { SupportedEmojis.isUnsupported(it) }
            .takeIf { it.isNotEmpty() }?.joinToString(",")?.let { emojiPopupSpecs[split.first()] = it }
    }
}

private fun getEmojiFileName(id: Int): String? {
    return when (id) {
        KeyboardId.ELEMENT_EMOJI_CATEGORY1 -> "SMILEYS_AND_EMOTION.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY2 -> "PEOPLE_AND_BODY.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY3 -> "ANIMALS_AND_NATURE.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY4 -> "FOOD_AND_DRINK.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY5 -> "TRAVEL_AND_PLACES.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY6 -> "ACTIVITIES.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY7 -> "OBJECTS.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY8 -> "SYMBOLS.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY9 -> "FLAGS.txt"
        KeyboardId.ELEMENT_EMOJI_CATEGORY10 -> "EMOTICONS.txt"
        else -> null
    }
}

private fun loadEmojiFile(emojiFileName: String, context: Context): List<String> =
    context.assets.open("emoji/$emojiFileName").reader().use { it.readLines() }

const val EMOJI_HINT_LABEL = "◥"

private var defaultSkinTone: String? = null
private val emojiDefaultVersions: MutableMap<String, String> = mutableMapOf()
private val emojiNeutralVersions: MutableMap<String, String> = mutableMapOf()
private val emojiPopupSpecs: MutableMap<String, String> = mutableMapOf()

fun getEmojiDefaultVersion(emoji: String): String = emojiDefaultVersions[emoji] ?: emoji
fun getEmojiNeutralVersion(emoji: String): String = emojiNeutralVersions[emoji] ?: emoji
fun getEmojiPopupSpec(emoji: String): String? = emojiPopupSpecs[emoji]
