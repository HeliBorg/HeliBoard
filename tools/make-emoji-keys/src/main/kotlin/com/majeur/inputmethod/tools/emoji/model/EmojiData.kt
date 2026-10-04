// SPDX-License-Identifier: GPL-3.0-only

package com.majeur.inputmethod.tools.emoji.model

import kotlin.collections.mutableSetOf

class EmojiData {

    var unicodeVersion = ""
    var dataDate = ""

    private var emojiGroups = mutableMapOf<EmojiGroup, MutableList<EmojiSpec>>()

    operator fun get(group: EmojiGroup) = emojiGroups.getValue(group)

    fun emojiCount(group: EmojiGroup): Int {
        var acc = 0
        emojiGroups.values.forEach { acc += it.size }
        return acc
    }

    fun emojiGroupCount(group: EmojiGroup) = emojiGroups[group]?.size ?: 0

    /** Unicode's emoji-test.txt only lists the composed country flags, but Android also supports
     *  the standalone regional indicator letters (two of them in a row combine into a flag).
     *  They were removed from the symbols group in #1680 with the promise to restore them if
     *  requested (#2627), so append them to the end of the flags group as suggested there. */
    fun appendRegionalIndicators() {
        (CP_REGIONAL_INDICATOR_SYMBOL_LETTER_A..CP_REGIONAL_INDICATOR_SYMBOL_LETTER_Z).forEach { cp ->
            val letter = 'a' + (cp - CP_REGIONAL_INDICATOR_SYMBOL_LETTER_A)
            insertEmoji(EmojiGroup.FLAGS, intArrayOf(cp), 2.0f, "regional indicator symbol letter $letter")
        }
    }

    fun insertEmoji(group: EmojiGroup, codes: IntArray, unicodeVer: Float, name: String): EmojiSpec {
        return EmojiSpec(codes, unicodeVer, name).also { emoji ->
            val baseEmoji = findBaseEmoji(group, emoji)
            if (baseEmoji != null && onEmojiVariantInserted(group, baseEmoji, emoji)) {
                baseEmoji.variants.add(emoji)
            } else if (onEmojiInserted(group, emoji)) {
                emojiGroups.getOrPut(group) { mutableListOf() }.add(emoji)
            }
        }
    }

    private fun onEmojiInserted(group: EmojiGroup, emoji: EmojiSpec): Boolean {
        // Some multi-skin-tone variants use a different base code than their non-multi-skin-tone counterparts,
        // so they don't get grouped. We drop them here, to prevent each variant from being displayed separately.
        return ! hasMultipleSkinModifiers(emoji.codes)
    }

    private fun hasMultipleSkinModifiers(codes: IntArray): Boolean {
        val tones = mutableSetOf<Int>()
        codes.forEach {
            when (it) {
                CP_LIGHT_SKIN_TONE, CP_MEDIUM_LIGHT_SKIN_TONE, CP_MEDIUM_SKIN_TONE,
                CP_MEDIUM_DARK_SKIN_TONE, CP_DARK_SKIN_TONE ->
                    tones.add(it)
            }
        }
        return tones.size > 1
    }

    private fun onEmojiVariantInserted(group: EmojiGroup, baseSpec: EmojiSpec, emojiSpec: EmojiSpec): Boolean {
        return true
    }

    private fun findBaseEmoji(group: EmojiGroup, emoji: EmojiSpec): EmojiSpec? {
        val (baseCodePoints, componentCode) = withoutComponentCodes(emoji.codes)

        // No component codes found, this emoji is a standalone one
        if (componentCode == CP_NUL) return null

        // Second try for emojis with U+FE0F suffix
        val baseCodePoints2 = baseCodePoints + CP_VARIANT_SELECTOR

        // Third try for emojis with U+FE0F prefix before an eventual ZWJ
        val baseCodePoints3 = emoji.codes.toMutableList()
                .apply { set(emoji.codes.indexOf(componentCode), CP_VARIANT_SELECTOR) }.toIntArray()

        return emojiGroups[group]?.firstOrNull { it.codes contentEquals baseCodePoints }
            ?: emojiGroups[group]?.firstOrNull { it.codes contentEquals baseCodePoints2 }
            ?: emojiGroups[group]?.firstOrNull { it.codes contentEquals baseCodePoints3 }
    }

    private fun withoutComponentCodes(codes: IntArray) : Pair<IntArray, Int> {
        var res = codes
        var tone = CP_NUL
        codes.forEach { code ->
            when (code) {
                CP_LIGHT_SKIN_TONE, CP_MEDIUM_LIGHT_SKIN_TONE, CP_MEDIUM_SKIN_TONE,
                CP_MEDIUM_DARK_SKIN_TONE, CP_DARK_SKIN_TONE -> {
                    res = res.asList().minus(code).toIntArray()
                    tone = code
                }
            }
        }
        return res to tone
    }

    companion object {
        const val CP_NUL = 0x0000

        private const val CP_ZWJ = 0x200D
        private const val CP_FEMALE_SIGN = 0x2640
        private const val CP_MALE_SIGN = 0x2642
        private const val CP_LIGHT_SKIN_TONE = 0x1F3FB
        private const val CP_MEDIUM_LIGHT_SKIN_TONE = 0x1F3FC
        private const val CP_MEDIUM_SKIN_TONE = 0x1F3FD
        private const val CP_MEDIUM_DARK_SKIN_TONE = 0x1F3FE
        private const val CP_DARK_SKIN_TONE = 0x1F3FF
        private const val CP_RED_HAIR = 0x1F9B0
        private const val CP_CURLY_HAIR = 0x1F9B1
        private const val CP_WHITE_HAIR = 0x1F9B3
        private const val CP_BARLD = 0x1F9B2
        private const val CP_VARIANT_SELECTOR = 0xFE0F

        private const val CP_REGIONAL_INDICATOR_SYMBOL_LETTER_A = 0x1F1E6
        private const val CP_REGIONAL_INDICATOR_SYMBOL_LETTER_Z = 0x1F1FF
    }
}
