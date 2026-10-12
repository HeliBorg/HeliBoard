// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.event

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.common.isEmoji

/**
 * Special combiner key type for the 12-key flick layout. This key type toggles the last input.
 * inherits from combiner.kt
 * used by FlickKeysPanel.kt
 *
 * Two specific use-cases:
 * dakuten (か→が→か), handakuten (は→ば→ぱ→は), small-kana (つ→っ→つ) toggles
 * vowel-cycle key, (な→の→ね→ぬ→に→な)
 */

class FlickDakutenCombiner : Combiner {

    private val composingWord = StringBuilder()

    override fun processEvent(previousEvents: ArrayList<Event>?, event: Event): Event {
        if (event.keyCode == KeyCode.SHIFT || isEmoji(event.codePoint)) return event
        if (Character.isWhitespace(event.codePoint)) {
            val text = combiningStateFeedback
            reset()
            return createEventChainFromSequence(text, event)
        } else if (event.isFunctionalKeyEvent) {
            if (event.keyCode == KeyCode.DELETE) {
                return if (composingWord.isNotEmpty()) {
                    composingWord.deleteCharAt(composingWord.lastIndex)
                    Event.createConsumedEvent(event)
                } else event
            }
            val text = combiningStateFeedback
            reset()
            return createEventChainFromSequence(text, event)
        }

        if (event.codePoint == DAKUTEN_TOGGLE_CODEPOINT) {
            if (composingWord.isNotEmpty()) {
                val last = composingWord.last()
                DAKUTEN_CYCLE[last]?.let { composingWord.setCharAt(composingWord.lastIndex, it) }
            }
            return Event.createConsumedEvent(event)
        }

        if (event.codePoint == VOWEL_CYCLE_CODEPOINT) {
            if (composingWord.isNotEmpty()) {
                val last = composingWord.last()
                VOWEL_BACK_CYCLE[last]?.let { composingWord.setCharAt(composingWord.lastIndex, it) }
            }
            return Event.createConsumedEvent(event)
        }

        composingWord.appendCodePoint(event.codePoint)
        return Event.createConsumedEvent(event)
    }

    override val combiningStateFeedback: CharSequence
        get() = composingWord.toString()

    override fun reset() {
        composingWord.setLength(0)
    }

    companion object {
        // dakuten/handakuten/small-kana cycle key to switch between related characters
        // "accent" (dakuten) cycle key
        const val DAKUTEN_TOGGLE_CODEPOINT = 0xE000
        // vowel-cycle "back" key
        const val VOWEL_CYCLE_CODEPOINT = 0xE001

        private fun createEventChainFromSequence(text: CharSequence, originalEvent: Event): Event =
            Event.createSoftwareTextEvent(text, KeyCode.MULTIPLE_CODE_POINTS, originalEvent)

        // each entry cycles to the next character
        // plain -> dakuten [-> handakuten] -> plain
        private val DAKUTEN_CYCLE: Map<Char, Char> = mapOf(
            // k-row
            'か' to 'が', 'き' to 'ぎ', 'く' to 'ぐ', 'け' to 'げ', 'こ' to 'ご',
            'が' to 'か', 'ぎ' to 'き', 'ぐ' to 'く', 'げ' to 'け', 'ご' to 'こ',
            // s-row
            'さ' to 'ざ', 'し' to 'じ', 'す' to 'ず', 'せ' to 'ぜ', 'そ' to 'ぞ',
            'ざ' to 'さ', 'じ' to 'し', 'ず' to 'す', 'ぜ' to 'せ', 'ぞ' to 'そ',
            // t-row (つ cycles through the sokuon っ rather than a dakuten form)
            'た' to 'だ', 'ち' to 'ぢ', 'つ' to 'っ', 'て' to 'で', 'と' to 'ど',
            'だ' to 'た', 'ぢ' to 'ち', 'で' to 'て', 'ど' to 'と', 'っ' to 'つ',
            // h-row: plain -> dakuten -> handakuten -> plain
            'は' to 'ば', 'ひ' to 'び', 'ふ' to 'ぶ', 'へ' to 'べ', 'ほ' to 'ぼ',
            'ば' to 'ぱ', 'び' to 'ぴ', 'ぶ' to 'ぷ', 'べ' to 'ぺ', 'ぼ' to 'ぽ',
            'ぱ' to 'は', 'ぴ' to 'ひ', 'ぷ' to 'ふ', 'ぺ' to 'へ', 'ぽ' to 'ほ',
            // small-kana toggles
            'あ' to 'ぁ', 'ぁ' to 'あ', 'い' to 'ぃ', 'ぃ' to 'い', 'う' to 'ぅ', 'ぅ' to 'う',
            'え' to 'ぇ', 'ぇ' to 'え', 'お' to 'ぉ', 'ぉ' to 'お',
            'や' to 'ゃ', 'ゃ' to 'や', 'ゆ' to 'ゅ', 'ゅ' to 'ゆ', 'よ' to 'ょ', 'ょ' to 'よ',
            'わ' to 'ゎ', 'ゎ' to 'わ',
        )

        // each entry steps backward through its row in gojuuon order
        // ex: for な/に/ぬ/ね/の (na/ni/nu/ne/no); pressing this key switches な => の
        private val VOWEL_BACK_CYCLE: Map<Char, Char> = mapOf(
            // a-row
            'あ' to 'お', 'お' to 'え', 'え' to 'う', 'う' to 'い', 'い' to 'あ',
            'ぁ' to 'ぉ', 'ぉ' to 'ぇ', 'ぇ' to 'ぅ', 'ぅ' to 'ぃ', 'ぃ' to 'ぁ',
            // k-row
            'か' to 'こ', 'こ' to 'け', 'け' to 'く', 'く' to 'き', 'き' to 'か',
            'が' to 'ご', 'ご' to 'げ', 'げ' to 'ぐ', 'ぐ' to 'ぎ', 'ぎ' to 'が',
            // s-row
            'さ' to 'そ', 'そ' to 'せ', 'せ' to 'す', 'す' to 'し', 'し' to 'さ',
            'ざ' to 'ぞ', 'ぞ' to 'ぜ', 'ぜ' to 'ず', 'ず' to 'じ', 'じ' to 'ざ',
            // t-row
            'た' to 'と', 'と' to 'て', 'て' to 'つ', 'つ' to 'ち', 'ち' to 'た',
            'だ' to 'ど', 'ど' to 'で', 'で' to 'づ', 'づ' to 'ぢ', 'ぢ' to 'だ',
            // n-row
            'な' to 'の', 'の' to 'ね', 'ね' to 'ぬ', 'ぬ' to 'に', 'に' to 'な',
            // h-row
            'は' to 'ほ', 'ほ' to 'へ', 'へ' to 'ふ', 'ふ' to 'ひ', 'ひ' to 'は',
            'ば' to 'ぼ', 'ぼ' to 'べ', 'べ' to 'ぶ', 'ぶ' to 'び', 'び' to 'ば',
            'ぱ' to 'ぽ', 'ぽ' to 'ぺ', 'ぺ' to 'ぷ', 'ぷ' to 'ぴ', 'ぴ' to 'ぱ',
            // m-row
            'ま' to 'も', 'も' to 'め', 'め' to 'む', 'む' to 'み', 'み' to 'ま',
            // y-row (yi/ye is replaced by small ゃ/ょ)
            'や' to 'よ', 'よ' to 'ょ', 'ょ' to 'ゆ', 'ゆ' to 'ゃ', 'ゃ' to 'や',
            // r-row
            'ら' to 'ろ', 'ろ' to 'れ', 'れ' to 'る', 'る' to 'り', 'り' to 'ら',
            // わ/ん/を are irregular; cycle between the 3 of them
            'わ' to 'を', 'を' to 'ん', 'ん' to 'わ',
        )
    }
}
