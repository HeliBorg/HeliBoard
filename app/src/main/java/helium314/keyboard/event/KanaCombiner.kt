// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.event

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.common.isEmoji
import java.lang.StringBuilder
import java.util.ArrayList

/**
 * Combiner that turns romaji typed into hiragana
 * Based on: https://github.com/fjkz/azit/blob/master/mozc_azit.txt
 * inherits combiner.kt
 *
 * moras and digraphs are matched as soon as possible
 * doubled consonants produce a sokuon (っ)
 * a lone 'n' not followed by a vowel/y/n resolves to ん
 */
class KanaCombiner : Combiner {

    private val composingWord = StringBuilder()
    private val pendingRomaji = StringBuilder()

    override fun processEvent(previousEvents: ArrayList<Event>?, event: Event): Event {
        if (event.keyCode == KeyCode.SHIFT || isEmoji(event.codePoint)) return event
        if (Character.isWhitespace(event.codePoint)) {
            flushPendingRomaji()
            val text = combiningStateFeedback
            reset()
            return createEventChainFromSequence(text, event)
        } else if (event.isFunctionalKeyEvent) {
            if (event.keyCode == KeyCode.DELETE) {
                return when {
                    pendingRomaji.isNotEmpty() -> {
                        pendingRomaji.deleteCharAt(pendingRomaji.lastIndex)
                        Event.createConsumedEvent(event)
                    }
                    composingWord.isNotEmpty() -> {
                        composingWord.deleteCharAt(composingWord.lastIndex)
                        Event.createConsumedEvent(event)
                    }
                    else -> event
                }
            }
            flushPendingRomaji()
            val text = combiningStateFeedback
            reset()
            return createEventChainFromSequence(text, event)
        }

        val lower = Character.toLowerCase(event.codePoint)
        if (lower < 'a'.code || lower > 'z'.code) {
            // non romaji letters to be ignored
            flushPendingRomaji()
            composingWord.appendCodePoint(event.codePoint)
            return Event.createConsumedEvent(event)
        }

        pendingRomaji.append(lower.toChar())
        resolvePendingRomaji()
        return Event.createConsumedEvent(event)
    }

    // repeatedly resolves as much of pendingRomaji into kana as possible
    // unresolved prefixes buffered for the next keystroke
    private fun resolvePendingRomaji() {
        while (pendingRomaji.isNotEmpty()) {
            val buffer = pendingRomaji.toString()
            val exact = ROMAJI_TO_KANA[buffer]
            if (exact != null) {
                composingWord.append(exact)
                pendingRomaji.setLength(0)
                return
            }
            if (buffer.length == 2 && buffer[0] == buffer[1] && buffer[0] !in VOWELS && buffer[0] != 'n') {
                // doubled consonant sokuon, keep the second consonant buffered
                composingWord.append('っ')
                pendingRomaji.deleteCharAt(0)
                continue
            }
            if (buffer.length >= 2 && buffer[0] == 'n' && buffer[1] != 'n' && buffer[1] !in VOWELS && buffer[1] != 'y') {
                // 'n' not followed by a vowel/y/n can't extend into any mora -> resolve as ん
                composingWord.append('ん')
                pendingRomaji.deleteCharAt(0)
                continue
            }
            // could still extend into a valid mora?
            if (ROMAJI_TO_KANA.keys.any { it.startsWith(buffer) }) return
            // flush the first character literally and retry with the rest
            composingWord.append(buffer[0])
            pendingRomaji.deleteCharAt(0)
        }
    }

    //check for 'n's, as its the edge case
    private fun flushPendingRomaji() {
        if (pendingRomaji.isEmpty()) return
        if (pendingRomaji.toString() == "n") composingWord.append('ん')
        else composingWord.append(pendingRomaji)
        pendingRomaji.setLength(0)
    }

    override val combiningStateFeedback: CharSequence
        get() = composingWord.toString() + pendingRomaji

    override fun reset() {
        composingWord.setLength(0)
        pendingRomaji.setLength(0)
    }

    companion object {
        private val VOWELS = charArrayOf('a', 'i', 'u', 'e', 'o')

        private fun createEventChainFromSequence(text: CharSequence, originalEvent: Event): Event {
            return Event.createSoftwareTextEvent(text, KeyCode.MULTIPLE_CODE_POINTS, originalEvent)
        }

        val ROMAJI_TO_KANA: Map<String, String> = mapOf(
            "a" to "あ", "i" to "い", "u" to "う", "e" to "え", "o" to "お",

            "ka" to "か", "ki" to "き", "ku" to "く", "ke" to "け", "ko" to "こ",
            "kya" to "きゃ", "kyu" to "きゅ", "kyo" to "きょ",
            "ga" to "が", "gi" to "ぎ", "gu" to "ぐ", "ge" to "げ", "go" to "ご",
            "gya" to "ぎゃ", "gyu" to "ぎゅ", "gyo" to "ぎょ",

            "sa" to "さ", "si" to "し", "shi" to "し", "su" to "す", "se" to "せ", "so" to "そ",
            "sha" to "しゃ", "sya" to "しゃ", "shu" to "しゅ", "syu" to "しゅ", "sho" to "しょ", "syo" to "しょ", "she" to "しぇ",
            "za" to "ざ", "zi" to "じ", "ji" to "じ", "zu" to "ず", "ze" to "ぜ", "zo" to "ぞ",
            "ja" to "じゃ", "zya" to "じゃ", "ju" to "じゅ", "zyu" to "じゅ", "jo" to "じょ", "zyo" to "じょ", "je" to "じぇ",

            "ta" to "た", "ti" to "ち", "chi" to "ち", "tu" to "つ", "tsu" to "つ", "te" to "て", "to" to "と",
            "cha" to "ちゃ", "tya" to "ちゃ", "chu" to "ちゅ", "tyu" to "ちゅ", "cho" to "ちょ", "tyo" to "ちょ",
            "tsa" to "つぁ", "tsi" to "つぃ", "tse" to "つぇ", "tso" to "つぉ",
            "da" to "だ", "di" to "ぢ", "du" to "づ", "de" to "で", "do" to "ど",
            "dya" to "ぢゃ", "dyu" to "ぢゅ", "dyo" to "ぢょ",

            "na" to "な", "ni" to "に", "nu" to "ぬ", "ne" to "ね", "no" to "の",
            "nya" to "にゃ", "nyu" to "にゅ", "nyo" to "にょ", "nn" to "ん",

            "ha" to "は", "hi" to "ひ", "hu" to "ふ", "fu" to "ふ", "he" to "へ", "ho" to "ほ",
            "hya" to "ひゃ", "hyu" to "ひゅ", "hyo" to "ひょ",
            "fa" to "ふぁ", "fi" to "ふぃ", "fe" to "ふぇ", "fo" to "ふぉ",
            "ba" to "ば", "bi" to "び", "bu" to "ぶ", "be" to "べ", "bo" to "ぼ",
            "bya" to "びゃ", "byu" to "びゅ", "byo" to "びょ",
            "pa" to "ぱ", "pi" to "ぴ", "pu" to "ぷ", "pe" to "ぺ", "po" to "ぽ",
            "pya" to "ぴゃ", "pyu" to "ぴゅ", "pyo" to "ぴょ",

            "ma" to "ま", "mi" to "み", "mu" to "む", "me" to "め", "mo" to "も",
            "mya" to "みゃ", "myu" to "みゅ", "myo" to "みょ",

            "ya" to "や", "yu" to "ゆ", "yo" to "よ", "ye" to "いぇ",

            "ra" to "ら", "ri" to "り", "ru" to "る", "re" to "れ", "ro" to "ろ",
            "rya" to "りゃ", "ryu" to "りゅ", "ryo" to "りょ",

            "wa" to "わ", "wo" to "を", "wi" to "うぃ", "we" to "うぇ",
            "whi" to "うぃ", "whe" to "うぇ", "who" to "うぉ",

            "va" to "ゔぁ", "vi" to "ゔぃ", "vu" to "ゔ", "ve" to "ゔぇ", "vo" to "ゔぉ",

            // small kana, via l/x prefix
            "la" to "ぁ", "li" to "ぃ", "lu" to "ぅ", "le" to "ぇ", "lo" to "ぉ",
            "xa" to "ぁ", "xi" to "ぃ", "xu" to "ぅ", "xe" to "ぇ", "xo" to "ぉ",
            "ltu" to "っ", "xtu" to "っ", "ltsu" to "っ", "xtsu" to "っ",
            "lya" to "ゃ", "lyu" to "ゅ", "lyo" to "ょ",
            "xya" to "ゃ", "xyu" to "ゅ", "xyo" to "ょ",
            "lwa" to "ゎ", "xwa" to "ゎ"
        )
    }
}
