// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.event

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.isEmoji
import java.lang.StringBuilder
import java.util.ArrayList

class HangulCombiner : Combiner {

    private val composingWord = StringBuilder()

    val history: MutableList<HangulSyllable> = mutableListOf()
    private val syllable: HangulSyllable? get() = history.lastOrNull()

    override fun processEvent(previousEvents: ArrayList<Event>?, event: Event): Event {
        if (event.keyCode == KeyCode.SHIFT || isEmoji(event.codePoint)) return event
        // previously we only used the combiner if codePoint > 0x1100 or codePoint == -1, but looks here it's not necessary
        val event = HangulEventDecoder.decodeSoftwareKeyEvent(event)
        if (Character.isWhitespace(event.codePoint)) {
            val text = combiningStateFeedback
            reset()
            return createEventChainFromSequence(text, event)
        } else if (event.isFunctionalKeyEvent) {
            if(event.keyCode == KeyCode.DELETE) {
                return when {
                    history.size == 1 && composingWord.isEmpty() || history.isEmpty() && composingWord.length == 1 -> {
                        reset()
                        Event.createHardwareKeypressEvent(0x20, Constants.CODE_SPACE, 0, event, event.isKeyRepeat)
                    }
                    history.isNotEmpty() -> {
                        history.removeAt(history.lastIndex)
                        Event.createConsumedEvent(event)
                    }
                    composingWord.isNotEmpty() -> {
                        composingWord.deleteCharAt(composingWord.lastIndex)
                        Event.createConsumedEvent(event)
                    }
                    else -> event
                }
            }
            val text = combiningStateFeedback
            reset()
            return createEventChainFromSequence(text, event)
        } else {
            val currentSyllable = syllable ?: HangulSyllable()
            val jamo = HangulJamo.of(event.codePoint)
            if (!event.isCombining || jamo is HangulJamo.NonHangul) {
                composingWord.append(currentSyllable.string)
                composingWord.append(jamo.string)
                history.clear()
            } else {
                when (jamo) {
                    is HangulJamo.Consonant -> {
                        val initial = jamo.toInitial()
                        val final = jamo.toFinal()
                        if (currentSyllable.initial != null && currentSyllable.medial != null) {
                            if (currentSyllable.final == null) {
                                val combination = COMBINATION_TABLE_DUBEOLSIK[currentSyllable.initial.codePoint to (initial?.codePoint ?: -1)]
                                history +=
                                    if (combination != null) {
                                        currentSyllable.copy(initial = HangulJamo.Initial(combination))
                                    } else {
                                        if (final != null) {
                                            currentSyllable.copy(final = final)
                                        } else {
                                            composingWord.append(currentSyllable.string)
                                            history.clear()
                                            HangulSyllable(initial = initial)
                                        }
                                    }
                            } else {
                                val pair = currentSyllable.final.codePoint to (final?.codePoint ?: -1)
                                val combination = COMBINATION_TABLE_DUBEOLSIK[pair]
                                history += if (combination != null) {
                                    currentSyllable.copy(final = HangulJamo.Final(combination, combinationPair = pair))
                                } else {
                                    composingWord.append(currentSyllable.string)
                                    history.clear()
                                    HangulSyllable(initial = initial)
                                }
                            }
                        } else {
                            composingWord.append(currentSyllable.string)
                            history.clear()
                            history += HangulSyllable(initial = initial)
                        }
                    }
                    is HangulJamo.Vowel -> {
                        val medial = jamo.toMedial()
                        if (currentSyllable.final == null) {
                            history +=
                                if (currentSyllable.medial != null) {
                                    val combination = COMBINATION_TABLE_DUBEOLSIK[currentSyllable.medial.codePoint to (medial?.codePoint ?: -1)]
                                    if (combination != null) {
                                        currentSyllable.copy(medial = HangulJamo.Medial(combination))
                                    } else {
                                        composingWord.append(currentSyllable.string)
                                        history.clear()
                                        HangulSyllable(medial = medial)
                                    }
                            } else {
                                currentSyllable.copy(medial = medial)
                            }
                        } else if (currentSyllable.final.combinationPair != null) {
                            val pair = currentSyllable.final.combinationPair

                            history.removeAt(history.lastIndex)
                            val final = HangulJamo.Final(pair.first)
                            history += currentSyllable.copy(final = final)
                            composingWord.append(syllable?.string ?: "")
                            history.clear()
                            val initial = HangulJamo.Final(pair.second).toConsonant()?.toInitial()
                            val newSyllable = HangulSyllable(initial = initial)
                            history += newSyllable
                            history += newSyllable.copy(medial = medial)
                        } else {
                            history.removeAt(history.lastIndex)
                            composingWord.append(syllable?.string ?: "")
                            history.clear()
                            val initial = currentSyllable.final.toConsonant()?.toInitial()
                            val newSyllable = HangulSyllable(initial = initial)
                            history += newSyllable
                            history += newSyllable.copy(medial = medial)
                        }
                    }
                    is HangulJamo.Initial -> {
                        history +=
                            if (currentSyllable.initial != null) {
                                val combination = COMBINATION_TABLE_SEBEOLSIK[currentSyllable.initial.codePoint to jamo.codePoint]
                                if (combination != null && currentSyllable.medial == null && currentSyllable.final == null) {
                                    currentSyllable.copy(initial = HangulJamo.Initial(combination))
                                } else {
                                    composingWord.append(currentSyllable.string)
                                    history.clear()
                                    HangulSyllable(initial = jamo)
                                }
                            } else {
                                currentSyllable.copy(initial = jamo)
                            }
                    }
                    is HangulJamo.Medial -> {
                        history +=
                            if (currentSyllable.medial != null) {
                                val combination = COMBINATION_TABLE_SEBEOLSIK[currentSyllable.medial.codePoint to jamo.codePoint]
                                if (combination != null) {
                                    currentSyllable.copy(medial = HangulJamo.Medial(combination))
                                } else {
                                    composingWord.append(currentSyllable.string)
                                    history.clear()
                                    HangulSyllable(medial = jamo)
                                }
                            } else {
                                currentSyllable.copy(medial = jamo)
                            }
                    }
                    is HangulJamo.Final -> {
                        history +=
                            if (currentSyllable.final != null) {
                                val combination = COMBINATION_TABLE_SEBEOLSIK[currentSyllable.final.codePoint to jamo.codePoint]
                                if (combination != null) {
                                    currentSyllable.copy(final = HangulJamo.Final(combination))
                                } else {
                                    composingWord.append(currentSyllable.string)
                                    history.clear()
                                    HangulSyllable(final = jamo)
                                }
                            } else {
                                currentSyllable.copy(final = jamo)
                            }
                    }
                    // compiler bug? when it's not added, compiler complains that it's missing
                    // but when added, linter (correctly) states it's unreachable anyway
                    is HangulJamo.NonHangul -> Unit
                }
            }
        }

        return Event.createConsumedEvent(event)
    }

    override val combiningStateFeedback: CharSequence
        get() = composingWord.toString() + (syllable?.string ?: "")

    override fun reset() {
        composingWord.setLength(0)
        history.clear()
    }

    sealed class HangulJamo {
        abstract val codePoint: Int
        abstract val modern: Boolean
        val string: String get() = codePoint.toChar().toString()
        data class NonHangul(override val codePoint: Int) : HangulJamo() {
            override val modern: Boolean get() = false
        }
        data class Initial(override val codePoint: Int) : HangulJamo() {
            override val modern: Boolean get() = codePoint in 0x1100 .. 0x1112
            val ordinal: Int get() = codePoint - 0x1100
            fun toConsonant(): Consonant? {
                val codePoint = COMPAT_CONSONANTS.getOrNull(CONVERT_INITIALS.indexOf(codePoint.toChar())) ?: return null
                if(codePoint.code == 0) return null
                return Consonant(codePoint.code)
            }
        }
        data class Medial(override val codePoint: Int) : HangulJamo() {
            override val modern: Boolean get() = codePoint in 1161 .. 0x1175
            val ordinal: Int get() = codePoint - 0x1161
            fun toVowel(): Vowel? {
                val codePoint = COMPAT_VOWELS.getOrNull(CONVERT_MEDIALS.indexOf(codePoint.toChar())) ?: return null
                return Vowel(codePoint.code)
            }
        }
        data class Final(override val codePoint: Int, val combinationPair: Pair<Int, Int>? = null) : HangulJamo() {
            override val modern: Boolean get() = codePoint in 0x11a8 .. 0x11c2
            val ordinal: Int get() = codePoint - 0x11a7
            fun toConsonant(): Consonant? {
                val codePoint = COMPAT_CONSONANTS.getOrNull(CONVERT_FINALS.indexOf(codePoint.toChar())) ?: return null
                if(codePoint.code == 0) return null
                return Consonant(codePoint.code)
            }
        }
        data class Consonant(override val codePoint: Int) : HangulJamo() {
            override val modern: Boolean get() = codePoint in 0x3131 .. 0x314e
            val ordinal: Int get() = codePoint - 0x3131
            fun toInitial(): Initial? {
                val codePoint = CONVERT_INITIALS.getOrNull(COMPAT_CONSONANTS.indexOf(codePoint.toChar())) ?: return null
                if(codePoint.code == 0) return null
                return Initial(codePoint.code)
            }
            fun toFinal(): Final? {
                val codePoint = CONVERT_FINALS.getOrNull(COMPAT_CONSONANTS.indexOf(codePoint.toChar())) ?: return null
                if(codePoint.code == 0) return null
                return Final(codePoint.code)
            }
        }
        data class Vowel(override val codePoint: Int) : HangulJamo() {
            override val modern: Boolean get() = codePoint in 0x314f .. 0x3163
            val ordinal: Int get() = codePoint - 0x314f1
            fun toMedial(): Medial? {
                val codePoint = CONVERT_MEDIALS.getOrNull(COMPAT_VOWELS.indexOf(codePoint.toChar())) ?: return null
                if(codePoint.code == 0) return null
                return Medial(codePoint.code)
            }
        }
        companion object {
            const val COMPAT_CONSONANTS = "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
            const val COMPAT_VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
            const val CONVERT_INITIALS = "ᄀᄁ\u0000ᄂ\u0000\u0000ᄃᄄᄅ\u0000\u0000\u0000\u0000\u0000\u0000\u0000ᄆᄇᄈ\u0000ᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ"
            const val CONVERT_MEDIALS = "ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵ"
            const val CONVERT_FINALS = "ᆨᆩᆪᆫᆬᆭᆮ\u0000ᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸ\u0000ᆹᆺᆻᆼᆽ\u0000ᆾᆿᇀᇁᇂ"
            fun of(codePoint: Int): HangulJamo {
                return when(codePoint) {
                    in 0x3131 .. 0x314e -> Consonant(codePoint)
                    in 0x314f .. 0x3163 -> Vowel(codePoint)
                    in 0x1100 .. 0x115f -> Initial(codePoint)
                    in 0x1160 .. 0x11a7 -> Medial(codePoint)
                    in 0x11a8 .. 0x11ff -> Final(codePoint)
                    else -> NonHangul(codePoint)
                }
            }
        }
    }

    data class HangulSyllable(
            val initial: HangulJamo.Initial? = null,
            val medial: HangulJamo.Medial? = null,
            val final: HangulJamo.Final? = null
    ) {
        val combinable: Boolean get() = (initial?.modern ?: false) && (medial?.modern ?: false) && (final?.modern ?: true)
        val combined: String get() = (0xac00 + (initial?.ordinal ?: 0) * 21 * 28
                + (medial?.ordinal ?: 0) * 28
                + (final?.ordinal ?: 0)).toChar().toString()
        val uncombined: String get() = (initial?.string ?: "") + (medial?.string ?: "") + (final?.string ?: "")
        val uncombinedCompat: String get() = (initial?.toConsonant()?.string ?: "") +
                (medial?.toVowel()?.string ?: "") + (final?.toConsonant()?.string ?: "")
        val string: String get() = if (this.combinable) this.combined else this.uncombinedCompat
    }

    companion object {
        val COMBINATION_TABLE_DUBEOLSIK = mapOf<Pair<Int, Int>, Int>(
                0x1169 to 0x1161 to 0x116a,
                0x1169 to 0x1162 to 0x116b,
                0x1169 to 0x1175 to 0x116c,
                0x116e to 0x1165 to 0x116f,
                0x116e to 0x1166 to 0x1170,
                0x116e to 0x1175 to 0x1171,
                0x1173 to 0x1175 to 0x1174,

                0x11a8 to 0x11ba to 0x11aa,
                0x11ab to 0x11bd to 0x11ac,
                0x11ab to 0x11c2 to 0x11ad,
                0x11af to 0x11a8 to 0x11b0,
                0x11af to 0x11b7 to 0x11b1,
                0x11af to 0x11b8 to 0x11b2,
                0x11af to 0x11ba to 0x11b3,
                0x11af to 0x11c0 to 0x11b4,
                0x11af to 0x11c1 to 0x11b5,
                0x11af to 0x11c2 to 0x11b6,
                0x11b8 to 0x11ba to 0x11b9
        )
        val COMBINATION_TABLE_SEBEOLSIK = mapOf<Pair<Int, Int>, Int>(
                0x1100 to 0x1100 to 0x1101,	// ㄲ
                0x1103 to 0x1103 to 0x1104,	// ㄸ
                0x1107 to 0x1107 to 0x1108,	// ㅃ
                0x1109 to 0x1109 to 0x110a,	// ㅆ
                0x110c to 0x110c to 0x110d,	// ㅉ

                0x1169 to 0x1161 to 0x116a,	// ㅘ
                0x1169 to 0x1162 to 0x116b,	// ㅙ
                0x1169 to 0x1175 to 0x116c,	// ㅚ
                0x116e to 0x1165 to 0x116f,	// ㅝ
                0x116e to 0x1166 to 0x1170,	// ㅞ
                0x116e to 0x1175 to 0x1171,	// ㅟ
                0x1173 to 0x1175 to 0x1174,	// ㅢ

                0x11a8 to 0x11a8 to 0x11a9,	// ㄲ
                0x11a8 to 0x11ba to 0x11aa,	// ㄳ
                0x11ab to 0x11bd to 0x11ac,	// ㄵ
                0x11ab to 0x11c2 to 0x11ad,	// ㄶ
                0x11af to 0x11a8 to 0x11b0,	// ㄺ
                0x11af to 0x11b7 to 0x11b1,	// ㄻ
                0x11af to 0x11b8 to 0x11b2,	// ㄼ
                0x11af to 0x11ba to 0x11b3,	// ㄽ
                0x11af to 0x11c0 to 0x11b4,	// ㄾ
                0x11af to 0x11c1 to 0x11b5,	// ㄿ
                0x11af to 0x11c2 to 0x11b6,	// ㅀ
                0x11b8 to 0x11ba to 0x11b9,	// ㅄ
                0x11ba to 0x11ba to 0x11bb	// ㅆ
        )
        private fun createEventChainFromSequence(text: CharSequence, originalEvent: Event): Event {
            return Event.createSoftwareTextEvent(text, KeyCode.MULTIPLE_CODE_POINTS, originalEvent)
        }
    }

}
