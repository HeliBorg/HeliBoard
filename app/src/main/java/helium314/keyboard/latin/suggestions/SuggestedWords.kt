/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.suggestions

import android.view.inputmethod.CompletionInfo
import helium314.keyboard.latin.InputStyle
import helium314.keyboard.latin.define.DebugFlags

open class SuggestedWords(
    val suggestedWordInfoList: List<SuggestedWordInfo>,
    val rawSuggestions: List<SuggestedWordInfo>?,
    val typedWordInfo: SuggestedWordInfo?,
    val typedWordValid: Boolean,
    /** Note: this INCLUDES cases where the word will auto-correct to itself. A good definition
    // of what this flag means would be "the top suggestion is strong enough to auto-correct",
    // whether this exactly matches the user entry or not. */
    val willAutoCorrect: Boolean,
    val isObsoleteSuggestions: Boolean,
    /** How the input for these suggested words was done by the user */
    val inputStyle: InputStyle,
    /** Sequence number for auto-commit */
    val sequenceNumber: Int
) : Collection<SuggestedWordInfo> {
    override fun isEmpty() = suggestedWordInfoList.isEmpty()

    override val size = suggestedWordInfoList.size

    val wordCountToShow get() = if (isPrediction) size else size - 1 // typed word

    /**
     * Get suggested word at `index`.
     * @param index The index of the suggested word.
     * @return The suggested word.
     */
    open fun getWord(index: Int): String = suggestedWordInfoList[index].word

    /**
     * Get displayed text at `index`.
     * In RTL languages, the displayed text on the suggestion strip may be different from the
     * suggested word that is returned from [.getWord]. For example the displayed text
     * of punctuation suggestion "(" should be ")".
     * @param index The index of the text to display.
     * @return The text to be displayed.
     */
    open fun getLabel(index: Int): String = suggestedWordInfoList[index].word

    /**
     * Get [SuggestedWordInfo] object at `index`.
     * @param index The index of the [SuggestedWordInfo].
     * @return The [SuggestedWordInfo] object.
     */
    open fun getInfo(index: Int): SuggestedWordInfo = suggestedWordInfoList[index]

    /**
     * Gets the suggestion index from the suggestions list.
     * @param suggestedWordInfo The [SuggestedWordInfo] to find the index.
     * @return The position of the suggestion in the suggestion list.
     */
    fun indexOf(suggestedWordInfo: SuggestedWordInfo?): Int = suggestedWordInfoList.indexOf(suggestedWordInfo)

    fun getDebugString(pos: Int) = if (!DebugFlags.DEBUG_ENABLED) null
        else getInfo(pos).debugString.takeIf { it.isNotEmpty() }

    // Pretty-print method to help debug
    override fun toString() = ("SuggestedWords:"
            + " typedWordValid=" + typedWordValid
            + " willAutoCorrect=" + willAutoCorrect
            + " inputStyle=" + inputStyle
            + " words=" + suggestedWordInfoList.toTypedArray().contentToString())

    val autoCommitCandidate: SuggestedWordInfo? get() {
        if (suggestedWordInfoList.isEmpty()) return null
        val candidate = suggestedWordInfoList[0]
        return if (candidate.isEligibleForAutoCommit) candidate else null
    }

    val isPrediction get() = inputStyle == InputStyle.PREDICTION || inputStyle == InputStyle.BEGINNING_OF_SENTENCE_PREDICTION

    /**
     * @return the [SuggestedWordInfo] which corresponds to the word that is originally
     * typed by the user. Otherwise returns `null`. Note that gesture input is not
     * considered to be a typed word.
     */
    val typedWordInfoOrNull: SuggestedWordInfo? get() {
        if (INDEX_OF_TYPED_WORD >= size) {
            return null
        }
        val info = getInfo(INDEX_OF_TYPED_WORD)
        return if (info.kind == SuggestedWordInfo.KIND_TYPED) info else null
    }

    override fun contains(element: SuggestedWordInfo) = suggestedWordInfoList.contains(element)

    override fun iterator() = suggestedWordInfoList.iterator()

    override fun containsAll(elements: Collection<SuggestedWordInfo>) = suggestedWordInfoList.containsAll(elements)

    companion object {
        const val INDEX_OF_TYPED_WORD: Int = 0
        const val INDEX_OF_AUTO_CORRECTION: Int = 1
        const val NOT_A_SEQUENCE_NUMBER: Int = -1

        // The maximum number of suggestions available.
        const val MAX_SUGGESTIONS: Int = 18

        private val EMPTY_WORD_INFO_LIST = ArrayList<SuggestedWordInfo>(0)
        @JvmStatic
        val emptyInstance = SuggestedWords(
            EMPTY_WORD_INFO_LIST, null, null, false,
            false, false, InputStyle.NONE, NOT_A_SEQUENCE_NUMBER
        )

        @JvmStatic
        val emptyBatchInstance = SuggestedWords(
            EMPTY_WORD_INFO_LIST, null, null, false,
            false, false, InputStyle.UPDATE_BATCH, NOT_A_SEQUENCE_NUMBER
        )

        @JvmStatic
        fun getFromApplicationSpecifiedCompletions(infos: Array<CompletionInfo?>) =
            infos.mapNotNull { if (it?.text == null) null else SuggestedWordInfo(it) }

        // Should get rid of the first one (what the user typed previously) from suggestions
        // and replace it with what the user currently typed.
        @JvmStatic
        fun getTypedWordAndPreviousSuggestions(typedWordInfo: SuggestedWordInfo, previousSuggestions: SuggestedWords): List<SuggestedWordInfo> {
            val suggestionsList = mutableListOf(typedWordInfo)
            val alreadySeen = hashSetOf(typedWordInfo.word)
            previousSuggestions.forEachIndexed { index, info ->
                if (index == 0 || index == previousSuggestions.size) return@forEachIndexed
                val prevWord = info.word
                // Filter out duplicate suggestions.
                if (!alreadySeen.contains(prevWord)) {
                    suggestionsList.add(info)
                    alreadySeen.add(prevWord)
                }
            }
            return suggestionsList
        }
    }
}
