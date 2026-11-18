/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.suggestions

import helium314.keyboard.keyboard.internal.KeySpecParser
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.InputStyle
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.dictionary.Dictionary

/**
 * The extended [SuggestedWords] class to represent punctuation suggestions.
 *
 * Each punctuation specification string is the key specification that can be parsed by
 * [KeySpecParser].
 */
class PunctuationSuggestions private constructor(punctuationsList: List<SuggestedWordInfo>) : SuggestedWords(
    punctuationsList, null, null, false, false,
    false, InputStyle.NONE, NOT_A_SEQUENCE_NUMBER
) {
    /**
     * {@inheritDoc}
     * Note that [SuggestedWords.getWord] returns a punctuation key specification text.
     * The suggested punctuation should be gotten by parsing the key specification.
     */
    override fun getWord(index: Int): String {
        val keySpec = super.getWord(index)
        val code = KeySpecParser.getCode(keySpec)
        return (if (code == KeyCode.MULTIPLE_CODE_POINTS)
            KeySpecParser.getOutputText(keySpec, code)!!
        else
            StringUtils.newSingleCodePointString(code))
    }

    /**
     * {@inheritDoc}
     * Note that [SuggestedWords.getWord] returns a punctuation key specification text.
     * The displayed text should be gotten by parsing the key specification.
     */
    override fun getLabel(index: Int): String = KeySpecParser.getLabel(super.getWord(index))!!

    /**
     * {@inheritDoc}
     * Note that [.getWord] returns a suggested punctuation. We should create a
     * [SuggestedWordInfo] object that represents a hard coded word.
     */
    override fun getInfo(index: Int) = newHardCodedWordInfo(getWord(index))

    override fun toString() = ("PunctuationSuggestions: words=" + suggestedWordInfoList.toTypedArray().contentToString())

    companion object {
        /**
         * Create new instance of [PunctuationSuggestions] from the array of punctuation key
         * specifications.
         *
         * @param punctuationSpecs The array of punctuation key specifications.
         * @return The [PunctuationSuggestions] object.
         */
        @JvmStatic
        fun newPunctuationSuggestions(punctuationSpecs: Array<String>) =
            PunctuationSuggestions(punctuationSpecs.map { newHardCodedWordInfo(it) })

        private fun newHardCodedWordInfo(keySpec: String) = SuggestedWordInfo(
            keySpec, "",
            SuggestedWordInfo.MAX_SCORE,
            SuggestedWordInfo.KIND_HARDCODED,
            Dictionary.DICTIONARY_HARDCODED,
            SuggestedWordInfo.NOT_AN_INDEX,
            SuggestedWordInfo.NOT_A_CONFIDENCE
        )
    }
}
