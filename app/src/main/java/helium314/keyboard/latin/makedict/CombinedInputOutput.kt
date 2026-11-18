/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.makedict

import helium314.keyboard.latin.makedict.FormatSpec.DictionaryOptions
import helium314.keyboard.latin.makedict.FusionDictionary.PtNodeArray
import helium314.keyboard.latin.utils.CombinedFormatUtils
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.FileReader
import java.io.IOException
import java.util.TreeSet

/** Reads and writes combined format for a FusionDictionary */
object CombinedInputOutput {
    private const val WHITELIST_TAG = "whitelist"
    private const val OPTIONS_TAG = "options"
    private const val COMMENT_LINE_STARTER = "#"
    private const val HISTORICAL_INFO_ELEMENT_COUNT = 3

    /**
     * Basic test to find out whether the file is in the combined format or not.
     * Concretely this only tests the header line.
     */
    fun isCombinedDictionary(filename: String?): Boolean {
        try {
            val headerRegex = "^${CombinedFormatUtils.DICTIONARY_TAG}=[^:]+(:[^=]+=[^:]+)*".toRegex()
            BufferedReader(FileReader(filename)).use { reader ->
                var firstLine = reader.readLine()
                while (firstLine.startsWith(COMMENT_LINE_STARTER)) {
                    firstLine = reader.readLine()
                }
                return firstLine.matches(headerRegex)
            }
        } catch (_: IOException) {
            return false
        }
    }

    /**
     * Reads a dictionary from a combined format file.
     * This is the public method that will read a combined file and return the corresponding memory
     * representation.
     */
    @Throws(IOException::class)
    fun readDictionaryCombined(reader: BufferedReader): FusionDictionary {
        var headerLine = reader.readLine()
        while (headerLine.startsWith(COMMENT_LINE_STARTER)) {
            headerLine = reader.readLine()
        }
        val header = headerLine.split(",")
        val attributes = HashMap<String, String>()
        for (item in header) {
            val keyValue = item.split("=")
            if (2 != keyValue.size) {
                throw RuntimeException("Wrong header format: $headerLine")
            }
            attributes[keyValue[0]] = keyValue[1]
        }

        attributes.remove(OPTIONS_TAG)
        val dict = FusionDictionary(PtNodeArray(), DictionaryOptions(attributes))

        var word: String? = null
        var probabilityInfo = ProbabilityInfo(0)
        var isNotAWord = false
        var isPossiblyOffensive = false
        val bigrams = ArrayList<WeightedString>()
        var shortcuts = ArrayList<WeightedString>()
        var line: String? = reader.readLine()?.trimStart()
        while (line != null) {
            // loop over lines
            // build word and additional data from following lines (bigrams, shortcuts)
            // on next word, add data of previous word to dictionary and reset variables
            if (line.startsWith(COMMENT_LINE_STARTER)) {
                line = reader.readLine()
                continue
            }
            val args = line.split(",")
            if (args[0].startsWith("${CombinedFormatUtils.WORD_TAG}=")) {
                if (!word.isNullOrEmpty()) { // empty words exist in history dictionary
                    dict.add(
                        word, probabilityInfo, shortcuts.takeIf { it.isNotEmpty() }, isNotAWord, isPossiblyOffensive
                    )
                    for (s in bigrams) {
                        dict.setBigram(word, s.mWord, s.mProbabilityInfo)
                    }
                }
                if (!shortcuts.isEmpty()) shortcuts = ArrayList() // list is kept in PtNode, so we need to create a new instance
                bigrams.clear()
                isNotAWord = false
                isPossiblyOffensive = false

                for (param in args) {
                    val params = param.split("=", limit = 2)
                    if (params.size != 2) throw RuntimeException("Wrong format: $line")
                    when (params[0]) {
                        CombinedFormatUtils.WORD_TAG -> word = params[1]
                        CombinedFormatUtils.PROBABILITY_TAG -> probabilityInfo = ProbabilityInfo(
                            params[1].toInt(), probabilityInfo.mTimestamp, probabilityInfo.mLevel, probabilityInfo.mCount
                        )
                        CombinedFormatUtils.HISTORICAL_INFO_TAG -> {
                            val historicalInfoParams = params[1].split(CombinedFormatUtils.HISTORICAL_INFO_SEPARATOR)
                            if (historicalInfoParams.size != HISTORICAL_INFO_ELEMENT_COUNT) {
                                throw RuntimeException("Wrong format (historical info): $line")
                            }
                            probabilityInfo = ProbabilityInfo(
                                probabilityInfo.mProbability,
                                historicalInfoParams[0].toInt(),
                                historicalInfoParams[1].toInt(),
                                historicalInfoParams[2].toInt()
                            )
                        }
                        CombinedFormatUtils.NOT_A_WORD_TAG -> isNotAWord = params[1].toBoolean()
                        CombinedFormatUtils.POSSIBLY_OFFENSIVE_TAG -> isPossiblyOffensive = params[1].toBoolean()
                    }
                }
            } else if (args[0].startsWith("${CombinedFormatUtils.SHORTCUT_TAG}=")) {
                var shortcut: String? = null
                var shortcutFreq = 0
                for (param in args) {
                    val params = param.split("=", limit = 2)
                    if (params.size != 2) throw RuntimeException("Wrong format: $line")
                    if (params[0] == CombinedFormatUtils.SHORTCUT_TAG) {
                        shortcut = params[1]
                    } else if (params[0] == CombinedFormatUtils.PROBABILITY_TAG) {
                        shortcutFreq = if (params[1] == WHITELIST_TAG) FormatSpec.SHORTCUT_WHITELIST_FREQUENCY
                        else params[1].toInt()
                    }
                }
                if (shortcut == null)
                    throw RuntimeException("Wrong format: $line")
                shortcuts.add(WeightedString(shortcut, shortcutFreq))
            } else if (args[0].startsWith("${CombinedFormatUtils.BIGRAM_TAG}=")) {
                var secondWordOfBigram: String? = null
                var bigramProbabilityInfo = ProbabilityInfo(0)
                for (param in args) {
                    val params = param.split("=", limit = 2)
                    if (params.size != 2) throw RuntimeException("Wrong format: $line")
                    if (CombinedFormatUtils.BIGRAM_TAG == params[0]) {
                        secondWordOfBigram = params[1]
                    } else if (CombinedFormatUtils.PROBABILITY_TAG == params[0]) {
                        bigramProbabilityInfo = ProbabilityInfo(
                            params[1].toInt(),
                            bigramProbabilityInfo.mTimestamp, bigramProbabilityInfo.mLevel,
                            bigramProbabilityInfo.mCount
                        )
                    } else if (CombinedFormatUtils.HISTORICAL_INFO_TAG == params[0]) {
                        val historicalInfoParams = params[1].split(CombinedFormatUtils.HISTORICAL_INFO_SEPARATOR)
                        if (historicalInfoParams.size != HISTORICAL_INFO_ELEMENT_COUNT) {
                            throw RuntimeException("Wrong format (historical info): $line")
                        }
                        bigramProbabilityInfo = ProbabilityInfo(
                            bigramProbabilityInfo.mProbability,
                            historicalInfoParams[0].toInt(),
                            historicalInfoParams[1].toInt(),
                            historicalInfoParams[2].toInt()
                        )
                    }
                }
                if (secondWordOfBigram == null)
                    throw RuntimeException("Wrong format: $line")
                bigrams.add(WeightedString(secondWordOfBigram, bigramProbabilityInfo))
            } else if (args[0].startsWith("${CombinedFormatUtils.NGRAM_TAG}=")) {
                // copy of above (could be done better), for ngram output of writeDictionaryCombined
                // difference: "ngram" instead of "bigram", and the previous word is in next line(s)
                var secondWordOfBigram: String? = null
                var bigramProbabilityInfo = ProbabilityInfo(0)
                for (param in args) {
                    val params = param.split("=", limit = 2)
                    if (params.size != 2) throw RuntimeException("Wrong format: $line")
                    if (CombinedFormatUtils.NGRAM_TAG == params[0]) {
                        secondWordOfBigram = params[1]
                    } else if (CombinedFormatUtils.PROBABILITY_TAG == params[0]) {
                        bigramProbabilityInfo = ProbabilityInfo(
                            params[1].toInt(),
                            bigramProbabilityInfo.mTimestamp, bigramProbabilityInfo.mLevel,
                            bigramProbabilityInfo.mCount
                        )
                    } else if (CombinedFormatUtils.HISTORICAL_INFO_TAG == params[0]) {
                        val historicalInfoParams = params[1].split(CombinedFormatUtils.HISTORICAL_INFO_SEPARATOR)
                        if (historicalInfoParams.size != HISTORICAL_INFO_ELEMENT_COUNT) {
                            throw RuntimeException("Wrong format (historical info): $line")
                        }
                        bigramProbabilityInfo = ProbabilityInfo(
                            bigramProbabilityInfo.mProbability,
                            historicalInfoParams[0].toInt(),
                            historicalInfoParams[1].toInt(),
                            historicalInfoParams[2].toInt()
                        )
                    }
                }
                line = reader.readLine()?.trimStart()
                if (line != "${CombinedFormatUtils.NGRAM_PREV_WORD_TAG}[0]=$word")
                    continue // only read the bigram part, expect ordered ngrams
                if (secondWordOfBigram == null)
                    throw RuntimeException("Wrong format: $line")
                bigrams.add(WeightedString(secondWordOfBigram, bigramProbabilityInfo))
            }
            line = reader.readLine()?.trimStart()
        }
        if (null != word) {
            dict.add(
                word, probabilityInfo, shortcuts.takeIf { it.isNotEmpty() }, isNotAWord, isPossiblyOffensive
            )
            for (s in bigrams) {
                dict.setBigram(word, s.mWord, s.mProbabilityInfo)
            }
        }

        return dict
    }

    /** Writes a dictionary to a combined file. Entries are sorted by frequency, then by word.*/
    @Throws(IOException::class)
    fun writeDictionaryCombined(destination: BufferedWriter, dict: FusionDictionary) {
        val wordPropertiesInDict = TreeSet<WordProperty>()
        for (wordProperty in dict) {
            // This for ordering by frequency, then by asciibetic order
            wordPropertiesInDict.add(wordProperty)
        }
        destination.write(CombinedFormatUtils.formatAttributeMap(dict.mOptions.mAttributes))
        for (wordProperty in wordPropertiesInDict) {
            destination.write(CombinedFormatUtils.formatWordProperty(wordProperty))
        }
    }
}
