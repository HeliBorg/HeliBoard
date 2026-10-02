// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.japanese

import java.io.DataInputStream

/**
 * Converts a hiragana reading into ranked kanji/word candidates
 *
 * prediction is based on Viterbi algorithm; TODO implement N-best candidates / n-gram?
 *
 * currently, this returns single best path plus whole-word fallbacks;
 * Having some issues with repetitive mistakes of suggesting the correct candidates; TODO implement history ranking?
 *
 */
class KanaKanjiEngine private constructor(
    private val trie: LoudsTrie,
    private val candidates: List<List<Candidate>>,
    private val posSize: Int,
    private val resolution: Int,
    private val connectionMatrix: ByteArray
) {
    data class Candidate(val surface: String, val cost: Int, val leftId: Int, val rightId: Int)

    private data class Node(val cost: Int, val prevPos: Int, val prevRightId: Int, val surface: String)

    private fun connectionCost(rid: Int, lid: Int): Int {
        // UNKNOWN_ID is a placeholder id for words that aren't in the dictionary (names)
        if (rid == UNKNOWN_ID || lid == UNKNOWN_ID) return UNKNOWN_CONNECTION_COST
        val quantized = connectionMatrix[rid * posSize + lid].toInt() and 0xFF
        return quantized * resolution
    }

    // returns ranked candidate strings for [reading]. never empty
    // Viterbi shortest path search
    // roughly based on https://github.com/KazumaProject/JapaneseKeyboard/blob/dev/app/src/main/java/com/kazumaproject/markdownhelperkeyboard/converter/path_algorithm/FindPath.kt
    fun convert(reading: String): List<String> {
        if (reading.isEmpty()) return emptyList()
        val n = reading.length
        // dictionary prediction -> find 'cheapest value' path
        val dp = Array(n + 1) { HashMap<Int, Node>() }
        dp[0][BOS_EOS_ID] = Node(0, -1, -1, "")

        // for every starting position [i], try every way to place a word
        for (i in 0 until n) {
            val states = dp[i]
            if (states.isEmpty()) continue
            for ((end, termId) in trie.commonPrefixSearch(reading, i)) {
                for (candidate in candidates[termId]) {
                    // viterbi
                    var bestCost = Int.MAX_VALUE
                    var bestPrevRightId = -1
                    for ((prevRightId, node) in states) {
                        val cost = node.cost + candidate.cost + connectionCost(prevRightId, candidate.leftId)
                        if (cost < bestCost) {
                            bestCost = cost
                            bestPrevRightId = prevRightId
                        }
                    }
                    // keep cost of candidate for later
                    val target = dp[end]
                    val existing = target[candidate.rightId]
                    if (existing == null || bestCost < existing.cost)
                        target[candidate.rightId] = Node(bestCost, i, bestPrevRightId, candidate.surface)
                }
            }
            // a single kana character renders as katakana
            val target = dp[i + 1]
            var bestCost = Int.MAX_VALUE
            var bestPrevRightId = -1
            for ((prevRightId, node) in states) {
                val cost = node.cost + UNKNOWN_WORD_COST + connectionCost(prevRightId, UNKNOWN_ID)
                if (cost < bestCost) {
                    bestCost = cost
                    bestPrevRightId = prevRightId
                }
            }
            val existing = target[UNKNOWN_ID]
            if (existing == null || bestCost < existing.cost)
                target[UNKNOWN_ID] = Node(bestCost, i, bestPrevRightId, toKatakana(reading[i].toString()))
        }

        var bestFinalCost = Int.MAX_VALUE
        var bestFinalRightId = -1
        // pick winner
        for ((rightId, node) in dp[n]) {
            val cost = node.cost + connectionCost(rightId, BOS_EOS_ID)
            if (cost < bestFinalCost) {
                bestFinalCost = cost
                bestFinalRightId = rightId
            }
        }

        val parts = mutableListOf<String>()
        var pos = n
        var rightId = bestFinalRightId
        // walk it back to make sure we picked the cheapest
        while (pos > 0) {
            val node = dp[pos].getValue(rightId)
            parts.add(node.surface)
            rightId = node.prevRightId
            pos = node.prevPos
        }
        val bestPath = parts.asReversed().joinToString("")

        val results = LinkedHashSet<String>()
        results.add(bestPath)
        trie.exactMatch(reading)?.let { termId ->
            candidates[termId].sortedBy { it.cost }.take(MAX_WHOLE_WORD_CANDIDATES).forEach { results.add(it.surface) }
        }
        results.add(toKatakana(reading))
        results.add(reading)
        return results.toList()
    }

    companion object {
        private const val UNKNOWN_WORD_COST = 3000
        private const val UNKNOWN_CONNECTION_COST = 3000
        private const val MAX_WHOLE_WORD_CANDIDATES = 4
        private const val BOS_EOS_ID = 0
        private const val UNKNOWN_ID = -1
        private const val MAGIC = "JAKD"
        private const val VERSION = 2

        fun load(input: DataInputStream): KanaKanjiEngine {
            val magicBytes = ByteArray(4)
            input.readFully(magicBytes)
            require(String(magicBytes, Charsets.US_ASCII) == MAGIC) { "not a kana-kanji dictionary" }
            val version = input.readUnsignedByte()
            require(version == VERSION) { "unsupported kana-kanji dictionary version $version" }

            val count = input.readInt()
            val readings = ArrayList<String>(count)
            val candidates = ArrayList<List<Candidate>>(count)
            repeat(count) {
                readings.add(input.readLengthPrefixedUtf8())
                val candidateCount = input.readUnsignedShort()
                val candidateList = ArrayList<Candidate>(candidateCount)
                repeat(candidateCount) {
                    val cost = input.readInt()
                    val leftId = input.readUnsignedShort()
                    val rightId = input.readUnsignedShort()
                    candidateList.add(Candidate(input.readLengthPrefixedUtf8(), cost, leftId, rightId))
                }
                candidates.add(candidateList)
            }
            val posSize = input.readUnsignedShort()
            val resolution = input.readUnsignedShort()
            val matrix = ByteArray(posSize * posSize)
            input.readFully(matrix)

            return KanaKanjiEngine(LoudsTrie.build(readings), candidates, posSize, resolution, matrix)
        }

        //reads a length (u16, byte count)-prefixed UTF-8 string, matching build_ja_dictionary.py's format
        private fun DataInputStream.readLengthPrefixedUtf8(): String {
            val len = readUnsignedShort()
            val bytes = ByteArray(len)
            readFully(bytes)
            return String(bytes, Charsets.UTF_8)
        }

        fun toKatakana(hiragana: String): String = buildString {
            for (c in hiragana) append(if (c in 'ぁ'..'ゖ') c + 0x60 else c)
        }
    }
}
