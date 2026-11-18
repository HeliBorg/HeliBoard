/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.suggestions

import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.define.ProductionFlags
import java.util.TreeSet

/**
 * A TreeSet of SuggestedWordInfo that is bounded in size, ordered by score, and throws everything that's smaller than its capacity
 */
class SuggestionResults(
    val capacity: Int,
    val ngramContext: NgramContext,
    val firstSuggestionExceedsConfidenceThreshold: Boolean // todo: this is always false -> use it or remove it
) : TreeSet<SuggestedWordInfo>(suggestedWordInfoComparator) {
    val rawSuggestions = if (ProductionFlags.INCLUDE_RAW_SUGGESTIONS) mutableListOf<SuggestedWordInfo>() else null

    override fun add(wordInfo: SuggestedWordInfo): Boolean {
        rawSuggestions?.add(wordInfo)
        if (size < capacity) return super.add(wordInfo)
        if (comparator().compare(wordInfo, last()) > 0) return false
        super.add(wordInfo)
        pollLast() // removes the last element
        return true
    }

    companion object {
        private val suggestedWordInfoComparator = Comparator<SuggestedWordInfo> { o1, o2 ->
            if (o1.score > o2.score) return@Comparator -1
            if (o1.score < o2.score) return@Comparator 1
            if (o1.codePointCount < o2.codePointCount) return@Comparator -1
            if (o1.codePointCount > o2.codePointCount) return@Comparator 1
            return@Comparator o1.word.compareTo(o2.word)
        }
    }
}
