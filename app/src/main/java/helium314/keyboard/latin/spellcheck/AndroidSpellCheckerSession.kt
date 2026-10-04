/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.spellcheck

import android.os.Binder
import android.text.TextUtils
import android.view.textservice.SentenceSuggestionsInfo
import android.view.textservice.SuggestionsInfo
import android.view.textservice.TextInfo
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.NgramContext.WordInfo
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.SpannableStringUtils

class AndroidSpellCheckerSession(service: AndroidSpellCheckerService) : AndroidWordLevelSpellCheckerSession(service) {
    private val mResources = service.resources
    private var mSentenceLevelAdapter: SentenceLevelAdapter? = null

    private fun fixWronglyInvalidatedWordWithSingleQuote(textInfo: TextInfo, sentenceSuggestionsInfo: SentenceSuggestionsInfo): SentenceSuggestionsInfo? {
        val typedText = textInfo.charSequence
        if (!typedText.toString().contains(AndroidSpellCheckerService.SINGLE_QUOTE)) {
            return null
        }
        val count = sentenceSuggestionsInfo.suggestionsCount
        val additionalOffsets = ArrayList<Int>()
        val additionalLengths = ArrayList<Int>()
        val additionalSuggestionsInfos = ArrayList<SuggestionsInfo?>()
        for (i in 0..<count) {
            val suggestionsInfo = sentenceSuggestionsInfo.getSuggestionsInfoAt(i)
            val flags = suggestionsInfo.suggestionsAttributes
            if ((flags and SuggestionsInfo.RESULT_ATTR_IN_THE_DICTIONARY) == 0) {
                continue
            }
            val offset = sentenceSuggestionsInfo.getOffsetAt(i)
            val length = sentenceSuggestionsInfo.getLengthAt(i)
            val subText = typedText.subSequence(offset, offset + length)
            if (!subText.toString().contains(AndroidSpellCheckerService.SINGLE_QUOTE)) {
                continue
            }
            // Split preserving spans.
            val splitTexts = SpannableStringUtils.split(subText,
                AndroidSpellCheckerService.SINGLE_QUOTE, true)
            if (splitTexts == null || splitTexts.size <= 1) {
                continue
            }
            for (splitText in splitTexts) {
                if (TextUtils.isEmpty(splitText)) {
                    continue
                }
                if (suggestionsCache.getSuggestionsFromCache(splitText.toString()) == null) {
                    continue
                }
                val newLength = splitText!!.length
                // Neither RESULT_ATTR_IN_THE_DICTIONARY nor RESULT_ATTR_LOOKS_LIKE_TYPO
                val newFlags = 0
                val newSuggestionsInfo = SuggestionsInfo(newFlags, EMPTY_STRING_ARRAY)
                newSuggestionsInfo.setCookieAndSequence(suggestionsInfo.cookie, suggestionsInfo.sequence)
                if (DBG)
                    Log.d(TAG, ("Override and remove old span over: $splitText, $offset,$newLength"))
                additionalOffsets.add(offset)
                additionalLengths.add(newLength)
                additionalSuggestionsInfos.add(newSuggestionsInfo)
            }
        }
        val additionalSize = additionalOffsets.size
        if (additionalSize == 0) {
            return null
        }
        val suggestionsSize = count + additionalSize
        val newOffsets = IntArray(suggestionsSize)
        val newLengths = IntArray(suggestionsSize)
        val newSuggestionsInfos = arrayOfNulls<SuggestionsInfo>(suggestionsSize)
        var i = 0
        while (i < count) {
            newOffsets[i] = sentenceSuggestionsInfo.getOffsetAt(i)
            newLengths[i] = sentenceSuggestionsInfo.getLengthAt(i)
            newSuggestionsInfos[i] = sentenceSuggestionsInfo.getSuggestionsInfoAt(i)
            ++i
        }
        while (i < suggestionsSize) {
            newOffsets[i] = additionalOffsets[i - count]
            newLengths[i] = additionalLengths[i - count]
            newSuggestionsInfos[i] = additionalSuggestionsInfos[i - count]
            ++i
        }
        return SentenceSuggestionsInfo(newSuggestionsInfos, newOffsets, newLengths)
    }

    override fun onGetSentenceSuggestionsMultiple(textInfos: Array<TextInfo?>, suggestionsLimit: Int): Array<SentenceSuggestionsInfo?>? {
        val retval = splitAndSuggest(textInfos, suggestionsLimit)
        if (retval == null || retval.size != textInfos.size) {
            return retval
        }
        for (i in retval.indices) {
            val tempSsi = fixWronglyInvalidatedWordWithSingleQuote(textInfos[i]!!, retval[i]!!)
            if (tempSsi != null) {
                retval[i] = tempSsi
            }
        }
        return retval
    }

    /**
     * Get sentence suggestions for specified texts in an array of TextInfo. This is taken from
     * SpellCheckerService#onGetSentenceSuggestionsMultiple that we can't use because it's
     * using private variables.
     * The default implementation splits the input text to words and returns
     * [SentenceSuggestionsInfo] which contains suggestions for each word.
     * This function will run on the incoming IPC thread.
     * So, this is not called on the main thread,
     * but will be called in series on another thread.
     * @param textInfos an array of the text metadata
     * @param suggestionsLimit the maximum number of suggestions to be returned
     * @return an array of [SentenceSuggestionsInfo] returned by
     * [android.service.textservice.SpellCheckerService.Session.onGetSuggestions]
     */
    private fun splitAndSuggest(textInfos: Array<TextInfo?>?, suggestionsLimit: Int): Array<SentenceSuggestionsInfo?>? {
        if (textInfos.isNullOrEmpty()) {
            return SentenceLevelAdapter.getEmptySentenceSuggestionsInfo()
        }
        var sentenceLevelAdapter: SentenceLevelAdapter?
        synchronized(this) {
            sentenceLevelAdapter = mSentenceLevelAdapter
            if (sentenceLevelAdapter == null) {
                val localeString = getLocale()
                if (!TextUtils.isEmpty(localeString)) {
                    sentenceLevelAdapter = SentenceLevelAdapter(mResources, localeString!!.constructLocale())
                    mSentenceLevelAdapter = sentenceLevelAdapter
                }
            }
        }
        if (sentenceLevelAdapter == null) {
            return SentenceLevelAdapter.getEmptySentenceSuggestionsInfo()
        }
        val infosSize = textInfos.size
        val retval = arrayOfNulls<SentenceSuggestionsInfo>(infosSize)
        for (i in 0..<infosSize) {
            val textInfoParams = sentenceLevelAdapter.getSplitWords(textInfos[i])
            val mItems = textInfoParams.mItems
            val itemsSize = mItems.size
            val splitTextInfos = Array(itemsSize) { mItems[it].mTextInfo }
            retval[i] = SentenceLevelAdapter.reconstructSuggestions(
                textInfoParams, onGetSuggestionsMultiple(
                    splitTextInfos, suggestionsLimit, true
                )
            )
        }
        return retval
    }

    override fun onGetSuggestionsMultiple(textInfos: Array<TextInfo>, suggestionsLimit: Int, sequentialWords: Boolean): Array<SuggestionsInfo?> {
        val ident = Binder.clearCallingIdentity()
        try {
            return Array(textInfos.size) { i ->
                val prevWord: CharSequence?
                if (sequentialWords && i > 0) {
                    val prevTextInfo = textInfos[i - 1]
                    val prevWordCandidate = prevTextInfo.charSequence
                    // Note that an empty string would be used to indicate the initial word in the future.
                    prevWord = if (TextUtils.isEmpty(prevWordCandidate)) null else prevWordCandidate
                } else {
                    prevWord = null
                }
                val ngramContext =
                    NgramContext(WordInfo(prevWord))
                val textInfo = textInfos[i]
                val suggestions = onGetSuggestionsInternal(textInfo, ngramContext, suggestionsLimit)
                suggestions.setCookieAndSequence(textInfo.cookie, textInfo.sequence)
                suggestions
            }
        } finally {
            Binder.restoreCallingIdentity(ident)
        }
    }

    companion object {
        private val TAG: String = AndroidSpellCheckerSession::class.java.simpleName
        private const val DBG = false
    }
}
