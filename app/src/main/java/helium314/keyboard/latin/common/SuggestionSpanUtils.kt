/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.common

import android.content.Context
import android.text.Spannable
import android.text.SpannableString
import android.text.Spanned
import android.text.style.SuggestionSpan
import helium314.keyboard.latin.suggestions.SuggestedWordInfo
import helium314.keyboard.latin.suggestions.SuggestedWords
import java.util.*

fun getTextWithAutoCorrectionIndicatorUnderline(context: Context?, text: String, locale: Locale?): CharSequence {
    if (text.isEmpty())
        return text
    val spannable: Spannable = SpannableString(text)
    val suggestionSpan = SuggestionSpan(context, locale, arrayOf(), SuggestionSpan.FLAG_AUTO_CORRECTION, null)
    spannable.setSpan(suggestionSpan, 0, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE or Spanned.SPAN_COMPOSING)
    return spannable
}

fun getTextWithSuggestionSpan(context: Context, pickedWord: String, suggestedWords: SuggestedWords, locale: Locale): CharSequence {
    if (pickedWord.isEmpty() || suggestedWords.isEmpty
        || suggestedWords.isPrediction || suggestedWords.isPunctuationSuggestions
    ) {
        return pickedWord
    }
    val suggestionsList = mutableListOf<String>()
    for (i in 0 until suggestedWords.size()) {
        if (suggestionsList.size >= SuggestionSpan.SUGGESTIONS_MAX_SIZE) {
            break
        }
        val info = suggestedWords.getInfo(i)
        if (info.isKindOf(SuggestedWordInfo.KIND_PREDICTION)) {
            continue
        }
        if (pickedWord != info.mWord) {
            suggestionsList.add(info.mWord)
        }
    }
    val suggestionSpan = SuggestionSpan(context, locale, suggestionsList.toTypedArray(), 0, null)
    val spannable = SpannableString(pickedWord)
    spannable.setSpan(suggestionSpan, 0, pickedWord.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
    return spannable
}
