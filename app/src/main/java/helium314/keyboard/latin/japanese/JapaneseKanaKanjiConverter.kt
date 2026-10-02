// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.japanese

import android.content.Context
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.utils.Log
import java.io.DataInputStream

/**
 * main logic for kanji candidates to show where word suggestions normally do
 *
 * Bridges [KanaKanjiEngine] and the suggestion strip: [getSuggestions] is called from
 * Suggest.kt for the "ja" locale and returns candidates as ordinary [SuggestedWordInfo]
 *
 * loads mozc dictionary [kana_kanji.dat] and puts it into a bit vector via the LoudsTrie.kt
 */
object JapaneseKanaKanjiConverter {
    private const val TAG = "JapaneseKanaKanjiConverter"
    //! ooga-booga code monkey! this is hard coded
    private const val ASSET_PATH = "dicts/ja/kana_kanji.dat"
    private const val BASE_SCORE = 5000
    private const val SCORE_STEP = 100

    private var appContext: Context? = null
    private val sourceDict = Dictionary.PhonyDictionary("japanese_kana_kanji")
    private val engine: KanaKanjiEngine? by lazy { loadEngine() }

    fun init(context: Context) {
        appContext = context.applicationContext
    }

    // call KanaKanjiEngine for suggestions
    fun getSuggestions(reading: String): ArrayList<SuggestedWordInfo> {
        val result = ArrayList<SuggestedWordInfo>()
        if (reading.isEmpty()) return result
        val candidates = engine?.convert(reading) ?: return result
        var score = BASE_SCORE
        for (candidate in candidates) {
            result.add(
                SuggestedWordInfo(
                    candidate, "", score, SuggestedWordInfo.KIND_CORRECTION,
                    sourceDict, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE
                )
            )
            score -= SCORE_STEP
        }
        return result
    }

    //load dictionary into bit-vector
    private fun loadEngine(): KanaKanjiEngine? {
        val context = appContext ?: return null
        return try {
            context.assets.open(ASSET_PATH).use { input ->
                KanaKanjiEngine.load(DataInputStream(input.buffered()))
            }
        } catch (e: Exception) {
            Log.e(TAG, "failed to load $ASSET_PATH", e)
            null
        }
    }
}
