/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.dictionary;

import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo;
import helium314.keyboard.latin.common.ComposedData;
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion;
import helium314.keyboard.latin.utils.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Class for a collection of dictionaries that behave like one dictionary.
 */
public final class DictionaryCollection extends Dictionary {
    private final String TAG = DictionaryCollection.class.getSimpleName();
    private final ArrayList<Dictionary> mDictionaries;
    private final float[] mWeights;

    public DictionaryCollection(final String dictType, final Locale locale,
            final Collection<Dictionary> dictionaries, final float[] weights) {
        super(dictType, locale);
        mDictionaries = new ArrayList<>(dictionaries);
        mDictionaries.removeAll(Collections.singleton(null));
        if (mDictionaries.size() > weights.length) {
            mWeights = new float[mDictionaries.size()];
            Arrays.fill(mWeights, 1f);
            Log.w(TAG, "got weights array of length " + weights.length + ", expected "+mDictionaries.size());
        } else mWeights = weights;
    }

    @Override
    public ArrayList<SuggestedWordInfo> getSuggestions(final ComposedData composedData,
            final NgramContext ngramContext, final long proximityInfoHandle,
            final SettingsValuesForSuggestion settingsValuesForSuggestion,
            final int sessionId, final float weightForLocale,
            final float[] inOutWeightOfLangModelVsSpatialModel) {
        final ArrayList<Dictionary> dictionaries = mDictionaries;
        if (dictionaries.isEmpty()) return null;
        // To avoid creating unnecessary objects, we get the list out of the first
        // dictionary and add the rest to it if not null, hence the get(0)
        ArrayList<SuggestedWordInfo> suggestions = dictionaries.get(0).getSuggestions(composedData,
                ngramContext, proximityInfoHandle, settingsValuesForSuggestion, sessionId,
                weightForLocale * mWeights[0], inOutWeightOfLangModelVsSpatialModel);
        if (null == suggestions) suggestions = new ArrayList<>();
        final int length = dictionaries.size();
        for (int i = 1; i < length; ++ i) {
            final ArrayList<SuggestedWordInfo> sugg = dictionaries.get(i).getSuggestions(
                    composedData, ngramContext, proximityInfoHandle, settingsValuesForSuggestion,
                    sessionId, weightForLocale * mWeights[i], inOutWeightOfLangModelVsSpatialModel);
            if (null != sugg) suggestions.addAll(sugg);
        }
        return suggestions;
    }

    @Override
    public boolean isInDictionary(final String word) {
        for (int i = mDictionaries.size() - 1; i >= 0; --i)
            if (mDictionaries.get(i).isInDictionary(word)) return true;
        return false;
    }

    @Override
    public int getFrequency(final String word) {
        int maxFreq = -1;
        for (int i = mDictionaries.size() - 1; i >= 0; --i) {
            final int tempFreq = mDictionaries.get(i).getFrequency(word);
            maxFreq = Math.max(tempFreq, maxFreq);
        }
        return maxFreq;
    }

    @Override
    public int getMaxFrequencyOfExactMatches(final String word) {
        int maxFreq = -1;
        for (int i = mDictionaries.size() - 1; i >= 0; --i) {
            final int tempFreq = mDictionaries.get(i).getMaxFrequencyOfExactMatches(word);
            maxFreq = Math.max(tempFreq, maxFreq);
        }
        return maxFreq;
    }

    @Override
    @androidx.annotation.NonNull
    public Map<String, Integer> getAllWordsWithFrequency() {
        Map<String, Integer> result = new HashMap<>();
        for (Dictionary dict : mDictionaries) result.putAll(dict.getAllWordsWithFrequency());
        return result;
    }

    @Override
    public boolean isInitialized() {
        return !mDictionaries.isEmpty();
    }

    @Override
    public void close() {
        for (final Dictionary dict : mDictionaries)
            dict.close();
    }
}
