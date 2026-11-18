/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import com.android.inputmethod.latin.BinaryDictionary;
import java.util.ArrayList;

/**
 * An implementation of binary dictionary decoder for version 4 binary dictionary.
 */
public class Ver4DictDecoder {
     public FusionDictionary readDictionaryBinary(BinaryDictionary binaryDictionary)
            throws UnsupportedFormatException {
        // dictType is not being used in dicttool. Passing an empty string.

        DictionaryHeader header = binaryDictionary.getHeader();
        FusionDictionary fusionDict = new FusionDictionary(new FusionDictionary.PtNodeArray(), header.mDictionaryOptions);
        int token = 0;
        ArrayList<WordProperty> wordProperties = new ArrayList<>();
        do {
            BinaryDictionary.GetNextWordPropertyResult result =
                    binaryDictionary.getNextWordProperty(token);
            WordProperty wordProperty = result.mWordProperty;
            if (wordProperty == null) {
                binaryDictionary.close();
                return null;
            }
            wordProperties.add(wordProperty);
            token = result.mNextToken;
        } while (token != 0);

        // Insert unigrams into the fusion dictionary.
        for (WordProperty wordProperty : wordProperties) {
            fusionDict.add(wordProperty.mWord, wordProperty.mProbabilityInfo,
                    wordProperty.mShortcutTargets, wordProperty.mIsNotAWord,
                    wordProperty.mIsPossiblyOffensive);
        }
        // Insert bigrams into the fusion dictionary.
        // TODO: Support ngrams.
        for (WordProperty wordProperty : wordProperties) {
            if (!wordProperty.mHasNgrams) {
                continue;
            }
            String word0 = wordProperty.mWord;
            for (WeightedString bigram : wordProperty.getBigrams()) {
                fusionDict.setBigram(word0, bigram.mWord, bigram.mProbabilityInfo);
            }
        }
        binaryDictionary.close();
        return fusionDict;
    }
}
