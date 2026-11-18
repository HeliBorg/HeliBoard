/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import com.android.inputmethod.latin.BinaryDictionary;
import helium314.keyboard.latin.makedict.FormatSpec.FormatOptions;
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.common.LocaleUtils;
import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.utils.Log;

/**
 * An implementation of DictEncoder for version 4 binary dictionary.
 * We probably won't need it because that's the larger and updateable format.
 */
public class Ver4DictEncoder implements DictEncoder {
    private static final String TAG = "Ver4DictEncoder";
    private final File mDictPlacedDir;
    public Ver4DictEncoder(File dictPlacedDir) {
        mDictPlacedDir = dictPlacedDir;
    }
    // TODO: This builds a FusionDictionary first and iterates it to add words to the binary
    // dictionary. However, it is possible to just add words directly to the binary dictionary
    // instead.
    // In the long run, when we stop supporting version 2, FusionDictionary will become deprecated
    // and we can remove it. Then we'll be able to just call BinaryDictionary directly.
    @Override
    public void writeDictionary(FusionDictionary dict, FormatOptions formatOptions)
        throws IOException, UnsupportedFormatException {
        if (formatOptions.mVersion != FormatSpec.VERSION4) {
            throw new UnsupportedFormatException("File header has a wrong version number : "
                + formatOptions.mVersion);
        }
        if (!mDictPlacedDir.isDirectory()) {
            throw new UnsupportedFormatException("Given path is not a directory.");
        }
        if (!BinaryDictionaryUtils.createEmptyDictFile(mDictPlacedDir.getAbsolutePath(),
            FormatSpec.VERSION4, LocaleUtils.constructLocale(dict.mOptions.mAttributes.get(DictionaryHeader.DICTIONARY_LOCALE_KEY)),
            dict.mOptions.mAttributes)) {
            throw new IOException("Cannot create dictionary file : "
                + mDictPlacedDir.getAbsolutePath());
        }
        BinaryDictionary binaryDict = new BinaryDictionary(
            mDictPlacedDir.getAbsolutePath(),
            0L,
            mDictPlacedDir.length(),
            true,
            LocaleUtils.constructLocale(dict.mOptions.mAttributes.get(DictionaryHeader.DICTIONARY_LOCALE_KEY)),
            Dictionary.TYPE_USER /* Dictionary type. Does not matter for us */,
            true /* isUpdatable */
        );
        if (!binaryDict.isValidDictionary()) {
            // Somehow createEmptyDictFile returned true, but the file was not created correctly
            throw new IOException("Cannot create dictionary file");
        }
        for (WordProperty wordProperty : dict) {
            // todo: no shortcuts
            if (!binaryDict.addUnigramEntry(wordProperty.mWord, wordProperty.getProbability(), null, 0,
                wordProperty.mIsBeginningOfSentence, wordProperty.mIsNotAWord,
                wordProperty.mIsPossiblyOffensive, 0 /* timestamp */)) {
                Log.e(TAG, "Cannot add unigram entry for " + wordProperty.mWord);
            }
            if (binaryDict.needsToRunGC(true /* mindsBlockByGC */)) {
                if (!binaryDict.flushWithGC()) {
                    Log.e(TAG, "Cannot flush dict with GC.");
                    return;
                }
            }
        }
        for (WordProperty word0Property : dict) {
            if (!word0Property.mHasNgrams) continue;
            // TODO: Support ngram.
            for (WeightedString word1 : word0Property.getBigrams()) {
                NgramContext ngramContext =
                    new NgramContext(new NgramContext.WordInfo(word0Property.mWord));
                if (!binaryDict.addNgramEntry(ngramContext, word1.mWord,
                    word1.getProbability(), 0 /* timestamp */)) {
                    Log.e(TAG, "Cannot add n-gram entry for "
                        + ngramContext + " -> " + word1.mWord);
                    return;
                }
                if (binaryDict.needsToRunGC(true /* mindsBlockByGC */)) {
                    if (!binaryDict.flushWithGC()) {
                        Log.e(TAG, "Cannot flush dict with GC.");
                        return;
                    }
                }
            }
        }
        if (!binaryDict.flushWithGC()) {
            Log.e(TAG, "Cannot flush dict with GC.");
            return;
        }
        binaryDict.close();
    }
    @Override
    public void setPosition(int position) {
    }
    @Override
    public int getPosition() {
        return 0;
    }
    @Override
    public void writePtNodeCount(int ptNodeCount) {
    }
    @Override
    public void writePtNode(FusionDictionary.PtNode ptNode, FusionDictionary dict,
                            HashMap<Integer, Integer> codePointToOneByteCodeMap) {
    }
}
