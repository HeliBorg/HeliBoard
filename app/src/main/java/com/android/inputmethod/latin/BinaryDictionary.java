/*
 * Copyright (C) 2008 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package com.android.inputmethod.latin;

import android.text.TextUtils;
import helium314.keyboard.latin.utils.ChecksumCalculator;
import helium314.keyboard.latin.utils.Log;
import android.util.SparseArray;

import androidx.annotation.NonNull;

import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo;
import helium314.keyboard.latin.common.ComposedData;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.FileUtils;
import helium314.keyboard.latin.common.InputPointers;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.makedict.DictionaryHeader;
import helium314.keyboard.latin.makedict.FormatSpec.DictionaryOptions;
import helium314.keyboard.latin.makedict.UnsupportedFormatException;
import helium314.keyboard.latin.makedict.WordProperty;
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion;
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils;
import helium314.keyboard.latin.utils.JniUtils;
import com.android.inputmethod.latin.utils.WordInputEventForPersonalization;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Implements a static, compacted, binary dictionary of standard words.
 */
// TODO: All methods which should be locked need to have a suffix "Locked".
public final class BinaryDictionary extends Dictionary {
    private static final String TAG = BinaryDictionary.class.getSimpleName();

    // The cutoff returned by native for auto-commit confidence.
    // Must be equal to CONFIDENCE_TO_AUTO_COMMIT in native/jni/src/defines.h
    private static final int CONFIDENCE_TO_AUTO_COMMIT = 1000000;

    public static final int DICTIONARY_MAX_WORD_LENGTH = 48;
    public static final int MAX_PREV_WORD_COUNT_FOR_N_GRAM = 3;

    public static final String UNIGRAM_COUNT_QUERY = "UNIGRAM_COUNT";
    public static final String BIGRAM_COUNT_QUERY = "BIGRAM_COUNT";
    public static final String MAX_UNIGRAM_COUNT_QUERY = "MAX_UNIGRAM_COUNT";
    public static final String MAX_BIGRAM_COUNT_QUERY = "MAX_BIGRAM_COUNT";

    public static final int NOT_A_VALID_TIMESTAMP = -1;

    // Format to get unigram flags from native side via getWordPropertyNative().
    private static final int FORMAT_WORD_PROPERTY_OUTPUT_FLAG_COUNT = 5;
    private static final int FORMAT_WORD_PROPERTY_IS_NOT_A_WORD_INDEX = 0;
    private static final int FORMAT_WORD_PROPERTY_IS_POSSIBLY_OFFENSIVE_INDEX = 1;
    private static final int FORMAT_WORD_PROPERTY_HAS_NGRAMS_INDEX = 2;
    private static final int FORMAT_WORD_PROPERTY_HAS_SHORTCUTS_INDEX = 3;
    private static final int FORMAT_WORD_PROPERTY_IS_BEGINNING_OF_SENTENCE_INDEX = 4;

    // Format to get probability and historical info from native side via getWordPropertyNative().
    public static final int FORMAT_WORD_PROPERTY_OUTPUT_PROBABILITY_INFO_COUNT = 4;
    public static final int FORMAT_WORD_PROPERTY_PROBABILITY_INDEX = 0;
    public static final int FORMAT_WORD_PROPERTY_TIMESTAMP_INDEX = 1;
    public static final int FORMAT_WORD_PROPERTY_LEVEL_INDEX = 2;
    public static final int FORMAT_WORD_PROPERTY_COUNT_INDEX = 3;

    public static final String DICT_FILE_NAME_SUFFIX_FOR_MIGRATION = ".migrate";
    public static final String DIR_NAME_SUFFIX_FOR_RECORD_MIGRATION = ".migrating";

    private long mNativeDict;
    private final long mDictSize;
    private final String mDictFilePath;
    private final boolean mUseFullEditDistance;
    private final boolean mIsUpdatable;
    private boolean mHasUpdated;

    private final SparseArray<DicTraverseSession> mDicTraverseSessions = new SparseArray<>();

    // TODO: There should be a way to remove used DicTraverseSession objects from
    // {@code mDicTraverseSessions}.
    private DicTraverseSession getTraverseSession(final int traverseSessionId) {
        synchronized(mDicTraverseSessions) {
            DicTraverseSession traverseSession = mDicTraverseSessions.get(traverseSessionId);
            if (traverseSession == null) {
                traverseSession = new DicTraverseSession(mLocale, mNativeDict, mDictSize);
                mDicTraverseSessions.put(traverseSessionId, traverseSession);
            }
            return traverseSession;
        }
    }

    /**
     * Constructs binary dictionary using existing dictionary file.
     * @param filename the name of the file to read through native code.
     * @param offset the offset of the dictionary data within the file.
     * @param length the length of the binary data.
     * @param useFullEditDistance whether to use the full edit distance in suggestions
     * @param dictType the dictionary type, as a human-readable string
     * @param isUpdatable whether to open the dictionary file in writable mode.
     */
    public BinaryDictionary(final String filename, final long offset, final long length,
            final boolean useFullEditDistance, final Locale locale, final String dictType,
            final boolean isUpdatable) {
        super(dictType, locale);
        mDictSize = length;
        mDictFilePath = filename;
        mIsUpdatable = isUpdatable;
        mHasUpdated = false;
        mUseFullEditDistance = useFullEditDistance;
        loadDictionary(filename, offset, length, isUpdatable);
    }

    /**
     * Constructs binary dictionary on memory.
     * @param filename the name of the file used to flush.
     * @param useFullEditDistance whether to use the full edit distance in suggestions
     * @param dictType the dictionary type, as a human-readable string
     * @param formatVersion the format version of the dictionary
     * @param attributeMap the attributes of the dictionary
     */
    public BinaryDictionary(final String filename, final boolean useFullEditDistance,
            final Locale locale, final String dictType, final long formatVersion,
            final Map<String, String> attributeMap) {
        super(dictType, locale);
        mDictSize = 0;
        mDictFilePath = filename;
        // On memory dictionary is always updatable.
        mIsUpdatable = true;
        mHasUpdated = false;
        mUseFullEditDistance = useFullEditDistance;
        final String[] keyArray = new String[attributeMap.size()];
        final String[] valueArray = new String[attributeMap.size()];
        int index = 0;
        for (final String key : attributeMap.keySet()) {
            keyArray[index] = key;
            valueArray[index] = attributeMap.get(key);
            index++;
        }
        mNativeDict = createOnMemoryNative(formatVersion, locale.toString(), keyArray, valueArray);
    }


    static {
        JniUtils.loadNativeLibrary();
    }

    private static native long openNative(String sourceDir, long dictOffset, long dictSize,
            boolean isUpdatable);
    private static native long createOnMemoryNative(long formatVersion,
            String locale, String[] attributeKeyStringArray, String[] attributeValueStringArray);
    private static native void getHeaderInfoNative(long dict, int[] outHeaderSize,
            int[] outFormatVersion, ArrayList<int[]> outAttributeKeys,
            ArrayList<int[]> outAttributeValues);
    private static native boolean flushNative(long dict, String filePath);
    private static native boolean needsToRunGCNative(long dict, boolean mindsBlockByGC);
    private static native boolean flushWithGCNative(long dict, String filePath);
    private static native void closeNative(long dict);
    private static native int getFormatVersionNative(long dict);
    private static native int getProbabilityNative(long dict, int[] word);
    private static native int getMaxProbabilityOfExactMatchesNative(long dict, int[] word);
    private static native int getNgramProbabilityNative(long dict, int[][] prevWordCodePointArrays,
            boolean[] isBeginningOfSentenceArray, int[] word);
    private static native void getWordPropertyNative(long dict, int[] word,
            boolean isBeginningOfSentence, int[] outCodePoints, boolean[] outFlags,
            int[] outProbabilityInfo, ArrayList<int[][]> outNgramPrevWordsArray,
            ArrayList<boolean[]> outNgramPrevWordIsBeginningOfSentenceArray,
            ArrayList<int[]> outNgramTargets, ArrayList<int[]> outNgramProbabilityInfo,
            ArrayList<int[]> outShortcutTargets, ArrayList<Integer> outShortcutProbabilities);
    private static native int getNextWordNative(long dict, int token, int[] outCodePoints,
            boolean[] outIsBeginningOfSentence);
    private static native void getSuggestionsNative(long dict, long proximityInfo,
            long traverseSession, int[] xCoordinates, int[] yCoordinates, int[] times,
            int[] pointerIds, int[] inputCodePoints, int inputSize, int[] suggestOptions,
            int[][] prevWordCodePointArrays, boolean[] isBeginningOfSentenceArray,
            int prevWordCount, int[] outputSuggestionCount, int[] outputCodePoints,
            int[] outputScores, int[] outputIndices, int[] outputTypes,
            int[] outputAutoCommitFirstWordConfidence,
            float[] inOutWeightOfLangModelVsSpatialModel);
    private static native boolean addUnigramEntryNative(long dict, int[] word, int probability,
            int[] shortcutTarget, int shortcutProbability, boolean isBeginningOfSentence,
            boolean isNotAWord, boolean isPossiblyOffensive, int timestamp);
    private static native boolean removeUnigramEntryNative(long dict, int[] word);
    private static native boolean addNgramEntryNative(long dict,
            int[][] prevWordCodePointArrays, boolean[] isBeginningOfSentenceArray,
            int[] word, int probability, int timestamp);
    private static native boolean removeNgramEntryNative(long dict,
            int[][] prevWordCodePointArrays, boolean[] isBeginningOfSentenceArray, int[] word);
    private static native boolean updateEntriesForWordWithNgramContextNative(long dict,
            int[][] prevWordCodePointArrays, boolean[] isBeginningOfSentenceArray,
            int[] word, boolean isValidWord, int count, int timestamp);
    private static native int updateEntriesForInputEventsNative(long dict,
            WordInputEventForPersonalization[] inputEvents, int startIndex);
    private static native String getPropertyNative(long dict, String query);
    private static native boolean isCorruptedNative(long dict);
    private static native boolean migrateNative(long dict, String dictFilePath,
            long newFormatVersion);

    // TODO: Move native dict into session
    private void loadDictionary(final String path, final long startOffset,
            final long length, final boolean isUpdatable) {
        mHasUpdated = false;
        mNativeDict = openNative(path, startOffset, length, isUpdatable);
    }

    // TODO: Check isCorrupted() for main dictionaries.
    public boolean isCorrupted() {
        if (!isValidDictionary()) {
            return false;
        }
        if (!isCorruptedNative(mNativeDict)) {
            return false;
        }
        // TODO: Record the corruption.
        Log.e(TAG, "BinaryDictionary (" + mDictFilePath + ") is corrupted.");
        Log.e(TAG, "locale: " + mLocale);
        Log.e(TAG, "dict size: " + mDictSize);
        Log.e(TAG, "updatable: " + mIsUpdatable);
        return true;
    }

    public DictionaryHeader getHeader() throws UnsupportedFormatException {
        if (mNativeDict == 0) {
            return null;
        }
        final int[] outHeaderSize = new int[1];
        final int[] outFormatVersion = new int[1];
        final ArrayList<int[]> outAttributeKeys = new ArrayList<>();
        final ArrayList<int[]> outAttributeValues = new ArrayList<>();
        getHeaderInfoNative(mNativeDict, outHeaderSize, outFormatVersion, outAttributeKeys,
                outAttributeValues);
        final HashMap<String, String> attributes = new HashMap<>();
        for (int i = 0; i < outAttributeKeys.size(); i++) {
            final String attributeKey = StringUtils.getStringFromNullTerminatedCodePointArray(
                    outAttributeKeys.get(i));
            final String attributeValue = StringUtils.getStringFromNullTerminatedCodePointArray(
                    outAttributeValues.get(i));
            attributes.put(attributeKey, attributeValue);
        }
        return new DictionaryHeader(new DictionaryOptions(attributes));
    }

    @Override
    public ArrayList<SuggestedWordInfo> getSuggestions(final ComposedData composedData,
            final NgramContext ngramContext, final long proximityInfoHandle,
            final SettingsValuesForSuggestion settingsValuesForSuggestion,
            final int sessionId, final float weightForLocale,
            final float[] inOutWeightOfLangModelVsSpatialModel) {
        if (!isValidDictionary()) {
            return null;
        }
        final DicTraverseSession session = getTraverseSession(sessionId);
        Arrays.fill(session.mInputCodePoints, Constants.NOT_A_CODE);
        ngramContext.outputToArray(session.mPrevWordCodePointArrays,
                session.mIsBeginningOfSentenceArray);
        final InputPointers inputPointers = composedData.mInputPointers;
        final boolean isGesture = composedData.mIsBatchMode;
        final int inputSize;
        if (!isGesture) {
            inputSize =
                    composedData.copyCodePointsExceptTrailingSingleQuotesAndReturnCodePointCount(
                        session.mInputCodePoints);
            if (inputSize < 0) {
                return null;
            }
        } else {
            inputSize = inputPointers.getPointerSize();
        }
        session.mNativeSuggestOptions.setUseFullEditDistance(mUseFullEditDistance);
        session.mNativeSuggestOptions.setIsGesture(isGesture);
        if (isGesture)
            session.mNativeSuggestOptions.setIsSpaceAwareGesture(settingsValuesForSuggestion.mSpaceAwareGesture);
        session.mNativeSuggestOptions.setBlockOffensiveWords(settingsValuesForSuggestion.mBlockPotentiallyOffensive);
        session.mNativeSuggestOptions.setWeightForLocale(weightForLocale);
        if (inOutWeightOfLangModelVsSpatialModel != null) {
            session.mInputOutputWeightOfLangModelVsSpatialModel[0] =
                    inOutWeightOfLangModelVsSpatialModel[0];
        } else {
            session.mInputOutputWeightOfLangModelVsSpatialModel[0] =
                    Dictionary.NOT_A_WEIGHT_OF_LANG_MODEL_VS_SPATIAL_MODEL;
        }
        // TOOD: Pass multiple previous words information for n-gram.
        getSuggestionsNative(mNativeDict, proximityInfoHandle,
                getTraverseSession(sessionId).getSession(), inputPointers.getXCoordinates(),
                inputPointers.getYCoordinates(), inputPointers.getTimes(),
                inputPointers.getPointerIds(), session.mInputCodePoints, inputSize,
                session.mNativeSuggestOptions.getOptions(), session.mPrevWordCodePointArrays,
                session.mIsBeginningOfSentenceArray, ngramContext.getPrevWordCount(),
                session.mOutputSuggestionCount, session.mOutputCodePoints, session.mOutputScores,
                session.mSpaceIndices, session.mOutputTypes,
                session.mOutputAutoCommitFirstWordConfidence,
                session.mInputOutputWeightOfLangModelVsSpatialModel);
        if (inOutWeightOfLangModelVsSpatialModel != null) {
            inOutWeightOfLangModelVsSpatialModel[0] =
                    session.mInputOutputWeightOfLangModelVsSpatialModel[0];
        }
        final int count = session.mOutputSuggestionCount[0];
        final ArrayList<SuggestedWordInfo> suggestions = new ArrayList<>();
        for (int j = 0; j < count; ++j) {
            final int start = j * DICTIONARY_MAX_WORD_LENGTH;
            int len = 0;
            while (len < DICTIONARY_MAX_WORD_LENGTH
                    && session.mOutputCodePoints[start + len] != 0) {
                ++len;
            }
            if (len > 0) {
                suggestions.add(new SuggestedWordInfo(
                        new String(session.mOutputCodePoints, start, len),
                        "" /* prevWordsContext */,
                        (int)(session.mOutputScores[j] * weightForLocale),
                        session.mOutputTypes[j],
                        this /* sourceDict */,
                        session.mSpaceIndices[j] /* indexOfTouchPointOfSecondWord */,
                        session.mOutputAutoCommitFirstWordConfidence[0]));
            }
        }
        return suggestions;
    }

    public boolean isValidDictionary() {
        return mNativeDict != 0;
    }

    public int getFormatVersion() {
        return getFormatVersionNative(mNativeDict);
    }

    @Override
    public boolean isInDictionary(final String word) {
        return getFrequency(word) != NOT_A_PROBABILITY;
    }

    @Override
    public int getFrequency(final String word) {
        if (TextUtils.isEmpty(word)) {
            return NOT_A_PROBABILITY;
        }
        final int[] codePoints = StringUtils.toCodePointArray(word);
        return getProbabilityNative(mNativeDict, codePoints);
    }

    @Override
    public int getMaxFrequencyOfExactMatches(final String word) {
        if (TextUtils.isEmpty(word)) {
            return NOT_A_PROBABILITY;
        }
        final int[] codePoints = StringUtils.toCodePointArray(word);
        return getMaxProbabilityOfExactMatchesNative(mNativeDict, codePoints);
    }

    public boolean isValidNgram(final NgramContext ngramContext, final String word) {
        return getNgramProbability(ngramContext, word) != NOT_A_PROBABILITY;
    }

    public int getNgramProbability(final NgramContext ngramContext, final String word) {
        if (!ngramContext.isValid() || TextUtils.isEmpty(word)) {
            return NOT_A_PROBABILITY;
        }
        final int[][] prevWordCodePointArrays = new int[ngramContext.getPrevWordCount()][];
        final boolean[] isBeginningOfSentenceArray = new boolean[ngramContext.getPrevWordCount()];
        ngramContext.outputToArray(prevWordCodePointArrays, isBeginningOfSentenceArray);
        final int[] wordCodePoints = StringUtils.toCodePointArray(word);
        return getNgramProbabilityNative(mNativeDict, prevWordCodePointArrays,
                isBeginningOfSentenceArray, wordCodePoints);
    }

    public WordProperty getWordProperty(final String word, final boolean isBeginningOfSentence) {
        if (word == null) {
            return null;
        }
        final int[] codePoints = StringUtils.toCodePointArray(word);
        final int[] outCodePoints = new int[DICTIONARY_MAX_WORD_LENGTH];
        final boolean[] outFlags = new boolean[FORMAT_WORD_PROPERTY_OUTPUT_FLAG_COUNT];
        final int[] outProbabilityInfo =
                new int[FORMAT_WORD_PROPERTY_OUTPUT_PROBABILITY_INFO_COUNT];
        final ArrayList<int[][]> outNgramPrevWordsArray = new ArrayList<>();
        final ArrayList<boolean[]> outNgramPrevWordIsBeginningOfSentenceArray =
                new ArrayList<>();
        final ArrayList<int[]> outNgramTargets = new ArrayList<>();
        final ArrayList<int[]> outNgramProbabilityInfo = new ArrayList<>();
        final ArrayList<int[]> outShortcutTargets = new ArrayList<>();
        final ArrayList<Integer> outShortcutProbabilities = new ArrayList<>();
        getWordPropertyNative(mNativeDict, codePoints, isBeginningOfSentence, outCodePoints,
                outFlags, outProbabilityInfo, outNgramPrevWordsArray,
                outNgramPrevWordIsBeginningOfSentenceArray, outNgramTargets,
                outNgramProbabilityInfo, outShortcutTargets, outShortcutProbabilities);
        return new WordProperty(codePoints,
                outFlags[FORMAT_WORD_PROPERTY_IS_NOT_A_WORD_INDEX],
                outFlags[FORMAT_WORD_PROPERTY_IS_POSSIBLY_OFFENSIVE_INDEX],
                outFlags[FORMAT_WORD_PROPERTY_HAS_NGRAMS_INDEX],
                outFlags[FORMAT_WORD_PROPERTY_HAS_SHORTCUTS_INDEX],
                outFlags[FORMAT_WORD_PROPERTY_IS_BEGINNING_OF_SENTENCE_INDEX], outProbabilityInfo,
                outNgramPrevWordsArray, outNgramPrevWordIsBeginningOfSentenceArray,
                outNgramTargets, outNgramProbabilityInfo, outShortcutTargets,
                outShortcutProbabilities);
    }

    public static class GetNextWordPropertyResult {
        public WordProperty mWordProperty;
        public int mNextToken;

        public GetNextWordPropertyResult(final WordProperty wordProperty, final int nextToken) {
            mWordProperty = wordProperty;
            mNextToken = nextToken;
        }
    }

    /**
     * Method to iterate all words in the dictionary for makedict.
     * If token is 0, this method newly starts iterating the dictionary.
     */
    public GetNextWordPropertyResult getNextWordProperty(final int token) {
        final int[] codePoints = new int[DICTIONARY_MAX_WORD_LENGTH];
        final boolean[] isBeginningOfSentence = new boolean[1];
        final int nextToken = getNextWordNative(mNativeDict, token, codePoints,
                isBeginningOfSentence);
        final String word = StringUtils.getStringFromNullTerminatedCodePointArray(codePoints);
        return new GetNextWordPropertyResult(
                getWordProperty(word, isBeginningOfSentence[0]), nextToken);
    }

    // Add a unigram entry to binary dictionary with unigram attributes in native code.
    public boolean addUnigramEntry(final String word, final int probability,
            final String shortcutTarget, final int shortcutProbability,
            final boolean isBeginningOfSentence, final boolean isNotAWord,
            final boolean isPossiblyOffensive, final int timestamp) {
        if (word == null || (word.isEmpty() && !isBeginningOfSentence)) {
            return false;
        }
        final int[] codePoints = StringUtils.toCodePointArray(word);
        final int[] shortcutTargetCodePoints = (shortcutTarget != null) ?
                StringUtils.toCodePointArray(shortcutTarget) : null;
        if (!addUnigramEntryNative(mNativeDict, codePoints, probability, shortcutTargetCodePoints,
                shortcutProbability, isBeginningOfSentence, isNotAWord, isPossiblyOffensive,
                timestamp)) {
            return false;
        }
        mHasUpdated = true;
        return true;
    }

    // Remove a unigram entry from the binary dictionary in native code.
    public boolean removeUnigramEntry(final String word) {
        if (TextUtils.isEmpty(word)) {
            return false;
        }
        final int[] codePoints = StringUtils.toCodePointArray(word);
        if (!removeUnigramEntryNative(mNativeDict, codePoints)) {
            return false;
        }
        mHasUpdated = true;
        return true;
    }

    // Add an n-gram entry to the binary dictionary with timestamp in native code.
    public boolean addNgramEntry(final NgramContext ngramContext, final String word,
            final int probability, final int timestamp) {
        if (!ngramContext.isValid() || TextUtils.isEmpty(word)) {
            return false;
        }
        final int[][] prevWordCodePointArrays = new int[ngramContext.getPrevWordCount()][];
        final boolean[] isBeginningOfSentenceArray = new boolean[ngramContext.getPrevWordCount()];
        ngramContext.outputToArray(prevWordCodePointArrays, isBeginningOfSentenceArray);
        final int[] wordCodePoints = StringUtils.toCodePointArray(word);
        if (!addNgramEntryNative(mNativeDict, prevWordCodePointArrays,
                isBeginningOfSentenceArray, wordCodePoints, probability, timestamp)) {
            return false;
        }
        mHasUpdated = true;
        return true;
    }

    // Update entries for the word occurrence with the ngramContext.
    public boolean updateEntriesForWordWithNgramContext(@NonNull final NgramContext ngramContext,
            final String word, final boolean isValidWord, final int count, final int timestamp) {
        if (TextUtils.isEmpty(word)) {
            return false;
        }
        final int[][] prevWordCodePointArrays = new int[ngramContext.getPrevWordCount()][];
        final boolean[] isBeginningOfSentenceArray = new boolean[ngramContext.getPrevWordCount()];
        ngramContext.outputToArray(prevWordCodePointArrays, isBeginningOfSentenceArray);
        final int[] wordCodePoints = StringUtils.toCodePointArray(word);
        if (!updateEntriesForWordWithNgramContextNative(mNativeDict, prevWordCodePointArrays,
                isBeginningOfSentenceArray, wordCodePoints, isValidWord, count, timestamp)) {
            return false;
        }
        mHasUpdated = true;
        return true;
    }

    public void updateEntriesForInputEvents(final WordInputEventForPersonalization[] inputEvents) {
        if (!isValidDictionary()) {
            return;
        }
        int processedEventCount = 0;
        while (processedEventCount < inputEvents.length) {
            if (needsToRunGC(true /* mindsBlockByGC */)) {
                flushWithGC();
            }
            processedEventCount = updateEntriesForInputEventsNative(mNativeDict, inputEvents,
                    processedEventCount);
            mHasUpdated = true;
            if (processedEventCount <= 0) {
                return;
            }
        }
    }

    private void reopen() {
        close();
        final File dictFile = new File(mDictFilePath);
        // WARNING: Because we pass 0 as the offset and file.length() as the length, this can
        // only be called for actual files. Right now it's only called by the flush() family of
        // functions, which require an updatable dictionary, so it's okay. But beware.
        loadDictionary(dictFile.getAbsolutePath(), 0 /* startOffset */,
                dictFile.length(), mIsUpdatable);
    }

    // Flush to dict file if the dictionary has been updated.
    public boolean flush() {
        if (!isValidDictionary()) {
            return false;
        }
        if (mHasUpdated) {
            if (!flushNative(mNativeDict, mDictFilePath)) {
                return false;
            }
            reopen();
        }
        return true;
    }

    // Run GC and flush to dict file if the dictionary has been updated.
    public boolean flushWithGCIfHasUpdated() {
        if (mHasUpdated) {
            return flushWithGC();
        }
        return true;
    }

    // Run GC and flush to dict file.
    public boolean flushWithGC() {
        if (!isValidDictionary()) {
            return false;
        }
        if (!flushWithGCNative(mNativeDict, mDictFilePath)) {
            return false;
        }
        reopen();
        return true;
    }

    /**
     * Checks whether GC is needed to run or not.
     * @param mindsBlockByGC Whether to mind operations blocked by GC. We don't need to care about
     * the blocking in some situations such as in idle time or just before closing.
     * @return whether GC is needed to run or not.
     */
    public boolean needsToRunGC(final boolean mindsBlockByGC) {
        if (!isValidDictionary()) {
            return false;
        }
        return needsToRunGCNative(mNativeDict, mindsBlockByGC);
    }

    public boolean migrateTo(final int newFormatVersion) {
        if (!isValidDictionary()) {
            return false;
        }
        final File isMigratingDir =
                new File(mDictFilePath + DIR_NAME_SUFFIX_FOR_RECORD_MIGRATION);
        if (isMigratingDir.exists()) {
            isMigratingDir.delete();
            Log.e(TAG, "Previous migration attempt failed probably due to a crash. "
                        + "Giving up using the old dictionary (" + mDictFilePath + ").");
            return false;
        }
        if (!isMigratingDir.mkdir()) {
            Log.e(TAG, "Cannot create a dir (" + isMigratingDir.getAbsolutePath()
                    + ") to record migration.");
            return false;
        }
        try {
            final String tmpDictFilePath = mDictFilePath + DICT_FILE_NAME_SUFFIX_FOR_MIGRATION;
            if (!migrateNative(mNativeDict, tmpDictFilePath, newFormatVersion)) {
                return false;
            }
            close();
            final File dictFile = new File(mDictFilePath);
            final File tmpDictFile = new File(tmpDictFilePath);
            if (!FileUtils.deleteRecursively(dictFile)) {
                return false;
            }
            if (!BinaryDictionaryUtils.renameDict(tmpDictFile, dictFile)) {
                return false;
            }
            loadDictionary(dictFile.getAbsolutePath(), 0 /* startOffset */,
                    dictFile.length(), mIsUpdatable);
            return true;
        } finally {
            isMigratingDir.delete();
        }
    }

    public String getPropertyForGettingStats(final String query) {
        if (!isValidDictionary()) {
            return "";
        }
        return getPropertyNative(mNativeDict, query);
    }

    @Override
    public boolean shouldAutoCommit(final SuggestedWordInfo candidate) {
        return candidate.mAutoCommitFirstWordConfidence > CONFIDENCE_TO_AUTO_COMMIT;
    }

    @Override
    public void close() {
        synchronized (mDicTraverseSessions) {
            final int sessionsSize = mDicTraverseSessions.size();
            for (int index = 0; index < sessionsSize; ++index) {
                final DicTraverseSession traverseSession = mDicTraverseSessions.valueAt(index);
                if (traverseSession != null) {
                    traverseSession.close();
                }
            }
            mDicTraverseSessions.clear();
        }
        closeInternalLocked();
    }

    private synchronized void closeInternalLocked() {
        if (mNativeDict != 0) {
            closeNative(mNativeDict);
            mNativeDict = 0;
        }
    }

    private String mDictFileHash;
    public String getHash() {
        if (mDictFileHash != null) return mDictFileHash;
        final File dict = new File(mDictFilePath);
        if (!dict.isFile()) {
            mDictFileHash = "";
            return mDictFileHash;
        }
        mDictFileHash = ChecksumCalculator.INSTANCE.checksum(dict);
        if (mDictFileHash == null)
            mDictFileHash = "";
        return mDictFileHash;
    }

    // TODO: Manage BinaryDictionary instances without using WeakReference or something.
    @Override
    protected void finalize() throws Throwable {
        try {
            closeInternalLocked();
        } finally {
            super.finalize();
        }
    }
}
