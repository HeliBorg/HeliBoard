/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.dictionary;

import android.content.Context;
import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.utils.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.android.inputmethod.latin.BinaryDictionary;

import helium314.keyboard.latin.common.ComposedData;
import helium314.keyboard.latin.common.FileUtils;
import helium314.keyboard.latin.define.DecoderSpecificConstants;
import helium314.keyboard.latin.makedict.DictionaryHeader;
import helium314.keyboard.latin.makedict.FormatSpec;
import helium314.keyboard.latin.makedict.WordProperty;
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion;
import helium314.keyboard.latin.suggestions.SuggestedWordInfo;
import helium314.keyboard.latin.utils.AsyncResultHolder;
import helium314.keyboard.latin.utils.ExecutorUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Abstract base class for an expandable dictionary that can be created and updated dynamically
 * during runtime. When updated it automatically generates a new binary dictionary to handle future
 * queries in native code. This binary dictionary is written to internal storage.
 * <p>
 * A class that extends this abstract class must have a static factory method named
 *   getDictionary(Context context, Locale locale, File dictFile, String dictNamePrefix)
 */
abstract public class ExpandableBinaryDictionary extends Dictionary {
    private static final boolean DEBUG = false;

    /** Used for Log actions from this class */
    private static final String TAG = ExpandableBinaryDictionary.class.getSimpleName();

    /** Whether to print debug output to log */
    private static final boolean DBG_STRESS_TEST = false;

    private static final int TIMEOUT_FOR_READ_OPS_IN_MILLISECONDS = 100;

    /**
     * The maximum length of a word in this dictionary.
     */
    protected static final int MAX_WORD_LENGTH =
            DecoderSpecificConstants.DICTIONARY_MAX_WORD_LENGTH;

    private static final int DICTIONARY_FORMAT_VERSION = FormatSpec.VERSION4;

    private static final WordProperty[] DEFAULT_WORD_PROPERTIES_FOR_SYNC =
            new WordProperty[0] /* default */;

    /** The application context. */
    public final Context mContext;

    /**
     * The binary dictionary generated dynamically from the fusion dictionary. This is used to
     * answer unigram and bigram queries.
     */
    private BinaryDictionary mBinaryDictionary;

    /**
     * The name of this dictionary, used as a part of the filename for storing the binary
     * dictionary.
     */
    private final String mDictName;

    /** Dictionary file */
    private final File mDictFile;

    /** Indicates whether a task for reloading the dictionary has been scheduled. */
    private final AtomicBoolean mIsReloading;

    private final ReentrantReadWriteLock mLock;

    /* A extension for a binary dictionary file. */
    protected static final String DICT_FILE_EXTENSION = ".dict";

    /**
     * Abstract method for loading initial contents of a given dictionary.
     */
    protected abstract void loadInitialContentsLocked();

    private static boolean needsToMigrateDictionary(final int formatVersion) {
        // When we bump up the dictionary format version, the old version should be added to here
        // for supporting migration. Note that native code has to support reading such formats.
        return formatVersion == FormatSpec.VERSION402;
    }

    public boolean isValidDictionaryLocked() {
        return mBinaryDictionary.isValidDictionary();
    }

    /**
     * Creates a new expandable binary dictionary.
     *
     * @param context The application context of the parent.
     * @param dictName The name of the dictionary. Multiple instances with the same
     *        name is supported.
     * @param locale the dictionary locale.
     * @param dictType the dictionary type, as a human-readable string
     * @param dictFile dictionary file path. if null, use default dictionary path based on
     *        dictionary type.
     */
    public ExpandableBinaryDictionary(final Context context, final String dictName,
            final Locale locale, final String dictType, final File dictFile) {
        super(dictType, locale);
        mDictName = dictName;
        mContext = context;
        mDictFile = getDictFile(context, dictName, dictFile);
        mBinaryDictionary = null;
        mIsReloading = new AtomicBoolean();
        mLock = new ReentrantReadWriteLock();
    }

    public static File getDictFile(final Context context, final String dictName,
            final File dictFile) {
        return (dictFile != null) ? dictFile
                : new File(context.getFilesDir(), dictName + DICT_FILE_EXTENSION);
    }

    public static String getDictName(final String name, final Locale locale,
            final File dictFile) {
        return dictFile != null ? dictFile.getName() : name + "." + locale.toLanguageTag();
    }

    private void asyncExecuteTaskWithWriteLock(final Runnable task) {
        asyncExecuteTaskWithLock(mLock.writeLock(), task);
    }

    private static void asyncExecuteTaskWithLock(final Lock lock, final Runnable task) {
        ExecutorUtils.getBackgroundExecutor(ExecutorUtils.KEYBOARD).execute(() -> {
            lock.lock();
            try {
                task.run();
            } finally {
                lock.unlock();
            }
        });
    }

    @Nullable
    BinaryDictionary getBinaryDictionary() {
        return mBinaryDictionary;
    }

    @Override
    public int getFrequency(final String word) {
        if (mLock.readLock().tryLock()) {
            try {
                return mBinaryDictionary.getFrequency(word);
            } finally {
                mLock.readLock().unlock();
            }
        }
        return NOT_A_PROBABILITY;
    }

    void closeBinaryDictionary() {
        if (mBinaryDictionary != null) {
            mBinaryDictionary.close();
            mBinaryDictionary = null;
        }
    }

    /**
     * Closes and cleans up the binary dictionary.
     */
    @Override
    public void close() {
        asyncExecuteTaskWithWriteLock(this::closeBinaryDictionary);
    }

    protected Map<String, String> getHeaderAttributeMap() {
        HashMap<String, String> attributeMap = new HashMap<>();
        attributeMap.put(DictionaryHeader.DICTIONARY_ID_KEY, mDictName);
        attributeMap.put(DictionaryHeader.DICTIONARY_LOCALE_KEY, mLocale.toString());
        attributeMap.put(DictionaryHeader.DICTIONARY_VERSION_KEY,
                String.valueOf(TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis())));
        return attributeMap;
    }

    private void removeBinaryDictionary() {
        asyncExecuteTaskWithWriteLock(this::removeBinaryDictionaryLocked);
    }

    void removeBinaryDictionaryLocked() {
        closeBinaryDictionary();
        if (mDictFile.exists() && !FileUtils.deleteRecursively(mDictFile)) {
            Log.e(TAG, "Can't remove a file: " + mDictFile.getName());
        }
    }

    private void openBinaryDictionaryLocked() {
        mBinaryDictionary = new BinaryDictionary(
                mDictFile.getAbsolutePath(), 0 /* offset */, mDictFile.length(),
                true /* useFullEditDistance */, mLocale, mDictType, true /* isUpdatable */);
    }

    void createOnMemoryBinaryDictionaryLocked() {
        mBinaryDictionary = new BinaryDictionary(
                mDictFile.getAbsolutePath(), true /* useFullEditDistance */, mLocale, mDictType,
                DICTIONARY_FORMAT_VERSION, getHeaderAttributeMap());
    }

    public void clear() {
        asyncExecuteTaskWithWriteLock(() -> {
            removeBinaryDictionaryLocked();
            createOnMemoryBinaryDictionaryLocked();
        });
    }

    /**
     * Check whether GC is needed and run GC if required.
     */
    public void runGCIfRequired(final boolean mindsBlockByGC) {
        asyncExecuteTaskWithWriteLock(() -> {
            if (getBinaryDictionary() == null) {
                return;
            }
            runGCIfRequiredLocked(mindsBlockByGC);
        });
    }

    protected void runGCIfRequiredLocked(final boolean mindsBlockByGC) {
        if (mBinaryDictionary.needsToRunGC(mindsBlockByGC)) {
            mBinaryDictionary.flushWithGC();
        }
    }

    private void updateDictionaryWithWriteLock(@NonNull final Runnable updateTask) {
        reloadDictionaryIfRequired();
        asyncExecuteTaskWithWriteLock(() -> {
            if (getBinaryDictionary() == null) {
                return;
            }
            runGCIfRequiredLocked(true /* mindsBlockByGC */);
            updateTask.run();
        });
    }

    /**
     * Adds unigram information of a word to the dictionary. May overwrite an existing entry.
     */
    public void addUnigramEntry(final String word, final int frequency,
            final String shortcutTarget, final int shortcutFreq, final boolean isNotAWord,
            final boolean isPossiblyOffensive, final int timestamp) {
        updateDictionaryWithWriteLock(() -> addUnigramLocked(word, frequency, shortcutTarget,
                shortcutFreq, isNotAWord, isPossiblyOffensive, timestamp));
    }

    protected void addUnigramLocked(final String word, final int frequency,
            final String shortcutTarget, final int shortcutFreq, final boolean isNotAWord,
            final boolean isPossiblyOffensive, final int timestamp) {
        if (!mBinaryDictionary.addUnigramEntry(word, frequency, shortcutTarget, shortcutFreq,
                false /* isBeginningOfSentence */, isNotAWord, isPossiblyOffensive, timestamp)) {
            Log.e(TAG, "Cannot add unigram entry. word: " + word);
        }
    }

    /**
     * Dynamically remove the unigram entry from the dictionary.
     */
    public void removeUnigramEntryDynamically(final String word) {
        reloadDictionaryIfRequired();
        asyncExecuteTaskWithWriteLock(() -> {
            final BinaryDictionary binaryDictionary = getBinaryDictionary();
            if (binaryDictionary == null) {
                return;
            }
            runGCIfRequiredLocked(true /* mindsBlockByGC */);
            if (!binaryDictionary.removeUnigramEntry(word)) {
                if (DEBUG) {
                    Log.i(TAG, "Cannot remove unigram entry: " + word);
                }
            }
        });
    }

    /**
     * Adds n-gram information of a word to the dictionary. May overwrite an existing entry.
     */
    public void addNgramEntry(@NonNull final NgramContext ngramContext, final String word,
            final int frequency, final int timestamp) {
        reloadDictionaryIfRequired();
        asyncExecuteTaskWithWriteLock(() -> {
            if (getBinaryDictionary() == null) {
                return;
            }
            runGCIfRequiredLocked(true /* mindsBlockByGC */);
            addNgramEntryLocked(ngramContext, word, frequency, timestamp);
        });
    }

    protected void addNgramEntryLocked(@NonNull final NgramContext ngramContext, final String word,
            final int frequency, final int timestamp) {
        if (!mBinaryDictionary.addNgramEntry(ngramContext, word, frequency, timestamp)) {
            if (DEBUG) {
                Log.i(TAG, "Cannot add n-gram entry.");
                Log.i(TAG, "  NgramContext: " + ngramContext + ", word: " + word);
            }
        }
    }

    /**
     * Update dictionary for the word with the ngramContext.
     */
    public void updateEntriesForWord(@NonNull final NgramContext ngramContext,
            final String word, final boolean isValidWord, final int count, final int timestamp) {
        updateDictionaryWithWriteLock(() -> {
            final BinaryDictionary binaryDictionary = getBinaryDictionary();
            if (binaryDictionary == null) {
                return;
            }
            if (!binaryDictionary.updateEntriesForWordWithNgramContext(ngramContext, word,
                    isValidWord, count, timestamp)) {
                if (DEBUG) {
                    Log.e(TAG, "Cannot update counter. word: " + word
                            + " context: " + ngramContext);
                }
            }
        });
    }

    @Override
    public ArrayList<SuggestedWordInfo> getSuggestions(final ComposedData composedData,
            final NgramContext ngramContext, final long proximityInfoHandle,
            final SettingsValuesForSuggestion settingsValuesForSuggestion, final int sessionId,
            final float weightForLocale, final float[] inOutWeightOfLangModelVsSpatialModel) {
        reloadDictionaryIfRequired();
        boolean lockAcquired = false;
        try {
            lockAcquired = mLock.readLock().tryLock(
                    TIMEOUT_FOR_READ_OPS_IN_MILLISECONDS, TimeUnit.MILLISECONDS);
            if (lockAcquired) {
                if (mBinaryDictionary == null) {
                    return null;
                }
                final ArrayList<SuggestedWordInfo> suggestions =
                        mBinaryDictionary.getSuggestions(composedData, ngramContext,
                                proximityInfoHandle, settingsValuesForSuggestion, sessionId,
                                weightForLocale, inOutWeightOfLangModelVsSpatialModel);
                if (mBinaryDictionary.isCorrupted()) {
                    Log.i(TAG, "Dictionary (" + mDictName +") is corrupted. "
                            + "Remove and regenerate it.");
                    removeBinaryDictionary();
                }
                return suggestions;
            }
        } catch (final InterruptedException e) {
            Log.e(TAG, "Interrupted tryLock() in getSuggestionsWithSessionId().", e);
        } finally {
            if (lockAcquired) {
                mLock.readLock().unlock();
            }
        }
        return null;
    }

    @Override
    public boolean isInDictionary(final String word) {
        reloadDictionaryIfRequired();
        boolean lockAcquired = false;
        try {
            lockAcquired = mLock.readLock().tryLock(
                    TIMEOUT_FOR_READ_OPS_IN_MILLISECONDS, TimeUnit.MILLISECONDS);
            if (lockAcquired) {
                if (mBinaryDictionary == null) {
                    return false;
                }
                return isInDictionaryLocked(word);
            }
        } catch (final InterruptedException e) {
            Log.e(TAG, "Interrupted tryLock() in isInDictionary().", e);
        } finally {
            if (lockAcquired) {
                mLock.readLock().unlock();
            }
        }
        return false;
    }

    protected boolean isInDictionaryLocked(final String word) {
        if (mBinaryDictionary == null) return false;
        return mBinaryDictionary.isInDictionary(word);
    }

    @Override
    public int getMaxFrequencyOfExactMatches(final String word) {
        reloadDictionaryIfRequired();
        boolean lockAcquired = false;
        try {
            lockAcquired = mLock.readLock().tryLock(
                    TIMEOUT_FOR_READ_OPS_IN_MILLISECONDS, TimeUnit.MILLISECONDS);
            if (lockAcquired) {
                if (mBinaryDictionary == null) {
                    return NOT_A_PROBABILITY;
                }
                return mBinaryDictionary.getMaxFrequencyOfExactMatches(word);
            }
        } catch (final InterruptedException e) {
            Log.e(TAG, "Interrupted tryLock() in getMaxFrequencyOfExactMatches().", e);
        } finally {
            if (lockAcquired) {
                mLock.readLock().unlock();
            }
        }
        return NOT_A_PROBABILITY;
    }


    /**
     * Loads the current binary dictionary from internal storage. Assumes the dictionary file
     * exists.
     */
    void loadBinaryDictionaryLocked() {
        if (DBG_STRESS_TEST) {
            // Test if this class does not cause problems when it takes long time to load binary
            // dictionary.
            try {
                Log.w(TAG, "Start stress in loading: " + mDictName);
                Thread.sleep(15000);
                Log.w(TAG, "End stress in loading");
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while loading: " + mDictName, e);
            }
        }
        final BinaryDictionary oldBinaryDictionary = mBinaryDictionary;
        openBinaryDictionaryLocked();
        if (oldBinaryDictionary != null) {
            oldBinaryDictionary.close();
        }
        if (mBinaryDictionary.isValidDictionary()
                && needsToMigrateDictionary(mBinaryDictionary.getFormatVersion())) {
            if (!mBinaryDictionary.migrateTo(DICTIONARY_FORMAT_VERSION)) {
                Log.e(TAG, "Dictionary migration failed: " + mDictName);
                removeBinaryDictionaryLocked();
            }
        }
    }

    /**
     * Create a new binary dictionary and load initial contents.
     */
    void createNewDictionaryLocked() {
        if (DebugFlags.DEBUG_ENABLED)
            Log.i(TAG, "creating new dictionary: " + this.getClass().getSimpleName() + " " + mLocale);
        removeBinaryDictionaryLocked();
        createOnMemoryBinaryDictionaryLocked();
        loadInitialContentsLocked();
        // Run GC and flush to file when initial contents have been loaded.
        mBinaryDictionary.flushWithGCIfHasUpdated();
    }

    private void clearNeedsToRecreate() {
        ExpandableDictionaryUtils.setHasUpdated(this);
    }

    private boolean needsToRecreate() {
        return ExpandableDictionaryUtils.needsUpdate(this);
    }

    String getDictName() {
        return mDictName;
    }

    /**
     * Load the current binary dictionary from internal storage. If the dictionary file doesn't
     * exists or needs to be regenerated, the new dictionary file will be asynchronously generated.
     * However, the dictionary itself is accessible even before the new dictionary file is actually
     * generated. It may return a null result for getSuggestions() in that case by design.
     */
    public final void reloadDictionaryIfRequired() {
        if (!isReloadRequired()) return;
        asyncReloadDictionary();
    }

    /**
     * Returns whether a dictionary reload is required.
     */
    private boolean isReloadRequired() {
        return mBinaryDictionary == null || needsToRecreate();
    }

    /**
     * Reloads the dictionary. Access is controlled on a per dictionary file basis.
     */
    private void asyncReloadDictionary() {
        final AtomicBoolean isReloading = mIsReloading;
        if (!isReloading.compareAndSet(false, true)) {
            return;
        }
        final File dictFile = mDictFile;
        asyncExecuteTaskWithWriteLock(() -> {
            try {
                if (!dictFile.exists() || needsToRecreate()) {
                    // If the dictionary file does not exist or contents have been updated,
                    // generate a new one.
                    createNewDictionaryLocked();
                } else if (getBinaryDictionary() == null) {
                    // Otherwise, load the existing dictionary.
                    loadBinaryDictionaryLocked();
                    BinaryDictionary binaryDictionary = getBinaryDictionary();
                    if (binaryDictionary != null && !(isValidDictionaryLocked())) {
                        // Binary dictionary or its format version is not valid. Regenerate
                        // the dictionary file. createNewDictionaryLocked will remove the
                        // existing files if appropriate.
                        createNewDictionaryLocked();
                    }
                }
                clearNeedsToRecreate();
            } finally {
                isReloading.set(false);
            }
        });
    }

    /**
     * Flush binary dictionary to dictionary file.
     */
    @Override
    public void onFinishInput() {
        asyncExecuteTaskWithWriteLock(() -> {
            final BinaryDictionary binaryDictionary = getBinaryDictionary();
            if (binaryDictionary == null) {
                return;
            }
            if (binaryDictionary.needsToRunGC(false /* mindsBlockByGC */)) {
                binaryDictionary.flushWithGCIfHasUpdated();
            } else {
                binaryDictionary.flush();
            }
        });
    }

    public DictionaryStats getDictionaryStats() {
        reloadDictionaryIfRequired();
        final String dictName = mDictName;
        final File dictFile = mDictFile;
        final AsyncResultHolder<DictionaryStats> result =
                new AsyncResultHolder<>("DictionaryStats");
        asyncExecuteTaskWithLock(mLock.readLock(), new Runnable() {
            @Override
            public void run() {
                result.set(new DictionaryStats(mLocale, dictName, dictName, dictFile, 0));
            }
        });
        return result.get(null /* defaultValue */, TIMEOUT_FOR_READ_OPS_IN_MILLISECONDS);
    }
}
