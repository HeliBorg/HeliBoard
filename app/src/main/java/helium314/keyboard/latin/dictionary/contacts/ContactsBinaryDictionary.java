/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.dictionary.contacts;

import android.Manifest;
import android.content.Context;

import androidx.annotation.NonNull;

import com.android.inputmethod.latin.BinaryDictionary;

import java.io.File;
import java.util.Locale;

import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.dictionary.ExpandableBinaryDictionary;
import helium314.keyboard.latin.permissions.PermissionsUtil;
import helium314.keyboard.latin.utils.Log;

public class ContactsBinaryDictionary extends ExpandableBinaryDictionary {
    private static final String TAG = ContactsBinaryDictionary.class.getSimpleName();
    public static final String NAME = "contacts";

    private static final boolean DEBUG = false;
    private static final boolean DEBUG_DUMP = false;

    /**
     * Whether to use "firstname lastname" in bigram predictions.
     */
    private final boolean mUseFirstLastBigrams;

    protected ContactsBinaryDictionary(final Context context, final Locale locale,
            final File dictFile, final String name) {
        super(context, getDictName(name, locale, dictFile), locale, Dictionary.TYPE_CONTACTS,
                dictFile);
        mUseFirstLastBigrams = ContactsDictionaryUtils.useFirstLastBigramsForLocale(locale);
        reloadDictionaryIfRequired();
    }

    public static ContactsBinaryDictionary getDictionary(final Context context, @NonNull final Locale locale,
            final File dictFile, final String dictNamePrefix) {
        return new ContactsBinaryDictionary(context, locale, dictFile, dictNamePrefix + NAME);
    }

    /**
     * Typically called whenever the dictionary is created for the first time or
     * recreated when we think that there are updates to the dictionary.
     * This is called asynchronously.
     */
    @Override
    public void loadInitialContentsLocked() {
        loadDictionaryLocked();
    }

    /**
     * Loads data within content providers to the dictionary.
     */
    private void loadDictionaryLocked() {
        if (!PermissionsUtil.checkAllPermissionsGranted(mContext, Manifest.permission.READ_CONTACTS)) {
            Log.i(TAG, "No permission to read contacts. Not loading the Dictionary.");
        }

        for (String name : ContactsUtils.INSTANCE.getNames(mContext)) {
            addNameLocked(name);
        }
    }

    /**
     * Adds the words in a name (e.g., firstname/lastname) to the binary dictionary along with their
     * bigrams depending on locale.
     */
    private void addNameLocked(final String name) {
        int len = StringUtils.codePointCount(name);
        NgramContext ngramContext = NgramContext.getEmptyPrevWordsContext(
                BinaryDictionary.MAX_PREV_WORD_COUNT_FOR_N_GRAM);
        // TODO: Better tokenization for non-Latin writing systems
        for (int i = 0; i < len; i++) {
            if (Character.isLetter(name.codePointAt(i))) {
                int end = ContactsDictionaryUtils.getWordEndPosition(name, len, i);
                String word = name.substring(i, end);
                if (DEBUG_DUMP) {
                    Log.d(TAG, "addName word = " + word);
                }
                i = end - 1;
                // Don't add single letter words, possibly confuses
                // capitalization of i.
                final int wordLen = StringUtils.codePointCount(word);
                if (wordLen <= MAX_WORD_LENGTH && wordLen > 1) {
                    if (DEBUG) {
                        Log.d(TAG, "addName " + name + ", " + word + ", "  + ngramContext);
                    }
                    runGCIfRequiredLocked(true);
                    addUnigramLocked(word, ContactsUtils.FREQUENCY_FOR_CONTACTS, null, 0,
                        false, false, BinaryDictionary.NOT_A_VALID_TIMESTAMP);
                    if (ngramContext.isValid() && mUseFirstLastBigrams) {
                        runGCIfRequiredLocked(true /* mindsBlockByGC */);
                        addNgramEntryLocked(ngramContext, word, ContactsUtils.FREQUENCY_FOR_CONTACTS_BIGRAM, BinaryDictionary.NOT_A_VALID_TIMESTAMP);
                    }
                    ngramContext = ngramContext.getNextNgramContext(
                            new NgramContext.WordInfo(word));
                }
            }
        }
    }
}
