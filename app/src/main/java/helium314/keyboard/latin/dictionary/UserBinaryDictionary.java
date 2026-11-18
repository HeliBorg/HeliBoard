/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.dictionary;

import static helium314.keyboard.latin.dictionary.DictionaryUtilsKt.createPersonalDictionaryRequest;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteException;
import android.provider.UserDictionary.Words;

import com.android.inputmethod.latin.BinaryDictionary;

import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.utils.Log;

import java.io.File;
import java.util.Arrays;
import java.util.Locale;

/**
 * An expandable dictionary that stores the words in the user dictionary provider into a binary
 * dictionary file to use it from native code.
 */
public class UserBinaryDictionary extends ExpandableBinaryDictionary {
    private static final String TAG = UserBinaryDictionary.class.getSimpleName();

    private static final int HISTORICAL_DEFAULT_USER_DICTIONARY_FREQUENCY = 250;
    private static final int LATINIME_DEFAULT_USER_DICTIONARY_FREQUENCY = 160;
    // Shortcut frequency is 0~15, with 15 = whitelist. We don't want user dictionary entries
    // to auto-correct, so we set this to the highest frequency that won't, i.e. 14.
    private static final int USER_DICT_SHORTCUT_FREQUENCY = 14;

    private static final String[] PROJECTION_QUERY_WITH_SHORTCUT = new String[] {
            Words.WORD,
            Words.SHORTCUT,
            Words.FREQUENCY,
    };

    public static final String NAME = "userunigram";

    final private boolean mAlsoUseMoreRestrictiveLocales;

    protected UserBinaryDictionary(Context context, Locale locale, boolean alsoUseMoreRestrictiveLocales, File dictFile, String name) {
        super(context, getDictName(name, locale, dictFile), locale, TYPE_USER, dictFile);
        mAlsoUseMoreRestrictiveLocales = alsoUseMoreRestrictiveLocales;

        reloadDictionaryIfRequired();
    }

    public static UserBinaryDictionary getDictionary(Context context, Locale locale, File dictFile, String dictNamePrefix) {
        return new UserBinaryDictionary(context, locale, true, dictFile, dictNamePrefix + NAME);
    }

    @Override
    public void loadInitialContentsLocked() {
        try {
            var requestAndArgs = createPersonalDictionaryRequest(mLocale, mAlsoUseMoreRestrictiveLocales);
            if (DebugFlags.DEBUG_ENABLED)
                Log.d(TAG, "checking personal dictionary locales "+ Arrays.asList(requestAndArgs.component2()));
            addWordsFromProjectionLocked(requestAndArgs.component1(), requestAndArgs.component2());
        } catch (Exception e) {
            Log.i(TAG, "error adding words to personal dictionary", e);
        }
    }

    private void addWordsFromProjectionLocked(String request, String[] requestArguments)
            throws IllegalArgumentException {
        Cursor cursor = null;
        try {
            cursor = mContext.getContentResolver().query(
                    Words.CONTENT_URI, PROJECTION_QUERY_WITH_SHORTCUT, request, requestArguments, null);
            addWordsLocked(cursor);
        } catch (SQLiteException e) {
            Log.e(TAG, "SQLiteException in the remote User dictionary process.", e);
        } finally {
            try {
                if (null != cursor) cursor.close();
            } catch (SQLiteException e) {
                Log.e(TAG, "SQLiteException in the remote User dictionary process.", e);
            }
        }
    }

    private static int scaleFrequencyFromDefaultToLatinIme(int defaultFrequency) {
        // The default frequency for the user dictionary is 250 for historical reasons.
        // Latin IME considers a good value for the default user dictionary frequency
        // is about 160 considering the scale we use. So we are scaling down the values.
        if (defaultFrequency > Integer.MAX_VALUE / LATINIME_DEFAULT_USER_DICTIONARY_FREQUENCY) {
            return (defaultFrequency / HISTORICAL_DEFAULT_USER_DICTIONARY_FREQUENCY)
                    * LATINIME_DEFAULT_USER_DICTIONARY_FREQUENCY;
        }
        return (defaultFrequency * LATINIME_DEFAULT_USER_DICTIONARY_FREQUENCY)
                / HISTORICAL_DEFAULT_USER_DICTIONARY_FREQUENCY;
    }

    private void addWordsLocked(Cursor cursor) {
        if (cursor == null) return;
        if (cursor.moveToFirst()) {
            int indexWord = cursor.getColumnIndex(Words.WORD);
            int indexShortcut = cursor.getColumnIndex(Words.SHORTCUT);
            int indexFrequency = cursor.getColumnIndex(Words.FREQUENCY);
            while (!cursor.isAfterLast()) {
                String word = cursor.getString(indexWord);
                String shortcut = cursor.getString(indexShortcut);
                int frequency = cursor.getInt(indexFrequency);
                int adjustedFrequency = scaleFrequencyFromDefaultToLatinIme(frequency);
                // Safeguard against adding really long words.
                if (word.length() <= MAX_WORD_LENGTH) {
                    runGCIfRequiredLocked(true);
                    addUnigramLocked(word, adjustedFrequency, null, 0,
                        false, false, BinaryDictionary.NOT_A_VALID_TIMESTAMP);
                    if (null != shortcut && shortcut.length() <= MAX_WORD_LENGTH) {
                        runGCIfRequiredLocked(true);
                        addUnigramLocked(shortcut, adjustedFrequency, word, USER_DICT_SHORTCUT_FREQUENCY,
                            true, false, BinaryDictionary.NOT_A_VALID_TIMESTAMP);
                    }
                }
                cursor.moveToNext();
            }
        }
    }
}
