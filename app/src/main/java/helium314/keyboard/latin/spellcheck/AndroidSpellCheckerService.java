/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.spellcheck;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.service.textservice.SpellCheckerService;
import android.text.InputType;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodSubtype;
import android.view.textservice.SuggestionsInfo;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.Keyboard;
import helium314.keyboard.keyboard.KeyboardElement;
import helium314.keyboard.keyboard.KeyboardLayoutSet;
import helium314.keyboard.latin.DictionaryFacilitator;
import helium314.keyboard.latin.DictionaryFacilitatorLruCache;
import helium314.keyboard.latin.InputAttributes;
import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.RichInputMethodSubtype;
import helium314.keyboard.latin.SuggestedWords;
import helium314.keyboard.latin.common.ComposedData;
import helium314.keyboard.latin.settings.Defaults;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.SubtypeSettings;
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional;
import helium314.keyboard.latin.utils.SuggestionResults;

import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;

/**
 * Service for spell checking, using LatinIME's dictionaries and mechanisms.
 */
public final class AndroidSpellCheckerService extends SpellCheckerService
        implements SharedPreferences.OnSharedPreferenceChangeListener {

    public static final int SPELLCHECKER_DUMMY_KEYBOARD_WIDTH = 480;
    public static final int SPELLCHECKER_DUMMY_KEYBOARD_HEIGHT = 301;

    private static final String DICTIONARY_NAME_PREFIX = "spellcheck_";

    private static final String[] EMPTY_STRING_ARRAY = new String[0];

    private final int MAX_NUM_OF_THREADS_READ_DICTIONARY = 2;
    private final Semaphore mSemaphore = new Semaphore(MAX_NUM_OF_THREADS_READ_DICTIONARY, true);
    // TODO: Make each spell checker session has its own session id.
    private final ConcurrentLinkedQueue<Integer> mSessionIdPool = new ConcurrentLinkedQueue<>();

    private final DictionaryFacilitatorLruCache mDictionaryFacilitatorCache =
            new DictionaryFacilitatorLruCache(this, DICTIONARY_NAME_PREFIX);
    private final ConcurrentHashMap<Locale, Keyboard> mKeyboardCache = new ConcurrentHashMap<>();

    // The threshold for a suggestion to be considered "recommended".
    private float mRecommendedThreshold;
    private SettingsValuesForSuggestion mSettingsValuesForSuggestion;

    public static final String SINGLE_QUOTE = "'";
    public static final String APOSTROPHE = "’";

    public AndroidSpellCheckerService() {
        super();
        for (int i = 0; i < MAX_NUM_OF_THREADS_READ_DICTIONARY; i++) {
            mSessionIdPool.add(i);
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        mRecommendedThreshold = Float.parseFloat(getString(R.string.spellchecker_recommended_threshold_value));
        final SharedPreferences prefs = KtxKt.prefs(this);
        prefs.registerOnSharedPreferenceChangeListener(this);
        onSharedPreferenceChanged(prefs, Settings.PREF_USE_CONTACTS);
        onSharedPreferenceChanged(prefs, Settings.PREF_USE_APPS);
        final boolean blockOffensive = prefs.getBoolean(Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE, Defaults.PREF_BLOCK_POTENTIALLY_OFFENSIVE);
        mSettingsValuesForSuggestion = new SettingsValuesForSuggestion(blockOffensive, false);
    }

    public float getRecommendedThreshold() {
        return mRecommendedThreshold;
    }

    @Override
    public void onSharedPreferenceChanged(final SharedPreferences prefs, final String key) {
        if (key != null) switch (key) {
        case Settings.PREF_USE_CONTACTS -> {
            final boolean useContactsDictionary = prefs.getBoolean(Settings.PREF_USE_CONTACTS, Defaults.PREF_USE_CONTACTS);
            mDictionaryFacilitatorCache.setUseContactsDictionary(useContactsDictionary);
        }
        case Settings.PREF_USE_APPS -> {
            final boolean useAppsDictionary = prefs.getBoolean(Settings.PREF_USE_APPS, Defaults.PREF_USE_APPS);
            mDictionaryFacilitatorCache.setUseAppsDictionary(useAppsDictionary);
        }
        case Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE -> {
            final boolean blockOffensive = prefs.getBoolean(Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE, Defaults.PREF_BLOCK_POTENTIALLY_OFFENSIVE);
            mSettingsValuesForSuggestion = new SettingsValuesForSuggestion(blockOffensive, false);
        }}
    }

    @Override
    public Session createSession() {
        // Should not refer to AndroidSpellCheckerSession directly considering
        // that AndroidSpellCheckerSession may be overlaid.
        return AndroidSpellCheckerSessionFactory.newInstance(this);
    }

    /**
     * Returns an empty SuggestionsInfo with flags signaling the word is not in the dictionary.
     * @param reportAsTypo whether this should include the flag LOOKS_LIKE_TYPO, for red underline.
     * @return the empty SuggestionsInfo with the appropriate flags set.
     */
    public static SuggestionsInfo getNotInDictEmptySuggestions(final boolean reportAsTypo) {
        return new SuggestionsInfo(reportAsTypo ? SuggestionsInfo.RESULT_ATTR_LOOKS_LIKE_TYPO : 0,
                EMPTY_STRING_ARRAY);
    }
    public static SuggestionsInfo getTypoNoUiSuggestions() {
        return (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
            ? new SuggestionsInfo(SuggestionsInfo.RESULT_ATTR_LOOKS_LIKE_TYPO | SuggestionsInfo.RESULT_ATTR_DONT_SHOW_UI_FOR_SUGGESTIONS,
                EMPTY_STRING_ARRAY)
           : new SuggestionsInfo(SuggestionsInfo.RESULT_ATTR_LOOKS_LIKE_TYPO, EMPTY_STRING_ARRAY); // should not happen unless user transfers settings to old phone
    }
    /**
     * Returns an empty suggestionInfo with flags signaling the word is in the dictionary.
     * @return the empty SuggestionsInfo with the appropriate flags set.
     */
    public static SuggestionsInfo getInDictEmptySuggestions() {
        return new SuggestionsInfo(SuggestionsInfo.RESULT_ATTR_IN_THE_DICTIONARY, EMPTY_STRING_ARRAY);
    }

    public boolean isValidWord(final Locale locale, final String word) {
        mSemaphore.acquireUninterruptibly();
        try {
            DictionaryFacilitator dictionaryFacilitatorForLocale = mDictionaryFacilitatorCache.get(locale);
            return dictionaryFacilitatorForLocale.isValidSpellingWord(word);
        } finally {
            mSemaphore.release();
        }
    }

    public SuggestionResults getSuggestionResults(final Locale locale,
            final ComposedData composedData, final NgramContext ngramContext,
            @NonNull final Keyboard keyboard) {
        Integer sessionId = null;
        mSemaphore.acquireUninterruptibly();
        try {
            sessionId = mSessionIdPool.poll();
            DictionaryFacilitator dictionaryFacilitatorForLocale = mDictionaryFacilitatorCache.get(locale);
            return dictionaryFacilitatorForLocale.getSuggestionResults(composedData, ngramContext,
                    keyboard, mSettingsValuesForSuggestion,
                    sessionId, SuggestedWords.INPUT_STYLE_TYPING);
        } finally {
            if (sessionId != null) {
                mSessionIdPool.add(sessionId);
            }
            mSemaphore.release();
        }
    }

    public boolean hasMainDictionaryForLocale(final Locale locale) {
        mSemaphore.acquireUninterruptibly();
        try {
            final DictionaryFacilitator dictionaryFacilitator =
                    mDictionaryFacilitatorCache.get(locale);
            return dictionaryFacilitator.hasAtLeastOneInitializedMainDictionary();
        } finally {
            mSemaphore.release();
        }
    }

    @Override
    public boolean onUnbind(final Intent intent) {
        mSemaphore.acquireUninterruptibly(MAX_NUM_OF_THREADS_READ_DICTIONARY);
        try {
            mDictionaryFacilitatorCache.closeDictionaries();
        } finally {
            mSemaphore.release(MAX_NUM_OF_THREADS_READ_DICTIONARY);
        }
        mKeyboardCache.clear();
        return false;
    }

    public Keyboard getKeyboardForLocale(final Locale locale) {
        Keyboard keyboard = mKeyboardCache.get(locale);
        if (keyboard == null) {
            keyboard = createKeyboardForLocale(locale);
            mKeyboardCache.put(locale, keyboard);
        }
        return keyboard;
    }

    private Keyboard createKeyboardForLocale(final Locale locale) {
        if (Settings.getValues() == null) {
            // creating a keyboard reads SettingsValues from Settings instance
            // maybe it would be "more correct" to create an instance of SettingsValues and use that one instead
            // but creating a global one if not existing should be fine too
            final EditorInfo editorInfo = new EditorInfo();
            editorInfo.inputType = InputType.TYPE_CLASS_TEXT;
            Settings.getInstance().loadSettings(this, locale, new InputAttributes(editorInfo, false, getPackageName()));
        }
        final String mainLayoutName = SubtypeSettings.INSTANCE.getMatchingMainLayoutNameForLocale(locale);
        final InputMethodSubtype subtype = SubtypeUtilsAdditional.INSTANCE.createDummyAdditionalSubtype(locale, mainLayoutName);
        final KeyboardLayoutSet keyboardLayoutSet = createKeyboardSetForSpellChecker(subtype);
        return keyboardLayoutSet.getKeyboard(KeyboardElement.ALPHABET);
    }

    private KeyboardLayoutSet createKeyboardSetForSpellChecker(final InputMethodSubtype subtype) {
        final EditorInfo editorInfo = new EditorInfo();
        editorInfo.inputType = InputType.TYPE_CLASS_TEXT;
        final KeyboardLayoutSet.Builder builder = new KeyboardLayoutSet.Builder(this, editorInfo, null);
        return builder
                .setKeyboardGeometry(SPELLCHECKER_DUMMY_KEYBOARD_WIDTH, SPELLCHECKER_DUMMY_KEYBOARD_HEIGHT)
                .setSubtype(RichInputMethodSubtype.Companion.get(subtype))
                .setIsSpellChecker(true)
                .disableTouchPositionCorrectionData()
                .build();
    }
}
