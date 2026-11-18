// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.dictionary;

import android.content.Context;
import android.content.pm.PackageManager;

import com.android.inputmethod.latin.BinaryDictionary;
import java.io.File;
import java.util.Locale;
import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.utils.AppsUtils;
import helium314.keyboard.latin.utils.Log;
import helium314.keyboard.latin.utils.SpacedTokens;

// todo: actually we only need a single AppsBinaryDictionary, but currently
//  have one for each language, and may even have multiple instances in multilingual typing
public class AppsBinaryDictionary extends ExpandableBinaryDictionary {
    private static final String TAG = AppsBinaryDictionary.class.getSimpleName();
    public static final String NAME = "apps";

    private static final int FREQUENCY_FOR_APPS = 100;
    private static final int FREQUENCY_FOR_APPS_BIGRAM = 200;

    private static final boolean DEBUG = false;
    private static final boolean DEBUG_DUMP = false;

    private final PackageManager mPackageManager;

    protected AppsBinaryDictionary(Context ctx, Locale locale, File dictFile, String name) {
        super(ctx, getDictName(name, locale, dictFile), locale, Dictionary.TYPE_APPS, dictFile);
        mPackageManager = ctx.getPackageManager();
        reloadDictionaryIfRequired();
    }

    public static AppsBinaryDictionary getDictionary(Context context, Locale locale, File dictFile, String dictNamePrefix) {
        return new AppsBinaryDictionary(context, locale, dictFile, dictNamePrefix + NAME);
    }

    @Override
    public synchronized void close() {
        super.close();
    }

    /**
     * Typically called whenever the dictionary is created for the first time or recreated when we
     * think that there are updates to the dictionary. This is called asynchronously.
     */
    @Override
    public void loadInitialContentsLocked() {
        loadDictionaryLocked();
    }

    /**
     * Loads app names to the dictionary.
     */
    private void loadDictionaryLocked() {
        for (String name : AppsUtils.INSTANCE.getNames(mPackageManager)) {
            addNameLocked(name);
        }
    }

    /**
     * Adds the words in an app label to the binary dictionary along with their n-grams.
     */
    private void addNameLocked(final String appLabel) {
        NgramContext ngramContext = NgramContext.getEmptyPrevWordsContext(
                BinaryDictionary.MAX_PREV_WORD_COUNT_FOR_N_GRAM);
        // TODO: Better tokenization for non-Latin writing systems
        for (final String word : new SpacedTokens(appLabel)) {
            if (DEBUG_DUMP) {
                Log.d(TAG, "addName word = " + word);
            }
            final int wordLen = StringUtils.codePointCount(word);
            // Don't add single letter words, possibly confuses capitalization of i.
            if (1 < wordLen && wordLen <= MAX_WORD_LENGTH) {
                if (DEBUG) {
                    Log.d(TAG, "addName " + appLabel + ", " + word + ", "  + ngramContext);
                }
                runGCIfRequiredLocked(true /* mindsBlockByGC */);
                addUnigramLocked(word, FREQUENCY_FOR_APPS,
                        null /* shortcut */, 0 /* shortcutFreq */, false /* isNotAWord */,
                        false /* isPossiblyOffensive */,
                        BinaryDictionary.NOT_A_VALID_TIMESTAMP);
                if (ngramContext.isValid()) {
                    runGCIfRequiredLocked(true /* mindsBlockByGC */);
                    addNgramEntryLocked(ngramContext,
                            word,
                            FREQUENCY_FOR_APPS_BIGRAM,
                            BinaryDictionary.NOT_A_VALID_TIMESTAMP);
                }
                ngramContext = ngramContext.getNextNgramContext(
                        new NgramContext.WordInfo(word));
            }
        }
    }
}
