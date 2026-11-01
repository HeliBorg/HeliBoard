// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.dictionary;

import helium314.keyboard.event.HangulCombiner;
import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.common.ComposedData;
import helium314.keyboard.latin.makedict.WordProperty;
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion;
import helium314.keyboard.latin.suggestions.SuggestedWordInfo;

import java.text.Normalizer;
import java.util.ArrayList;

/*
 * For Korean dictionary, there are too many cases of characters to store on dictionary, which makes it slow.
 * To solve that, Unicode normalization is used to decompose Hangul syllables into Hangul jamos.
 */
public class KoreanDictionary extends Dictionary {

    private static final String COMPAT_JAMO = HangulCombiner.HangulJamo.COMPAT_CONSONANTS + HangulCombiner.HangulJamo.COMPAT_VOWELS;
    private static final String STANDARD_JAMO = HangulCombiner.HangulJamo.CONVERT_INITIALS + HangulCombiner.HangulJamo.CONVERT_MEDIALS;

    private final Dictionary mDictionary;

    public KoreanDictionary(Dictionary dictionary) {
        super(dictionary.mDictType, dictionary.mLocale);
        mDictionary = dictionary;
    }

    private String processInput(String input) {
        String normalized = Normalizer.normalize(input, Normalizer.Form.NFD);
        StringBuilder result = new StringBuilder();
        for (char c : normalized.toCharArray()) {
            int index = COMPAT_JAMO.indexOf(c);
            if (index == -1) result.append(c);
            else result.append(STANDARD_JAMO.charAt(index));
        }
        return result.toString();
    }

    private String processOutput(String output) {
        return Normalizer.normalize(output, Normalizer.Form.NFC);
    }

    @Override
    public ArrayList<SuggestedWordInfo> getSuggestions(ComposedData composedData,
                  NgramContext ngramContext, long proximityInfoHandle, SettingsValuesForSuggestion settingsValuesForSuggestion,
                  int sessionId, float weightForLocale, float[] inOutWeightOfLangModelVsSpatialModel) {
        composedData = new ComposedData(composedData.mInputPointers,
                composedData.mIsBatchMode, processInput(composedData.mTypedWord));
        ArrayList<SuggestedWordInfo> suggestions = mDictionary.getSuggestions(composedData,
                ngramContext, proximityInfoHandle, settingsValuesForSuggestion, sessionId,
                weightForLocale, inOutWeightOfLangModelVsSpatialModel);
        ArrayList<SuggestedWordInfo> result = new ArrayList<>();
        for (SuggestedWordInfo info : suggestions) {
            result.add(new SuggestedWordInfo(processOutput(info.mWord), info.mPrevWordsContext,
                    info.mScore, info.mKindAndFlags, info.mSourceDict, info.mIndexOfTouchPointOfSecondWord, info.mAutoCommitFirstWordConfidence));
        }
        return result;
    }

    @Override
    public boolean isInDictionary(String word) {
        return mDictionary.isInDictionary(processInput(word));
    }

    @Override
    public int getFrequency(String word) {
        return mDictionary.getFrequency(processInput(word));
    }

    @Override
    public int getMaxFrequencyOfExactMatches(String word) {
        return mDictionary.getMaxFrequencyOfExactMatches(processInput(word));
    }

    @Override
    public WordProperty getWordProperty(String word, boolean isBeginningOfSentence) {
        return mDictionary.getWordProperty(processInput(word), isBeginningOfSentence);
    }

    @Override
    protected boolean same(char[] word, int length, String typedWord) {
        word = processInput(new String(word)).toCharArray();
        typedWord = processInput(typedWord);
        return mDictionary.same(word, length, typedWord);
    }

    @Override
    public void close() {
        mDictionary.close();
    }

    @Override
    public void onFinishInput() {
        mDictionary.onFinishInput();
    }

    @Override
    public boolean isInitialized() {
        return mDictionary.isInitialized();
    }

    @Override
    public boolean shouldAutoCommit(SuggestedWordInfo candidate) {
        return mDictionary.shouldAutoCommit(candidate);
    }

    @Override
    public boolean isUserSpecific() {
        return mDictionary.isUserSpecific();
    }
}
