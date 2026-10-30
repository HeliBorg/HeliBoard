/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.keyboard.internal.KeySpecParser;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.dictionary.Dictionary;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * The extended {@link SuggestedWords} class to represent punctuation suggestions.
 * <p>
 * Each punctuation specification string is the key specification that can be parsed by
 * {@link KeySpecParser}.
 */
public final class PunctuationSuggestions extends SuggestedWords {
    private PunctuationSuggestions(ArrayList<SuggestedWordInfo> punctuationsList) {
        super(punctuationsList, null, null, false, false,
            false, InputStyle.NONE, SuggestedWords.NOT_A_SEQUENCE_NUMBER);
    }

    /**
     * Create new instance of {@link PunctuationSuggestions} from the array of punctuation key
     * specifications.
     *
     * @param punctuationSpecs The array of punctuation key specifications.
     * @return The {@link PunctuationSuggestions} object.
     */
    public static PunctuationSuggestions newPunctuationSuggestions(
            @Nullable final String[] punctuationSpecs) {
        if (punctuationSpecs == null || punctuationSpecs.length == 0) {
            return new PunctuationSuggestions(new ArrayList<>(0));
        }
        final ArrayList<SuggestedWordInfo> punctuationList =
                new ArrayList<>(punctuationSpecs.length);
        for (String spec : punctuationSpecs) {
            punctuationList.add(newHardCodedWordInfo(spec));
        }
        return new PunctuationSuggestions(punctuationList);
    }

    /**
     * {@inheritDoc}
     * Note that {@link SuggestedWords#getWord(int)} returns a punctuation key specification text.
     * The suggested punctuation should be gotten by parsing the key specification.
     */
    @Override
    public String getWord(final int index) {
        final String keySpec = super.getWord(index);
        final int code = KeySpecParser.getCode(keySpec);
        return (code == KeyCode.MULTIPLE_CODE_POINTS)
                ? KeySpecParser.getOutputText(keySpec, code)
                : StringUtils.newSingleCodePointString(code);
    }

    /**
     * {@inheritDoc}
     * Note that {@link SuggestedWords#getWord(int)} returns a punctuation key specification text.
     * The displayed text should be gotten by parsing the key specification.
     */
    @Override
    public String getLabel(final int index) {
        final String keySpec = super.getWord(index);
        return KeySpecParser.getLabel(keySpec);
    }

    /**
     * {@inheritDoc}
     * Note that {@link #getWord(int)} returns a suggested punctuation. We should create a
     * {@link SuggestedWords.SuggestedWordInfo} object that represents a hard coded word.
     */
    @Override
    public SuggestedWordInfo getInfo(final int index) {
        return newHardCodedWordInfo(getWord(index));
    }

    /**
     * The predicator to tell whether this object represents punctuation suggestions.
     * @return true if this object represents punctuation suggestions.
     */
    @Override
    public boolean isPunctuationSuggestions() {
        return true;
    }

    @Override
    @NonNull public String toString() {
        return "PunctuationSuggestions: "
                + " words=" + Arrays.toString(mSuggestedWordInfoList.toArray());
    }

    private static SuggestedWordInfo newHardCodedWordInfo(final String keySpec) {
        return new SuggestedWordInfo(keySpec, "" /* prevWordsContext */,
                SuggestedWordInfo.MAX_SCORE,
                SuggestedWordInfo.KIND_HARDCODED,
                Dictionary.DICTIONARY_HARDCODED,
                SuggestedWordInfo.NOT_AN_INDEX /* indexOfTouchPointOfSecondWord */,
                SuggestedWordInfo.NOT_A_CONFIDENCE /* autoCommitFirstWordConfidence */);
    }
}
