/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import com.android.inputmethod.latin.utils.BinaryDictionaryUtils;

import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.suggestions.SuggestedWordInfo;

public final class AutoCorrectionUtils {
    private static final String TAG = AutoCorrectionUtils.class.getSimpleName();

    private AutoCorrectionUtils() {
        // Purely static class: can't instantiate.
    }

    public static boolean suggestionExceedsThreshold(final SuggestedWordInfo suggestion,
            final String consideredWord, final float threshold) {
        if (null != suggestion) {
            // Shortlist a whitelisted word
            if (suggestion.isKindOf(SuggestedWordInfo.KIND_WHITELIST)) {
                return true;
            }
            if (!suggestion.isAppropriateForAutoCorrection()) {
                return false;
            }
            final int autoCorrectionSuggestionScore = suggestion.mScore;
            // TODO: when the normalized score of the first suggestion is nearly equals to
            //       the normalized score of the second suggestion, behave less aggressive.
            final float normalizedScore = BinaryDictionaryUtils.calcNormalizedScore(
                    consideredWord, suggestion.mWord, autoCorrectionSuggestionScore);
            if (DebugFlags.DEBUG_ENABLED) {
                Log.d(TAG, "Normalized " + consideredWord + "," + suggestion + ","
                        + autoCorrectionSuggestionScore + ", " + normalizedScore
                        + " (" + threshold + ")");
            }
            if (normalizedScore >= threshold) {
                if (DebugFlags.DEBUG_ENABLED) {
                    Log.d(TAG, "Exceeds threshold.");
                }
                return true;
            }
        }
        return false;
    }
}
