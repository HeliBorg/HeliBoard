/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import java.util.Locale;

/**
 * The status of the current recapitalize process.
 */
public class RecapitalizeStatus {
    // We store the location of the cursor and the string that was there before the recapitalize
    // action was done, and the location of the cursor and the string that was there after.
    private String mText;
    private TextPlacement mTextPlacement;
    private RecapitalizeMode mCurrentMode;
    private boolean mSkipOriginalMixedCaseMode;
    private Locale mLocale;
    private int[] mSortedSeparators;
    private boolean mIsStarted;
    private boolean mIsEnabled = true;

    public RecapitalizeStatus() {
        // By default, initialize with dummy values that won't match any real recapitalize.
        start("", -1, Locale.getDefault(), new int[0]);
        stop();
    }

    public void start(final String text, final int cursorStart, final Locale locale, final int[] sortedSeparators) {
        if (!mIsEnabled) {
            return;
        }
        mText = text;
        mTextPlacement = new TextPlacement(text, cursorStart);
        mCurrentMode = RecapitalizeMode.of(mText, sortedSeparators);
        mSkipOriginalMixedCaseMode = RecapitalizeMode.ORIGINAL_MIXED_CASE != mCurrentMode;
        mLocale = locale;
        mSortedSeparators = sortedSeparators;
        mIsStarted = true;
    }

    public void stop() {
        mIsStarted = false;
    }

    public boolean isStarted() {
        return mIsStarted;
    }

    public void enable() {
        mIsEnabled = true;
    }

    public void disable() {
        mIsEnabled = false;
    }

    public boolean isEnabled() {
        return mIsEnabled;
    }

    public boolean isSetAt(final int cursorStart, final int cursorEnd) {
        return cursorStart == mTextPlacement.selectionStart && cursorEnd == mTextPlacement.selectionEnd();
    }

    /**
     * Rotate through the different possible capitalization modes.
     */
    public void rotate() {
        final int modeCount = RecapitalizeMode.count();
        String replacement;
        int i = 0; // Protection against infinite loop.
        do {
            mCurrentMode = mCurrentMode.rotate(mSkipOriginalMixedCaseMode);
            replacement = mCurrentMode.apply(mText, mSortedSeparators, mLocale);
            ++i;
        } while (replacement.equals(mTextPlacement.text) && i <= modeCount);
        mTextPlacement.text = replacement;
    }

    public TextPlacement textReplacement() {
        return mTextPlacement;
    }

    public RecapitalizeMode getCurrentMode() {
        return mCurrentMode;
    }
}
