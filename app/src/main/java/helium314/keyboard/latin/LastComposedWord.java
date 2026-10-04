/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin;

import android.text.TextUtils;

import helium314.keyboard.event.Event;
import helium314.keyboard.latin.common.InputPointers;
import helium314.keyboard.latin.define.DecoderSpecificConstants;

import java.util.ArrayList;

/**
 * This class encapsulates data about a word previously composed, but that has been
 * committed already. This is used for resuming suggestion, and cancel auto-correction.
 */
public final class LastComposedWord {
    // COMMIT_TYPE_USER_TYPED_WORD is used when the word committed is the exact typed word, with
    // no hinting from the IME. It happens when some external event happens (rotating the device,
    // for example) or when auto-correction is off by settings or editor attributes.
    public static final int COMMIT_TYPE_USER_TYPED_WORD = 0;
    // COMMIT_TYPE_MANUAL_PICK is used when the user pressed a field in the suggestion strip.
    public static final int COMMIT_TYPE_MANUAL_PICK = 1;
    // COMMIT_TYPE_DECIDED_WORD is used when the IME commits the word it decided was best
    // for the current user input. It may be different from what the user typed (true auto-correct)
    // or it may be exactly what the user typed if it's in the dictionary or the IME does not have
    // enough confidence in any suggestion to auto-correct (auto-correct to typed word).
    public static final int COMMIT_TYPE_DECIDED_WORD = 2;
    // COMMIT_TYPE_CANCEL_AUTO_CORRECT is used upon committing back the old word upon cancelling
    // an auto-correction.
    public static final int COMMIT_TYPE_CANCEL_AUTO_CORRECT = 3;

    public static final String NOT_A_SEPARATOR = "";

    public final ArrayList<Event> mEvents;
    public final String mTypedWord;
    public final CharSequence mCommittedWord;
    public final String mSeparatorString;
    public final NgramContext mNgramContext;
    public final CapsMode mCapitalizedMode;
    public final InputPointers mInputPointers =
            new InputPointers(DecoderSpecificConstants.DICTIONARY_MAX_WORD_LENGTH);

    private boolean mActive;

    public static final LastComposedWord NOT_A_COMPOSED_WORD =
            new LastComposedWord(new ArrayList<Event>(), null, "", "",
            NOT_A_SEPARATOR, null, CapsMode.OFF);

    // Warning: this is using the passed objects as is and fully expects them to be
    // immutable. Do not fiddle with their contents after you passed them to this constructor.
    public LastComposedWord(ArrayList<Event> events, InputPointers inputPointers, String typedWord,
            CharSequence committedWord, String separatorString, NgramContext ngramContext, CapsMode capitalizedMode) {
        if (inputPointers != null) {
            mInputPointers.copy(inputPointers);
        }
        mTypedWord = typedWord;
        mEvents = new ArrayList<>(events);
        mCommittedWord = committedWord;
        mSeparatorString = separatorString;
        mActive = true;
        mNgramContext = ngramContext;
        mCapitalizedMode = capitalizedMode;
    }

    public void deactivate() {
        mActive = false;
    }

    public boolean canRevertCommit() {
        return mActive && !TextUtils.isEmpty(mCommittedWord) && !didCommitTypedWord();
    }

    private boolean didCommitTypedWord() {
        return TextUtils.equals(mTypedWord, mCommittedWord);
    }
}
