/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.inputlogic;

import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;

import helium314.keyboard.latin.InputStyle;
import helium314.keyboard.latin.LatinIME;
import helium314.keyboard.latin.SuggestedWords;
import helium314.keyboard.latin.common.InputPointers;

/**
 * A helper to manage deferred tasks for the input logic.
 */
class InputLogicHandler implements Handler.Callback {
    final Handler mNonUIThreadHandler;
    final LatinIME.UIHandler mLatinIMEHandler;
    final InputLogic mInputLogic;
    private final Object mLock = new Object();
    private boolean mInBatchInput; // synchronized using {@link #mLock}.

    private static final int MSG_GET_SUGGESTED_WORDS = 1;

    public InputLogicHandler(final LatinIME.UIHandler latinIMEHandler, final InputLogic inputLogic) {
        final HandlerThread handlerThread = new HandlerThread(
                InputLogicHandler.class.getSimpleName());
        handlerThread.start();
        mNonUIThreadHandler = new Handler(handlerThread.getLooper(), this);
        mLatinIMEHandler = latinIMEHandler;
        mInputLogic = inputLogic;
    }

    public void reset() {
        mNonUIThreadHandler.removeCallbacksAndMessages(null);
    }

    /**
     * Handle a message.
     * @see android.os.Handler.Callback#handleMessage(android.os.Message)
     */
    // Called on the Non-UI handler thread by the Handler code.
    @Override
    public boolean handleMessage(final Message msg) {
        if (msg.what == MSG_GET_SUGGESTED_WORDS)
            ((Runnable)msg.obj).run();
        return true;
    }

    // Called on the UI thread by InputLogic.
    public void onStartBatchInput() {
        synchronized (mLock) {
            mInBatchInput = true;
        }
    }

    public boolean isInBatchInput() {
        return mInBatchInput;
    }

    /**
     * Fetch suggestions corresponding to an update of a batch input.
     * @param batchPointers the updated pointers, including the part that was passed last time.
     * @param sequenceNumber the sequence number associated with this batch input.
     * @param isTailBatchInput true if this is the end of a batch input, false if it's an update.
     */
    // This method can be called from any thread and will see to it that the correct threads
    // are used for parts that require it. This method will send a message to the Non-UI handler
    // thread to pull suggestions, and get the inlined callback to get called on the Non-UI
    // handler thread. If this is the end of a batch input, the callback will then proceed to
    // send a message to the UI handler in LatinIME so that showing suggestions can be done on
    // the UI thread.
    private void updateBatchInput(final InputPointers batchPointers,
            final int sequenceNumber, final boolean isTailBatchInput) {
        synchronized (mLock) {
            if (!mInBatchInput) {
                // Batch input has ended or canceled while the message was being delivered.
                return;
            }
            mInputLogic.mWordComposer.setBatchInputPointers(batchPointers);
            getSuggestedWords(() -> mInputLogic.getSuggestedWords(
                isTailBatchInput ? InputStyle.TAIL_BATCH : InputStyle.UPDATE_BATCH, sequenceNumber,
                suggestedWords -> showGestureSuggestionsWithPreviewVisuals(suggestedWords, isTailBatchInput))
            );
        }
    }

    private void showGestureSuggestionsWithPreviewVisuals(final SuggestedWords suggestedWordsForBatchInput,
            final boolean isTailBatchInput) {
        final SuggestedWords suggestedWordsToShowSuggestions;
        // We're now inside the callback. This always runs on the Non-UI thread,
        // no matter what thread updateBatchInput was originally called on.
        if (suggestedWordsForBatchInput.isEmpty()) {
            // Use old suggestions if we don't have any new ones.
            // Previous suggestions are found in InputLogic#mSuggestedWords.
            // Since these are the most recent ones and we just recomputed
            // new ones to update them, then the previous ones are there.
            suggestedWordsToShowSuggestions = mInputLogic.mSuggestedWords;
        } else {
            suggestedWordsToShowSuggestions = suggestedWordsForBatchInput;
        }
        mLatinIMEHandler.showGesturePreviewAndSetSuggestions(suggestedWordsToShowSuggestions, isTailBatchInput);
        if (isTailBatchInput) {
            mInBatchInput = false;
            // The following call schedules onEndBatchInputInternal
            // to be called on the UI thread.
            mLatinIMEHandler.showTailBatchInputResult(suggestedWordsToShowSuggestions);
        }
    }

    /**
     * Update a batch input.
     * <p>
     * This fetches suggestions and updates the suggestion strip and the floating text preview.
     *
     * @param batchPointers the updated batch pointers.
     * @param sequenceNumber the sequence number associated with this batch input.
     */
    // Called on the UI thread by InputLogic.
    public void onUpdateBatchInput(final InputPointers batchPointers,
            final int sequenceNumber) {
        updateBatchInput(batchPointers, sequenceNumber, false);
    }

    /**
     * Cancel a batch input.
     * <p>
     * Note that as opposed to updateTailBatchInput, we do the UI side of this immediately on the
     * same thread, rather than get this to call a method in LatinIME. This is because
     * canceling a batch input does not necessitate the long operation of pulling suggestions.
     */
    // Called on the UI thread by InputLogic.
    public void onCancelBatchInput() {
        synchronized (mLock) {
            mInBatchInput = false;
        }
    }

    /**
     * Trigger an update for a tail batch input.
     * <p>
     * A tail batch input is the last update for a gesture, the one that is triggered after the
     * user lifts their finger. This method schedules fetching suggestions on the non-UI thread,
     * then when the suggestions are computed it comes back on the UI thread to update the
     * suggestion strip, commit the first suggestion, and dismiss the floating text preview.
     *
     * @param batchPointers the updated batch pointers.
     * @param sequenceNumber the sequence number associated with this batch input.
     */
    // Called on the UI thread by InputLogic.
    public void updateTailBatchInput(final InputPointers batchPointers,
            final int sequenceNumber) {
        updateBatchInput(batchPointers, sequenceNumber, true);
    }

    public void getSuggestedWords(final Runnable callback) {
        mNonUIThreadHandler.obtainMessage(MSG_GET_SUGGESTED_WORDS, callback).sendToTarget();
    }
}
