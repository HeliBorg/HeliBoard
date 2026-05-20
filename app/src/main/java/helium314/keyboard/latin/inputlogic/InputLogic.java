/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.inputlogic;

import static helium314.keyboard.latin.common.SuggestionSpanUtilsKt.getTextWithSuggestionSpan;

import android.graphics.Color;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.InputType;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.TextUtils;
import android.text.style.BackgroundColorSpan;
import android.text.style.SuggestionSpan;
import android.view.KeyCharacterMap;
import android.view.KeyEvent;
import android.view.inputmethod.CorrectionInfo;
import android.view.inputmethod.EditorInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.event.Event;
import helium314.keyboard.event.InputTransaction;
import helium314.keyboard.keyboard.Keyboard;
import helium314.keyboard.keyboard.KeyboardLayoutSet;
import helium314.keyboard.keyboard.KeyboardSwitcher;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.DictionaryFacilitator;
import helium314.keyboard.latin.dictionary.DictionaryFactory;
import helium314.keyboard.latin.LastComposedWord;
import helium314.keyboard.latin.LatinIME;
import helium314.keyboard.latin.NgramContext;
import helium314.keyboard.latin.RichInputConnection;
import helium314.keyboard.latin.SingleDictionaryFacilitator;
import helium314.keyboard.latin.Suggest;
import helium314.keyboard.latin.Suggest.OnGetSuggestedWordsCallback;
import helium314.keyboard.latin.SuggestedWords;
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo;
import helium314.keyboard.latin.WordComposer;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.InputPointers;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.common.StringUtilsKt;
import helium314.keyboard.latin.common.SuggestionSpanUtilsKt;
import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.settings.SettingsValues;
import helium314.keyboard.latin.settings.SpacingAndPunctuations;
import helium314.keyboard.latin.suggestions.SuggestionStripViewAccessor;
import helium314.keyboard.latin.utils.AsyncResultHolder;
import helium314.keyboard.latin.utils.DictionaryInfoUtils;
import helium314.keyboard.latin.utils.InputTypeUtils;
import helium314.keyboard.latin.utils.IntentUtils;
import helium314.keyboard.latin.utils.Log;
import helium314.keyboard.latin.utils.RecapitalizeMode;
import helium314.keyboard.latin.utils.RecapitalizeStatus;
import helium314.keyboard.latin.utils.ScriptUtils;
import helium314.keyboard.latin.utils.StatsUtils;
import helium314.keyboard.latin.utils.TextPlacement;
import helium314.keyboard.latin.utils.TextRange;
import helium314.keyboard.latin.utils.TimestampKt;

import java.util.ArrayList;
import java.util.Locale;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;

/**
 * This class manages the input logic.
 */
public final class InputLogic {
    private static final String TAG = InputLogic.class.getSimpleName();
    private static final char INLINE_EMOJI_SEARCH_MARKER = ':';
    private static final int[] EMPTY_CODE_POINTS = new int[0];

    // TODO : Remove this member when we can.
    final LatinIME mLatinIME;
    private final SuggestionStripViewAccessor mSuggestionStripViewAccessor;

    @NonNull private final InputLogicHandler mInputLogicHandler;

    // TODO : make all these fields private as soon as possible.
    // Current space state of the input method. This can be any of the above constants.
    private int mSpaceState;
    // Never null
    public SuggestedWords mSuggestedWords = SuggestedWords.getEmptyInstance();
    public Suggest mSuggest; // non-final for active gesture data gathering, revert when data gathering phase is done (end of 2026 latest)
    public DictionaryFacilitator mDictionaryFacilitator; // non-final for active gesture data gathering, revert when data gathering phase is done (end of 2026 latest)
    private SingleDictionaryFacilitator mEmojiDictionaryFacilitator;
    public void setFacilitator(DictionaryFacilitator facilitator) { // only for active gesture data gathering, remove when data gathering phase is done (end of 2026 latest)
        if (mDictionaryFacilitator == facilitator) return;
        mDictionaryFacilitator = facilitator;
        mSuggest = new Suggest(mDictionaryFacilitator);
    }

    public LastComposedWord mLastComposedWord = LastComposedWord.NOT_A_COMPOSED_WORD;
    // This has package visibility so it can be accessed from InputLogicHandler.
    /* package */ final WordComposer mWordComposer;
    public final RichInputConnection mConnection;
    private final RecapitalizeStatus mRecapitalizeStatus = new RecapitalizeStatus();

    private int mDeleteCount;
    private long mLastKeyTime;
    // todo: this is not used, so either remove it or do something with it
    public final TreeSet<Long> mCurrentlyPressedHardwareKeys = new TreeSet<>();

    // Two-thumb typing (#1.4): when {@code mGestureTapPromotionMs > 0} AND the user starts a
    // gesture within that window of their last letter tap, the gesture extends the existing
    // composing word instead of replacing it (the manual-spacing-extend path). Flag is set in
    // {@link #onStartBatchInput} and consumed at the end of {@link #onUpdateTailBatchInputCompleted}.
    // This lets us make the "extend or not" decision based on the timing AT THE MOMENT OF
    // GESTURE-START, not gesture-end (so a long gesture doesn't lose the promotion).
    private boolean mGestureExtendsByTapPromotion;

    // Snapshot of {@code keyboardSwitcher.getKeyboardShiftMode()} captured at the start of
    // each gesture. Used by {@link #onUpdateTailBatchInputCompleted} to capitalize the
    // recognizer's lowercase output. We can't read getKeyboardShiftMode() at gesture-end
    // because the keyboard typically auto-clears the shifted state during the gesture.
    private int mShiftModeAtGestureStart = WordComposer.CAPS_MODE_OFF;
    /** Set to true at the end of {@link #onCombiningGraceExpired} when an autospace was
     *  written, so the next punctuation tap in {@link #handleSeparatorEvent} can strip it
     *  (yielding "hello," not "hello ,"). One-shot — cleared on next non-separator letter
     *  tap, on backspace, on cancel, and right after consumption. NOT a SpaceState flag
     *  because PHANTOM would make the next letter ALSO write a space, double-spacing. */
    private boolean mAutospaceJustWritten;

    // Two-thumb typing (#1.1, sub-option PREF_GESTURE_FRAGMENT_BACKSPACE): char-offsets that
    // mark the END of each "fragment" appended to the current composing word under manual
    // spacing. A fragment is one gesture's output OR one tap's letter. With the pref on,
    // backspace pops the most recent fragment in one keystroke instead of deleting one
    // character at a time. The list is kept in sync with {@code mWordComposer.getTypedWord()}:
    // entries past the current length are filtered at read time, and the list is cleared
    // outright whenever the composing word is committed or reset.
    private final java.util.ArrayList<Integer> mGestureFragmentBoundaries = new java.util.ArrayList<>();

    // ---- Unified combining-mode state machine ----------------------------------------------
    // After every composing-word-extending event (tap of a letter OR gesture completion),
    // schedule a Handler callback in {@code mCombiningGraceMs} milliseconds. The callback
    // commits the current composing word (optionally with autocorrect) and inserts an
    // auto-space. Any new tap/gesture before the callback cancels it AND extends the same
    // composing word, so users can build a word from multiple fragments — e.g. tap "s",
    // then two-finger swipe "il"+"ver" to get "silver". Short words ("I", "so") still get
    // an autospace because the timer fires once the user pauses.
    //
    // Visual: while a commit is pending, the spacebar shows a countdown progress bar
    // (the {@link MainKeyboardView#setCombiningMode} call) — replaces the older
    // PREF_AUTOSPACE_VISUAL_HINT flash, which was decoupled from the state that caused it.
    //
    // All access on the main thread (touch events + Handler posts to main looper).
    private final Handler mCombiningHandler = new Handler(Looper.getMainLooper());
    @Nullable private Runnable mPendingCombiningCommit;
    private boolean mInCombiningMode;

    // Keeps track of most recently inserted text (multi-character key) for
    // reverting
    private String mEnteredText;

    // TODO: This boolean is persistent state and causes large side effects at unexpected times.
    // Find a way to remove it for readability.
    private boolean mIsAutoCorrectionIndicatorOn;
    private long mDoubleSpacePeriodCountdownStart;

    // The word being corrected while the cursor is in the middle of the word.
    // Note: This does not have a composing span, so it must be handled separately.
    private String mWordBeingCorrectedByCursor = null;

    private boolean mJustRevertedACommit = false;

    /**
     * Create a new instance of the input logic.
     * @param latinIME the instance of the parent LatinIME. We should remove this when we can.
     * @param suggestionStripViewAccessor an object to access the suggestion strip view.
     * @param dictionaryFacilitator facilitator for getting suggestions and updating user history
     * dictionary.
     */
    public InputLogic(final LatinIME latinIME,
            final SuggestionStripViewAccessor suggestionStripViewAccessor,
            final DictionaryFacilitator dictionaryFacilitator) {
        mLatinIME = latinIME;
        mSuggestionStripViewAccessor = suggestionStripViewAccessor;
        mWordComposer = new WordComposer();
        mConnection = new RichInputConnection(latinIME);
        mInputLogicHandler = new InputLogicHandler(mLatinIME.mHandler, this);
        mSuggest = new Suggest(dictionaryFacilitator);
        mDictionaryFacilitator = dictionaryFacilitator;
    }

    /**
     * Initializes the input logic for input in an editor.
     * <p>
     * Call this when input starts or restarts in some editor (typically, in onStartInputView).
     *
     * @param combiningSpec the combining spec string for this subtype (from extra value)
     * @param settingsValues the current settings values
     */
    public void startInput(final String combiningSpec, final SettingsValues settingsValues) {
        mEnteredText = null;
        mWordBeingCorrectedByCursor = null;
        mConnection.onStartInput();
        if (!mWordComposer.getTypedWord().isEmpty()) {
            // For messaging apps that offer send button, the IME does not get the opportunity
            // to capture the last word. This block should capture those uncommitted words.
            // The timestamp at which it is captured is not accurate but close enough.
            StatsUtils.onWordCommitUserTyped(mWordComposer.getTypedWord(), mWordComposer.isBatchMode());
        }
        mWordComposer.restartCombining(combiningSpec);
        resetComposingState(true /* alsoResetLastComposedWord */);
        mDeleteCount = 0;
        mSpaceState = SpaceState.NONE;
        mRecapitalizeStatus.disable(); // Do not perform recapitalize until the cursor is moved once
        mCurrentlyPressedHardwareKeys.clear();
        mSuggestedWords = SuggestedWords.getEmptyInstance();
        // In some cases (e.g. after rotation of the device, or when scrolling the text before bringing up keyboard)
        // editorInfo.initialSelStart is not the actual cursor position, so we try using some heuristics to find the correct position.
        mConnection.tryFixIncorrectCursorPosition();
        cancelDoubleSpacePeriodCountdown();
        mInputLogicHandler.reset();
        mConnection.requestCursorUpdates(true, true);
        setInlineEmojiSearchAction(false);
    }

    /**
     * Call this when the subtype changes.
     * @param combiningSpec the spec string for the combining rules
     * @param settingsValues the current settings values
     */
    public void onSubtypeChanged(final String combiningSpec, final SettingsValues settingsValues) {
        finishInput();
        startInput(combiningSpec, settingsValues);
    }

    /**
     * Call this when the orientation changes.
     * @param settingsValues the current values of the settings.
     */
    public void onOrientationChange(final SettingsValues settingsValues) {
        // If !isComposingWord, #commitTyped() is a no-op, but still, it's better to avoid
        // the useless IPC of {begin,end}BatchEdit.
        if (mWordComposer.isComposingWord()) {
            mConnection.beginBatchEdit();
            // If we had a composition in progress, we need to commit the word so that the
            // suggestionsSpan will be added. This will allow resuming on the same suggestions
            // after rotation is finished.
            commitTyped(settingsValues, LastComposedWord.NOT_A_SEPARATOR);
            mConnection.endBatchEdit();
        }
    }

    /**
     * Clean up the input logic after input is finished.
     */
    public void finishInput() {
        if (mWordComposer.isComposingWord()) {
            mConnection.finishComposingText();
            StatsUtils.onWordCommitUserTyped(mWordComposer.getTypedWord(), mWordComposer.isBatchMode());
        }
        resetComposingState(true);
        mInputLogicHandler.reset();
        mSpaceState = SpaceState.NONE;
    }

    /**
     * React to a string input.
     * <p>
     * This is triggered by keys that input many characters at once, like the ".com" key or
     * some additional keys for example.
     *
     * @param settingsValues the current values of the settings.
     * @param event the input event containing the data.
     * @return the complete transaction object
     */
    public InputTransaction onTextInput(final SettingsValues settingsValues, final Event event,
            final int keyboardShiftMode, final LatinIME.UIHandler handler) {
        final String rawText = event.getTextToCommit().toString();
        final InputTransaction inputTransaction = new InputTransaction(settingsValues, event,
                SystemClock.uptimeMillis(), mSpaceState,
                getActualCapsMode(settingsValues, keyboardShiftMode));
        mConnection.beginBatchEdit();
        if (mWordComposer.isComposingWord()) {
            if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) {
                // stop composing, otherwise the text will end up at the end of the current word
                mConnection.finishComposingText();
                resetComposingState(false);
            } else {
                commitCurrentAutoCorrection(settingsValues, rawText, handler);
                addToHistoryIfEmoji(rawText, settingsValues); // add emoji after committing text
            }
        } else {
            addToHistoryIfEmoji(rawText, settingsValues); // add emoji before resetting, otherwise lastComposedWord is empty
            resetComposingState(true /* alsoResetLastComposedWord */);
        }
        handler.postUpdateSuggestionStrip(SuggestedWords.INPUT_STYLE_TYPING);
        final String text = performSpecificTldProcessingOnTextInput(rawText);
        if (SpaceState.PHANTOM == mSpaceState) {
            insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
        }
        mConnection.commitText(text, 1);
        StatsUtils.onWordCommitUserTyped(mEnteredText, mWordComposer.isBatchMode());
        mConnection.endBatchEdit();
        // Space state must be updated before calling updateShiftState
        mSpaceState = SpaceState.NONE;
        mEnteredText = text;
        mWordBeingCorrectedByCursor = null;
        inputTransaction.setDidAffectContents();
        inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
        return inputTransaction;
    }

    /**
     * A suggestion was picked from the suggestion strip.
     * @param settingsValues the current values of the settings.
     * @param suggestionInfo the suggestion info.
     * @param keyboardShiftState the shift state of the keyboard, as returned by
     *     {@link helium314.keyboard.keyboard.KeyboardSwitcher#getKeyboardShiftMode()}
     * @return the complete transaction object
     */
    // Called from {@link SuggestionStripView} through the {@link SuggestionStripView#Listener}
    // interface
    public InputTransaction onPickSuggestionManually(final SettingsValues settingsValues,
            final SuggestedWordInfo suggestionInfo, final int keyboardShiftState,
            final String currentKeyboardScript, final LatinIME.UIHandler handler) {
        if (isInlineEmojiSearchAction()) {
            deleteTextReplacedByEmoji();
        }
        // Combining-mode revert: if the strip is still showing alternatives for a word that
        // the grace timer just auto-committed (keep_alternatives / alternatives_then_next_word
        // modes), the suggestion the user picked is meant to REPLACE that word, not append
        // after it. Delete the auto-committed text (word + autospace) before continuing into
        // the normal pick path, so commitChosenWord lands at the original word's position.
        if (mAutoCommitRevertLength > 0 && !mWordComposer.isComposingWord()) {
            mConnection.beginBatchEdit();
            mConnection.deleteTextBeforeCursor(mAutoCommitRevertLength);
            mConnection.endBatchEdit();
            mAutoCommitRevertLength = 0;
            // The autocommit's autospace got deleted along with the word. Clear PHANTOM so
            // commitChosenWord doesn't try to insert another one in addition to the one
            // re-inserted at the bottom of this method by mAutospaceAfterSuggestion.
            mSpaceState = SpaceState.NONE;
            // Re-insert the trailing space ourselves at the bottom of this method (after
            // commitChosenWord). PHANTOM-via-mAutospaceAfterSuggestion would only fire on the
            // next character; the user expects the cursor to land past the space immediately.
            mInsertTrailingSpaceAfterPick = true;
        }

        final SuggestedWords suggestedWords = mSuggestedWords;
        final String suggestion = suggestionInfo.mWord;
        // If this is a punctuation picked from the suggestion strip, pass it to onCodeInput
        if (suggestion.length() == 1 && suggestedWords.isPunctuationSuggestions()) {
            // We still want to log a suggestion click.
            StatsUtils.onPickSuggestionManually(mSuggestedWords, suggestionInfo, mDictionaryFacilitator);
            // Word separators are suggested before the user inputs something.
            // Rely on onCodeInput to do the complicated swapping/stripping logic consistently.
            final Event event = Event.createPunctuationSuggestionPickedEvent(suggestionInfo);
            return onCodeInput(settingsValues, event, keyboardShiftState, currentKeyboardScript, handler);
        }

        final Event event = Event.createSuggestionPickedEvent(suggestionInfo);
        final InputTransaction inputTransaction = new InputTransaction(settingsValues,
                event, SystemClock.uptimeMillis(), mSpaceState, keyboardShiftState);
        // Manual pick affects the contents of the editor, so we take note of this. It's important
        // for the sequence of language switching.
        inputTransaction.setDidAffectContents();
        mConnection.beginBatchEdit();
        if (SpaceState.PHANTOM == mSpaceState && suggestion.length() > 0
                // In the batch input mode, a manually picked suggested word should just replace
                // the current batch input text and there is no need for a phantom space.
                && !mWordComposer.isBatchMode()
                // when a commit was reverted and user chose a different suggestion, we don't want
                // to insert a space before the picked word
                && !mJustRevertedACommit) {
            final int firstChar = Character.codePointAt(suggestion, 0);
            if (!settingsValues.isWordSeparator(firstChar)
                    || settingsValues.isUsuallyPrecededBySpace(firstChar)) {
                insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
            }
        }
        mJustRevertedACommit = false;

        // TODO: We should not need the following branch. We should be able to take the same
        // code path as for other kinds, use commitChosenWord, and do everything normally. We will
        // however need to reset the suggestion strip right away, because we know we can't take
        // the risk of calling commitCompletion twice because we don't know how the app will react.
        if (suggestionInfo.isKindOf(SuggestedWordInfo.KIND_APP_DEFINED)) {
            mSuggestedWords = SuggestedWords.getEmptyInstance();
            mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
            inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
            resetComposingState(true /* alsoResetLastComposedWord */);
            mConnection.commitCompletion(suggestionInfo.mApplicationSpecifiedCompletionInfo);
            mConnection.endBatchEdit();
            return inputTransaction;
        }

        commitChosenWord(settingsValues, suggestion, LastComposedWord.COMMIT_TYPE_MANUAL_PICK, LastComposedWord.NOT_A_SEPARATOR);
        mConnection.endBatchEdit();
        // Combining-mode revert: the auto-committed word's trailing space was wiped along
        // with the word at the top of this method; re-insert it now so cursor lands at
        // "the |" rather than "the|". Uses the same helper as the timer's autospace so
        // URL / e-mail / phantom guards apply consistently.
        if (mInsertTrailingSpaceAfterPick) {
            mInsertTrailingSpaceAfterPick = false;
            insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
            // Don't ALSO set PHANTOM below — we already inserted a real space.
            mSpaceState = SpaceState.NONE;
        } else if (settingsValues.mAutospaceAfterSuggestion) {
            mSpaceState = SpaceState.PHANTOM;
        }
        // Don't allow cancellation of manual pick
        mLastComposedWord.deactivate();
        inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
        setInlineEmojiSearchAction(false);

        // If we're not showing the "Touch again to save", then update the suggestion strip.
        // That's going to be predictions (or punctuation suggestions), so INPUT_STYLE_NONE.
        handler.postUpdateSuggestionStrip(SuggestedWords.INPUT_STYLE_NONE);

        StatsUtils.onPickSuggestionManually(mSuggestedWords, suggestionInfo, mDictionaryFacilitator);
        StatsUtils.onWordCommitSuggestionPickedManually(suggestionInfo.mWord, mWordComposer.isBatchMode());
        return inputTransaction;
    }

    /**
     * Consider an update to the cursor position. Evaluate whether this update has happened as
     * part of normal typing or whether it was an explicit cursor move by the user. In any case,
     * do the necessary adjustments.
     * @param oldSelStart old selection start
     * @param oldSelEnd old selection end
     * @param newSelStart new selection start
     * @param newSelEnd new selection end
     * @param settingsValues the current values of the settings.
     * @return whether the cursor has moved as a result of user interaction.
     */
    public boolean onUpdateSelection(final int oldSelStart, final int oldSelEnd, final int newSelStart,
             final int newSelEnd, final int composingSpanStart, final int composingSpanEnd, final SettingsValues settingsValues) {
        if (mConnection.isBelatedExpectedUpdate(oldSelStart, newSelStart, oldSelEnd, newSelEnd, composingSpanStart, composingSpanEnd)) {
            return false;
        }
        // TODO: the following is probably better done in resetEntireInputState().
        // it should only happen when the cursor moved, and the very purpose of the
        // test below is to narrow down whether this happened or not. Likewise with
        // the call to updateShiftState.
        // We set this to NONE because after a cursor move, we don't want the space
        // state-related special processing to kick in.
        mSpaceState = SpaceState.NONE;

        final boolean selectionChangedOrSafeToReset =
                oldSelStart != newSelStart || oldSelEnd != newSelEnd // selection changed
                || !mWordComposer.isComposingWord(); // safe to reset
        final boolean hasOrHadSelection = (oldSelStart != oldSelEnd || newSelStart != newSelEnd);
        final int moveAmount = newSelStart - oldSelStart;
        // As an added small gift from the framework, it happens upon rotation when there
        // is a selection that we get a wrong cursor position delivered to startInput() that
        // does not get reflected in the oldSel{Start,End} parameters to the next call to
        // onUpdateSelection. In this case, we may have set a composition, and when we're here
        // we realize we shouldn't have. In theory, in this case, selectionChangedOrSafeToReset
        // should be true, but that is if the framework had taken that wrong cursor position
        // into account, which means we have to reset the entire composing state whenever there
        // is or was a selection regardless of whether it changed or not.
        if (hasOrHadSelection || !settingsValues.needsToLookupSuggestions()
                || (selectionChangedOrSafeToReset
                        && !mWordComposer.moveCursorByAndReturnIfInsideComposingWord(moveAmount))) {
            // If we are composing a word and moving the cursor, we would want to set a
            // suggestion span for recorrection to work correctly. Unfortunately, that
            // would involve the keyboard committing some new text, which would move the
            // cursor back to where it was. Latin IME could then fix the position of the cursor
            // again, but the asynchronous nature of the calls results in this wreaking havoc
            // with selection on double tap and the like.
            // Another option would be to send suggestions each time we set the composing
            // text, but that is probably too expensive to do, so we decided to leave things
            // as is.
            // Also, we're posting a resume suggestions message, and this will update the
            // suggestions strip in a few milliseconds, so if we cleared the suggestion strip here
            // we'd have the suggestion strip noticeably janky. To avoid that, we don't clear
            // it here, which means we'll keep outdated suggestions for a split second but the
            // visual result is better.
            resetEntireInputState(newSelStart, newSelEnd, false /* clearSuggestionStrip */);
            // If the user is in the middle of correcting a word, we should learn it before moving
            // the cursor away.
            if (!TextUtils.isEmpty(mWordBeingCorrectedByCursor)) {
                performAdditionToUserHistoryDictionary(settingsValues, mWordBeingCorrectedByCursor,
                        NgramContext.EMPTY_PREV_WORDS_INFO);
            }
        } else {
            // resetEntireInputState calls resetCachesUponCursorMove, but forcing the
            // composition to end. But in all cases where we don't reset the entire input
            // state, we still want to tell the rich input connection about the new cursor
            // position so that it can update its caches.
            mConnection.resetCachesUponCursorMoveAndReturnSuccess(
                    newSelStart, newSelEnd, false /* shouldFinishComposition */);
        }

        // The cursor has been moved : we now accept to perform recapitalization
        mRecapitalizeStatus.enable();
        // We moved the cursor. If we are touching a word, we need to resume suggestion.
        mLatinIME.mHandler.postResumeSuggestions(true /* shouldDelay */);
        // Stop the last recapitalization, if started.
        mRecapitalizeStatus.stop();
        mWordBeingCorrectedByCursor = null;
        return true;
    }

    public boolean moveCursorByAndReturnIfInsideComposingWord(int distance) {
        return mWordComposer.moveCursorByAndReturnIfInsideComposingWord(distance);
    }

    /**
     * React to a code input. It may be a code point to insert, or a symbolic value that influences
     * the keyboard behavior.
     * <p>
     * Typically, this is called whenever a key is pressed on the software keyboard. This is not
     * the entry point for gesture input; see the onBatchInput* family of functions for this.
     *
     * @param settingsValues the current settings values.
     * @param event the event to handle.
     * @param keyboardShiftMode the current shift mode of the keyboard, as returned by
     *     {@link helium314.keyboard.keyboard.KeyboardSwitcher#getKeyboardShiftMode()}
     * @return the complete transaction object
     */
    public InputTransaction onCodeInput(final SettingsValues settingsValues,
            @NonNull final Event event, final int keyboardShiftMode,
            final String currentKeyboardScript, final LatinIME.UIHandler handler) {
        mWordBeingCorrectedByCursor = null;
        mJustRevertedACommit = false;
        final Event processedEvent = mWordComposer.processEvent(event);
        final InputTransaction inputTransaction = new InputTransaction(settingsValues,
                processedEvent, SystemClock.uptimeMillis(), mSpaceState,
                getActualCapsMode(settingsValues, keyboardShiftMode));
        if (processedEvent.getKeyCode() != KeyCode.DELETE
                || inputTransaction.getTimestamp() > mLastKeyTime + Constants.LONG_PRESS_MILLISECONDS) {
            mDeleteCount = 0;
        }
        mLastKeyTime = inputTransaction.getTimestamp();
        mConnection.beginBatchEdit();
        if (!mWordComposer.isComposingWord()) {
            // TODO: is this useful? It doesn't look like it should be done here, but rather after
            // a word is committed.
            mIsAutoCorrectionIndicatorOn = false;
        }

        // TODO: Consolidate the double-space period timer, mLastKeyTime, and the space state.
        if (processedEvent.getCodePoint() != Constants.CODE_SPACE) {
            cancelDoubleSpacePeriodCountdown();
        }

        Event currentEvent = processedEvent;
        while (null != currentEvent) {
            if (currentEvent.isConsumed()) {
                handleConsumedEvent(currentEvent, inputTransaction);
            } else if (currentEvent.isFunctionalKeyEvent()) {
                handleFunctionalEvent(currentEvent, inputTransaction, currentKeyboardScript, handler);
            } else {
                handleNonFunctionalEvent(currentEvent, inputTransaction, handler);
            }
            currentEvent = currentEvent.getNextEvent();
        }
        // Try to record the word being corrected when the user enters a word character or
        // the backspace key.
        if (!mConnection.hasSlowInputConnection() && !mWordComposer.isComposingWord()
                && (settingsValues.isWordCodePoint(processedEvent.getCodePoint())
                    || processedEvent.getKeyCode() == KeyCode.DELETE)
                ) {
            mWordBeingCorrectedByCursor = getWordAtCursor(settingsValues, currentKeyboardScript);
        }
        if (!inputTransaction.didAutoCorrect() && processedEvent.getKeyCode() != KeyCode.SHIFT
                && processedEvent.getKeyCode() != KeyCode.CAPS_LOCK
                && processedEvent.getKeyCode() != KeyCode.SYMBOL_ALPHA
                && processedEvent.getKeyCode() != KeyCode.ALPHA
                && processedEvent.getKeyCode() != KeyCode.SYMBOL)
            mLastComposedWord.deactivate();
        if (KeyCode.DELETE != processedEvent.getKeyCode()) {
            mEnteredText = null;
        }
        mConnection.endBatchEdit();
        return inputTransaction;
    }

    public void onStartBatchInput(final SettingsValues settingsValues,
            final KeyboardSwitcher keyboardSwitcher, final LatinIME.UIHandler handler) {
        // Snapshot the keyboard's shift mode BEFORE any state mutates — the shifted indicator
        // typically auto-clears once the gesture starts moving, so by the time
        // onUpdateTailBatchInputCompleted fires the live mode reads as UNSHIFTED.
        // We compute the *actual* caps mode (resolves AUTO_SHIFTED into AUTO_SHIFT_LOCKED if
        // the input field is in all-caps), so a true all-caps field gives the right answer.
        mShiftModeAtGestureStart = getActualCapsMode(settingsValues, keyboardSwitcher.getKeyboardShiftMode());
        mWordBeingCorrectedByCursor = null;
        mInputLogicHandler.onStartBatchInput();
        handler.showGesturePreviewAndSetSuggestions(SuggestedWords.getEmptyBatchInstance(), false);
        handler.cancelUpdateSuggestionStrip();
        ++mAutoCommitSequenceNumber;
        // Combining mode: snapshot before cancel; the gesture will re-arm the timer on completion.
        final boolean wasInCombiningMode = mInCombiningMode;
        cancelCombiningMode();
        mConnection.beginBatchEdit();
        // Two-thumb typing (#1.1 + combining-mode): two ways the gesture can EXTEND an existing
        // composing word instead of replacing it:
        //   * manual spacing (always extends, user explicitly opted out of autospace), or
        //   * combining mode active at gesture-start (a tap or prior gesture has armed the
        //     grace timer; this gesture should append to the same word — e.g. "tap s, swipe
        //     ilver" → "silver", or "swipe tech, swipe nology" → "technology"). The window
        //     is enforced by the Handler-driven timer, so a long gesture never "ages out".
        // Both gate on cursor-at-end of an existing composing word; if the cursor isn't there
        // we can't safely extend.
        final boolean cursorAtEndOfComposingWord = mWordComposer.isComposingWord()
                && !mWordComposer.isCursorFrontOrMiddleOfComposingWord();
        final boolean manualSpacingExtend = settingsValues.mGestureManualSpacing
                && cursorAtEndOfComposingWord;
        final boolean combiningExtend = wasInCombiningMode && cursorAtEndOfComposingWord;
        // Legacy tap-promotion flag retained so onUpdateTailBatchInputCompleted reuses the
        // same decision (kept here so a long gesture doesn't fall out of the window between
        // start and end). When combining mode supersedes tap-promotion the flag just tracks
        // "we want to extend".
        mGestureExtendsByTapPromotion = combiningExtend;
        final boolean extendComposingWord = manualSpacingExtend || combiningExtend;
        if (mWordComposer.isComposingWord()) {
            if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) {
                // If we are in the middle of a recorrection, we need to commit the recorrection
                // first so that we can insert the batch input at the current cursor position.
                // We also need to unlearn the original word that is now being corrected.
                unlearnWord(mWordComposer.getTypedWord(), settingsValues, Constants.EVENT_BACKSPACE);
                resetEntireInputState(mConnection.getExpectedSelectionStart(), mConnection.getExpectedSelectionEnd(),
                        true);
            } else if (extendComposingWord) {
                // Manual spacing OR tap-promotion-extend: keep the existing composing word;
                // the new gesture's result will be APPENDED in onUpdateTailBatchInputCompleted
                // instead of replacing it.
                //
                // Multi-part composition (#1.6): snapshot the prior fragments' input
                // pointers (gesture trail or tap-coords) so each setBatchInputPointers call
                // during this gesture merges them in. Without this the lib only sees the
                // SECOND fragment and gives bogus words ('biology' for 'nology'). The
                // merge re-times the base so the stream looks like one continuous stroke.
                if (settingsValues.mMultipartAutoExtendInCombining
                        && settingsValues.mCombiningGraceMs > 0) {
                    mWordComposer.setExtendBatchInputBase(mWordComposer.getInputPointers());
                }
            } else if (mWordComposer.isSingleLetter() && !isInlineEmojiSearchAction()) {
                // We auto-correct the previous (typed, not gestured) string iff it's one
                // character
                // long. The reason for this is, even in the middle of gesture typing, you'll
                // still
                // tap one-letter words and you want them auto-corrected (typically, "i" in
                // English
                // should become "I"). However for any longer word, we assume that the reason
                // for
                // tapping probably is that the word you intend to type is not in the
                // dictionary,
                // so we do not attempt to correct, on the assumption that if that was a
                // dictionary
                // word, the user would probably have gestured instead.
                commitCurrentAutoCorrection(settingsValues, LastComposedWord.NOT_A_SEPARATOR,
                        handler);
            } else {
                commitTyped(settingsValues, LastComposedWord.NOT_A_SEPARATOR);
            }
        } else if (mConnection.hasSelection()) {
            final CharSequence selectedText = mConnection.getSelectedText(0);
            if (selectedText != null)
                // set selected text as rejected to avoid glide typing resulting in exactly the selected word again
                mWordComposer.setRejectedBatchModeSuggestion(selectedText.toString());
        }
        final int codePointBeforeCursor = mConnection.getCodePointBeforeCursor();
        if (Character.isLetterOrDigit(codePointBeforeCursor)
                || settingsValues.isUsuallyFollowedBySpace(codePointBeforeCursor)) {
            final boolean autoShiftHasBeenOverriden = keyboardSwitcher.getKeyboardShiftMode() !=
                    getCurrentAutoCapsState(settingsValues);
            if (settingsValues.mAutospaceBeforeGestureTyping)
                mSpaceState = SpaceState.PHANTOM;
            if (!autoShiftHasBeenOverriden) {
                // When we change the space state, we need to update the shift state of the
                // keyboard unless it has been overridden manually. This is happening for example
                // after typing some letters and a period, then gesturing; the keyboard is not in
                // caps mode yet, but since a gesture is starting, it should go in caps mode,
                // unless the user explictly said it should not.
                keyboardSwitcher.requestUpdatingShiftState(getCurrentAutoCapsState(settingsValues),
                        getCurrentRecapitalizeState());
            }
        }
        mConnection.endBatchEdit();
        mWordComposer.setCapitalizedModeAtStartComposingTime(
                getActualCapsMode(settingsValues, keyboardSwitcher.getKeyboardShiftMode()));
    }

    /* The sequence number member is only used in onUpdateBatchInput. It is increased each time
     * auto-commit happens. The reason we need this is, when auto-commit happens we trim the
     * input pointers that are held in a singleton, and to know how much to trim we rely on the
     * results of the suggestion process that is held in mSuggestedWords.
     * However, the suggestion process is asynchronous, and sometimes we may enter the
     * onUpdateBatchInput method twice without having recomputed suggestions yet, or having
     * received new suggestions generated from not-yet-trimmed input pointers. In this case, the
     * mIndexOfTouchPointOfSecondWords member will be out of date, and we must not use it lest we
     * remove an unrelated number of pointers (possibly even more than are left in the input
     * pointers, leading to a crash).
     * To avoid that, we increase the sequence number each time we auto-commit and trim the
     * input pointers, and we do not use any suggested words that have been generated with an
     * earlier sequence number.
     */
    private int mAutoCommitSequenceNumber = 1;
    public void onUpdateBatchInput(final InputPointers batchPointers) {
        mInputLogicHandler.onUpdateBatchInput(batchPointers, mAutoCommitSequenceNumber);
    }

    public void onEndBatchInput(final InputPointers batchPointers) {
        mInputLogicHandler.updateTailBatchInput(batchPointers, mAutoCommitSequenceNumber);
        ++mAutoCommitSequenceNumber;
    }

    public void onCancelBatchInput(final LatinIME.UIHandler handler) {
        mInputLogicHandler.onCancelBatchInput();
        // Combining mode: cancelled gesture wipes the would-be-fragment from the composing
        // word — drop the timer so we don't fire an autospace based on stale state.
        cancelCombiningMode();
        // Drop any seed codepoint stashed by PointerTracker so the next gesture doesn't
        // strip its first letter against a stale seed.
        helium314.keyboard.keyboard.PointerTracker.consumeGestureSeedCodepoint();
        // Tap-promotion-extend was a per-gesture decision; clear on cancel so the NEXT
        // gesture re-evaluates against current timing.
        mGestureExtendsByTapPromotion = false;
        // Two-thumb typing (#1.1): if a gesture got far enough into onUpdateBatchInput before
        // being cancelled, {@link InputLogicHandler#updateBatchInput} has already flipped the
        // composer into batch mode and replaced its input pointers with the cancelled
        // gesture's. Under manual spacing that composer is the user's persistent composing
        // word — leaving it in batch mode would cause backspace to delete the whole thing
        // in one keystroke (see InputLogic.handleBackspaceEvent isBatchMode branch). Restore
        // typed-word semantics so subsequent edits behave normally.
        if (mWordComposer.isComposingWord() && mWordComposer.isBatchMode()) {
            mWordComposer.unsetBatchMode();
        }
        handler.showGesturePreviewAndSetSuggestions(
                SuggestedWords.getEmptyInstance(), true /* dismissGestureFloatingPreviewText */);
    }

    // ---- Two-thumb typing (#1.1): fragment-boundary tracking for PREF_GESTURE_FRAGMENT_BACKSPACE.
    // The list grows AFTER each fragment is appended to the composing word, recording the
    // word's length at that point. Backspace pops the most-recent entry and shrinks the
    // composing word to the previous one (or 0 if empty), giving the user "undo one swipe /
    // tap" rather than "delete one character" semantics. List is cleared whenever the
    // composing word is committed or reset; stale entries beyond the current length are
    // filtered in {@link #tryFragmentBackspace} as a defensive measure.

    /** Record a fragment boundary at the current composing-word length. No-op when not in manual+fragment mode. */
    private void recordFragmentBoundaryIfTracking(final SettingsValues sv) {
        // Two paths into fragment tracking:
        //   * Legacy manual-spacing mode (#1.1) — gated on both prefs.
        //   * Multi-part word composition (#1.6) — combining mode is the trigger; fragment
        //     backspace works regardless of manual-spacing so the user can pop the last
        //     swipe/tap they joined onto the word.
        final boolean legacyTracking = sv.mGestureManualSpacing && sv.mGestureFragmentBackspace;
        final boolean multipartTracking = sv.mMultipartAutoExtendInCombining
                && sv.mCombiningGraceMs > 0
                && sv.mGestureFragmentBackspace;
        if (!legacyTracking && !multipartTracking) return;
        if (!mWordComposer.isComposingWord()) return;
        final int len = mWordComposer.getTypedWord().length();
        // Don't record duplicates (e.g. the same fragment appended twice in quick succession).
        if (!mGestureFragmentBoundaries.isEmpty()
                && mGestureFragmentBoundaries.get(mGestureFragmentBoundaries.size() - 1) == len) {
            return;
        }
        mGestureFragmentBoundaries.add(len);
    }

    /** Clear all recorded fragment boundaries. Call after committing / resetting the composing word. */
    private void clearFragmentBoundaries() {
        if (!mGestureFragmentBoundaries.isEmpty()) mGestureFragmentBoundaries.clear();
    }

    // ---- Unified combining-mode helpers --------------------------------------------------

    /**
     * Enter or refresh combining mode: cancel any prior pending commit, schedule a new one,
     * and notify the keyboard view to draw the countdown progress bar. No-op when grace is 0.
     *
     * @param fromTap when true, the trigger was a letter tap and we add
     *     {@code mCombiningTapExtraMs} on top of the base grace (peck-typers need more time
     *     between letters than swipers do between gestures). When false (gesture trigger),
     *     only the base grace applies.
     */
    private void enterCombiningMode(final SettingsValues settingsValues, final boolean fromTap) {
        // Any new input that arms combining mode also invalidates the "next space swaps to
        // next-word predictions" one-shot — the user moved on to typing the next word.
        mAutospaceAlternativesPending = false;
        mAutoCommitRevertLength = 0;
        mLastGestureCommittedLength = 0;
        mAutospaceJustWritten = false;
        final int baseGraceMs = settingsValues.mCombiningGraceMs;
        if (baseGraceMs <= 0) return;
        // We only arm the timer if there's actually a composing word — for tap-only events on
        // separators / cursor-front recompositions there's nothing to auto-commit, and arming
        // the timer would draw a spurious progress bar.
        if (!mWordComposer.isComposingWord()) return;
        final int graceMs = baseGraceMs + Math.max(0, settingsValues.mCombiningTapExtraMs);
        cancelCombiningTimerOnly();
        mInCombiningMode = true;
        final long startTime = SystemClock.uptimeMillis();
        mPendingCombiningCommit = () -> onCombiningGraceExpired();
        mCombiningHandler.postDelayed(mPendingCombiningCommit, graceMs);
        final MainKeyboardView kv = KeyboardSwitcher.getInstance().getMainKeyboardView();
        if (kv != null) kv.setCombiningMode(true, startTime, graceMs);
    }

    /**
     * Cancel a pending combining-mode commit without firing it. Used on backspace,
     * separators, cursor moves, explicit commits, and any place that wipes the composing
     * word — callers there commit through their own paths and don't want the timer to
     * race them.
     */
    void cancelCombiningMode() {
        cancelCombiningTimerOnly();
        // Backspace / separator / cursor-move / cancel paths also drop the one-shot — they
        // represent the user explicitly leaving the just-auto-committed context.
        mAutospaceAlternativesPending = false;
        mAutoCommitRevertLength = 0;
        mLastGestureCommittedLength = 0;
        mAutospaceJustWritten = false;
        if (mInCombiningMode) {
            mInCombiningMode = false;
            final MainKeyboardView kv = KeyboardSwitcher.getInstance().getMainKeyboardView();
            if (kv != null) kv.setCombiningMode(false, 0L, 0);
        }
    }

    private void cancelCombiningTimerOnly() {
        if (mPendingCombiningCommit != null) {
            mCombiningHandler.removeCallbacks(mPendingCombiningCommit);
            mPendingCombiningCommit = null;
        }
    }

    /** Combining-mode seeding helper: last codepoint of the current composing word, or 0
     *  if no composing word. Currently unused but kept available for future seeding work. */
    @SuppressWarnings("unused")
    private int lastCodepointOfTypedWord() {
        if (!mWordComposer.isComposingWord()) return 0;
        final String w = mWordComposer.getTypedWord();
        if (w.isEmpty()) return 0;
        return w.codePointBefore(w.length());
    }

    /** Public accessor so PointerTracker / KeyboardActionListenerImpl can ask "are we extending right now?". */
    public boolean isInCombiningMode() {
        return mInCombiningMode;
    }

    /** One-shot flag set by {@link #onCombiningGraceExpired} when the user picked behavior #3
     *  ("alternatives_then_next_word"): the strip currently shows the just-committed word's
     *  alternatives, and the NEXT space tap should switch to next-word predictions instead
     *  of inserting a second space. Consumed in {@link #handleSeparatorEvent} for CODE_SPACE. */
    private boolean mAutospaceAlternativesPending;
    /** Number of characters the timer-driven auto-commit wrote into the editor (typed word +
     *  optional autospace). When greater than zero AND the strip is showing the
     *  just-auto-committed word's alternatives, picking a suggestion should REPLACE the
     *  auto-committed text instead of appending — we delete this many chars before re-committing.
     *  Cleared on any new input that arms combining mode, on cancel, on the next space tap,
     *  and after a suggestion-pick. */
    private int mAutoCommitRevertLength;
    /** Length of the most recent gesture-driven commit (typed word + autospace). Set in
     *  {@link #onCombiningGraceExpired} when {@code mWordComposer.isBatchMode()} was true at
     *  commit time. Consumed by the first backspace tap when
     *  {@code PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD} is on, deleting the whole word
     *  + space in one keystroke (unless an autocorrect-revert applies first — that always
     *  wins). Cleared on any new input that arms combining mode. */
    private int mLastGestureCommittedLength;
    /** Set in {@link #onPickSuggestionManually} when the picker reverted an auto-committed
     *  word; consumed at the bottom of the same method to insert a visible trailing space so
     *  the cursor lands at "the |" instead of "the|". */
    private boolean mInsertTrailingSpaceAfterPick;

    /**
     * Fired by the Handler when the grace timer expires. Commit the current composing word
     * (with or without autocorrect per {@code PREF_COMBINING_AUTOCORRECT_ON_AUTOSPACE}), then
     * insert an autospace via the existing helper. Clears the visual indicator.
     */
    private void onCombiningGraceExpired() {
        mPendingCombiningCommit = null;
        mInCombiningMode = false;
        final MainKeyboardView kv = KeyboardSwitcher.getInstance().getMainKeyboardView();
        if (kv != null) kv.setCombiningMode(false, 0L, 0);
        final SettingsValues sv = Settings.getInstance().getCurrent();
        if (!mWordComposer.isComposingWord()) return;
        // Capture whether the word being committed by this timer came from a gesture. We
        // use this AFTER commit to decide whether to arm the "backspace deletes whole word"
        // behavior. Tap-only words don't qualify — char-by-char backspace is fine for those.
        final boolean wasBatchMode = mWordComposer.isBatchMode();
        // Snapshot the typed word's length BEFORE commit — the composing text is already in
        // the editor (added via setComposingTextInternal during the tap/gesture path), so the
        // commit itself doesn't move the cursor. We need the typed-word length plus whatever
        // chars the autospace path writes to know how much to undo on a suggestion-pick.
        final String typedWordAtCommit = mWordComposer.getTypedWord();
        final int cursorBefore = mConnection.getExpectedSelectionEnd();
        mConnection.beginBatchEdit();
        if (sv.mCombiningAutocorrectOnAutospace) {
            // Use the same path as a regular space-commit: it runs autocorrect when the
            // composer has a stronger suggestion, and falls back to typed-word otherwise.
            commitCurrentAutoCorrection(sv, LastComposedWord.NOT_A_SEPARATOR, mLatinIME.mHandler);
        } else {
            commitTyped(sv, LastComposedWord.NOT_A_SEPARATOR);
        }
        // Track whether the helper actually wrote a space (skipped for URL / e-mail / phantom).
        final int beforeSpace = mConnection.getExpectedSelectionEnd();
        insertAutomaticSpaceIfOptionsAndTextAllow(sv);
        final boolean autospaceInserted = mConnection.getExpectedSelectionEnd() > beforeSpace;
        // If we DID insert an autospace, fix up mLastComposedWord so revertCommit (backspace +
        // PREF_BACKSPACE_REVERTS_AUTOCORRECT) deletes the space along with the word. Without
        // this the existing revert code's `deleteLength = cancelLength + separatorLength`
        // would only delete the word, and in DEBUG builds the bundled assertion against
        // `getTextBeforeCursor(...).subSequence(0, cancelLength) equals committedWord` throws
        // because the last cancelLength chars now include the trailing space, not the word.
        if (autospaceInserted && mLastComposedWord != null
                && mLastComposedWord != LastComposedWord.NOT_A_COMPOSED_WORD
                && Constants.STRING_SPACE.equals(mLastComposedWord.mSeparatorString) == false) {
            mLastComposedWord = new LastComposedWord(
                    mLastComposedWord.mEvents,
                    mLastComposedWord.mInputPointers,
                    mLastComposedWord.mTypedWord,
                    mLastComposedWord.mCommittedWord,
                    Constants.STRING_SPACE,
                    mLastComposedWord.mNgramContext,
                    mLastComposedWord.mCapitalizedMode);
        }
        // Don't set PHANTOM here — we already wrote the space to the editor. PHANTOM would
        // make the next letter call insertAutomaticSpaceIfOptionsAndTextAllow AGAIN (see
        // handleNonSeparatorEvent line ~1760), giving a double space. Instead, set a
        // dedicated one-shot flag that handleSeparatorEvent uses to strip the autospace if
        // a punctuation character follows. The flag is cleared by enterCombiningMode (next
        // input took over), cancelCombiningMode (backspace etc), or once consumed.
        mAutospaceJustWritten = autospaceInserted;
        mSpaceState = SpaceState.NONE;
        mConnection.endBatchEdit();
        final int cursorAfter = mConnection.getExpectedSelectionEnd();
        // The commit doesn't move the cursor for the composing text itself (it was already
        // on-screen via setComposingTextInternal). The only cursor delta is from autospace
        // (0 or 1 char). We additionally need to undo the COMMITTED word, whose length comes
        // from mLastComposedWord.mCommittedWord — that captures the autocorrect-applied
        // value correctly (e.g. "teh" → "the" gives 3-char committed; "u" → "you" gives
        // 3-char committed). Fall back to the typed-word length if for any reason the last
        // commit didn't populate mCommittedWord.
        final int autospaceChars = Math.max(0, cursorAfter - cursorBefore);
        final int committedLen = (mLastComposedWord != null && mLastComposedWord.mCommittedWord != null)
                ? mLastComposedWord.mCommittedWord.length()
                : typedWordAtCommit.length();
        final int writtenChars = committedLen + autospaceChars;
        // Behavior selection for the suggestion strip after auto-commit:
        //   "next_word"                    : drop the just-committed-word alternatives and
        //                                     ask for next-word predictions, like a normal space.
        //   "keep_alternatives"            : leave whatever the gesture/typing strip showed —
        //                                     user can tap one to revert the auto-correction.
        //   "alternatives_then_next_word"  : keep alternatives for now, but arm a one-shot so
        //                                     the next space tap swaps to next-word predictions
        //                                     instead of inserting a second space.
        mAutospaceAlternativesPending = false;
        mAutoCommitRevertLength = 0;
        final String mode = sv.mCombiningAutospaceSuggestions;
        if ("next_word".equals(mode)) {
            if (mSuggestionStripViewAccessor != null) {
                setSuggestedWords(SuggestedWords.getEmptyInstance());
            }
            mLatinIME.mHandler.postUpdateSuggestionStrip(SuggestedWords.INPUT_STYLE_NONE);
        } else if ("alternatives_then_next_word".equals(mode)) {
            // Leave the strip alone; arm the swap-on-next-space behavior AND the revert window.
            mAutospaceAlternativesPending = true;
            mAutoCommitRevertLength = writtenChars;
        } else if ("keep_alternatives".equals(mode)) {
            // Arm the revert window so the next suggestion pick replaces the auto-committed
            // word rather than appending to it.
            mAutoCommitRevertLength = writtenChars;
        }
        // Independent of the strip behavior: if the timer just committed a GESTURE word,
        // arm "first backspace deletes the whole word" — see handleBackspaceEvent. Stays
        // 0 for tap-only commits, where char-by-char delete is the right behavior.
        if (wasBatchMode) {
            mLastGestureCommittedLength = writtenChars;
        }
        // "keep_alternatives" — fall through, do nothing.
    }

    /**
     * Try to handle a backspace as a fragment-pop rather than a char-delete. Returns true if
     * handled (the caller should then return without touching the editor further); false if
     * the caller should fall through to its normal char-by-char path.
     *
     * <p>Gating: both prefs on, composing word at cursor-end, at least one boundary recorded.
     * When the last boundary is past the composing word's current length (which can happen if
     * the user externally edited the word), we just drop it and try again — eventually we
     * either find a valid boundary or run out and fall through.
     */
    private boolean tryFragmentBackspace(final SettingsValues sv) {
        final boolean legacyTracking = sv.mGestureManualSpacing && sv.mGestureFragmentBackspace;
        final boolean multipartTracking = sv.mMultipartAutoExtendInCombining
                && sv.mCombiningGraceMs > 0
                && sv.mGestureFragmentBackspace;
        if (!legacyTracking && !multipartTracking) return false;
        if (mGestureFragmentBoundaries.isEmpty()) return false;
        if (!mWordComposer.isComposingWord()) {
            clearFragmentBoundaries();
            return false;
        }
        if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) return false;

        final int currentLen = mWordComposer.getTypedWord().length();
        // Filter out stale boundaries past the current length.
        while (!mGestureFragmentBoundaries.isEmpty()
                && mGestureFragmentBoundaries.get(mGestureFragmentBoundaries.size() - 1) >= currentLen) {
            mGestureFragmentBoundaries.remove(mGestureFragmentBoundaries.size() - 1);
        }
        if (mGestureFragmentBoundaries.isEmpty()) return false;

        // Pop one boundary and shrink to the previous one (or 0 if no more).
        mGestureFragmentBoundaries.remove(mGestureFragmentBoundaries.size() - 1);
        final int newLen = mGestureFragmentBoundaries.isEmpty()
                ? 0
                : mGestureFragmentBoundaries.get(mGestureFragmentBoundaries.size() - 1);

        final String oldWord = mWordComposer.getTypedWord();
        final String newWord = newLen <= 0
                ? ""
                : oldWord.substring(0, Math.min(newLen, oldWord.length()));

        mConnection.beginBatchEdit();
        if (newWord.isEmpty()) {
            // Composing word fully cleared: commit empty and reset state. Treat as if a normal
            // word-delete brought us to a no-composing state.
            mConnection.commitText("", 1);
            mWordComposer.reset();
            clearFragmentBoundaries();
        } else {
            // Re-seed the composer with the truncated text. unsetBatchMode() so the next tap
            // appends correctly and the suggestion strip looks at it as typed text.
            mWordComposer.setBatchInputWord(newWord);
            mWordComposer.unsetBatchMode();
            setComposingTextInternal(newWord, 1);
        }
        mConnection.endBatchEdit();
        // Bump the delete count so any caller watching it (key-repeat etc.) sees progress.
        mDeleteCount++;
        return true;
    }

    // TODO: on the long term, this method should become private, but it will be
    // difficult.
    // Especially, how do we deal with InputMethodService.onDisplayCompletions?
    public void setSuggestedWords(final SuggestedWords suggestedWords) {
        if (!suggestedWords.isEmpty()) {
            final SuggestedWordInfo suggestedWordInfo;
            if (suggestedWords.mWillAutoCorrect) {
                suggestedWordInfo = suggestedWords.getInfo(SuggestedWords.INDEX_OF_AUTO_CORRECTION);
            } else {
                // We can't use suggestedWords.getWord(SuggestedWords.INDEX_OF_TYPED_WORD)
                // because it may differ from mWordComposer.mTypedWord.
                suggestedWordInfo = suggestedWords.mTypedWordInfo;
            }
            mWordComposer.setAutoCorrection(suggestedWordInfo);
        }
        mSuggestedWords = suggestedWords;
        final boolean newAutoCorrectionIndicator = suggestedWords.mWillAutoCorrect;

        // Put a blue underline to a word in TextView which will be auto-corrected.
        if (mIsAutoCorrectionIndicatorOn != newAutoCorrectionIndicator && mWordComposer.isComposingWord()) {
            mIsAutoCorrectionIndicatorOn = newAutoCorrectionIndicator;
            final CharSequence textWithUnderline = getTextWithUnderline(mWordComposer.getTypedWord());
            // TODO: when called from an updateSuggestionStrip() call that results from a posted
            // message, this is called outside any batch edit. Potentially, this may result in some
            // janky flickering of the screen, although the display speed makes it unlikely in
            // the practice.
            setComposingTextInternal(textWithUnderline, 1);
        }
    }

    /**
     * Handle a consumed event.
     * <p>
     * Consumed events represent events that have already been consumed, typically by the
     * combining chain.
     *
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     */
    private void handleConsumedEvent(final Event event, final InputTransaction inputTransaction) {
        // A consumed event may have text to commit and an update to the composing state, so
        // we evaluate both. With some combiners, it's possible than an event contains both
        // and we enter both of the following if clauses.
        final CharSequence textToCommit = event.getTextToCommit();
        if (!TextUtils.isEmpty(textToCommit)) {
            mConnection.commitText(textToCommit, 1);
            inputTransaction.setDidAffectContents();
        }
        if (mWordComposer.isComposingWord()) {
            // Khipro auto-space after suggestion: when user picks a suggestion and starts composing the next word,
            // insert space automatically, but skip it if the next character is punctuation (. , ; : ! ?) or word connector.
            final int codePoint = event.getCodePoint();
            final SettingsValues settingsValues = inputTransaction.getSettingsValues();
            if (SpaceState.PHANTOM == inputTransaction.getSpaceState()
                    && "bn_khipro".equals(mWordComposer.getCombiningSpec())
                    && !settingsValues.isWordConnector(codePoint)
                    && !settingsValues.isUsuallyFollowedBySpace(codePoint)) {
                insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
                mSpaceState = SpaceState.NONE;
            }
            setComposingTextInternal(mWordComposer.getTypedWord(), 1);
            inputTransaction.setDidAffectContents();
            inputTransaction.setRequiresUpdateSuggestions();
        }
    }

    /**
     * Handles the action of pasting content from the clipboard.
     * Retrieves content from the clipboard history manager and commits it to the input connection.
     *
     */
    private void handleClipboardPaste() {
        final String clipboardContent = mLatinIME.getClipboardHistoryManager().retrieveClipboardContent().toString();
        if (!clipboardContent.isEmpty()) {
            mLatinIME.onTextInput(clipboardContent);
        }
    }

    /**
     * Handle a functional key event.
     * <p>
     * A functional event is a special key, like delete, shift, emoji, or the settings key.
     * Non-special keys are those that generate a single code point.
     * This includes all letters, digits, punctuation, separators, emoji. It excludes keys that
     * manage keyboard-related stuff like shift, language switch, settings, layout switch, or
     * any key that results in multiple code points like the ".com" key.
     *
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     */
    private void handleFunctionalEvent(Event event, InputTransaction inputTransaction, String currentKeyboardScript, LatinIME.UIHandler handler) {
        int keyCode = event.getKeyCode();
        SettingsValues sv = inputTransaction.getSettingsValues();
        if (sv.mIsLocked && KeyCode.isIsBlockedWhenLocked(keyCode)) {
            Log.w(TAG, "Blocked keycode while device was locked, this should not happen");
        }
        switch (keyCode) {
            case KeyCode.DELETE:
                handleBackspaceEvent(event, inputTransaction, currentKeyboardScript);
                // Backspace is a functional key, but it affects the contents of the editor.
                inputTransaction.setDidAffectContents();
                break;
            case KeyCode.SHIFT:
                if (KeyboardSwitcher.getInstance().getKeyboard() != null && !KeyboardSwitcher.getInstance().getKeyboard().mId.isAlphabetKeyboard())
                    break; // recapitalization and follow-up code should only trigger for alphabet shift, see #1256
                performRecapitalization(sv);
                inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
                inputTransaction.setRequiresUpdateSuggestions();
                if (mSpaceState == SpaceState.PHANTOM && sv.mShiftRemovesAutospace)
                    mSpaceState = SpaceState.NONE;
                break;
            case KeyCode.CAPS_LOCK:
                if (KeyboardSwitcher.getInstance().getKeyboard() == null || KeyboardSwitcher.getInstance().getKeyboard().mId.isAlphabetKeyboard())
                    inputTransaction.setRequiresUpdateSuggestions();
                break;
            case KeyCode.SETTINGS:
                onSettingsKeyPressed();
                break;
            case KeyCode.ACTION_NEXT:
                performEditorAction(EditorInfo.IME_ACTION_NEXT);
                break;
            case KeyCode.ACTION_PREVIOUS:
                performEditorAction(EditorInfo.IME_ACTION_PREVIOUS);
                break;
            case KeyCode.LANGUAGE_SWITCH:
                handleLanguageSwitchKey();
                break;
            case KeyCode.CLIPBOARD:
                // Note: If clipboard history is enabled, switching to clipboard keyboard
                // is being handled in {@link KeyboardState#onEvent(Event,int)}.
                // If disabled, current clipboard content is committed.
                if (!sv.mClipboardHistoryEnabled) {
                    handleClipboardPaste();
                }
                break;
            case KeyCode.CLIPBOARD_PASTE:
                handleClipboardPaste();
                break;
            case KeyCode.SHIFT_ENTER:
                // todo: try using sendDownUpKeyEventWithMetaState() and remove the key code maybe
                final Event tmpEvent = Event.createSoftwareKeypressEvent(Constants.CODE_ENTER,
                        keyCode, 0, event.getX(), event.getY(), event.isKeyRepeat());
                handleNonSpecialCharacterEvent(tmpEvent, inputTransaction, handler);
                // Shift + Enter is treated as a functional key but it results in adding a new
                // line, so that does affect the contents of the editor.
                inputTransaction.setDidAffectContents();
                break;
            case KeyCode.MULTIPLE_CODE_POINTS:
                // added in the hangul branch, createEventChainFromSequence
                // this introduces issues like space being added behind cursor, or input deleting
                // a word, but the keepCursorPosition applyProcessedEvent seems to help here
                mWordComposer.applyProcessedEvent(event, true);
                break;
            case KeyCode.CLIPBOARD_SELECT_ALL:
                mConnection.selectAll();
                break;
            case KeyCode.CLIPBOARD_SELECT_WORD:
                mConnection.selectWord(sv.mSpacingAndPunctuations, currentKeyboardScript);
                break;
            case KeyCode.CLIPBOARD_COPY:
                mConnection.copyText(true);
                break;
            case KeyCode.CLIPBOARD_COPY_ALL:
                mConnection.copyText(false);
                break;
            case KeyCode.CLIPBOARD_CLEAR_HISTORY:
                mLatinIME.getClipboardHistoryManager().clearHistory();
                break;
            case KeyCode.CLIPBOARD_CUT:
                if (mConnection.hasSelection()) {
                    mConnection.copyText(true);
                    // fake delete keypress to remove the text
                    final Event backspaceEvent = Event.createSoftwareKeypressEvent(KeyCode.DELETE, 0,
                            event.getX(), event.getY(), event.isKeyRepeat());
                    handleBackspaceEvent(backspaceEvent, inputTransaction, currentKeyboardScript);
                    inputTransaction.setDidAffectContents();
                }
                break;
            case KeyCode.WORD_LEFT:
                sendDownUpKeyEventWithMetaState(
                    ScriptUtils.isScriptRtl(currentKeyboardScript) ? KeyEvent.KEYCODE_DPAD_RIGHT : KeyEvent.KEYCODE_DPAD_LEFT,
                    KeyEvent.META_CTRL_ON | event.getMetaState());
                break;
            case KeyCode.WORD_RIGHT:
                sendDownUpKeyEventWithMetaState(
                    ScriptUtils.isScriptRtl(currentKeyboardScript) ? KeyEvent.KEYCODE_DPAD_LEFT : KeyEvent.KEYCODE_DPAD_RIGHT,
                    KeyEvent.META_CTRL_ON | event.getMetaState());
                break;
            case KeyCode.MOVE_START_OF_PAGE:
                final int selectionEnd1 = mConnection.getExpectedSelectionEnd();
                final int selectionStart1 = mConnection.getExpectedSelectionStart();
                sendDownUpKeyEventWithMetaState(KeyEvent.KEYCODE_MOVE_HOME, KeyEvent.META_CTRL_ON | event.getMetaState());
                if (mConnection.getExpectedSelectionStart() == selectionStart1 && mConnection.getExpectedSelectionEnd() == selectionEnd1) {
                    // unchanged -> try a different method (necessary for compose fields)
                    final int newEnd = (event.getMetaState() & KeyEvent.META_SHIFT_MASK) != 0 ? selectionEnd1 : 0;
                    mConnection.setSelection(0, newEnd);
                }
                break;
            case KeyCode.MOVE_END_OF_PAGE:
                final int selectionStart2 = mConnection.getExpectedSelectionStart();
                final int selectionEnd2 = mConnection.getExpectedSelectionEnd();
                sendDownUpKeyEventWithMetaState(KeyEvent.KEYCODE_MOVE_END, KeyEvent.META_CTRL_ON | event.getMetaState());
                if (mConnection.getExpectedSelectionStart() == selectionStart2 && mConnection.getExpectedSelectionEnd() == selectionEnd2) {
                    // unchanged, try fallback e.g. for compose fields that don't care about ctrl + end
                    // we just move to a very large index, and hope the field is prepared to deal with this
                    // getting the actual length of the text for setting the correct position can be tricky for some apps...
                    try {
                        final int newStart = (event.getMetaState() & KeyEvent.META_SHIFT_MASK) != 0 ? selectionStart2 : Integer.MAX_VALUE;
                        mConnection.setSelection(newStart, Integer.MAX_VALUE);
                    } catch (Exception e) {
                        // better catch potential errors and just do nothing in this case
                        Log.i(TAG, "error when trying to move cursor to last position: " + e);
                    }
                }
                break;
            case KeyCode.UNDO:
                sendDownUpKeyEventWithMetaState(KeyEvent.KEYCODE_Z, KeyEvent.META_CTRL_ON);
                break;
            case KeyCode.REDO:
                sendDownUpKeyEventWithMetaState(KeyEvent.KEYCODE_Z, KeyEvent.META_CTRL_ON | KeyEvent.META_SHIFT_ON);
                break;
            case KeyCode.SPLIT_LAYOUT:
                KeyboardSwitcher.getInstance().toggleSplitKeyboardMode();
                break;
            case KeyCode.TIMESTAMP:
                mLatinIME.onTextInput(TimestampKt.getTimestamp(mLatinIME));
                break;
            case KeyCode.EMOJI_SEARCH:
                commitTyped(sv, LastComposedWord.NOT_A_SEPARATOR);
                mLatinIME.launchEmojiSearch();
                break;
            case KeyCode.SEND_INTENT_ONE, KeyCode.SEND_INTENT_TWO, KeyCode.SEND_INTENT_THREE:
                IntentUtils.handleSendIntentKey(mLatinIME, event.getKeyCode());
            case KeyCode.IME_HIDE_UI:
                mLatinIME.requestHideSelf(0);
                break;
            case KeyCode.INLINE_EMOJI_SEARCH_DONE:
                setInlineEmojiSearchAction(false);
                inputTransaction.setRequiresUpdateSuggestions();
                break;
            case KeyCode.SYSTEM_INPUT_METHOD_PICKER:
                mLatinIME.showInputPickerDialog();
                break;
            case KeyCode.VOICE_INPUT:
                // switching to shortcut IME, shift state, keyboard,... is handled by LatinIME,
                // {@link KeyboardSwitcher#onEvent(Event)}, or {@link #onPressKey(int,int,boolean)} and {@link #onReleaseKey(int,boolean)}.
                // We need to switch to the shortcut IME. This is handled by LatinIME since the
                // input logic has no business with IME switching.
            case KeyCode.EMOJI, KeyCode.TOGGLE_ONE_HANDED_MODE, KeyCode.SWITCH_ONE_HANDED_MODE, KeyCode.TOGGLE_FLOATING_WINDOW:
                break;
            default:
                if (KeyCode.INSTANCE.isModifier(keyCode))
                    return; // continuation of previous switch case above, but modifiers are held in a separate place
                final int keyEventCode = keyCode > 0
                    ? keyCode
                    : event.getCodePoint() >= 0
                        ? KeyCode.codePointToKeyEventCode(event.getCodePoint())
                        : KeyCode.keyCodeToKeyEventCode(keyCode);
                if (keyEventCode != KeyEvent.KEYCODE_UNKNOWN) {
                    sendDownUpKeyEventWithMetaState(keyEventCode, event.getMetaState());
                    return;
                }
                if (event.getMetaState() != 0 && event.getCodePoint() >= 32) {
                    // Try handling it as normal key event, this essentially just ignore the meta state.
                    // The conversion is good, as this way we are able to use e.g. ctrl + V. But if we don't have a codePointToKeyEventCode
                    // for that key, we will just input the normal key now, e.g. ctrl + @ will (probably) do nothing, but ctrl + _ will input _
                    // Maybe we should just return instead?
                    handleNonFunctionalEvent(event, inputTransaction, handler);
                    return;
                }
                // unknown event
                Log.e(TAG, "unknown event, key code: "+keyCode+", codepoint "+event.getCodePoint()+", meta: "+event.getMetaState());
                if (DebugFlags.DEBUG_ENABLED)
                    throw new RuntimeException("Unknown event");
        }
    }

    /**
     * Handle an event that is not a functional event.
     * <p>
     * These events are generally events that cause input, but in some cases they may do other
     * things like trigger an editor action.
     *
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     */
    private void handleNonFunctionalEvent(final Event event, final InputTransaction inputTransaction, final LatinIME.UIHandler handler) {
        inputTransaction.setDidAffectContents();
        if (event.getCodePoint() == Constants.CODE_ENTER) {
            final EditorInfo editorInfo = getCurrentInputEditorInfo();
            final int imeOptionsActionId = InputTypeUtils.getImeOptionsActionIdFromEditorInfo(editorInfo);
            if (InputTypeUtils.IME_ACTION_CUSTOM_LABEL == imeOptionsActionId) {
                // Either we have an actionLabel and we should performEditorAction with
                // actionId regardless of its value.
                performEditorAction(editorInfo.actionId);
            } else if (EditorInfo.IME_ACTION_NONE != imeOptionsActionId) {
                // We didn't have an actionLabel, but we had another action to execute.
                // EditorInfo.IME_ACTION_NONE explicitly means no action. In contrast,
                // EditorInfo.IME_ACTION_UNSPECIFIED is the default value for an action, so it
                // means there should be an action and the app didn't bother to set a specific
                // code for it - presumably it only handles one. It does not have to be treated
                // in any specific way: anything that is not IME_ACTION_NONE should be sent to
                // performEditorAction.
                performEditorAction(imeOptionsActionId);
            } else {
                // No action label, and the action from imeOptions is NONE: this is a regular
                // enter key that should input a carriage return.
                handleNonSpecialCharacterEvent(event, inputTransaction, handler);
            }
        } else {
            handleNonSpecialCharacterEvent(event, inputTransaction, handler);
        }
    }

    /**
     * Handle inputting a code point to the editor.
     * <p>
     * Non-special keys are those that generate a single code point.
     * This includes all letters, digits, punctuation, separators, emoji. It excludes keys that
     * manage keyboard-related stuff like shift, language switch, settings, layout switch, or
     * any key that results in multiple code points like the ".com" key.
     *
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     */
    private void handleNonSpecialCharacterEvent(final Event event,
            final InputTransaction inputTransaction,
            final LatinIME.UIHandler handler) {
        final int codePoint = event.getCodePoint();
        mSpaceState = SpaceState.NONE;
        final SettingsValues sv = inputTransaction.getSettingsValues();

        // wrap / unwrap selected text in codepoint pairs
        if (!mWordComposer.isComposingWord() && mConnection.hasSelection()) { // we should never be composing when something is selected
            final int pairedCodepoint = sv.mSpacingAndPunctuations.getSecondInSymbolPair(codePoint);
            if (pairedCodepoint != Constants.NOT_A_CODE) {
                wrapSelection(codePoint, pairedCodepoint);
                inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
                return;
            }
        }

        // don't treat separators as for handling URLs and similar
        //  otherwise it would work too, but whenever a separator is entered, the word is not selected
        //  until the next character is entered, and the word is added to history
        //  -> the changing selection would be confusing, and adding partial URLs to history is probably bad
        if (Character.getType(codePoint) == Character.OTHER_SYMBOL
                || (Character.getType(codePoint) == Character.UNASSIGNED && StringUtils.mightBeEmoji(codePoint)) // outdated java doesn't detect some emojis
                || (sv.isWordSeparator(codePoint)
                    && (Character.isWhitespace(codePoint) // whitespace is always a separator
                        || !textBeforeCursorMayBeUrlOrSimilar(sv, false) // if text before is not URL or similar, it's a separator
                        || (codePoint == '/' && mWordComposer.lastChar() == '/') // break composing at 2 consecutive slashes
                    )
                )
        ) {
            handleSeparatorEvent(event, inputTransaction, handler);
            addToHistoryIfEmoji(StringUtils.newSingleCodePointString(codePoint), sv);
        } else {
            if (SpaceState.PHANTOM == inputTransaction.getSpaceState()) {
                if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) {
                    // If we are in the middle of a recorrection, we need to commit the recorrection
                    // first so that we can insert the character at the current cursor position.
                    // We also need to unlearn the original word that is now being corrected.
                    unlearnWord(mWordComposer.getTypedWord(), sv, Constants.EVENT_BACKSPACE);
                    resetEntireInputState(mConnection.getExpectedSelectionStart(),
                            mConnection.getExpectedSelectionEnd(), true /* clearSuggestionStrip */);
                } else {
                    commitTyped(sv, LastComposedWord.NOT_A_SEPARATOR);
                }
            }
            handleNonSeparatorEvent(event, sv, inputTransaction);
        }
    }

    /**
     * Handle a non-separator.
     * @param event The event to handle.
     * @param settingsValues The current settings values.
     * @param inputTransaction The transaction in progress.
     */
    private void handleNonSeparatorEvent(final Event event, final SettingsValues settingsValues,
            final InputTransaction inputTransaction) {
        // Any non-separator letter input means the user accepted the autospace and is
        // typing the next word — the strip-on-next-punctuation one-shot no longer applies.
        mAutospaceJustWritten = false;
        final int codePoint = event.getCodePoint();
        // TODO: refactor this method to stop flipping isComposingWord around all the time, and
        // make it shorter (possibly cut into several pieces). Also factor
        // handleNonSpecialCharacterEvent which has the same name as other handle* methods but is
        // not the same.
        boolean isComposingWord = mWordComposer.isComposingWord();
        mWordComposer.unsetBatchMode(); // relevant in case we continue a batch word with normal typing

        // if we continue directly after a sometimesWordConnector, restart suggestions for the whole word
        // (only with URL detection and suggestions enabled)
        if (settingsValues.mUrlDetectionEnabled && settingsValues.needsToLookupSuggestions()
                && !isComposingWord && SpaceState.NONE == inputTransaction.getSpaceState()
                && settingsValues.mSpacingAndPunctuations.isSometimesWordConnector(mConnection.getCodePointBeforeCursor())
                // but not if there are two consecutive sometimesWordConnectors (e.g. "...bla")
                && !settingsValues.mSpacingAndPunctuations.isSometimesWordConnector(mConnection.getCharBeforeBeforeCursor())
                // and not if there is no letter before the separator
                && mConnection.hasLetterBeforeLastSpaceBeforeCursor()
        ) {
            final CharSequence text = mConnection.textBeforeCursorUntilLastWhitespaceOrDoubleSlash();
            final TextRange range = new TextRange(text, 0, text.length(), text.length(), false);
            isComposingWord = true;
            restartSuggestions(range);
        }
        // TODO: remove isWordConnector() and use isUsuallyFollowedBySpace() instead.
        // See onStartBatchInput() to see how to do it.
        if (SpaceState.PHANTOM == inputTransaction.getSpaceState()
                && !settingsValues.isWordConnector(codePoint)
                && !settingsValues.isUsuallyFollowedBySpace(codePoint) // only relevant in rare cases
        ) {
            if (isComposingWord) {
                // Sanity check
                throw new RuntimeException("Should not be composing here");
            }
            insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
        }

        if (mWordComposer.isCursorInFrontOfComposingWord()) {
            // we add something in front of the composing word, this is likely for adding something
            // and not for a correction
            // keep composing and don't unlearn word in this case
            resetEntireInputState(mConnection.getExpectedSelectionStart(), mConnection.getExpectedSelectionEnd(), false);
            isComposingWord = false;
        } else if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) {
            // If we are in the middle of a recorrection, we need to commit the recorrection
            // first so that we can insert the character at the current cursor position.
            // We also need to unlearn the original word that is now being corrected.
            unlearnWord(mWordComposer.getTypedWord(), inputTransaction.getSettingsValues(), Constants.EVENT_BACKSPACE);
            resetEntireInputState(mConnection.getExpectedSelectionStart(), mConnection.getExpectedSelectionEnd(), true);
            isComposingWord = false;
        }
        // We want to find out whether to start composing a new word with this character. If so,
        // we need to reset the composing state and switch isComposingWord. The order of the
        // tests is important for good performance.
        // We only start composing if we're not already composing.
        if (!isComposingWord
        // We only start composing if this is a word code point. Essentially that means it's a
        // a letter or a word connector.
                && settingsValues.isWordCodePoint(codePoint)
        // We never go into composing state if suggestions are not requested.
                && settingsValues.needsToLookupSuggestions() &&
        // In languages with spaces, we only start composing a word when we are not already
        // in the middle or at the end of a word. In languages without spaces, the above conditions are sufficient.
        // NOTE: If the InputConnection is slow, we skip the text-after-cursor check since it
        // can incur a very expensive getTextAfterCursor() lookup, potentially making the
        // keyboard UI slow and non-responsive.
        // TODO: Cache the text after the cursor so we don't need to go to the InputConnection
        // each time. We are already doing this for getTextBeforeCursor().
                (!settingsValues.mSpacingAndPunctuations.mCurrentLanguageHasSpaces
                        || !mConnection.isCursorTouchingWord(settingsValues.mSpacingAndPunctuations,
                                !mConnection.hasSlowInputConnection() /* checkTextAfter */)
                        || isCursorAtStartOrAfterSeparator(settingsValues))) {
            // Reset entirely the composing state anyway, then start composing a new word unless
            // the character is a word connector. The idea here is, word connectors are not
            // separators and they should be treated as normal characters, except in the first
            // position where they should not start composing a word.
            isComposingWord = !settingsValues.mSpacingAndPunctuations.isWordConnector(codePoint);
            // Here we don't need to reset the last composed word. It will be reset
            // when we commit this one, if we ever do; if on the other hand we backspace
            // it entirely and resume suggestions on the previous word, we'd like to still
            // have touch coordinates for it.
            resetComposingState(false /* alsoResetLastComposedWord */);
        }

        enterInlineEmojiSearchIfNeeded(codePoint, settingsValues);

        if (isComposingWord) {
            mWordComposer.applyProcessedEvent(event);
            // If it's the first letter, make note of auto-caps state
            if (mWordComposer.isSingleLetter()) {
                mWordComposer.setCapitalizedModeAtStartComposingTime(inputTransaction.getShiftState());
            }
            setComposingTextInternal(getTextWithUnderline(mWordComposer.getTypedWord()), 1);
            // Two-thumb typing (#1.1): record this tap as a fragment boundary so a future
            // backspace under PREF_GESTURE_FRAGMENT_BACKSPACE can pop the whole tap.
            recordFragmentBoundaryIfTracking(settingsValues);
            // Combining mode: arm/refresh the grace timer for the next input.
            enterCombiningMode(settingsValues, true /* fromTap, unused — kept for clarity */);
        } else {
            final boolean swapWeakSpace = tryStripSpaceAndReturnWhetherShouldSwapInstead(event, inputTransaction);

            if (swapWeakSpace && trySwapSwapperAndSpace(event, inputTransaction)) {
                mSpaceState = SpaceState.WEAK;
            } else if ((settingsValues.mInputAttributes.mInputType & InputType.TYPE_MASK_CLASS) != InputType.TYPE_CLASS_TEXT
                    && codePoint >= '0' && codePoint <= '9') {
                // weird issue when committing text: https://github.com/HeliBorg/HeliBoard/issues/585
                // but at the same time we don't always want to do it for numbers because it might interfere with url detection
                // todo: consider always using sendDownUpKeyEvent for non-text-inputType
                sendDownUpKeyEvent(codePoint - '0' + KeyEvent.KEYCODE_0);
            } else {
                mConnection.commitCodePoint(codePoint);
            }
        }
        inputTransaction.setRequiresUpdateSuggestions();
    }

    private boolean isCursorAtStartOrAfterSeparator(SettingsValues settingsValues) {
        var codePointBeforeCursor = mConnection.getCodePointBeforeCursor();
        return codePointBeforeCursor == Constants.NOT_A_CODE
                || settingsValues.mSpacingAndPunctuations.isWordSeparator(codePointBeforeCursor);
    }

    /**
     * Handle input of a separator code point.
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     */
    private void handleSeparatorEvent(final Event event, final InputTransaction inputTransaction,
            final LatinIME.UIHandler handler) {
        // Combining mode behavior #3: if the prior auto-commit left the strip showing the
        // just-committed word's alternatives, the FIRST space tap after that should swap to
        // next-word predictions instead of inserting a second space. Check this BEFORE
        // cancelCombiningMode() so the flag survives long enough to be consumed.
        final int rawCodePoint = event.getCodePoint();
        if (Constants.CODE_SPACE == rawCodePoint && mAutospaceAlternativesPending) {
            mAutospaceAlternativesPending = false;
            // Don't cancel combining mode visuals — there are none active, but be safe.
            cancelCombiningMode();
            if (mSuggestionStripViewAccessor != null) {
                setSuggestedWords(SuggestedWords.getEmptyInstance());
            }
            handler.postUpdateSuggestionStrip(SuggestedWords.INPUT_STYLE_NONE);
            // Eat the keystroke entirely — no second space, no shift update, no commit.
            return;
        }
        // Combining-mode punctuation strip: when the timer-driven autospace just wrote a
        // trailing space (mAutospaceJustWritten was set) AND the user taps a punctuation
        // character that's "usually followed by space" (`,` `.` `;` `:` `!` `?` …), delete
        // the preceding space so the comma lands tight against the word: "hello," instead of
        // "hello ,". The PHANTOM-after path then sets state so the next non-separator letter
        // gets a fresh autospace. Excludes space and Enter (those would defeat the
        // autospace).
        final boolean justAutoSpaced = mAutospaceJustWritten;
        mAutospaceJustWritten = false;
        // Combining mode: explicit separator (space, punctuation, Enter) supersedes the
        // grace timer — the user is committing the word themselves.
        cancelCombiningMode();
        final int codePoint = event.getCodePoint();
        final SettingsValues settingsValues = inputTransaction.getSettingsValues();
        if (justAutoSpaced
                && Constants.CODE_SPACE != codePoint
                && Constants.CODE_ENTER != codePoint
                && settingsValues.isUsuallyFollowedBySpace(codePoint)
                && Constants.CODE_SPACE == mConnection.getCodePointBeforeCursor()) {
            mConnection.removeTrailingSpace();
            // Leave mSpaceState=NONE; the existing PHANTOM-keep path at line 2000ish will
            // re-establish PHANTOM if appropriate after the comma is committed.
        }
        final boolean wasComposingWord = mWordComposer.isComposingWord();
        // We avoid sending spaces in languages without spaces if we were composing.
        final boolean shouldAvoidSendingCode = Constants.CODE_SPACE == codePoint
                && !settingsValues.mSpacingAndPunctuations.mCurrentLanguageHasSpaces
                && wasComposingWord;

        if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) {
            // If we are in the middle of a recorrection, we need to commit the recorrection
            // first so that we can insert the separator at the current cursor position.
            // We also need to unlearn the original word that is now being corrected.
            unlearnWord(mWordComposer.getTypedWord(), inputTransaction.getSettingsValues(), Constants.EVENT_BACKSPACE);
            resetEntireInputState(mConnection.getExpectedSelectionStart(),
                    mConnection.getExpectedSelectionEnd(), true /* clearSuggestionStrip */);
        }
        // isComposingWord() may have changed since we stored wasComposing
        if (mWordComposer.isComposingWord()) {
            if (settingsValues.mAutoCorrectEnabled && ! isInlineEmojiSearchAction()) {
                final String separator = shouldAvoidSendingCode ? LastComposedWord.NOT_A_SEPARATOR
                        : StringUtils.newSingleCodePointString(codePoint);
                commitCurrentAutoCorrection(settingsValues, separator, handler);
                inputTransaction.setDidAutoCorrect();
            } else {
                commitTyped(settingsValues, StringUtils.newSingleCodePointString(codePoint));
            }
        }

        final boolean swapWeakSpace = tryStripSpaceAndReturnWhetherShouldSwapInstead(event, inputTransaction);

        final boolean isInsideDoubleQuoteOrAfterDigit = Constants.CODE_DOUBLE_QUOTE == codePoint
                && mConnection.isInsideDoubleQuoteOrAfterDigit();

        final boolean needsPrecedingSpace;
        if (SpaceState.PHANTOM != inputTransaction.getSpaceState()) {
            needsPrecedingSpace = false;
        } else if (Constants.CODE_DOUBLE_QUOTE == codePoint) {
            // Double quotes behave like they are usually preceded by space iff we are
            // not inside a double quote or after a digit.
            needsPrecedingSpace = !isInsideDoubleQuoteOrAfterDigit;
        } else if (settingsValues.mSpacingAndPunctuations.isClusteringSymbol(codePoint)
                && settingsValues.mSpacingAndPunctuations.isClusteringSymbol(
                        mConnection.getCodePointBeforeCursor())) {
            needsPrecedingSpace = false;
        } else {
            needsPrecedingSpace = settingsValues.isUsuallyPrecededBySpace(codePoint) || StringUtilsKt.isEmoji(codePoint);
        }

        if (needsPrecedingSpace) {
            insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
        }

        if (tryPerformDoubleSpacePeriod(event, inputTransaction)) {
            mSpaceState = SpaceState.DOUBLE;
            inputTransaction.setRequiresUpdateSuggestions();
            StatsUtils.onDoubleSpacePeriod();
        } else if (swapWeakSpace && trySwapSwapperAndSpace(event, inputTransaction)) {
            mSpaceState = SpaceState.SWAP_PUNCTUATION;
            mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
        } else if (Constants.CODE_SPACE == codePoint) {
            if (!mSuggestedWords.isPunctuationSuggestions()) {
                mSpaceState = SpaceState.WEAK;
            }

            startDoubleSpacePeriodCountdown(inputTransaction);
            if (wasComposingWord || mSuggestedWords.isEmpty()) {
                inputTransaction.setRequiresUpdateSuggestions();
            }

            if (!shouldAvoidSendingCode) {
                mConnection.commitCodePoint(codePoint);
            }
        } else {
            if (SpaceState.PHANTOM == inputTransaction.getSpaceState()
                    && (settingsValues.isUsuallyFollowedBySpace(codePoint) || isInsideDoubleQuoteOrAfterDigit)) {
                // If we are in phantom space state, and the user presses a separator, we want to
                // stay in phantom space state so that the next keypress has a chance to add the
                // space. For example, if I type "Good dat", pick "day" from the suggestion strip
                // then insert a comma and go on to typing the next word, I want the space to be
                // inserted automatically before the next word, the same way it is when I don't
                // input the comma. Also when closing a quote the phantom state should be preserved.
                // The case is a little different if the separator is a space stripper. Such a
                // separator does not normally need a space on the right (that's the difference
                // between swappers and strippers), so we should not stay in phantom space state if
                // the separator is a stripper. Hence the additional test above.
                mSpaceState = SpaceState.PHANTOM;
            } else {
                // mSpaceState is still SpaceState.NONE, but some characters should typically
                // be followed by space. Set phantom space state for such characters if the user
                // enabled the setting and was not composing a word. The latter avoids setting
                // phantom space state when typing decimal numbers, with the drawback of not
                // setting phantom space state after ending a sentence with a non-word.
                // A double quote behaves like it's usually followed by space if we're inside
                // a double quote.
                if (wasComposingWord
                        && settingsValues.mAutospaceAfterPunctuation
                        && (settingsValues.isUsuallyFollowedBySpace(codePoint) || isInsideDoubleQuoteOrAfterDigit)) {
                    mSpaceState = SpaceState.PHANTOM;
                }
            }

            enterInlineEmojiSearchIfNeeded(codePoint, settingsValues);

            mConnection.commitCodePoint(codePoint);

            if (isInlineEmojiSearchAction()) {
                inputTransaction.setRequiresUpdateSuggestions();
            } else {
                // Set punctuation right away. onUpdateSelection will fire but tests whether it is
                // already displayed or not, so it's okay.
                mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
            }
        }

        inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
    }

    /**
     * Handle a press on the backspace key.
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     */
    private void handleBackspaceEvent(final Event event, final InputTransaction inputTransaction) {
        // Combining mode: snapshot the gesture-word-length flag BEFORE cancelCombiningMode
        // clears it. If non-zero (the previous commit was a gesture), this backspace MIGHT
        // delete the whole word — see further down, after the autocorrect-revert branch.
        final int gestureCommittedLen = mLastGestureCommittedLength;
        // Combining mode: a backspace always cancels the pending commit. The user is
        // explicitly retracting input; we don't want the timer to fire mid-correction.
        cancelCombiningMode();
        final String currentKeyboardScript = inputTransaction.getSettingsValues().mCurrentKeyboardScript;
        // Two-thumb typing (#1.1, PREF_GESTURE_FRAGMENT_BACKSPACE): try to pop the most-recent
        // fragment from the composing word as one keystroke. Returns true if handled — in
        // which case we still need to update the space-state + shift behaviour like a normal
        // backspace, so do those AFTER the fragment-pop succeeds, then return early instead
        // of falling through to the usual per-character path.
        if (tryFragmentBackspace(inputTransaction.getSettingsValues())) {
            mSpaceState = SpaceState.NONE;
            inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
            inputTransaction.setRequiresUpdateSuggestions();
            return;
        }
        mSpaceState = SpaceState.NONE;
        mDeleteCount++;

        // In many cases after backspace, we need to update the shift state. Normally we need
        // to do this right away to avoid the shift state being out of date in case the user types
        // backspace then some other character very fast. However, in the case of backspace key
        // repeat, this can lead to flashiness when the cursor flies over positions where the
        // shift state should be updated, so if this is a key repeat, we update after a small delay.
        // Then again, even in the case of a key repeat, if the cursor is at start of text, it
        // can't go any further back, so we can update right away even if it's a key repeat.
        final int shiftUpdateKind = event.isKeyRepeat() && mConnection.getExpectedSelectionStart() > 0
                ? InputTransaction.SHIFT_UPDATE_LATER
                : InputTransaction.SHIFT_UPDATE_NOW;
        inputTransaction.requireShiftUpdate(shiftUpdateKind);

        if (mWordComposer.isCursorFrontOrMiddleOfComposingWord()) {
            // If we are in the middle of a recorrection, we need to commit the recorrection
            // first so that we can remove the character at the current cursor position.
            // We also need to unlearn the original word that is now being corrected.
            unlearnWord(mWordComposer.getTypedWord(), inputTransaction.getSettingsValues(),
                    Constants.EVENT_BACKSPACE);
            resetEntireInputState(mConnection.getExpectedSelectionStart(),
                    mConnection.getExpectedSelectionEnd(), true /* clearSuggestionStrip */);
            // When we exit this if-clause, mWordComposer.isComposingWord() will return false.
        }
        if (mWordComposer.isComposingWord()) {
            if (mWordComposer.isBatchMode()) {
                final String rejectedSuggestion = mWordComposer.getTypedWord();
                mWordComposer.reset();
                mWordComposer.setRejectedBatchModeSuggestion(rejectedSuggestion);
                if (!TextUtils.isEmpty(rejectedSuggestion)) {
                    unlearnWord(rejectedSuggestion, inputTransaction.getSettingsValues(),
                            Constants.EVENT_REJECTION);
                }
                StatsUtils.onBackspaceWordDelete(rejectedSuggestion.length());
            } else {
                mWordComposer.applyProcessedEvent(event);
                StatsUtils.onBackspacePressed(1);
            }
            if (mWordComposer.isComposingWord()) {
                setComposingTextInternal(getTextWithUnderline(mWordComposer.getTypedWord()), 1);
            } else {
                mConnection.commitText("", 1);
            }
            updateInlineEmojiSearch();
            inputTransaction.setRequiresUpdateSuggestions();
        } else {
            if (mLastComposedWord.canRevertCommit() && inputTransaction.getSettingsValues().mBackspaceRevertsAutocorrect) {
                final String lastComposedWord = mLastComposedWord.mTypedWord;
                revertCommit(inputTransaction);
                StatsUtils.onRevertAutoCorrect();
                StatsUtils.onWordCommitUserTyped(lastComposedWord, mWordComposer.isBatchMode());
                // Restart suggestions when backspacing into a reverted word. This is required for
                // the final corrected word to be learned, as learning only occurs when suggestions
                // are active.
                //
                // Note: restartSuggestionsOnWordTouchedByCursor is already called for normal
                // (non-revert) backspace handling.
                if (inputTransaction.getSettingsValues().needsToLookupSuggestions()
                        && inputTransaction.getSettingsValues().mSpacingAndPunctuations.mCurrentLanguageHasSpaces) {
                    restartSuggestionsOnWordTouchedByCursor(inputTransaction.getSettingsValues(), currentKeyboardScript);
                }
                return;
            }
            // Combining mode: if no autocorrect-revert applied and the last commit was a
            // gesture word, this backspace deletes the whole word + autospace in one go
            // (gated by PREF_COMBINING_BACKSPACE_DELETES_GESTURE_WORD, default on). Tap
            // words don't qualify — gestureCommittedLen is 0 for those. We delete via
            // deleteTextBeforeCursor instead of the usual single-char path so the user
            // doesn't have to mash backspace to undo a swiped word.
            if (gestureCommittedLen > 0
                    && inputTransaction.getSettingsValues().mCombiningBackspaceDeletesGestureWord) {
                mConnection.beginBatchEdit();
                mConnection.deleteTextBeforeCursor(gestureCommittedLen);
                mConnection.endBatchEdit();
                StatsUtils.onBackspaceWordDelete(gestureCommittedLen);
                inputTransaction.setRequiresUpdateSuggestions();
                return;
            }
            // todo: this is currently disabled, as it causes inconsistencies with
            // textInput, depending whether the end
            // is part of a word (where we start composing) or not (where we end in code
            // below)
            // see https://github.com/Helium314/HeliBoard/issues/1019
            // with better emoji detection on backspace (getFullEmojiAtEnd), this
            // functionality might not be necessary
            // -> enable again if there are issues, otherwise delete the code, together with
            // mEnteredText
            if (false && mEnteredText != null && mConnection.sameAsTextBeforeCursor(mEnteredText)) {
                // Cancel multi-character input: remove the text we just entered.
                // This is triggered on backspace after a key that inputs multiple characters,
                // like the smiley key or the .com key.
                mConnection.deleteTextBeforeCursor(mEnteredText.length());
                StatsUtils.onDeleteMultiCharInput(mEnteredText.length());
                mEnteredText = null;
                // If we have mEnteredText, then we know that mHasUncommittedTypedChars == false.
                // In addition we know that spaceState is false, and that we should not be
                // reverting any autocorrect at this point. So we can safely return.
                return;
            }
            if (SpaceState.DOUBLE == inputTransaction.getSpaceState()) {
                cancelDoubleSpacePeriodCountdown();
                if (mConnection.revertDoubleSpacePeriod(inputTransaction.getSettingsValues().mSpacingAndPunctuations)) {
                    // No need to reset mSpaceState, it has already be done (that's why we
                    // receive it as a parameter)
                    inputTransaction.setRequiresUpdateSuggestions();
                    mWordComposer.setCapitalizedModeAtStartComposingTime(WordComposer.CAPS_MODE_OFF);
                    StatsUtils.onRevertDoubleSpacePeriod();
                    return;
                }
            } else if (SpaceState.SWAP_PUNCTUATION == inputTransaction.getSpaceState()) {
                if (mConnection.revertSwapPunctuation()) {
                    StatsUtils.onRevertSwapPunctuation();
                    // Likewise
                    return;
                }
            }

            boolean hasUnlearnedWordBeingDeleted = false;

            // No cancelling of commit/double space/swap: we have a regular backspace.
            // We should backspace one char and restart suggestion if at the end of a word.
            if (mConnection.hasSelection()) {
                // If there is a selection, remove it.
                // We also need to unlearn the selected text.
                final CharSequence selection = mConnection.getSelectedText(0 /* 0 for no styles */);
                if (!TextUtils.isEmpty(selection)) {
                    unlearnWord(selection.toString(), inputTransaction.getSettingsValues(),
                            Constants.EVENT_BACKSPACE);
                    hasUnlearnedWordBeingDeleted = true;
                }
                final int numCharsDeleted = mConnection.getExpectedSelectionEnd()
                        - mConnection.getExpectedSelectionStart();
                mConnection.setSelection(mConnection.getExpectedSelectionEnd(),
                        mConnection.getExpectedSelectionEnd());
                mConnection.deleteTextBeforeCursor(numCharsDeleted);
                StatsUtils.onBackspaceSelectedText(numCharsDeleted);
            } else {
                // There is no selection, just delete one character.
                if (inputTransaction.getSettingsValues().mInputAttributes.isTypeNull()
                        || Constants.NOT_A_CURSOR_POSITION == mConnection.getExpectedSelectionEnd()) {
                    // There are three possible reasons to send a key event: either the field has
                    // type TYPE_NULL, in which case the keyboard should send events, or we are
                    // running in backward compatibility mode, or we don't know the cursor position.
                    // Before Jelly bean, the keyboard would simulate a hardware keyboard event on
                    // pressing enter or delete. This is bad for many reasons (there are race
                    // conditions with commits) but some applications are relying on this behavior
                    // so we continue to support it for older apps, so we retain this behavior if
                    // the app has target SDK < JellyBean.
                    // As for the case where we don't know the cursor position, it can happen
                    // because of bugs in the framework. But the framework should know, so the next
                    // best thing is to leave it to whatever it thinks is best.
                    sendDownUpKeyEvent(KeyEvent.KEYCODE_DEL);
                    int totalDeletedLength = 1;
                    if (mDeleteCount > Constants.DELETE_ACCELERATE_AT) {
                        // If this is an accelerated (i.e., double) deletion, then we need to
                        // consider unlearning here because we may have already reached
                        // the previous word, and will lose it after next deletion.
                        hasUnlearnedWordBeingDeleted |= unlearnWordBeingDeleted(
                                inputTransaction.getSettingsValues(), currentKeyboardScript);
                        sendDownUpKeyEvent(KeyEvent.KEYCODE_DEL);
                        totalDeletedLength++;
                    }
                    StatsUtils.onBackspacePressed(totalDeletedLength);
                } else {
                    final int codePointBeforeCursor = mConnection.getCodePointBeforeCursor();
                    if (codePointBeforeCursor == Constants.NOT_A_CODE) {
                        // HACK for backward compatibility with broken apps that haven't realized
                        // yet that hardware keyboards are not the only way of inputting text.
                        // Nothing to delete before the cursor. We should not do anything, but many
                        // broken apps expect something to happen in this case so that they can
                        // catch it and have their broken interface react. If you need the keyboard
                        // to do this, you're doing it wrong -- please fix your app.
                        //  To make this more interesting, web browsers, and apps that are basically
                        // browsers under the hood, in too many cases don't understand "deleteSurroundingText".
                        // So we try to send a backspace keypress instead.
                        if ((getCurrentInputEditorInfo().inputType & InputType.TYPE_MASK_VARIATION)
                                == InputType.TYPE_TEXT_VARIATION_WEB_EDIT_TEXT)
                            sendDownUpKeyEvent(KeyEvent.KEYCODE_DEL);
                        else mConnection.deleteTextBeforeCursor(1);
                        // TODO: Add a new StatsUtils method onBackspaceWhenNoText()
                        return;
                    }
                    int lengthToDelete = mConnection.getCharCountToDeleteBeforeCursor();
                    mConnection.deleteTextBeforeCursor(lengthToDelete);
                    int totalDeletedLength = lengthToDelete;
                    if (mDeleteCount > Constants.DELETE_ACCELERATE_AT) {
                        // If this is an accelerated (i.e., double) deletion, then we need to
                        // consider unlearning here because we may have already reached
                        // the previous word, and will lose it after next deletion.
                        hasUnlearnedWordBeingDeleted |= unlearnWordBeingDeleted(
                                inputTransaction.getSettingsValues(), currentKeyboardScript);
                        final int codePointBeforeCursorToDeleteAgain =
                                mConnection.getCodePointBeforeCursor();
                        if (codePointBeforeCursorToDeleteAgain != Constants.NOT_A_CODE) {
                            int lengthToDeleteAgain = mConnection.getCharCountToDeleteBeforeCursor();
                            mConnection.deleteTextBeforeCursor(lengthToDeleteAgain);
                            totalDeletedLength += lengthToDeleteAgain;
                        }
                    }
                    StatsUtils.onBackspacePressed(totalDeletedLength);
                }
            }
            if (!hasUnlearnedWordBeingDeleted) {
                // Consider unlearning the word being deleted (if we have not done so already).
                unlearnWordBeingDeleted(
                        inputTransaction.getSettingsValues(), currentKeyboardScript);
            }
            if (mConnection.hasSlowInputConnection()) {
                mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
            } else if (inputTransaction.getSettingsValues().needsToLookupSuggestions()
                    && inputTransaction.getSettingsValues().mSpacingAndPunctuations.mCurrentLanguageHasSpaces) {
                restartSuggestionsOnWordTouchedByCursor(inputTransaction.getSettingsValues(), currentKeyboardScript);
            }
        }
    }

    String getWordAtCursor(final SettingsValues settingsValues, final String currentKeyboardScript) {
        if (!mConnection.hasSelection()
                && settingsValues.needsToLookupSuggestions()
                && settingsValues.mSpacingAndPunctuations.mCurrentLanguageHasSpaces) {
            final TextRange range = mConnection.getWordRangeAtCursor(settingsValues.mSpacingAndPunctuations, currentKeyboardScript);
            if (range != null) {
                return range.mWord.toString();
            }
        }
        return "";
    }

    boolean unlearnWordBeingDeleted(
            final SettingsValues settingsValues, final String currentKeyboardScript) {
        if (mConnection.hasSlowInputConnection()) {
            // TODO: Refactor unlearning so that it does not incur any extra calls
            // to the InputConnection. That way it can still be performed on a slow
            // InputConnection.
            Log.w(TAG, "Skipping unlearning due to slow InputConnection.");
            return false;
        }
        // If we just started backspacing to delete a previous word (but have not
        // entered the composing state yet), unlearn the word.
        // TODO: Consider tracking whether or not this word was typed by the user.
        if (!mConnection.isCursorFollowedByWordCharacter(settingsValues.mSpacingAndPunctuations)) {
            final String wordBeingDeleted = getWordAtCursor(settingsValues, currentKeyboardScript);
            if (!TextUtils.isEmpty(wordBeingDeleted)) {
                unlearnWord(wordBeingDeleted, settingsValues, Constants.EVENT_BACKSPACE);
                return true;
            }
        }
        return false;
    }

    void unlearnWord(final String word, final SettingsValues settingsValues, final int eventType) {
        final NgramContext ngramContext = mConnection.getNgramContextFromNthPreviousWord(settingsValues.mSpacingAndPunctuations, 2);
        final long timeStampInSeconds = TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis());
        mDictionaryFacilitator.unlearnFromUserHistory(word, ngramContext, timeStampInSeconds, eventType);
    }

    /**
     * Handle a press on the language switch key (the "globe key")
     */
    private void handleLanguageSwitchKey() {
        mLatinIME.switchToNextSubtype();
    }

    /**
     * Swap a space with a space-swapping punctuation sign.
     * <p>
     * This method will check that there are two characters before the cursor and that the first
     * one is a space before it does the actual swapping.
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     * @return true if the swap has been performed, false if it was prevented by preliminary checks.
     */
    private boolean trySwapSwapperAndSpace(final Event event,
            final InputTransaction inputTransaction) {
        final int codePointBeforeCursor = mConnection.getCodePointBeforeCursor();
        if (Constants.CODE_SPACE != codePointBeforeCursor) {
            return false;
        }
        mConnection.deleteTextBeforeCursor(1);
        final String text = event.getTextToCommit() + " ";
        mConnection.commitText(text, 1);
        inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
        return true;
    }

    /*
     * Strip a trailing space if necessary and returns whether it's a swap weak space situation.
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     * @return whether we should swap the space instead of removing it.
     */
    private boolean tryStripSpaceAndReturnWhetherShouldSwapInstead(final Event event,
            final InputTransaction inputTransaction) {
        final int codePoint = event.getCodePoint();
        final boolean isFromSuggestionStrip = event.isSuggestionStripPress();
        if (Constants.CODE_ENTER == codePoint &&
                SpaceState.SWAP_PUNCTUATION == inputTransaction.getSpaceState()) {
            mConnection.removeTrailingSpace();
            return false;
        }
        if ((SpaceState.WEAK == inputTransaction.getSpaceState()
                || SpaceState.SWAP_PUNCTUATION == inputTransaction.getSpaceState())
                && isFromSuggestionStrip) {
            if (inputTransaction.getSettingsValues().isUsuallyPrecededBySpace(codePoint)) {
                return false;
            }
            if (inputTransaction.getSettingsValues().isUsuallyFollowedBySpace(codePoint)) {
                return true;
            }
            mConnection.removeTrailingSpace();
        }
        return false;
    }

    public void startDoubleSpacePeriodCountdown(final InputTransaction inputTransaction) {
        mDoubleSpacePeriodCountdownStart = inputTransaction.getTimestamp();
    }

    public void cancelDoubleSpacePeriodCountdown() {
        mDoubleSpacePeriodCountdownStart = 0;
    }

    public boolean isDoubleSpacePeriodCountdownActive(final InputTransaction inputTransaction) {
        return inputTransaction.getTimestamp() - mDoubleSpacePeriodCountdownStart
                < inputTransaction.getSettingsValues().mDoubleSpacePeriodTimeout;
    }

    /**
     * Apply the double-space-to-period transformation if applicable.
     * <p>
     * The double-space-to-period transformation means that we replace two spaces with a
     * period-space sequence of characters. This typically happens when the user presses space
     * twice in a row quickly.
     * This method will check that the double-space-to-period is active in settings, that the
     * two spaces have been input close enough together, that the typed character is a space
     * and that the previous character allows for the transformation to take place. If all of
     * these conditions are fulfilled, this method applies the transformation and returns true.
     * Otherwise, it does nothing and returns false.
     *
     * @param event The event to handle.
     * @param inputTransaction The transaction in progress.
     * @return true if we applied the double-space-to-period transformation, false otherwise.
     */
    private boolean tryPerformDoubleSpacePeriod(final Event event,
            final InputTransaction inputTransaction) {
        // Check the setting, the typed character and the countdown. If any of the conditions is
        // not fulfilled, return false.
        if (!inputTransaction.getSettingsValues().mUseDoubleSpacePeriod
                || Constants.CODE_SPACE != event.getCodePoint()
                || !isDoubleSpacePeriodCountdownActive(inputTransaction)) {
            return false;
        }
        // We only do this when we see one space and an accepted code point before the cursor.
        // The code point may be a surrogate pair but the space may not, so we need 3 chars.
        final CharSequence lastTwo = mConnection.getTextBeforeCursor(3, 0);
        if (null == lastTwo) return false;
        final int length = lastTwo.length();
        if (length < 2) return false;
        if (lastTwo.charAt(length - 1) != Constants.CODE_SPACE) {
            return false;
        }
        // We know there is a space in pos -1, and we have at least two chars. If we have only two
        // chars, isSurrogatePairs can't return true as charAt(1) is a space, so this is fine.
        final int firstCodePoint = Character.isSurrogatePair(lastTwo.charAt(0), lastTwo.charAt(1))
                        ? Character.codePointAt(lastTwo, length - 3)
                        : lastTwo.charAt(length - 2);
        if (canBeFollowedByDoubleSpacePeriod(firstCodePoint)) {
            cancelDoubleSpacePeriodCountdown();
            mConnection.deleteTextBeforeCursor(1);
            final String textToInsert = inputTransaction.getSettingsValues().mSpacingAndPunctuations
                    .mSentenceSeparatorAndSpace;
            mConnection.commitText(textToInsert, 1);
            inputTransaction.requireShiftUpdate(InputTransaction.SHIFT_UPDATE_NOW);
            inputTransaction.setRequiresUpdateSuggestions();
            return true;
        }
        return false;
    }

    /**
     * Returns whether this code point can be followed by the double-space-to-period transformation.
     * <p>
     * See #maybeDoubleSpaceToPeriod for details.
     * Generally, most word characters can be followed by the double-space-to-period transformation,
     * while most punctuation can't. Some punctuation however does allow for this to take place
     * after them, like the closing parenthesis for example.
     *
     * @param codePoint the code point after which we may want to apply the transformation
     * @return whether it's fine to apply the transformation after this code point.
     */
    private static boolean canBeFollowedByDoubleSpacePeriod(final int codePoint) {
        // TODO: This should probably be a blacklist rather than a whitelist.
        // TODO: This should probably be language-dependant...
        return Character.isLetterOrDigit(codePoint)
                || codePoint == Constants.CODE_SINGLE_QUOTE
                || codePoint == Constants.CODE_DOUBLE_QUOTE
                || codePoint == Constants.CODE_CLOSING_PARENTHESIS
                || codePoint == Constants.CODE_CLOSING_SQUARE_BRACKET
                || codePoint == Constants.CODE_CLOSING_CURLY_BRACKET
                || codePoint == Constants.CODE_CLOSING_ANGLE_BRACKET
                || codePoint == Constants.CODE_PLUS
                || codePoint == Constants.CODE_PERCENT
                || Character.getType(codePoint) == Character.OTHER_SYMBOL;
    }

    /**
     * Performs a recapitalization event.
     * @param settingsValues The current settings values.
     */
    private void performRecapitalization(final SettingsValues settingsValues) {
        if (!mConnection.hasSelection() || !mRecapitalizeStatus.isEnabled()) {
            return; // No selection or recapitalize is disabled for now
        }
        final int selectionStart = mConnection.getExpectedSelectionStart();
        final int selectionEnd = mConnection.getExpectedSelectionEnd();
        final int numCharsSelected = selectionEnd - selectionStart;
        if (numCharsSelected > Constants.MAX_CHARACTERS_FOR_RECAPITALIZATION) {
            // We bail out if we have too many characters for performance reasons. We don't want
            // to suck possibly multiple-megabyte data.
            return;
        }
        // If we have a recapitalize in progress, use it; otherwise, start a new one.
        if (!mRecapitalizeStatus.isStarted()
                || !mRecapitalizeStatus.isSetAt(selectionStart, selectionEnd)) {
            final CharSequence selectedText =
                    mConnection.getSelectedText(0 /* flags, 0 for no styles */);
            if (TextUtils.isEmpty(selectedText)) return; // Race condition with the input connection
            mRecapitalizeStatus.start(selectedText.toString(), selectionStart, settingsValues.mLocale,
                    settingsValues.mSpacingAndPunctuations.mSortedWordSeparators);
        }
        mConnection.finishComposingText();
        mRecapitalizeStatus.rotate();
        mConnection.setSelection(selectionEnd, selectionEnd);
        mConnection.deleteTextBeforeCursor(numCharsSelected);
        final TextPlacement replacement = mRecapitalizeStatus.textReplacement();
        mConnection.commitText(replacement.text, 0);
        mConnection.setSelection(replacement.selectionStart, replacement.selectionEnd());
    }

    private void performAdditionToUserHistoryDictionary(final SettingsValues settingsValues,
            final String suggestion, @NonNull final NgramContext ngramContext) {
        // For addition to user history we want suggestions (even if just for autocorrect) or a gestured word.
        // That's to avoid unintended additions in some sensitive fields, or fields that
        // expect to receive non-words.
        if ((!settingsValues.needsToLookupSuggestions() && !mWordComposer.isBatchMode()) || TextUtils.isEmpty(suggestion))
            return;
        boolean wasAutoCapitalized = mWordComposer.wasAutoCapitalized() && !mWordComposer.isMostlyCaps();
        String word = StringUtilsKt.stripTrailingSeparatorsAndConnectors(suggestion, settingsValues.mSpacingAndPunctuations);
        if (settingsValues.mIncognitoModeEnabled) {
            // don't add to history, but still adjust confidences
            // otherwise incognito input fields can be very annoying when the wrong language is active
            mDictionaryFacilitator.adjustConfidences(word, wasAutoCapitalized);
            return;
        }
        if (mConnection.hasSlowInputConnection()) {
            // Since we don't unlearn when the user backspaces on a slow InputConnection,
            // turn off learning to guard against adding typos that the user later deletes.
            Log.w(TAG, "Skipping learning due to slow InputConnection.");
            // but we still want to adjust confidences for multilingual typing
            mDictionaryFacilitator.adjustConfidences(word, wasAutoCapitalized);
            return;
        }
        final int timeStampInSeconds = (int)TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis());
        mDictionaryFacilitator.addToUserHistory(word, wasAutoCapitalized, ngramContext,
                timeStampInSeconds, settingsValues.mBlockPotentiallyOffensive);
    }

    private void addToHistoryIfEmoji(final String text, final SettingsValues settingsValues) {
        if (mLastComposedWord == LastComposedWord.NOT_A_COMPOSED_WORD // we want a last composed word, also to avoid storing consecutive emojis
            || mWordComposer.isComposingWord() // emoji will be part of the word in this case, better do nothing
            || !settingsValues.mBigramPredictionEnabled // this is only for next word suggestions, so they need to be enabled
            || settingsValues.mIncognitoModeEnabled
            || !settingsValues.needsToLookupSuggestions()
            || !StringUtilsKt.isEmoji(text)
            || mConnection.hasSlowInputConnection()
        ) return;
        mLastComposedWord = LastComposedWord.NOT_A_COMPOSED_WORD; // avoid storing consecutive emojis

        // commit emoji to dictionary, so it ends up in history and can be suggested as next word
        mDictionaryFacilitator.addToUserHistory(
            text,
            false,
            mConnection.getNgramContextFromNthPreviousWord(settingsValues.mSpacingAndPunctuations, 2),
            (int) TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis()),
            settingsValues.mBlockPotentiallyOffensive
        );
    }

    public void performUpdateSuggestionStripSync(final SettingsValues settingsValues, final int inputStyle) {
        long startTimeMillis = 0;
        if (DebugFlags.DEBUG_ENABLED) {
            startTimeMillis = SystemClock.elapsedRealtime();
            Log.d(TAG, "performUpdateSuggestionStripSync()");
        }
        // Check if we have a suggestion engine attached.
        if (!settingsValues.needsToLookupSuggestions()) {
            if (mWordComposer.isComposingWord()) {
                Log.w(TAG, "Called updateSuggestionsOrPredictions but suggestions were not "
                        + "requested!");
            }
            // Clear the suggestions strip.
            mSuggestionStripViewAccessor.setSuggestions(SuggestedWords.getEmptyInstance());
            return;
        }

        if (!mWordComposer.isComposingWord() && !settingsValues.mBigramPredictionEnabled) {
            mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
            return;
        }

        final AsyncResultHolder<SuggestedWords> holder = new AsyncResultHolder<>("Suggest");
        mInputLogicHandler.getSuggestedWords(() -> getSuggestedWords(
            inputStyle, SuggestedWords.NOT_A_SEQUENCE_NUMBER,
            suggestedWords -> {
                final String typedWordString = mWordComposer.getTypedWord();
                final SuggestedWordInfo typedWordInfo = new SuggestedWordInfo(
                    typedWordString, "", SuggestedWordInfo.MAX_SCORE, SuggestedWordInfo.KIND_TYPED,
                    Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE
                );
                // Show new suggestions if we have at least one. Otherwise keep the old
                // suggestions with the new typed word. Exception: if the length of the
                // typed word is <= 1 (after a deletion typically) we clear old suggestions.
                if (suggestedWords.size() > 1 || typedWordString.length() <= 1) {
                    holder.set(suggestedWords);
                } else {
                    holder.set(retrieveOlderSuggestions(typedWordInfo, mSuggestedWords));
                }
            }
        ));
        // This line may cause the current thread to wait.
        final SuggestedWords suggestedWords = holder.get(null,
                Constants.GET_SUGGESTED_WORDS_TIMEOUT);
        if (suggestedWords != null) {
            // Prefer clipboard suggestions (if available and setting is enabled) over beginning of sentence predictions.
            if (!(suggestedWords.mInputStyle == SuggestedWords.INPUT_STYLE_BEGINNING_OF_SENTENCE_PREDICTION
                    && mLatinIME.tryShowClipboardSuggestion())) {
                mSuggestionStripViewAccessor.setSuggestions(suggestedWords);
            }
            if (!suggestedWords.isEmpty() && settingsValues.mSuggestionsEnabled && isInlineEmojiSearchAction()) {
                mSuggestionStripViewAccessor.showSuggestionStrip();
            }
        }
        if (DebugFlags.DEBUG_ENABLED) {
            long runTimeMillis = SystemClock.elapsedRealtime() - startTimeMillis;
            Log.d(TAG, "performUpdateSuggestionStripSync() : " + runTimeMillis + " ms to finish");
        }
    }

    /**
     * Check if the cursor is touching a word. If so, restart suggestions on this word, else
     * do nothing.
     *
     * @param settingsValues the current values of the settings.
     */
    public void restartSuggestionsOnWordTouchedByCursor(final SettingsValues settingsValues,
            // TODO: remove this argument, put it into settingsValues
            final String currentKeyboardScript) {
        // HACK: We may want to special-case some apps that exhibit bad behavior in case of
        // recorrection. This is a temporary, stopgap measure that will be removed later.
        // TODO: remove this.
        if (!settingsValues.mSpacingAndPunctuations.mCurrentLanguageHasSpaces
                // If no suggestions are requested, don't try restarting suggestions.
                || !settingsValues.needsToLookupSuggestions()
                // If we are currently in a batch input, we must not resume suggestions, or the result
                // of the batch input will replace the new composition. This may happen in the corner case
                // that the app moves the cursor on its own accord during a batch input.
                || mInputLogicHandler.isInBatchInput()
                // If the cursor is not touching a word, or if there is a selection, return right away.
                || mConnection.hasSelection()
                // If we don't know the cursor location, return.
                || mConnection.getExpectedSelectionStart() < 0) {
            mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
            return;
        }

        updateInlineEmojiSearch();
        if (isInlineEmojiSearchAction()) {
            mInputLogicHandler.getSuggestedWords(() -> getSuggestedWords(SuggestedWords.INPUT_STYLE_TYPING,
                SuggestedWords.NOT_A_SEQUENCE_NUMBER, this::doShowSuggestionsAndClearAutoCorrectionIndicator));
            return;
        }

        if (!mConnection.isCursorTouchingWord(settingsValues.mSpacingAndPunctuations, true /* checkTextAfter */)) {
            // Show predictions.
            mWordComposer.setCapitalizedModeAtStartComposingTime(WordComposer.CAPS_MODE_OFF);
            mLatinIME.mHandler.postUpdateSuggestionStrip(SuggestedWords.INPUT_STYLE_RECORRECTION);
            // "unselect" the previous text
            mConnection.finishComposingText();
            return;
        }
        final TextRange range = mConnection.getWordRangeAtCursor(settingsValues.mSpacingAndPunctuations, currentKeyboardScript);
        if (null == range) return; // Happens if we don't have an input connection at all
        if (range.length() <= 0) {
            // Race condition, or touching a word in a non-supported script.
            mLatinIME.setNeutralSuggestionStrip();
            mConnection.finishComposingText();
            return;
        }
        // If for some strange reason (editor bug or so) we measure the text before the cursor as
        // longer than what the entire text is supposed to be, the safe thing to do is bail out.
        if (range.mHasUrlSpans) return;
        // If there are links, we don't resume suggestions. Making
        // edits to a linkified text through batch commands would ruin the URL spans, and unless
        // we take very complicated steps to preserve the whole link, we can't do things right so
        // we just do not resume because it's safer.
        if (!isResumableWord(settingsValues, range.mWord.toString())) {
            mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
            // "unselect" the previous text
            mConnection.finishComposingText();
            return;
        }
        restartSuggestions(range);
    }

    private void restartSuggestions(final TextRange range) {
        final int numberOfCharsInWordBeforeCursor = range.getNumberOfCharsInWordBeforeCursor();
        final int expectedCursorPosition = mConnection.getExpectedSelectionStart();
        if (numberOfCharsInWordBeforeCursor > expectedCursorPosition) return;
        final ArrayList<SuggestedWordInfo> suggestions = new ArrayList<>();
        final String typedWordString = range.mWord.toString();
        final SuggestedWordInfo typedWordInfo = new SuggestedWordInfo(typedWordString,
                "" /* prevWordsContext */, SuggestedWords.MAX_SUGGESTIONS + 1,
                SuggestedWordInfo.KIND_TYPED, Dictionary.DICTIONARY_USER_TYPED,
                SuggestedWordInfo.NOT_AN_INDEX /* indexOfTouchPointOfSecondWord */,
                SuggestedWordInfo.NOT_A_CONFIDENCE /* autoCommitFirstWordConfidence */);
        suggestions.add(typedWordInfo);
        int i = 0;
        for (final SuggestionSpan span : range.getSuggestionSpansAtWord()) {
            for (final String s : span.getSuggestions()) {
                ++i;
                if (!TextUtils.equals(s, typedWordString)) {
                    suggestions.add(new SuggestedWordInfo(s,
                            "" /* prevWordsContext */, SuggestedWords.MAX_SUGGESTIONS - i,
                            SuggestedWordInfo.KIND_RESUMED, Dictionary.DICTIONARY_RESUMED,
                            SuggestedWordInfo.NOT_AN_INDEX /* indexOfTouchPointOfSecondWord */,
                            SuggestedWordInfo.NOT_A_CONFIDENCE
                                    /* autoCommitFirstWordConfidence */));
                }
            }
        }
        final int[] codePoints = StringUtils.toCodePointArray(typedWordString);
        mWordComposer.setComposingWord(codePoints, mLatinIME.getCoordinatesForCurrentKeyboard(codePoints));
        mWordComposer.setCursorPositionWithinWord(typedWordString.codePointCount(0, numberOfCharsInWordBeforeCursor));
        mConnection.setComposingRegion(expectedCursorPosition - numberOfCharsInWordBeforeCursor,
                expectedCursorPosition + range.getNumberOfCharsInWordAfterCursor());
        if (suggestions.size() <= 1) {
            // If there weren't any suggestion spans on this word, suggestions#size() will be 1
            // if shouldIncludeResumedWordInSuggestions is true, 0 otherwise. In this case, we
            // have no useful suggestions, so we will try to compute some for it instead.
            mInputLogicHandler.getSuggestedWords(() -> getSuggestedWords(SuggestedWords.INPUT_STYLE_TYPING,
                SuggestedWords.NOT_A_SEQUENCE_NUMBER, this::doShowSuggestionsAndClearAutoCorrectionIndicator));
        } else {
            // We found suggestion spans in the word. We'll create the SuggestedWords out of
            // them, and make willAutoCorrect false. We make typedWordValid false, because the
            // color of the word in the suggestion strip changes according to this parameter,
            // and false gives the correct color.
            final SuggestedWords suggestedWords = new SuggestedWords(suggestions,
                    null /* rawSuggestions */, typedWordInfo, false /* typedWordValid */,
                    false /* willAutoCorrect */, false /* isObsoleteSuggestions */,
                    SuggestedWords.INPUT_STYLE_RECORRECTION, SuggestedWords.NOT_A_SEQUENCE_NUMBER);
            doShowSuggestionsAndClearAutoCorrectionIndicator(suggestedWords);
        }
    }

    private void doShowSuggestionsAndClearAutoCorrectionIndicator(final SuggestedWords suggestedWords) {
        mIsAutoCorrectionIndicatorOn = false;
        mLatinIME.mHandler.setSuggestions(suggestedWords);
    }

    /**
     * Reverts a previous commit with auto-correction.
     * <p>
     * This is triggered upon pressing backspace just after a commit with auto-correction.
     *
     * @param inputTransaction The transaction in progress.
     */
    private void revertCommit(final InputTransaction inputTransaction) {
        final CharSequence originallyTypedWord = mLastComposedWord.mTypedWord;
        final CharSequence committedWord = mLastComposedWord.mCommittedWord;
        final String committedWordString = committedWord.toString();
        final int cancelLength = committedWord.length();
        final String separatorString = mLastComposedWord.mSeparatorString;
        // If our separator is a space, we won't actually commit it,
        // but set the space state to PHANTOM so that a space will be inserted
        // on the next keypress
        final boolean usePhantomSpace = separatorString.equals(Constants.STRING_SPACE);
        // We want java chars, not codepoints for the following.
        final int separatorLength = separatorString.length();
        // TODO: should we check our saved separator against the actual contents of the text view?
        final int deleteLength = cancelLength + separatorLength;
        if (DebugFlags.DEBUG_ENABLED) {
            if (mWordComposer.isComposingWord()) {
                throw new RuntimeException("revertCommit, but we are composing a word");
            }
            final CharSequence wordBeforeCursor =
                    mConnection.getTextBeforeCursor(deleteLength, 0).subSequence(0, cancelLength);
            if (!TextUtils.equals(committedWord, wordBeforeCursor)) {
                throw new RuntimeException("revertCommit check failed: we thought we were "
                        + "reverting \"" + committedWord
                        + "\", but before the cursor we found \"" + wordBeforeCursor + "\"");
            }
        }
        mConnection.deleteTextBeforeCursor(deleteLength);
        if (!TextUtils.isEmpty(committedWord)) {
            unlearnWord(committedWordString, inputTransaction.getSettingsValues(),
                    Constants.EVENT_REVERT);
        }
        final String stringToCommit = originallyTypedWord +
                (usePhantomSpace ? "" : separatorString);
        final SpannableString textToCommit = new SpannableString(stringToCommit);
        if (committedWord instanceof SpannableString committedWordWithSuggestionSpans) {
            final Object[] spans = committedWordWithSuggestionSpans.getSpans(0,
                    committedWord.length(), Object.class);
            final int lastCharIndex = textToCommit.length() - 1;
            // We will collect all suggestions in the following array.
            final ArrayList<String> suggestions = new ArrayList<>();
            // First, add the committed word to the list of suggestions.
            suggestions.add(committedWordString);
            for (final Object span : spans) {
                // If this is a suggestion span, we check that the word is not the committed word.
                // That should mostly be the case.
                // Given this, we add it to the list of suggestions, otherwise we discard it.
                if (span instanceof final SuggestionSpan suggestionSpan) {
                    for (final String suggestion : suggestionSpan.getSuggestions()) {
                        if (!suggestion.equals(committedWordString)) {
                            suggestions.add(suggestion);
                        }
                    }
                } else {
                    // If this is not a suggestion span, we just add it as is.
                    textToCommit.setSpan(span, 0, lastCharIndex, committedWordWithSuggestionSpans.getSpanFlags(span));
                }
            }
            // Add the suggestion list to the list of suggestions.
            textToCommit.setSpan(new SuggestionSpan(mLatinIME, inputTransaction.getSettingsValues().mLocale,
                    suggestions.toArray(new String[0]), 0, null),
                    0, lastCharIndex, 0);
        }

        if (inputTransaction.getSettingsValues().mSpacingAndPunctuations.mCurrentLanguageHasSpaces) {
            mConnection.commitText(textToCommit, 1);
            if (usePhantomSpace) {
                mJustRevertedACommit = true;
                mSpaceState = SpaceState.PHANTOM;
            }
        } else {
            // For languages without spaces, we revert the typed string but the cursor is flush
            // with the typed word, so we need to resume suggestions right away.
            final int[] codePoints = StringUtils.toCodePointArray(stringToCommit);
            mWordComposer.setComposingWord(codePoints, mLatinIME.getCoordinatesForCurrentKeyboard(codePoints));
            setComposingTextInternal(textToCommit, 1);
        }
        // Don't restart suggestion yet. We'll restart if the user deletes the separator.
        mLastComposedWord = LastComposedWord.NOT_A_COMPOSED_WORD;

        // We have a separator between the word and the cursor: we should show predictions.
        inputTransaction.setRequiresUpdateSuggestions();
    }

    /**
     * Factor in auto-caps and manual caps and compute the current caps mode.
     * @param settingsValues the current settings values.
     * @param keyboardShiftMode the current shift mode of the keyboard. See
     *   KeyboardSwitcher#getKeyboardShiftMode() for possible values.
     * @return the actual caps mode the keyboard is in right now.
     */
    private int getActualCapsMode(final SettingsValues settingsValues,
            final int keyboardShiftMode) {
        if (keyboardShiftMode != WordComposer.CAPS_MODE_AUTO_SHIFTED) {
            return keyboardShiftMode;
        }
        final int auto = getCurrentAutoCapsState(settingsValues);
        if (0 != (auto & TextUtils.CAP_MODE_CHARACTERS)) {
            return WordComposer.CAPS_MODE_AUTO_SHIFT_LOCKED;
        }
        if (0 != auto) {
            return WordComposer.CAPS_MODE_AUTO_SHIFTED;
        }
        return WordComposer.CAPS_MODE_OFF;
    }

    /**
     * Gets the current auto-caps state, factoring in the space state.
     * <p>
     * This method tries its best to do this in the most efficient possible manner. It avoids
     * getting text from the editor if possible at all.
     * This is called from the KeyboardSwitcher (through a trampoline in LatinIME) because it
     * needs to know auto caps state to display the right layout.
     *
     * @param settingsValues the relevant settings values
     * @return a caps mode from TextUtils.CAP_MODE_* or Constants.TextUtils.CAP_MODE_OFF.
     */
    public int getCurrentAutoCapsState(final SettingsValues settingsValues) {
        if (!settingsValues.mAutoCap) return Constants.TextUtils.CAP_MODE_OFF;

        final EditorInfo ei = getCurrentInputEditorInfo();
        if (ei == null) return Constants.TextUtils.CAP_MODE_OFF;
        final int inputType = ei.inputType;
        // Warning: this depends on mSpaceState, which may not be the most current value. If
        // mSpaceState gets updated later, whoever called this may need to be told about it.
        return mConnection.getCursorCapsMode(inputType, settingsValues.mSpacingAndPunctuations,
                SpaceState.PHANTOM == mSpaceState);
    }

    @Nullable
    public RecapitalizeMode getCurrentRecapitalizeState() {
        if (!mRecapitalizeStatus.isStarted()
                || !mRecapitalizeStatus.isSetAt(mConnection.getExpectedSelectionStart(),
                        mConnection.getExpectedSelectionEnd())) {
            // Not recapitalizing at the moment
            return null;
        }
        return mRecapitalizeStatus.getCurrentMode();
    }

    /**
     * @return the editor info for the current editor
     */
    private EditorInfo getCurrentInputEditorInfo() {
        return mLatinIME.getCurrentInputEditorInfo();
    }

    /**
     * Get n-gram context from the nth previous word before the cursor as context
     * for the suggestion process.
     * @param spacingAndPunctuations the current spacing and punctuations settings.
     * @param nthPreviousWord reverse index of the word to get (1-indexed)
     * @return the information of previous words
     */
    public NgramContext getNgramContextFromNthPreviousWordForSuggestion(
            final SpacingAndPunctuations spacingAndPunctuations, final int nthPreviousWord) {
        if (spacingAndPunctuations.mCurrentLanguageHasSpaces) {
            // If we are typing in a language with spaces we can just look up the previous
            // word information from textview.
            return mConnection.getNgramContextFromNthPreviousWord(spacingAndPunctuations, nthPreviousWord);
        }
        if (LastComposedWord.NOT_A_COMPOSED_WORD == mLastComposedWord) {
            return NgramContext.BEGINNING_OF_SENTENCE;
        }
        return new NgramContext(new NgramContext.WordInfo(mLastComposedWord.mCommittedWord.toString()));
    }

    /**
     * Tests the passed word for resumability.
     * <p>
     * We can resume suggestions on words whose first code point is a word code point (with some
     * nuances: check the code for details).
     *
     * @param settings the current values of the settings.
     * @param word the word to evaluate.
     * @return whether it's fine to resume suggestions on this word.
     */
    private static boolean isResumableWord(final SettingsValues settings, final String word) {
        final int firstCodePoint = word.codePointAt(0);
        return settings.isWordCodePoint(firstCodePoint)
                && Constants.CODE_SINGLE_QUOTE != firstCodePoint
                && Constants.CODE_DASH != firstCodePoint;
    }

    /**
     * @param actionId the action to perform
     */
    private void performEditorAction(final int actionId) {
        mConnection.performEditorAction(actionId);
    }

    /**
     * Perform the processing specific to inputting TLDs.
     * <p>
     * Some keys input a TLD (specifically, the ".com" key) and this warrants some specific
     * processing. First, if this is a TLD, we ignore PHANTOM spaces -- this is done by type
     * of character in onCodeInput, but since this gets inputted as a whole string we need to
     * do it here specifically. Then, if the last character before the cursor is a period, then
     * we cut the dot at the start of ".com". This is because humans tend to type "www.google."
     * and then press the ".com" key and instinctively don't expect to get "www.google..com".
     *
     * @param text the raw text supplied to onTextInput
     * @return the text to actually send to the editor
     */
    private String performSpecificTldProcessingOnTextInput(final String text) {
        if (text.length() <= 1 || text.charAt(0) != Constants.CODE_PERIOD
                || !Character.isLetter(text.charAt(1))) {
            // Not a tld: do nothing.
            return text;
        }
        // We have a TLD (or something that looks like this): make sure we don't add
        // a space even if currently in phantom mode.
        mSpaceState = SpaceState.NONE;
        final int codePointBeforeCursor = mConnection.getCodePointBeforeCursor();
        // If no code point, #getCodePointBeforeCursor returns NOT_A_CODE_POINT.
        if (Constants.CODE_PERIOD == codePointBeforeCursor) {
            return text.substring(1);
        }
        return text;
    }

    /**
     * Handle a press on the settings key.
     */
    private void onSettingsKeyPressed() {
        mLatinIME.displaySettingsDialog();
    }

    /**
     * Resets the whole input state to the starting state.
     * <p>
     * This will clear the composing word, reset the last composed word, clear the suggestion
     * strip and tell the input connection about it so that it can refresh its caches.
     *
     * @param newSelStart the new selection start, in java characters.
     * @param newSelEnd the new selection end, in java characters.
     * @param clearSuggestionStrip whether this method should clear the suggestion strip.
     */
    // TODO: how is this different from startInput ?!
    private void resetEntireInputState(final int newSelStart, final int newSelEnd,
            final boolean clearSuggestionStrip) {
        final boolean shouldFinishComposition = mWordComposer.isComposingWord();
        resetComposingState(true /* alsoResetLastComposedWord */);
        if (clearSuggestionStrip) {
            mSuggestionStripViewAccessor.setNeutralSuggestionStrip();
        }
        mConnection.resetCachesUponCursorMoveAndReturnSuccess(newSelStart, newSelEnd, shouldFinishComposition);
    }

    /**
     * Resets only the composing state.
     * <p>
     * Compare #resetEntireInputState, which also clears the suggestion strip and resets the
     * input connection caches. This only deals with the composing state.
     *
     * @param alsoResetLastComposedWord whether to also reset the last composed word.
     */
    private void resetComposingState(final boolean alsoResetLastComposedWord) {
        mWordComposer.reset();
        // Two-thumb typing (#1.1): composing word is wiped — drop the matching boundaries.
        clearFragmentBoundaries();
        // Combining mode is keyed on a composing word existing; if we're wiping it, the
        // pending timer would commit nothing useful, so cancel.
        cancelCombiningMode();
        if (alsoResetLastComposedWord) {
            mLastComposedWord = LastComposedWord.NOT_A_COMPOSED_WORD;
        }
    }

    /**
     * Make a {@link helium314.keyboard.latin.SuggestedWords} object containing a typed word
     * and obsolete suggestions.
     * See {@link helium314.keyboard.latin.SuggestedWords#getTypedWordAndPreviousSuggestions(
     *      SuggestedWordInfo, helium314.keyboard.latin.SuggestedWords)}.
     * @param typedWordInfo The typed word as a SuggestedWordInfo.
     * @param previousSuggestedWords The previously suggested words.
     * @return Obsolete suggestions with the newly typed word.
     */
    static SuggestedWords retrieveOlderSuggestions(final SuggestedWordInfo typedWordInfo,
            final SuggestedWords previousSuggestedWords) {
        final SuggestedWords oldSuggestedWords = previousSuggestedWords.isPunctuationSuggestions()
                ? SuggestedWords.getEmptyInstance() : previousSuggestedWords;
        final ArrayList<SuggestedWords.SuggestedWordInfo> typedWordAndPreviousSuggestions =
                SuggestedWords.getTypedWordAndPreviousSuggestions(typedWordInfo, oldSuggestedWords);
        return new SuggestedWords(typedWordAndPreviousSuggestions, null /* rawSuggestions */,
                typedWordInfo, false /* typedWordValid */, false /* hasAutoCorrectionCandidate */,
                true /* isObsoleteSuggestions */, oldSuggestedWords.mInputStyle,
                SuggestedWords.NOT_A_SEQUENCE_NUMBER);
    }

    /**
     * @return the current {@link Locale} of the {@link #mDictionaryFacilitator} if available. Otherwise
     * {@link Locale#ROOT}.
     */
    @NonNull
    private Locale getDictionaryFacilitatorLocale() {
        return mDictionaryFacilitator != null ? mDictionaryFacilitator.getCurrentLocale() : Locale.ROOT;
    }

    /**
     * Gets a chunk of text with or the auto-correction indicator underline span as appropriate.
     * <p>
     * This method looks at the old state of the auto-correction indicator to put or not put
     * the underline span as appropriate. It is important to note that this does not correspond
     * exactly to whether this word will be auto-corrected to or not: what's important here is
     * to keep the same indication as before.
     * When we add a new code point to a composing word, we don't know yet if we are going to
     * auto-correct it until the suggestions are computed. But in the mean time, we still need
     * to display the character and to extend the previous underline. To avoid any flickering,
     * the underline should keep the same color it used to have, even if that's not ultimately
     * the correct color for this new word. When the suggestions are finished evaluating, we
     * will call this method again to fix the color of the underline.
     *
     * @param text the text on which to maybe apply the span.
     * @return the same text, with the auto-correction underline span if that's appropriate.
     */
    // TODO: Shouldn't this go in some *Utils class instead?
    private CharSequence getTextWithUnderline(final String text) {
        // TODO: Locale should be determined based on context and the text given.
        return mIsAutoCorrectionIndicatorOn
                ? SuggestionSpanUtilsKt.getTextWithAutoCorrectionIndicatorUnderline(
                        mLatinIME, text, getDictionaryFacilitatorLocale())
                : text;
    }

    /**
     * Sends a DOWN key event followed by an UP key event to the editor.
     * <p>
     * If possible at all, avoid using this method. It causes all sorts of race conditions with
     * the text view because it goes through a different, asynchronous binder. Also, batch edits
     * are ignored for key events. Use the normal software input methods instead.
     *
     * @param keyCode the key code to send inside the key event.
     */
    public void sendDownUpKeyEvent(final int keyCode) {
        sendDownUpKeyEventWithMetaState(keyCode, 0);
    }

    /**
     * Sends a DOWN key event followed by an UP key event to the editor.
     * <p>
     * If possible at all, avoid using this method. It causes all sorts of race conditions with
     * the text view because it goes through a different, asynchronous binder. Also, batch edits
     * are ignored for key events. Use the normal software input methods instead.
     *
     * @param keyCode the key code to send inside the key event.
     * @param metaState the meta state to send inside the key event, e.g. KeyEvent.META_CTRL_ON
     */
    public void sendDownUpKeyEventWithMetaState(final int keyCode, final int metaState) {
        final long eventTime = SystemClock.uptimeMillis();
        mConnection.sendKeyEvent(new KeyEvent(eventTime, eventTime,
                KeyEvent.ACTION_DOWN, keyCode, 0, metaState, KeyCharacterMap.VIRTUAL_KEYBOARD, 0,
                KeyEvent.FLAG_SOFT_KEYBOARD | KeyEvent.FLAG_KEEP_TOUCH_MODE));
        mConnection.sendKeyEvent(new KeyEvent(SystemClock.uptimeMillis(), eventTime,
                KeyEvent.ACTION_UP, keyCode, 0, metaState, KeyCharacterMap.VIRTUAL_KEYBOARD, 0,
                KeyEvent.FLAG_SOFT_KEYBOARD | KeyEvent.FLAG_KEEP_TOUCH_MODE));
    }

    /**
     * Insert an automatic space, if the options allow it.
     * <p>
     * This checks the options and the text before the cursor are appropriate before inserting
     * an automatic space.
     *
     * @param settingsValues the current values of the settings.
     */
    private void insertAutomaticSpaceIfOptionsAndTextAllow(final SettingsValues settingsValues) {
        if (settingsValues.shouldInsertSpacesAutomatically()
                && settingsValues.mSpacingAndPunctuations.mCurrentLanguageHasSpaces
                && !textBeforeCursorMayBeUrlOrSimilar(settingsValues, true)
                && !mConnection.textBeforeCursorLooksLikeURL() // adding this check to textBeforeCursorMayBeUrlOrSimilar might not be wanted for word continuation (see effect on unit tests)
                && !(mConnection.getCodePointBeforeCursor() == Constants.CODE_PERIOD && mConnection.wordBeforeCursorMayBeEmail())
        ) {
            mConnection.commitCodePoint(Constants.CODE_SPACE);
            // todo: why not remove phantom space state?
        }
    }

    /**
     * Multi-part word composition (#1.6): predicate for "looks like a plain English word"
     * — only letters and apostrophes, no hyphens, no mid-word capitals, no digits. Used to
     * filter out the lib's occasional obscure hyphenated/CamelCase dictionary entries
     * (e.g. "technon-U", "techMolly") that outrank the obvious answer.
     */
    private static boolean isPlainLetterWord(final String s) {
        if (s == null || s.isEmpty()) return false;
        for (int i = 0; i < s.length(); ) {
            final int cp = s.codePointAt(i);
            if (cp == '\'') {
                // apostrophes OK (don't, it's, etc.)
            } else if (!Character.isLetter(cp)) {
                return false;
            } else if (i > 0 && Character.isUpperCase(cp)) {
                // mid-word capital (CamelCase) — reject
                return false;
            }
            i += Character.charCount(cp);
        }
        return true;
    }

    private boolean textBeforeCursorMayBeUrlOrSimilar(final SettingsValues settingsValues, final Boolean forAutoSpace) {
        // URL / mail field and no space -> may be URL
        if (InputTypeUtils.isUriOrEmailType(settingsValues.mInputAttributes.mInputType) &&
                // we never want to commit the first part of the url, but we want to insert autospace if text might be a normal word
                (forAutoSpace ? mConnection.nonWordCodePointAndNoSpaceBeforeCursor(settingsValues.mSpacingAndPunctuations) // avoid detecting URL if it could be a word
                : !mConnection.spaceBeforeCursor()))
            return true;
        // already contains a SometimesWordConnector -> may be URL (not so sure, only do with detection enabled
        if (settingsValues.mUrlDetectionEnabled && settingsValues.mSpacingAndPunctuations.containsSometimesWordConnector(mWordComposer.getTypedWord()))
            return true;
        // "://" before typed word -> very much looks like URL
        final CharSequence textBeforeCursor = mConnection.getTextBeforeCursor(mWordComposer.getTypedWord().length() + 3, 0);
        if (textBeforeCursor != null && textBeforeCursor.toString().startsWith("://"))
            return true;
        return false;
    }

    /**
     * Do the final processing after a batch input has ended. This commits the word to the editor.
     * @param settingsValues the current values of the settings.
     * @param suggestedWords suggestedWords to use.
     */
    public void onUpdateTailBatchInputCompleted(final SettingsValues settingsValues,
            final SuggestedWords suggestedWords, final KeyboardSwitcher keyboardSwitcher) {
        String batchInputText = suggestedWords.isEmpty() ? null : suggestedWords.getWord(0);
        // Multi-part word composition (#1.6): when the merged-trail extend-base path was
        // used, the lib returns the full word but its top pick is sometimes an obscure
        // hyphenated/mid-cased dictionary entry ("technon-U") that ranks just above the
        // obvious answer ("technology"). Prefer the highest-ranked plain-letters suggestion
        // when extend-base is active.
        if (mWordComposer.isExtendBatchInputBaseSet() && !TextUtils.isEmpty(batchInputText)
                && !isPlainLetterWord(batchInputText)) {
            for (int i = 1; i < suggestedWords.size(); i++) {
                final String cand = suggestedWords.getWord(i);
                if (cand != null && !cand.isEmpty() && isPlainLetterWord(cand)) {
                    batchInputText = cand;
                    break;
                }
            }
        }
        if (TextUtils.isEmpty(batchInputText)) {
            // Still need to clear the seed slot so it doesn't leak into the next gesture.
            helium314.keyboard.keyboard.PointerTracker.consumeGestureSeedCodepoint();
            return;
        }
        // Combining-mode seeding: if PointerTracker seeded this gesture with a prior tap's
        // coords, the recognizer typically includes that letter as the first char of the
        // result. We strip it (case-insensitive) so the existing concat below doesn't
        // double-count it. See PointerTracker for the full rationale.
        //
        // Multi-part word composition (#1.6): the top suggestion isn't always the seeded
        // continuation — e.g. swiping "nology" after "tech" might recognize as "biology"
        // because the second swipe's path can match both. When a seed is set, prefer any
        // suggestion (in score order) whose first letter matches the seed; only fall back
        // to the top suggestion if none match.
        // Combining-mode seeding: if PointerTracker seeded this gesture with a prior tap's
        // coords, the recognizer typically includes that letter as the first char of the
        // result. We strip it (case-insensitive) so the existing concat below doesn't
        // double-count it. See PointerTracker for the full rationale.
        final int seedCp = helium314.keyboard.keyboard.PointerTracker.consumeGestureSeedCodepoint();
        if (seedCp > 0 && batchInputText.length() > 0) {
            final int firstCp = batchInputText.codePointAt(0);
            if (Character.toLowerCase(firstCp) == Character.toLowerCase(seedCp)) {
                batchInputText = batchInputText.substring(Character.charCount(firstCp));
            }
        }
        if (batchInputText.isEmpty()) return;
        mConnection.beginBatchEdit();
        // Two-thumb typing (#1.1 + #1.4): when either manual spacing OR tap-promotion-extend
        // applies, a follow-up gesture EXTENDS the existing composing word rather than
        // replacing it. We snapshot what the word composer currently holds and concatenate
        // before handing the combined string to {@link WordComposer#setBatchInputWord} and
        // the editor. The composer's reset() inside setBatchInputWord does not touch
        // mInputPointers (see WordComposer.java:106 vs the comment at line 279), so
        // re-feeding a longer string is safe.
        //
        // {@code mGestureExtendsByTapPromotion} was set in onStartBatchInput so the
        // promotion-window decision is made at gesture-start time (a long gesture shouldn't
        // age out of the promotion window between start and end).
        // Multi-part word composition (#1.6): when combining mode is armed AND a composing
        // word exists, the next gesture extends it (tech + nology -> technology). This
        // unifies what was previously only possible via manual-spacing or tap-promotion.
        final boolean combiningExtendsSwipe = settingsValues.mMultipartAutoExtendInCombining
                && settingsValues.mCombiningGraceMs > 0
                && mInCombiningMode;
        final boolean extendExistingCompose = (settingsValues.mGestureManualSpacing
                        || mGestureExtendsByTapPromotion
                        || combiningExtendsSwipe)
                && mWordComposer.isComposingWord()
                && !mWordComposer.isCursorFrontOrMiddleOfComposingWord();
        // Multi-part word composition (#1.6): when the lib was fed the MERGED prior+new
        // trail via the WordComposer extend-base mechanism, its top suggestion already
        // represents the WHOLE word (e.g. 'technology' rather than 'biology'). In that
        // case prevTypedWord must NOT be concatenated again or we'd get 'techtechnology'.
        final boolean usedMergedTrail = mWordComposer.isExtendBatchInputBaseSet();
        // Clear the extend base now that we've used it — gesture is committing.
        mWordComposer.setExtendBatchInputBase(null);
        final String prevTypedWord = (extendExistingCompose && !usedMergedTrail)
                ? mWordComposer.getTypedWord() : "";
        // Auto-capitalize the first letter of a fresh-word gesture when the keyboard is in
        // auto-shifted / manual-shifted / shift-locked state. The gesture-recognizer always
        // returns lowercase, so without this fix swiping "Hello" at sentence-start types
        // "hello". We deliberately skip this when extending an existing composing word, since
        // those continuation gestures should append in the casing the user already chose for
        // the start of the word.
        if (!extendExistingCompose && !batchInputText.isEmpty()) {
            // Use the shift mode captured at gesture-start, not the live mode — the
            // keyboard auto-clears the shifted indicator during the gesture, so a live
            // read here always returns UNSHIFTED.
            final int shiftMode = mShiftModeAtGestureStart;
            if (shiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFTED
                    || shiftMode == WordComposer.CAPS_MODE_AUTO_SHIFTED) {
                batchInputText = StringUtils.capitalizeFirstCodePoint(batchInputText, settingsValues.mLocale);
            } else if (shiftMode == WordComposer.CAPS_MODE_AUTO_SHIFT_LOCKED
                    || shiftMode == WordComposer.CAPS_MODE_MANUAL_SHIFT_LOCKED) {
                batchInputText = batchInputText.toUpperCase(settingsValues.mLocale);
            }
        }
        // Clear so a stale value from a previous gesture can't leak into a non-gesture
        // commit later.
        mShiftModeAtGestureStart = WordComposer.CAPS_MODE_OFF;
        final String composedText = prevTypedWord + batchInputText;
        // Two-thumb typing (#1.1 + #1.4): never silently prepend an autospace when extending an
        // existing composing word. {@code mSpaceState} can carry PHANTOM in from prior
        // punctuation / suggestion-pick paths; clearing it here (without inserting) is the
        // only way to guarantee the user-visible "no autospace" promise.
        if (SpaceState.PHANTOM == mSpaceState) {
            insertAutomaticSpaceIfOptionsAndTextAllow(settingsValues);
            mSpaceState = SpaceState.NONE;
        }
        enterInlineEmojiSearchIfNeeded(batchInputText.codePointAt(0), settingsValues);
        mWordComposer.setBatchInputWord(composedText);
        setComposingTextInternal(composedText, 1);
        if (extendExistingCompose) {
            // Two-thumb typing (#1.1 + #1.4): downgrade the composer out of batch mode so
            // future dictionary lookups treat the (possibly already-concatenated) composing
            // word as typed text instead of using only the last gesture's input pointers.
            // Otherwise the suggestion strip would keep showing suggestions for the most
            // recent fragment, and picking one would replace the WHOLE composing span with a
            // single-fragment alternative — catastrophic data loss for the user.
            mWordComposer.unsetBatchMode();
            // Multi-part word composition (#1.6): the suggestion strip was computed during
            // the gesture and reflects only the LAST fragment, which would mislead the user
            // if they tapped it (replacing the whole composing span with one fragment). Two
            // options: blank the strip (legacy) or repopulate it for the full composing word.
            if (settingsValues.mMultipartFullWordSuggestions) {
                performUpdateSuggestionStripSync(settingsValues, SuggestedWords.INPUT_STYLE_TYPING);
            } else {
                setSuggestedWords(SuggestedWords.getEmptyInstance());
            }
            // PREF_GESTURE_FRAGMENT_BACKSPACE: record this gesture as a fragment boundary
            // so backspace can pop the whole word at once. When extending an existing
            // composing word, both the prior fragments AND this new one are already in the
            // boundaries list (prior fragments were recorded at their own append time);
            // recordFragmentBoundaryIfTracking adds the NEW boundary at the end. No-op if
            // PREF_GESTURE_FRAGMENT_BACKSPACE isn't on.
            recordFragmentBoundaryIfTracking(settingsValues);
        } else {
            // Non-extend path replaces the whole composing word each gesture; any prior
            // fragment boundaries become meaningless.
            clearFragmentBoundaries();
        }
        // Tap-promotion-extend is a one-shot decision per gesture; clear the flag so the
        // next gesture re-evaluates the timing.
        mGestureExtendsByTapPromotion = false;
        mConnection.endBatchEdit();
        // Space state must be updated before calling updateShiftState. Two-thumb typing
        // (#1.1 + #1.4): skip the after-gesture autospace when we're extending — neither
        // manual spacing nor tap-promotion should auto-insert a space after the word.
        // Combining mode (#1.5): when the grace timer is active, IT owns autospace timing.
        // Setting PHANTOM here would cause the very next letter tap (e.g. "tech" + "y") to
        // commit the composing word and insert a space before the new letter via
        // handleNonSpecialCharacterEvent's PHANTOM branch, producing "tech y" instead of
        // letting the user extend the composing word to "techy".
        if (settingsValues.mAutospaceAfterGestureTyping && !extendExistingCompose
                && settingsValues.mCombiningGraceMs <= 0)
            mSpaceState = SpaceState.PHANTOM;
        keyboardSwitcher.requestUpdatingShiftState(getCurrentAutoCapsState(settingsValues), getCurrentRecapitalizeState());

        if (isInlineEmojiSearchAction()) {
            searchForEmojiInline(SuggestedWords.NOT_A_SEQUENCE_NUMBER, mLatinIME::setSuggestions);
        }
        // Combining mode: arm the grace timer after a gesture.
        enterCombiningMode(settingsValues, false /* fromTap, unused — kept for clarity */);
    }

    /**
     * Commit the typed string to the editor.
     * <p>
     * This is typically called when we should commit the currently composing word without applying
     * auto-correction to it. Typically, we come here upon pressing a separator when the keyboard
     * is configured to not do auto-correction at all (because of the settings or the properties of
     * the editor). In this case, `separatorString' is set to the separator that was pressed.
     * We also come here in a variety of cases with external user action. For example, when the
     * cursor is moved while there is a composition, or when the keyboard is closed, or when the
     * user presses the Send button for an SMS, we don't auto-correct as that would be unexpected.
     * In this case, `separatorString' is set to NOT_A_SEPARATOR.
     *
     * @param settingsValues the current values of the settings.
     * @param separatorString the separator that's causing the commit, or NOT_A_SEPARATOR if none.
     */
    public void commitTyped(final SettingsValues settingsValues, final String separatorString) {
        if (!mWordComposer.isComposingWord()) return;
        final String typedWord = mWordComposer.getTypedWord();
        if (typedWord.length() > 0) {
            final boolean isBatchMode = mWordComposer.isBatchMode();
            commitChosenWord(settingsValues, typedWord, LastComposedWord.COMMIT_TYPE_USER_TYPED_WORD, separatorString);
            StatsUtils.onWordCommitUserTyped(typedWord, isBatchMode);
        }
    }

    /**
     * Commit the current auto-correction.
     * <p>
     * This will commit the best guess of the keyboard regarding what the user meant by typing
     * the currently composing word. The IME computes suggestions and assigns a confidence score
     * to each of them; when it's confident enough in one suggestion, it replaces the typed string
     * by this suggestion at commit time. When it's not confident enough, or when it has no
     * suggestions, or when the settings or environment does not allow for auto-correction, then
     * this method just commits the typed string.
     * Note that if suggestions are currently being computed in the background, this method will
     * block until the computation returns. This is necessary for consistency (it would be very
     * strange if pressing space would commit a different word depending on how fast you press).
     *
     * @param settingsValues the current value of the settings.
     * @param separator the separator that's causing the commit to happen.
     */
    private void commitCurrentAutoCorrection(final SettingsValues settingsValues,
            final String separator, final LatinIME.UIHandler handler) {
        // Complete any pending suggestions query first
        if (handler.hasPendingUpdateSuggestions()) {
            handler.cancelUpdateSuggestionStrip();
            // To know the input style here, we should retrieve the in-flight "update suggestions"
            // message and read its arg1 member here. However, the Handler class does not let
            // us retrieve this message, so we can't do that. But in fact, we notice that
            // we only ever come here when the input style was typing. In the case of batch
            // input, we update the suggestions synchronously when the tail batch comes. Likewise
            // for application-specified completions. As for recorrections, we never auto-correct,
            // so we don't come here either. Hence, the input style is necessarily
            // INPUT_STYLE_TYPING.
            performUpdateSuggestionStripSync(settingsValues, SuggestedWords.INPUT_STYLE_TYPING);
        }
        final SuggestedWordInfo autoCorrectionOrNull = mWordComposer.getAutoCorrectionOrNull();
        final String typedWord = mWordComposer.getTypedWord();
        final String stringToCommit = (autoCorrectionOrNull != null) ? autoCorrectionOrNull.mWord : typedWord;
        if (stringToCommit != null) {
            final boolean isBatchMode = mWordComposer.isBatchMode();
            commitChosenWord(settingsValues, stringToCommit, LastComposedWord.COMMIT_TYPE_DECIDED_WORD, separator);
            if (!typedWord.equals(stringToCommit)) {
                // This will make the correction flash for a short while as a visual clue
                // to the user that auto-correction happened. It has no other effect; in particular
                // note that this won't affect the text inside the text field AT ALL: it only makes
                // the segment of text starting at the supplied index and running for the length
                // of the auto-correction flash. At this moment, the "typedWord" argument is
                // ignored by TextView.
                mConnection.commitCorrection(new CorrectionInfo(
                        mConnection.getExpectedSelectionEnd() - stringToCommit.length(),
                        typedWord, stringToCommit));
                final String prevWordsContext = (autoCorrectionOrNull != null)
                        ? autoCorrectionOrNull.mPrevWordsContext
                        : "";
                StatsUtils.onAutoCorrection(typedWord, stringToCommit, isBatchMode,
                        mDictionaryFacilitator, prevWordsContext);
                StatsUtils.onWordCommitAutoCorrect(stringToCommit, isBatchMode);
            } else {
                StatsUtils.onWordCommitUserTyped(stringToCommit, isBatchMode);
            }
        }
    }

    /**
     * Commits the chosen word to the text field and saves it for later retrieval.
     *
     * @param settingsValues the current values of the settings.
     * @param chosenWord the word we want to commit.
     * @param commitType the type of the commit, as one of LastComposedWord.COMMIT_TYPE_*
     * @param separatorString the separator that's causing the commit, or NOT_A_SEPARATOR if none.
     */
    private void commitChosenWord(final SettingsValues settingsValues, final String chosenWord,
            final int commitType, final String separatorString) {
        long startTimeMillis = 0;
        if (DebugFlags.DEBUG_ENABLED) {
            startTimeMillis = SystemClock.elapsedRealtime();
            Log.d(TAG, "commitChosenWord() : [" + chosenWord + "]");
        }
        // essentially reverted https://github.com/lineageos/android_packages_inputmethods_LatinIME/commit/ee6de1466bc98e27bd414c9a7451f2aee3f9e721
        // can't find any drawback (performance, neither when setting nor when reading)
        final CharSequence chosenWordWithSuggestions = getTextWithSuggestionSpan(mLatinIME, chosenWord,
                mSuggestedWords, getDictionaryFacilitatorLocale());
        if (DebugFlags.DEBUG_ENABLED) {
            long runTimeMillis = SystemClock.elapsedRealtime() - startTimeMillis;
            Log.d(TAG, "commitChosenWord() : " + runTimeMillis + " ms to run "
                    + "SuggestionSpanUtils.getTextWithSuggestionSpan()");
            startTimeMillis = SystemClock.elapsedRealtime();
        }
        // When we are composing word, get n-gram context from the 2nd previous word because the
        // 1st previous word is the word to be committed. Otherwise get n-gram context from the 1st
        // previous word.
        final NgramContext ngramContext = mConnection.getNgramContextFromNthPreviousWord(
                settingsValues.mSpacingAndPunctuations, mWordComposer.isComposingWord() ? 2 : 1);
        if (DebugFlags.DEBUG_ENABLED) {
            long runTimeMillis = SystemClock.elapsedRealtime() - startTimeMillis;
            Log.d(TAG, "commitChosenWord() : " + runTimeMillis + " ms to run "
                    + "Connection.getNgramContextFromNthPreviousWord()");
            Log.d(TAG, "commitChosenWord() : NgramContext = " + ngramContext);
            startTimeMillis = SystemClock.elapsedRealtime();
        }
        mConnection.commitText(chosenWordWithSuggestions, 1);
        if (DebugFlags.DEBUG_ENABLED) {
            long runTimeMillis = SystemClock.elapsedRealtime() - startTimeMillis;
            Log.d(TAG, "commitChosenWord() : " + runTimeMillis + " ms to run "
                    + "Connection.commitText");
            startTimeMillis = SystemClock.elapsedRealtime();
        }
        // Add the word to the user history dictionary
        performAdditionToUserHistoryDictionary(settingsValues, chosenWord, ngramContext);
        if (DebugFlags.DEBUG_ENABLED) {
            long runTimeMillis = SystemClock.elapsedRealtime() - startTimeMillis;
            Log.d(TAG, "commitChosenWord() : " + runTimeMillis + " ms to run "
                    + "performAdditionToUserHistoryDictionary()");
            startTimeMillis = SystemClock.elapsedRealtime();
        }
        // TODO: figure out here if this is an auto-correct or if the best word is actually
        // what user typed. Note: currently this is done much later in
        // LastComposedWord#didCommitTypedWord by string equality of the remembered
        // strings.
        mLastComposedWord = mWordComposer.commitWord(commitType, chosenWord, separatorString, ngramContext);
        if (DebugFlags.DEBUG_ENABLED) {
            long runTimeMillis = SystemClock.elapsedRealtime() - startTimeMillis;
            Log.d(TAG, "commitChosenWord() : " + runTimeMillis + " ms to run "
                    + "WordComposer.commitWord()");
        }
    }

    /**
     *  Wraps the selected text into the codepoints. If the same codepoints are
     *  already present before and after the selection, they are removed instead.
     *  (unfortunately Android long-press selection may select one of those codepoints, so unwrap may not work well)
     */
    private void wrapSelection(final int start, final int end) {
        final CharSequence selected = mConnection.getSelectedText(0);
        if (mConnection.getCodePointBeforeCursor() == start) {
            // maybe we want to revert it
            final CharSequence afterCursor = mConnection.getTextAfterCursor(1, 0);
            if (afterCursor.length() > 0 && afterCursor.charAt(0) == end) {
                mConnection.setSelection(mConnection.getExpectedSelectionStart() - 1, mConnection.getExpectedSelectionEnd() + 1);
                mConnection.commitText(selected, 1);
                return;
            }
        }
        mConnection.commitText(StringUtils.newSingleCodePointString(start) + selected + StringUtils.newSingleCodePointString(end), 1);
    }

    /**
     * Retry resetting caches in the rich input connection.
     * <p>
     * When the editor can't be accessed we can't reset the caches, so we schedule a retry.
     * This method handles the retry, and re-schedules a new retry if we still can't access.
     * We only retry up to 5 times before giving up.
     *
     * @param tryResumeSuggestions Whether we should resume suggestions or not.
     * @param remainingTries How many times we may try again before giving up.
     * @return whether true if the caches were successfully reset, false otherwise.
     */
    public boolean retryResetCachesAndReturnSuccess(final boolean tryResumeSuggestions,
            final int remainingTries, final LatinIME.UIHandler handler) {
        final boolean shouldFinishComposition = mConnection.hasSelection()
                || !mConnection.isCursorPositionKnown();
        if (!mConnection.resetCachesUponCursorMoveAndReturnSuccess(
                mConnection.getExpectedSelectionStart(), mConnection.getExpectedSelectionEnd(),
                shouldFinishComposition)) {
            if (0 < remainingTries) {
                handler.postResetCaches(tryResumeSuggestions, remainingTries - 1);
                return false;
            }
            // If remainingTries is 0, we should stop waiting for new tries, however we'll still
            // return true as we need to perform other tasks (for example, loading the keyboard).
        }
        mConnection.tryFixIncorrectCursorPosition();
        if (tryResumeSuggestions) {
            handler.postResumeSuggestions(true /* shouldDelay */);
        }
        return true;
    }

    // we used to provide keyboard, settingsValues and keyboardShiftMode, but every time read it from current instance anyway
    void getSuggestedWords(final int inputStyle, final int sequenceNumber, final OnGetSuggestedWordsCallback callback) {
        final Keyboard keyboard = KeyboardSwitcher.getInstance().getKeyboard();
        if (keyboard == null) {
            callback.onGetSuggestedWords(SuggestedWords.getEmptyInstance());
            return;
        }
        if (inputStyle != SuggestedWords.INPUT_STYLE_UPDATE_BATCH && inputStyle != SuggestedWords.INPUT_STYLE_TAIL_BATCH
                        && isInlineEmojiSearchAction()) {
            searchForEmojiInline(sequenceNumber, callback);
            return;
        }
        final SettingsValues settingsValues = Settings.getValues();
        mWordComposer.adviseCapitalizedModeBeforeFetchingSuggestions(
                getActualCapsMode(settingsValues, KeyboardSwitcher.getInstance().getKeyboardShiftMode()));
        try {
            final SuggestedWords suggestedWords = mSuggest.getSuggestedWords(mWordComposer,
                    getNgramContextFromNthPreviousWordForSuggestion(
                    settingsValues.mSpacingAndPunctuations,
                    // Get the word on which we should search the bigrams. If we are composing
                    // a word, it's whatever is *before* the half-committed word in the buffer,
                    // hence 2; if we aren't, we should just skip whitespace if any, so 1.
                    mWordComposer.isComposingWord() ? 2 : 1),
                    keyboard,
                    settingsValues.mSettingsValuesForSuggestion,
                    settingsValues.mAutoCorrectEnabled,
                    inputStyle, sequenceNumber);
            callback.onGetSuggestedWords(suggestedWords);
        } catch (Exception e) {
            // better go without suggestions than have the keyboard crash
            Log.e(TAG, "Error fetching suggested words, using empty words instead", e);
            callback.onGetSuggestedWords(SuggestedWords.getEmptyInstance());
            KeyboardSwitcher.getInstance().showToast("Error getting suggestions", true);
        }
    }

    /**
     * Used as an injection point for each call of
     * {@link RichInputConnection#setComposingText(CharSequence, int)}.
     *
     * <p>Currently using this method is optional and you can still directly call
     * {@link RichInputConnection#setComposingText(CharSequence, int)}, but it is recommended to
     * use this method whenever possible.<p>
     * <p>TODO: Should we move this mechanism to {@link RichInputConnection}?</p>
     *
     * @param newComposingText the composing text to be set
     * @param newCursorPosition the new cursor position
     */
    private void setComposingTextInternal(final CharSequence newComposingText,
            final int newCursorPosition) {
        setComposingTextInternalWithBackgroundColor(newComposingText, newCursorPosition,
                Color.TRANSPARENT, newComposingText.length());
    }

    /**
     * Equivalent to {@link #setComposingTextInternal(CharSequence, int)} except that this method
     * allows to set {@link BackgroundColorSpan} to the composing text with the given color.
     *
     * <p>TODO: Currently the background color is exclusive with the black underline, which is
     * automatically added by the framework. We need to change the framework if we need to have both
     * of them at the same time.</p>
     * <p>TODO: Should we move this method to {@link RichInputConnection}?</p>
     *
     * @param newComposingText the composing text to be set
     * @param newCursorPosition the new cursor position
     * @param backgroundColor the background color to be set to the composing text. Set
     * {@link Color#TRANSPARENT} to disable the background color.
     * @param coloredTextLength the length of text, in Java chars, which should be rendered with
     * the given background color.
     */
    private void setComposingTextInternalWithBackgroundColor(final CharSequence newComposingText,
            final int newCursorPosition, final int backgroundColor, final int coloredTextLength) {
        final CharSequence composingTextToBeSet;
        if (backgroundColor == Color.TRANSPARENT) {
            composingTextToBeSet = newComposingText;
        } else {
            final SpannableString spannable = new SpannableString(newComposingText);
            final BackgroundColorSpan backgroundColorSpan = new BackgroundColorSpan(backgroundColor);
            final int spanLength = Math.min(coloredTextLength, spannable.length());
            spannable.setSpan(backgroundColorSpan, 0, spanLength, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE | Spanned.SPAN_COMPOSING);
            composingTextToBeSet = spannable;
        }
        if (!mConnection.setComposingText(composingTextToBeSet, newCursorPosition))
            // inconsistency in set and found composing text, better cancel composing (should be restarted automatically)
            mWordComposer.reset();
    }

    /**
     * Gets an object allowing private IME commands to be sent to the
     * underlying editor.
     * @return An object for sending private commands to the underlying editor.
     */
    public PrivateCommandPerformer getPrivateCommandPerformer() {
        return mConnection;
    }

    /**
     * Gets the expected index of the first char of the composing span within the editor's text.
     * Returns a negative value in case there appears to be no valid composing span.
     *
     * @see #getComposingLength()
     * @see RichInputConnection#hasSelection()
     * @see RichInputConnection#isCursorPositionKnown()
     * @see RichInputConnection#getExpectedSelectionStart()
     * @see RichInputConnection#getExpectedSelectionEnd()
     * @return The expected index in Java chars of the first char of the composing span.
     */
    // TODO: try and see if we can get rid of this method. Ideally the users of this class should
    // never need to know this.
    public int getComposingStart() {
        if (!mConnection.isCursorPositionKnown() || mConnection.hasSelection()) {
            return -1;
        }
        return mConnection.getExpectedSelectionStart() - mWordComposer.size();
    }

    /**
     * Gets the expected length in Java chars of the composing span.
     * May be 0 if there is no valid composing span.
     * @see #getComposingStart()
     * @return The expected length of the composing span.
     */
    // TODO: try and see if we can get rid of this method. Ideally the users of this class should
    // never need to know this.
    public int getComposingLength() {
        return mWordComposer.size();
    }

    private void enterInlineEmojiSearchIfNeeded(int codePoint, SettingsValues settingsValues) {
        if (mEmojiDictionaryFacilitator == null || isInlineEmojiSearchAction()) {
            return;
        }

        if (isStartOfInlineEmojiSearch(codePoint, mConnection.getCodePointBeforeCursor(), mConnection.getCharBeforeBeforeCursor(),
                                       settingsValues)) {
            setInlineEmojiSearchAction(true);
        }
    }

    private void updateInlineEmojiSearch() {
        setInlineEmojiSearchAction(getInlineEmojiSearchString() != null);
    }

    private void setInlineEmojiSearchAction(boolean on) {
        if (on != isInlineEmojiSearchAction()) {
            KeyboardSwitcher.getInstance().loadKeyboard(mLatinIME.getCurrentInputEditorInfo(), Settings.getValues(),
                            mLatinIME.getCurrentAutoCapsState(), mLatinIME.getCurrentRecapitalizeState(),
                            on? new KeyboardLayoutSet.InternalAction(KeyCode.INLINE_EMOJI_SEARCH_DONE,"!icon/close_history") : null);
        }
    }

    private static boolean isInlineEmojiSearchAction() {
        var keyboard = KeyboardSwitcher.getInstance().getKeyboard();
        var internalAction = keyboard != null? keyboard.mId.mInternalAction : null;
        return internalAction != null && internalAction.code() == KeyCode.INLINE_EMOJI_SEARCH_DONE;
    }

    private void searchForEmojiInline(int sequenceNumber, OnGetSuggestedWordsCallback callback) {
        var input = getInlineEmojiSearchString();
        if (StringUtils.isEmpty(input)) {
            callback.onGetSuggestedWords(SuggestedWords.getEmptyInstance());
            return;
        }

        var suggestions = mEmojiDictionaryFacilitator.getSuggestions(StringUtilsKt.splitOnWhitespace(input));
        if (suggestions.isEmpty()) {
            callback.onGetSuggestedWords(SuggestedWords.getEmptyInstance());
            return;
        }

        var typedWordInfo = new SuggestedWordInfo(input, "", SuggestedWordInfo.MAX_SCORE, SuggestedWordInfo.KIND_TYPED,
                                 Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE);
        var suggestedWordInfos = new ArrayList<SuggestedWordInfo>(suggestions.size() + 1);
        suggestedWordInfos.add(typedWordInfo);
        for (var suggestion: suggestions) {
            if (suggestion.isEmoji()) {
                Suggest.addDebugInfo(suggestion, input);
                suggestedWordInfos.add(Suggest.useDefaultEmojiSkinTone(suggestion));
            }
        }
        callback.onGetSuggestedWords(new SuggestedWords(suggestedWordInfos, suggestions.mRawSuggestions, typedWordInfo,
                                     false /* typedWordValid */, false /* autoCorrectEnabled */,
                                     false /* isObsoleteSuggestions */, SuggestedWords.INPUT_STYLE_TYPING, sequenceNumber));
    }

    private void deleteTextReplacedByEmoji() {
        mConnection.finishComposingText();
        var inlineEmojiSearchString = getInlineEmojiSearchString();
        if (inlineEmojiSearchString != null) {
            mConnection.deleteTextBeforeCursor(inlineEmojiSearchString.length() + 1);
        } else {
            Log.e("inlineEmojiSearch", "Inconsistent state - inlineEmojiSearchString is null");
        }
    }

    private String getInlineEmojiSearchString() {
        if (mEmojiDictionaryFacilitator == null) {
            return null;
        }

        return getInlineEmojiSearchString(mConnection.getTextBeforeCursor(50, 0));
    }

    /**
     * Gets the inline emoji search string. Rules:
     * - string starts with last colon before the cursor
     * - the character before the colon has to be non-word, non-digit
     * - the character after the colon has to be non-space
     * - the string cannot contain newlines
     * <p>
     * Public for testing.
     */
    public static String getInlineEmojiSearchString(CharSequence textBeforeCursor) {
        if (textBeforeCursor == null) {
            return null;
        }

        var text = textBeforeCursor.toString();
        var markerIndex = text.lastIndexOf(INLINE_EMOJI_SEARCH_MARKER);
        if (markerIndex < 0 || text.length() < markerIndex + 2) {
            return null;
        }

        if (markerIndex > 0 && ! isValidInlineEmojiSearchPreviousChar(text.codePointAt(markerIndex - 1), Settings.getValues())) {
            return null;
        }

        if (Character.isWhitespace(text.codePointAt(markerIndex + 1))) {
            return null;
        }

        if (text.indexOf('\n', markerIndex + 2) >= 0) {
            return null;
        }

        return text.substring(markerIndex + 1);
    }

    // public for testing
    public static boolean isStartOfInlineEmojiSearch(int codePoint, int codePointBeforeCursor, int charBeforeBeforeCursor,
                                                      SettingsValues settingsValues) {
        return codePointBeforeCursor == INLINE_EMOJI_SEARCH_MARKER && codePoint != INLINE_EMOJI_SEARCH_MARKER
                && ! Character.isWhitespace(codePoint) && isValidInlineEmojiSearchPreviousChar(charBeforeBeforeCursor, settingsValues);
    }

    private static boolean isValidInlineEmojiSearchPreviousChar(int charBeforeBeforeCursor, SettingsValues settingsValues) {
        return ! Character.isDigit(charBeforeBeforeCursor) && ! settingsValues.isWordCodePoint(charBeforeBeforeCursor);
    }

    public void updateEmojiDictionary(Locale locale) {
        if (Settings.getValues().mInlineEmojiSearch && Settings.getValues().needsToLookupSuggestions() && ! mLatinIME.isEmojiSearch()) {
            if (mEmojiDictionaryFacilitator == null || ! mEmojiDictionaryFacilitator.isForLocale(locale)) {
                closeEmojiDictionary();
                var dictFile = DictionaryInfoUtils.getCachedDictForLocaleAndType(locale, "emoji", mLatinIME);
                var dictionary = dictFile != null? DictionaryFactory.getDictionary(dictFile, locale) : null;
                mEmojiDictionaryFacilitator = dictionary != null? new SingleDictionaryFacilitator(dictionary) : null;
            }
        } else {
            closeEmojiDictionary();
        }
    }

    private void closeEmojiDictionary() {
        if (mEmojiDictionaryFacilitator != null) {
            mEmojiDictionaryFacilitator.closeDictionaries();
            mEmojiDictionaryFacilitator = null;
        }
    }
}
