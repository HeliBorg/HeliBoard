/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import android.view.inputmethod.InputMethodSubtype;

import helium314.keyboard.latin.DictionaryFacilitator;
import helium314.keyboard.latin.RichInputMethodManager;
import helium314.keyboard.latin.settings.SettingsValues;
import helium314.keyboard.latin.suggestions.SuggestedWordInfo;
import helium314.keyboard.latin.suggestions.SuggestedWords;

@SuppressWarnings("unused")
public final class StatsUtils {

    private StatsUtils() {
        // Intentional empty constructor.
    }

    public static void onCreate(final SettingsValues settingsValues,
            RichInputMethodManager richImm) {
    }

    public static void onPickSuggestionManually(final SuggestedWords suggestedWords,
            final SuggestedWordInfo suggestionInfo,
            final DictionaryFacilitator dictionaryFacilitator) {
    }

    public static void onBackspaceWordDelete(int wordLength) {
    }

    public static void onBackspacePressed(int lengthToDelete) {
    }

    public static void onBackspaceSelectedText(int selectedTextLength) {
    }

    public static void onDeleteMultiCharInput(int multiCharLength) {
    }

    public static void onRevertAutoCorrect() {
    }

    public static void onRevertDoubleSpacePeriod() {
    }

    public static void onRevertSwapPunctuation() {
    }

    public static void onFinishInputView() {
    }

    public static void onCreateInputView() {
    }

    public static void onStartInputView(int inputType, int displayOrientation, boolean restarting) {
    }

    public static void onAutoCorrection(final String typedWord, final String autoCorrectionWord,
            final boolean isBatchInput, final DictionaryFacilitator dictionaryFacilitator,
            final String prevWordsContext) {
    }

    public static void onWordCommitUserTyped(final String commitWord, final boolean isBatchMode) {
    }

    public static void onWordCommitAutoCorrect(final String commitWord, final boolean isBatchMode) {
    }

    public static void onWordCommitSuggestionPickedManually(
            final String commitWord, final boolean isBatchMode) {
    }

    public static void onDoubleSpacePeriod() {
    }

    public static void onLoadSettings(SettingsValues settingsValues) {
    }

    public static void onInvalidWordIdentification(final String invalidWord) {
    }

    public static void onSubtypeChanged(final InputMethodSubtype oldSubtype,
            final InputMethodSubtype newSubtype) {
    }

    public static void onSettingsActivity(final String entryPoint) {
    }

    public static void onInputConnectionLaggy(final int operation, final long duration) {
    }

    public static void onDecoderLaggy(final int operation, final long duration) {
    }
}
