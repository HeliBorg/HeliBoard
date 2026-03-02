/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.settings;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.text.InputType;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodSubtype;

import androidx.annotation.NonNull;
import androidx.core.util.TypedValueCompat;

import helium314.keyboard.compat.ConfigurationCompatKt;
import helium314.keyboard.keyboard.KeyboardTheme;
import helium314.keyboard.keyboard.internal.keyboard_parser.LocaleKeyboardInfosKt;
import helium314.keyboard.latin.InputAttributes;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.RichInputMethodManager;
import helium314.keyboard.latin.common.Colors;
import helium314.keyboard.latin.permissions.PermissionsUtil;
import helium314.keyboard.latin.utils.FoldableUtils;
import helium314.keyboard.latin.utils.InputTypeUtils;
import helium314.keyboard.latin.utils.JniUtils;
import helium314.keyboard.latin.utils.ScriptUtils;
import helium314.keyboard.latin.utils.SubtypeSettings;
import helium314.keyboard.latin.utils.SubtypeUtilsKt;
import helium314.keyboard.latin.utils.ToolbarMode;

import java.util.List;
import java.util.Locale;

/**
 * When you call the constructor of this class, you may want to change the current system locale by
 * using {@link helium314.keyboard.latin.utils.RunInLocaleKt}.
 */
// Non-final for testing via mock library.
public class SettingsValues {
    // From resources:
    public final SpacingAndPunctuations mSpacingAndPunctuations;
    public final long mDoubleSpacePeriodTimeout;
    // From configuration:
    public final Locale mLocale;
    public final boolean mHasHardwareKeyboard;
    public final int mDisplayOrientation;
    // From preferences
    public final boolean mAutoCap;
    public final boolean mVibrateOn;
    public final boolean mVibrateInDndMode;
    public final boolean mSoundOn;
    public final boolean mSuggestEmojis;
    public final boolean mInlineEmojiSearch;
    public final boolean mShowEmojiDescriptions;
    public final boolean mKeyPreviewPopupOn;
    public final boolean mShowsVoiceInputKey;
    public final boolean mLanguageSwitchKeyToOtherImes;
    public final boolean mLanguageSwitchKeyToOtherSubtypes;
    private final boolean mShowsLanguageSwitchKey;
    public final boolean mShowsNumberRow;
    public final boolean mShowsNumberRowInSymbols;
    public final boolean mLocalizedNumberRow;
    public final boolean mShowNumberRowHints;
    public final boolean mShowsHints;
    public final boolean mShowsPopupHints;
    public final boolean mShowTldPopupKeys;
    public final boolean mSpaceForLangChange;
    public final boolean mShowsEmojiKey;
    public final boolean mVarToolbarDirection;
    public final boolean mUsePersonalizedDicts;
    public final boolean mUseDoubleSpacePeriod;
    public final boolean mBlockPotentiallyOffensive;
    public final int mSpaceSwipeHorizontal;
    public final int mSpaceSwipeVertical;
    public final int mLanguageSwipeDistance;
    public final boolean mDeleteSwipeEnabled;
    public final boolean mAutospaceAfterPunctuation;
    public final boolean mAutospaceAfterSuggestion;
    public final boolean mAutospaceAfterGestureTyping;
    public final boolean mAutospaceBeforeGestureTyping;
    public final boolean mShiftRemovesAutospace;
    public final boolean mClipboardHistoryEnabled;
    public final long mClipboardHistoryRetentionTime;
    public final boolean mClipboardHistoryPinnedFirst;
    public final boolean mOneHandedModeEnabled;
    public final int mOneHandedModeGravity;
    public final float mOneHandedModeScale;
    public final boolean mNarrowKeyGaps;
    public final String mShowMorePopupKeys;
    public final List<String> mPopupKeyTypes;
    public final List<String> mPopupKeyLabelSources;
    public final List<Locale> mSecondaryLocales;
    public final boolean mBigramPredictionEnabled;// Use bigrams to predict the next word when there is no input for it yet
    public final boolean mSuggestPunctuation;
    public final boolean mCenterSuggestionTextToEnter;
    public final boolean mGestureInputEnabled;
    public final boolean mGestureTrailEnabled;
    public final boolean mGestureFloatingPreviewTextEnabled;
    public final boolean mGestureFloatingPreviewDynamicEnabled;
    public final int mGestureFastTypingCooldown;
    public final int mGestureTrailFadeoutDuration;
    public final boolean mSlidingKeyInputPreviewEnabled;
    public final int mKeyLongpressTimeout;
    public final boolean mEnableEmojiAltPhysicalKey;
    public final boolean mIsSplitKeyboardEnabled;
    public final float mSplitKeyboardSpacerRelativeWidth;
    public final boolean mQuickPinToolbarKeys;
    public final int mScreenMetrics;
    public final boolean mAddToPersonalDictionary;
    public final boolean mUseContactsDictionary;
    public final boolean mUseAppsDictionary;
    public final boolean mCustomNavBarColor;
    public final float mKeyboardHeightScale;
    public final float mBottomRowScale;
    public final boolean mUrlDetectionEnabled;
    public final float mBottomPaddingScale;
    public final float mSidePaddingScale;
    public final ToolbarMode mToolbarMode;
    public final boolean mToolbarHidingGlobal;
    public final boolean mAutoShowToolbar;
    public final boolean mAutoHideToolbar;
    public final boolean mAlphaAfterEmojiInEmojiView;
    public final boolean mAlphaAfterClipHistoryEntry;
    public final boolean mAlphaAfterSymbolAndSpace;
    public final boolean mAlphaAfterNumpadAndSpace;
    public final boolean mRemoveRedundantPopups;
    public final String mSpaceBarText;
    public final float mFontSizeMultiplier;
    public final float mFontSizeMultiplierEmoji;
    public final boolean mEmojiKeyFit;

    // From the input box
    @NonNull
    public final InputAttributes mInputAttributes;

    // Deduced settings
    public final boolean mSuggestionStripHiddenPerUserSettings;
    public final boolean mSecondaryStripVisible;
    public final int mKeypressVibrationDuration;
    public final float mKeypressSoundVolume;
    public final boolean mAutoCorrectionEnabledPerUserSettings;
    public final boolean mAutoCorrectEnabled;
    public final float mAutoCorrectionThreshold;
    public final boolean mBackspaceRevertsAutocorrect;
    public final int mScoreLimitForAutocorrect;
    public final boolean mAutoCorrectShortcuts;
    private final boolean mSuggestionsEnabledPerUserSettings;
    private final boolean mOverrideShowingSuggestions;
    public final boolean mSuggestClipboardContent;
    public final SettingsValuesForSuggestion mSettingsValuesForSuggestion;
    public final boolean mIncognitoModeEnabled;
    public final boolean mLongPressSymbolsForNumpad;

    // User-defined colors
    public final Colors mColors;

    // creation of Colors and SpacingAndPunctuations are the slowest parts in here, but still ok
    public SettingsValues(final Context context, final SharedPreferences prefs, final Resources res,
                          @NonNull final InputAttributes inputAttributes) {
        mLocale = ConfigurationCompatKt.locale(res.getConfiguration());
        mDisplayOrientation = res.getConfiguration().orientation;
        final InputMethodSubtype selectedSubtype = SubtypeSettings.INSTANCE.getSelectedSubtype(prefs);

        // Store the input attributes
        mInputAttributes = inputAttributes;

        // Get the settings preferences
        mToolbarMode = Settings.readToolbarMode(prefs);
        mToolbarHidingGlobal = prefs.getBoolean(Settings.PREF_TOOLBAR_HIDING_GLOBAL, Defaults.PREF_TOOLBAR_HIDING_GLOBAL);
        mAutoCap = prefs.getBoolean(Settings.PREF_AUTO_CAP, Defaults.PREF_AUTO_CAP) && ScriptUtils.scriptSupportsUppercase(mLocale);
        mVibrateOn = Settings.readVibrationEnabled(prefs);
        mVibrateInDndMode = prefs.getBoolean(Settings.PREF_VIBRATE_IN_DND_MODE, Defaults.PREF_VIBRATE_IN_DND_MODE);
        mSoundOn = prefs.getBoolean(Settings.PREF_SOUND_ON, Defaults.PREF_SOUND_ON);
        mSuggestEmojis = prefs.getBoolean(Settings.PREF_SUGGEST_EMOJIS, Defaults.PREF_SUGGEST_EMOJIS);
        mInlineEmojiSearch = prefs.getBoolean(Settings.PREF_INLINE_EMOJI_SEARCH, Defaults.PREF_INLINE_EMOJI_SEARCH);
        mShowEmojiDescriptions = prefs.getBoolean(Settings.PREF_SHOW_EMOJI_DESCRIPTIONS, Defaults.PREF_SHOW_EMOJI_DESCRIPTIONS);
        mKeyPreviewPopupOn = prefs.getBoolean(Settings.PREF_POPUP_ON, Defaults.PREF_POPUP_ON);
        mSlidingKeyInputPreviewEnabled = prefs.getBoolean(
                DebugSettings.PREF_SLIDING_KEY_INPUT_PREVIEW, Defaults.PREF_SLIDING_KEY_INPUT_PREVIEW);
        mShowsVoiceInputKey = mInputAttributes.mShouldShowVoiceInputKey;
        final String languagePref = prefs.getString(Settings.PREF_LANGUAGE_SWITCH_KEY, Defaults.PREF_LANGUAGE_SWITCH_KEY);
        mLanguageSwitchKeyToOtherImes = languagePref.equals("input_method") || languagePref.equals("both");
        mLanguageSwitchKeyToOtherSubtypes = languagePref.equals("internal") || languagePref.equals("both");
        mShowsLanguageSwitchKey = prefs.getBoolean(Settings.PREF_SHOW_LANGUAGE_SWITCH_KEY, Defaults.PREF_SHOW_LANGUAGE_SWITCH_KEY);
        mShowsNumberRow = prefs.getBoolean(Settings.PREF_SHOW_NUMBER_ROW, Defaults.PREF_SHOW_NUMBER_ROW);
        mShowsNumberRowInSymbols = prefs.getBoolean(Settings.PREF_SHOW_NUMBER_ROW_IN_SYMBOLS, Defaults.PREF_SHOW_NUMBER_ROW_IN_SYMBOLS);
        mLocalizedNumberRow = SubtypeUtilsKt.getHasLocalizedNumberRow(selectedSubtype, prefs);
        mShowNumberRowHints = prefs.getBoolean(Settings.PREF_SHOW_NUMBER_ROW_HINTS, Defaults.PREF_SHOW_NUMBER_ROW_HINTS);
        mShowsHints = prefs.getBoolean(Settings.PREF_SHOW_HINTS, Defaults.PREF_SHOW_HINTS);
        mShowsPopupHints = prefs.getBoolean(Settings.PREF_SHOW_POPUP_HINTS, Defaults.PREF_SHOW_POPUP_HINTS);
        mShowTldPopupKeys = prefs.getBoolean(Settings.PREF_SHOW_TLD_POPUP_KEYS, Defaults.PREF_SHOW_TLD_POPUP_KEYS);
        mSpaceForLangChange = prefs.getBoolean(Settings.PREF_SPACE_TO_CHANGE_LANG, Defaults.PREF_SPACE_TO_CHANGE_LANG);
        mShowsEmojiKey = prefs.getBoolean(Settings.PREF_SHOW_EMOJI_KEY, Defaults.PREF_SHOW_EMOJI_KEY);
        mVarToolbarDirection = mToolbarMode != ToolbarMode.HIDDEN && prefs.getBoolean(Settings.PREF_VARIABLE_TOOLBAR_DIRECTION, Defaults.PREF_VARIABLE_TOOLBAR_DIRECTION);
        mUsePersonalizedDicts = prefs.getBoolean(Settings.PREF_KEY_USE_PERSONALIZED_DICTS, Defaults.PREF_KEY_USE_PERSONALIZED_DICTS);
        mUseDoubleSpacePeriod = prefs.getBoolean(Settings.PREF_KEY_USE_DOUBLE_SPACE_PERIOD, Defaults.PREF_KEY_USE_DOUBLE_SPACE_PERIOD)
                && inputAttributes.mIsGeneralTextInput;
        mBlockPotentiallyOffensive = prefs.getBoolean(Settings.PREF_BLOCK_POTENTIALLY_OFFENSIVE, Defaults.PREF_BLOCK_POTENTIALLY_OFFENSIVE);
        mUrlDetectionEnabled = prefs.getBoolean(Settings.PREF_URL_DETECTION, Defaults.PREF_URL_DETECTION);
        mAutoCorrectionEnabledPerUserSettings = prefs.getBoolean(Settings.PREF_AUTO_CORRECTION, Defaults.PREF_AUTO_CORRECTION);
        mAutoCorrectEnabled = mAutoCorrectionEnabledPerUserSettings
                && (mInputAttributes.mInputTypeShouldAutoCorrect || prefs.getBoolean(Settings.PREF_MORE_AUTO_CORRECTION, Defaults.PREF_MORE_AUTO_CORRECTION))
                && (mUrlDetectionEnabled || !InputTypeUtils.isUriOrEmailType(mInputAttributes.mInputType));
        mCenterSuggestionTextToEnter = prefs.getBoolean(Settings.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER, Defaults.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER);
        mAutoCorrectionThreshold = mAutoCorrectEnabled
                ? prefs.getFloat(Settings.PREF_AUTO_CORRECT_THRESHOLD, Defaults.PREF_AUTO_CORRECT_THRESHOLD)
                : Float.MAX_VALUE;
        mScoreLimitForAutocorrect = (mAutoCorrectionThreshold < 0) ? 600000 // very aggressive
                : (mAutoCorrectionThreshold < 0.07 ? 800000 : 950000); // aggressive or modest
        mAutoCorrectShortcuts = prefs.getBoolean(Settings.PREF_AUTOCORRECT_SHORTCUTS, Defaults.PREF_AUTOCORRECT_SHORTCUTS);
        mBackspaceRevertsAutocorrect = prefs.getBoolean(Settings.PREF_BACKSPACE_REVERTS_AUTOCORRECT, Defaults.PREF_BACKSPACE_REVERTS_AUTOCORRECT);
        mBigramPredictionEnabled = prefs.getBoolean(Settings.PREF_BIGRAM_PREDICTIONS, Defaults.PREF_BIGRAM_PREDICTIONS);
        mSuggestPunctuation = prefs.getBoolean(Settings.PREF_SUGGEST_PUNCTUATION, Defaults.PREF_SUGGEST_PUNCTUATION);
        mSuggestClipboardContent = prefs.getBoolean(Settings.PREF_SUGGEST_CLIPBOARD_CONTENT, Defaults.PREF_SUGGEST_CLIPBOARD_CONTENT);
        mDoubleSpacePeriodTimeout = 1100; // ms
        mHasHardwareKeyboard = Settings.readHasHardwareKeyboard(res.getConfiguration());
        final boolean isLandscape = mDisplayOrientation == Configuration.ORIENTATION_LANDSCAPE;
        final float displayWidthDp = TypedValueCompat.pxToDp(res.getDisplayMetrics().widthPixels, res.getDisplayMetrics());
        mIsSplitKeyboardEnabled = Settings.readSplitKeyboardEnabled(prefs, isLandscape);
        // determine spacerWidth from display width and scale setting
        final boolean isFolded = FoldableUtils.isFolded;
        mSplitKeyboardSpacerRelativeWidth = mIsSplitKeyboardEnabled
                ? Math.min(Math.max((displayWidthDp - 600) / 600f + 0.15f, 0.15f), 0.35f) * Settings.readSplitSpacerScale(prefs, isLandscape, isFolded)
                : 0f;
        mQuickPinToolbarKeys = mToolbarMode == ToolbarMode.EXPANDABLE && prefs.getBoolean(Settings.PREF_QUICK_PIN_TOOLBAR_KEYS, Defaults.PREF_QUICK_PIN_TOOLBAR_KEYS);
        mScreenMetrics = Settings.readScreenMetrics(res);

        // Compute other readable settings
        mKeyLongpressTimeout = prefs.getInt(Settings.PREF_KEY_LONGPRESS_TIMEOUT, Defaults.PREF_KEY_LONGPRESS_TIMEOUT);
        mKeypressVibrationDuration = prefs.getInt(Settings.PREF_VIBRATION_DURATION_SETTINGS, Defaults.PREF_VIBRATION_DURATION_SETTINGS);
        mKeypressSoundVolume = prefs.getFloat(Settings.PREF_KEYPRESS_SOUND_VOLUME, Defaults.PREF_KEYPRESS_SOUND_VOLUME);
        mEnableEmojiAltPhysicalKey = prefs.getBoolean(Settings.PREF_ENABLE_EMOJI_ALT_PHYSICAL_KEY, Defaults.PREF_ENABLE_EMOJI_ALT_PHYSICAL_KEY);
        mGestureInputEnabled = JniUtils.sHaveGestureLib && prefs.getBoolean(Settings.PREF_GESTURE_INPUT, Defaults.PREF_GESTURE_INPUT);
        mGestureTrailEnabled = prefs.getBoolean(Settings.PREF_GESTURE_PREVIEW_TRAIL, Defaults.PREF_GESTURE_PREVIEW_TRAIL);
        mGestureFloatingPreviewTextEnabled = !mInputAttributes.mDisableGestureFloatingPreviewText
                && prefs.getBoolean(Settings.PREF_GESTURE_FLOATING_PREVIEW_TEXT, Defaults.PREF_GESTURE_FLOATING_PREVIEW_TEXT);
        mGestureFloatingPreviewDynamicEnabled = Settings.readGestureDynamicPreviewEnabled(prefs);
        mGestureFastTypingCooldown = prefs.getInt(Settings.PREF_GESTURE_FAST_TYPING_COOLDOWN, Defaults.PREF_GESTURE_FAST_TYPING_COOLDOWN);
        mGestureTrailFadeoutDuration = prefs.getInt(Settings.PREF_GESTURE_TRAIL_FADEOUT_DURATION, Defaults.PREF_GESTURE_TRAIL_FADEOUT_DURATION);
        mSuggestionStripHiddenPerUserSettings = mToolbarMode == ToolbarMode.HIDDEN || mToolbarMode == ToolbarMode.TOOLBAR_KEYS;
        mOverrideShowingSuggestions = mInputAttributes.mMayOverrideShowingSuggestions
                && prefs.getBoolean(Settings.PREF_ALWAYS_SHOW_SUGGESTIONS, Defaults.PREF_ALWAYS_SHOW_SUGGESTIONS)
                && ((inputAttributes.mInputType & InputType.TYPE_MASK_VARIATION) != InputType.TYPE_TEXT_VARIATION_WEB_EDIT_TEXT
                  || !prefs.getBoolean(Settings.PREF_ALWAYS_SHOW_SUGGESTIONS_EXCEPT_WEB_TEXT, Defaults.PREF_ALWAYS_SHOW_SUGGESTIONS_EXCEPT_WEB_TEXT));
        final boolean suggestionsEnabled = prefs.getBoolean(Settings.PREF_SHOW_SUGGESTIONS, Defaults.PREF_SHOW_SUGGESTIONS);
        mSuggestionsEnabledPerUserSettings = suggestionsEnabled && (mInputAttributes.mShouldShowSuggestions || mOverrideShowingSuggestions)
                && !mSuggestionStripHiddenPerUserSettings;
        mSecondaryStripVisible = mToolbarMode != ToolbarMode.HIDDEN || ! mToolbarHidingGlobal;
        mIncognitoModeEnabled = prefs.getBoolean(Settings.PREF_ALWAYS_INCOGNITO_MODE, Defaults.PREF_ALWAYS_INCOGNITO_MODE) || mInputAttributes.mNoLearning
                || mInputAttributes.mIsPasswordField;
        mKeyboardHeightScale = Settings.readHeightScale(prefs, isLandscape, isFolded);
        mBottomRowScale = Settings.readBottomRowScale(prefs, isLandscape);
        mSpaceSwipeHorizontal = Settings.readHorizontalSpaceSwipe(prefs);
        mSpaceSwipeVertical = Settings.readVerticalSpaceSwipe(prefs);
        mLanguageSwipeDistance = prefs.getInt(Settings.PREF_LANGUAGE_SWIPE_DISTANCE, Defaults.PREF_LANGUAGE_SWIPE_DISTANCE);
        mDeleteSwipeEnabled = prefs.getBoolean(Settings.PREF_DELETE_SWIPE, Defaults.PREF_DELETE_SWIPE);
        mAutospaceAfterPunctuation = prefs.getBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, Defaults.PREF_AUTOSPACE_AFTER_PUNCTUATION);
        mAutospaceAfterSuggestion = prefs.getBoolean(Settings.PREF_AUTOSPACE_AFTER_SUGGESTION, Defaults.PREF_AUTOSPACE_AFTER_SUGGESTION);
        mAutospaceAfterGestureTyping = prefs.getBoolean(Settings.PREF_AUTOSPACE_AFTER_GESTURE_TYPING, Defaults.PREF_AUTOSPACE_AFTER_GESTURE_TYPING);
        mAutospaceBeforeGestureTyping = prefs.getBoolean(Settings.PREF_AUTOSPACE_BEFORE_GESTURE_TYPING, Defaults.PREF_AUTOSPACE_BEFORE_GESTURE_TYPING);
        mShiftRemovesAutospace = prefs.getBoolean(Settings.PREF_SHIFT_REMOVES_AUTOSPACE, Defaults.PREF_SHIFT_REMOVES_AUTOSPACE);
        mClipboardHistoryEnabled = prefs.getBoolean(Settings.PREF_ENABLE_CLIPBOARD_HISTORY, Defaults.PREF_ENABLE_CLIPBOARD_HISTORY);
        mClipboardHistoryRetentionTime = prefs.getInt(Settings.PREF_CLIPBOARD_HISTORY_RETENTION_TIME, Defaults.PREF_CLIPBOARD_HISTORY_RETENTION_TIME);
        mClipboardHistoryPinnedFirst = prefs.getBoolean(Settings.PREF_CLIPBOARD_HISTORY_PINNED_FIRST, Defaults.PREF_CLIPBOARD_HISTORY_PINNED_FIRST);

        mOneHandedModeEnabled = Settings.readOneHandedModeEnabled(prefs, isLandscape, mIsSplitKeyboardEnabled);
        mOneHandedModeGravity = Settings.readOneHandedModeGravity(prefs, isLandscape, mIsSplitKeyboardEnabled);
        if (mOneHandedModeEnabled) {
            final float baseScale = res.getFraction(R.fraction.config_one_handed_mode_width, 1, 1);
            final float extraScale = Settings.readOneHandedModeScale(prefs, isLandscape, mIsSplitKeyboardEnabled);
            mOneHandedModeScale = 1 - (1 - baseScale) * extraScale;
        } else
            mOneHandedModeScale = 1f;
        mSecondaryLocales = SubtypeUtilsKt.getSecondaryLocales(selectedSubtype.getExtraValue());
        mShowMorePopupKeys = selectedSubtype.isAsciiCapable()
                ? SubtypeUtilsKt.getMoreKeys(selectedSubtype, prefs)
                : LocaleKeyboardInfosKt.POPUP_KEYS_NORMAL;
        mColors = KeyboardTheme.getColorsForCurrentTheme(context);

        mPopupKeyTypes = SubtypeUtilsKt.getPopupKeyTypes(selectedSubtype, prefs);
        mPopupKeyLabelSources = SubtypeUtilsKt.getPopupKeyLabelSources(selectedSubtype, prefs);
        mAddToPersonalDictionary = prefs.getBoolean(Settings.PREF_ADD_TO_PERSONAL_DICTIONARY, Defaults.PREF_ADD_TO_PERSONAL_DICTIONARY);
        mUseContactsDictionary = SettingsValues.readUseContactsEnabled(prefs, context);
        mUseAppsDictionary = prefs.getBoolean(Settings.PREF_USE_APPS, Defaults.PREF_USE_APPS);
        mCustomNavBarColor = prefs.getBoolean(Settings.PREF_NAVBAR_COLOR, Defaults.PREF_NAVBAR_COLOR);
        mNarrowKeyGaps = prefs.getBoolean(Settings.PREF_NARROW_KEY_GAPS, Defaults.PREF_NARROW_KEY_GAPS);
        mSettingsValuesForSuggestion = new SettingsValuesForSuggestion(
                mBlockPotentiallyOffensive,
                prefs.getBoolean(Settings.PREF_GESTURE_SPACE_AWARE, Defaults.PREF_GESTURE_SPACE_AWARE)
        );
        mSpacingAndPunctuations = new SpacingAndPunctuations(res, mUrlDetectionEnabled);
        mBottomPaddingScale = Settings.readBottomPaddingScale(prefs, isLandscape, isFolded);
        mSidePaddingScale = Settings.readSidePaddingScale(prefs, isLandscape, mIsSplitKeyboardEnabled, isFolded);
        mLongPressSymbolsForNumpad = prefs.getBoolean(Settings.PREFS_LONG_PRESS_SYMBOLS_FOR_NUMPAD, Defaults.PREFS_LONG_PRESS_SYMBOLS_FOR_NUMPAD);
        mAutoShowToolbar = mToolbarMode == ToolbarMode.EXPANDABLE && prefs.getBoolean(Settings.PREF_AUTO_SHOW_TOOLBAR, Defaults.PREF_AUTO_SHOW_TOOLBAR);
        mAutoHideToolbar = mSuggestionsEnabledPerUserSettings && prefs.getBoolean(Settings.PREF_AUTO_HIDE_TOOLBAR, Defaults.PREF_AUTO_HIDE_TOOLBAR);
        mAlphaAfterEmojiInEmojiView = prefs.getBoolean(Settings.PREF_ABC_AFTER_EMOJI, Defaults.PREF_ABC_AFTER_EMOJI);
        mAlphaAfterClipHistoryEntry = prefs.getBoolean(Settings.PREF_ABC_AFTER_CLIP, Defaults.PREF_ABC_AFTER_CLIP);
        mAlphaAfterSymbolAndSpace = prefs.getBoolean(Settings.PREF_ABC_AFTER_SYMBOL_SPACE, Defaults.PREF_ABC_AFTER_SYMBOL_SPACE);
        mAlphaAfterNumpadAndSpace = prefs.getBoolean(Settings.PREF_ABC_AFTER_NUMPAD_SPACE, Defaults.PREF_ABC_AFTER_NUMPAD_SPACE);
        mRemoveRedundantPopups = prefs.getBoolean(Settings.PREF_REMOVE_REDUNDANT_POPUPS, Defaults.PREF_REMOVE_REDUNDANT_POPUPS);
        mSpaceBarText = prefs.getString(Settings.PREF_SPACE_BAR_TEXT, Defaults.PREF_SPACE_BAR_TEXT);
        mFontSizeMultiplier = prefs.getFloat(Settings.PREF_FONT_SCALE, Defaults.PREF_FONT_SCALE);
        mFontSizeMultiplierEmoji = prefs.getFloat(Settings.PREF_EMOJI_FONT_SCALE, Defaults.PREF_EMOJI_FONT_SCALE);
        mEmojiKeyFit = prefs.getBoolean(Settings.PREF_EMOJI_KEY_FIT, Defaults.PREF_EMOJI_KEY_FIT);
    }

    public boolean isApplicationSpecifiedCompletionsOn() {
        return mInputAttributes.mApplicationSpecifiedCompletionOn;
    }

    public boolean needsToLookupSuggestions() {
        return (mInputAttributes.mShouldShowSuggestions || mOverrideShowingSuggestions)
                && (mAutoCorrectEnabled || mSuggestionsEnabledPerUserSettings);
    }

    public boolean isSuggestionsEnabledPerUserSettings() {
        return mSuggestionsEnabledPerUserSettings;
    }

    public boolean isWordSeparator(final int code) {
        return mSpacingAndPunctuations.isWordSeparator(code);
    }

    public boolean isWordConnector(final int code) {
        return mSpacingAndPunctuations.isWordConnector(code);
    }

    public boolean isWordCodePoint(final int code) {
        return mSpacingAndPunctuations.isWordCodePoint(code);
    }

    public boolean isUsuallyPrecededBySpace(final int code) {
        return mSpacingAndPunctuations.isUsuallyPrecededBySpace(code);
    }

    public boolean isUsuallyFollowedBySpace(final int code) {
        return mSpacingAndPunctuations.isUsuallyFollowedBySpace(code);
    }

    public boolean shouldInsertSpacesAutomatically() {
        return mInputAttributes.mShouldInsertSpacesAutomatically;
    }

    public boolean isLanguageSwitchKeyEnabled() {
        if (!mShowsLanguageSwitchKey) {
            return false;
        }
        final RichInputMethodManager imm = RichInputMethodManager.getInstance();
        if (!mLanguageSwitchKeyToOtherSubtypes) {
            return imm.hasMultipleEnabledIMEsOrSubtypes(false /* include aux subtypes */);
        }
        if (!mLanguageSwitchKeyToOtherImes) {
            return imm.hasMultipleEnabledSubtypesInThisIme(false /* include aux subtypes */);
        }
        return imm.hasMultipleEnabledSubtypesInThisIme(false /* include aux subtypes */)
            || imm.hasMultipleEnabledIMEsOrSubtypes(false /* include aux subtypes */);
    }

    public boolean isSameInputType(final EditorInfo editorInfo) {
        return mInputAttributes.isSameInputType(editorInfo);
    }

    public boolean hasSameOrientation(final Configuration configuration) {
        return mDisplayOrientation == configuration.orientation;
    }

    private static boolean readUseContactsEnabled(final SharedPreferences prefs, final Context ctx) {
        final boolean setting = prefs.getBoolean(Settings.PREF_USE_CONTACTS, Defaults.PREF_USE_CONTACTS);
        if (!setting) return false;
        if (PermissionsUtil.checkAllPermissionsGranted(ctx, Manifest.permission.READ_CONTACTS)) {
            return true;
        }
        // disable if permission not granted
        prefs.edit().putBoolean(Settings.PREF_USE_CONTACTS, false).apply();
        return false;
    }

    public String dump() {
        final StringBuilder sb = new StringBuilder("Current settings :");
        sb.append("\n   mSpacingAndPunctuations = ");
        sb.append("" + mSpacingAndPunctuations.dump());
        sb.append("\n   mAutoCap = ");
        sb.append("" + mAutoCap);
        sb.append("\n   mVibrateOn = ");
        sb.append("" + mVibrateOn);
        sb.append("\n   mSoundOn = ");
        sb.append("" + mSoundOn);
        sb.append("\n   mKeyPreviewPopupOn = ");
        sb.append("" + mKeyPreviewPopupOn);
        sb.append("\n   mShowsVoiceInputKey = ");
        sb.append("" + mShowsVoiceInputKey);
        sb.append("\n   mLanguageSwitchKeyToOtherImes = ");
        sb.append("" + mLanguageSwitchKeyToOtherImes);
        sb.append("\n   mLanguageSwitchKeyToOtherSubtypes = ");
        sb.append("" + mLanguageSwitchKeyToOtherSubtypes);
        sb.append("\n   mUsePersonalizedDicts = ");
        sb.append("" + mUsePersonalizedDicts);
        sb.append("\n   mUseDoubleSpacePeriod = ");
        sb.append("" + mUseDoubleSpacePeriod);
        sb.append("\n   mBlockPotentiallyOffensive = ");
        sb.append("" + mBlockPotentiallyOffensive);
        sb.append("\n   mBigramPredictionEnabled = ");
        sb.append("" + mBigramPredictionEnabled);
        sb.append("\n   mGestureInputEnabled = ");
        sb.append("" + mGestureInputEnabled);
        sb.append("\n   mGestureTrailEnabled = ");
        sb.append("" + mGestureTrailEnabled);
        sb.append("\n   mGestureFloatingPreviewTextEnabled = ");
        sb.append("" + mGestureFloatingPreviewTextEnabled);
        sb.append("\n   mSlidingKeyInputPreviewEnabled = ");
        sb.append("" + mSlidingKeyInputPreviewEnabled);
        sb.append("\n   mKeyLongpressTimeout = ");
        sb.append("" + mKeyLongpressTimeout);
        sb.append("\n   mLocale = ");
        sb.append("" + mLocale);
        sb.append("\n   mInputAttributes = ");
        sb.append("" + mInputAttributes);
        sb.append("\n   mKeypressVibrationDuration = ");
        sb.append("" + mKeypressVibrationDuration);
        sb.append("\n   mKeypressSoundVolume = ");
        sb.append("" + mKeypressSoundVolume);
        sb.append("\n   mAutoCorrectEnabled = ");
        sb.append("" + mAutoCorrectEnabled);
        sb.append("\n   mAutoCorrectionThreshold = ");
        sb.append("" + mAutoCorrectionThreshold);
        sb.append("\n   mAutoCorrectionEnabledPerUserSettings = ");
        sb.append("" + mAutoCorrectionEnabledPerUserSettings);
        sb.append("\n   mSuggestionsEnabledPerUserSettings = ");
        sb.append("" + mSuggestionsEnabledPerUserSettings);
        sb.append("\n   mDisplayOrientation = ");
        sb.append("" + mDisplayOrientation);
        sb.append("\n   mAppWorkarounds = ");
        return sb.toString();
    }
}
