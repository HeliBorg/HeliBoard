/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.settings;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.ApplicationInfo;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.view.ContextThemeWrapper;
import android.view.inputmethod.EditorInfo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.StringRes;

import helium314.keyboard.compat.ConfigurationCompatKt;
import helium314.keyboard.keyboard.KeyboardActionListener;
import helium314.keyboard.latin.AudioAndHapticFeedbackManager;
import helium314.keyboard.latin.InputAttributes;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.RichInputMethodManager;
import helium314.keyboard.latin.RichInputMethodSubtype;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.utils.DeviceProtectedUtils;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.LayoutType;
import helium314.keyboard.latin.utils.Log;
import helium314.keyboard.latin.utils.ResourceUtils;
import helium314.keyboard.latin.utils.RunInLocaleKt;
import helium314.keyboard.latin.utils.StatsUtils;
import helium314.keyboard.latin.utils.SubtypeSettings;
import helium314.keyboard.latin.utils.ToolbarKey;
import helium314.keyboard.latin.utils.ToolbarUtilsKt;
import helium314.keyboard.latin.utils.ToolbarMode;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.concurrent.locks.ReentrantLock;

public final class Settings implements SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = Settings.class.getSimpleName();

    // theme-related stuff
    public static final String PREF_THEME_STYLE = "theme_style";
    public static final String PREF_ICON_STYLE = "icon_style";
    public static final String PREF_THEME_COLORS = "theme_colors";
    public static final String PREF_THEME_COLORS_NIGHT = "theme_colors_night";
    public static final String PREF_THEME_KEY_BORDERS = "theme_key_borders";
    public static final String PREF_THEME_DAY_NIGHT = "theme_auto_day_night";
    public static final String PREF_USER_COLORS_PREFIX = "user_colors_";
    public static final String PREF_USER_ALL_COLORS_PREFIX = "user_all_colors_";
    public static final String PREF_USER_MORE_COLORS_PREFIX = "user_more_colors_";

    public static final String PREF_CUSTOM_ICON_NAMES = "custom_icon_names";
    public static final String PREF_TOOLBAR_CUSTOM_KEY_CODES = "toolbar_custom_key_codes";
    public static final String PREF_LAYOUT_PREFIX = "layout_";

    public static final String PREF_AUTO_CAP = "auto_cap";
    public static final String PREF_VIBRATE_ON = "vibrate_on";
    public static final String PREF_VIBRATE_IN_DND_MODE = "vibrate_in_dnd_mode";
    public static final String PREF_SOUND_ON = "sound_on";
    public static final String PREF_SUGGEST_EMOJIS = "suggest_emojis";
    public static final String PREF_INLINE_EMOJI_SEARCH = "inline_emoji_search";
    public static final String PREF_SHOW_EMOJI_DESCRIPTIONS = "show_emoji_descriptions";
    public static final String PREF_POPUP_ON = "popup_on";
    public static final String PREF_AUTO_CORRECTION = "auto_correction";
    public static final String PREF_MORE_AUTO_CORRECTION = "more_auto_correction";
    public static final String PREF_AUTO_CORRECT_THRESHOLD = "auto_correct_threshold";
    public static final String PREF_AUTOCORRECT_SHORTCUTS = "autocorrect_shortcuts";
    public static final String PREF_BACKSPACE_REVERTS_AUTOCORRECT = "backspace_reverts_autocorrect";
    public static final String PREF_CENTER_SUGGESTION_TEXT_TO_ENTER = "center_suggestion_text_to_enter";
    public static final String PREF_SHOW_SUGGESTIONS = "show_suggestions";
    public static final String PREF_ALWAYS_SHOW_SUGGESTIONS = "always_show_suggestions";
    public static final String PREF_ALWAYS_SHOW_SUGGESTIONS_EXCEPT_WEB_TEXT = "always_show_suggestions_except_web_text";
    public static final String PREF_KEY_USE_PERSONALIZED_DICTS = "use_personalized_dicts";
    public static final String PREF_KEY_USE_DOUBLE_SPACE_PERIOD = "use_double_space_period";
    public static final String PREF_BLOCK_POTENTIALLY_OFFENSIVE = "block_potentially_offensive";
    public static final String PREF_SHOW_LANGUAGE_SWITCH_KEY = "show_language_switch_key";
    public static final String PREF_LANGUAGE_SWITCH_KEY = "language_switch_key";
    public static final String PREF_SHOW_EMOJI_KEY = "show_emoji_key";
    public static final String PREF_VARIABLE_TOOLBAR_DIRECTION = "var_toolbar_direction";
    public static final String PREF_ADDITIONAL_SUBTYPES = "additional_subtypes";
    public static final String PREF_ENABLE_SPLIT_KEYBOARD = "split_keyboard";
    public static final String PREF_ENABLE_SPLIT_KEYBOARD_LANDSCAPE = "split_keyboard_landscape";
    public static final String PREF_SPLIT_SPACER_SCALE_PREFIX = "split_spacer_scale";
    public static final String PREF_KEYBOARD_HEIGHT_SCALE_PREFIX = "keyboard_height_scale";
    public static final String PREF_BOTTOM_ROW_SCALE_PREFIX = "bottom_row_scale";
    public static final String PREF_BOTTOM_PADDING_SCALE_PREFIX = "bottom_padding_scale";
    public static final String PREF_SIDE_PADDING_SCALE_PREFIX = "side_padding_scale";
    public static final String PREF_FONT_SCALE = "font_scale";
    public static final String PREF_EMOJI_FONT_SCALE = "emoji_font_scale";
    public static final String PREF_EMOJI_KEY_FIT = "emoji_key_fit";
    public static final String PREF_EMOJI_SKIN_TONE = "emoji_skin_tone";
    public static final String PREF_SPACE_HORIZONTAL_SWIPE = "horizontal_space_swipe";
    public static final String PREF_SPACE_VERTICAL_SWIPE = "vertical_space_swipe";
    public static final String PREF_DELETE_SWIPE = "delete_swipe";
    public static final String PREF_AUTOSPACE_AFTER_PUNCTUATION = "autospace_after_punctuation";
    public static final String PREF_AUTOSPACE_AFTER_SUGGESTION = "autospace_after_suggestion";
    public static final String PREF_AUTOSPACE_AFTER_GESTURE_TYPING = "autospace_after_gesture_typing";
    public static final String PREF_AUTOSPACE_BEFORE_GESTURE_TYPING = "autospace_before_gesture_typing";
    public static final String PREF_SHIFT_REMOVES_AUTOSPACE = "shift_removes_autospace";
    public static final String PREF_ALWAYS_INCOGNITO_MODE = "always_incognito_mode";
    public static final String PREF_BIGRAM_PREDICTIONS = "next_word_prediction";
    public static final String PREF_SUGGEST_PUNCTUATION = "suggest_punctuation";
    public static final String PREF_SUGGEST_CLIPBOARD_CONTENT = "suggest_clipboard_content";
    public static final String PREF_GESTURE_INPUT = "gesture_input";
    public static final String PREF_VIBRATION_DURATION_SETTINGS = "vibration_duration_settings";
    public static final String PREF_KEYPRESS_SOUND_VOLUME = "keypress_sound_volume";
    public static final String PREF_KEY_LONGPRESS_TIMEOUT = "key_longpress_timeout";
    public static final String PREF_ENABLE_EMOJI_ALT_PHYSICAL_KEY = "enable_emoji_alt_physical_key";
    public static final String PREF_ENABLE_HARDWARE_KEYBOARD = "enable_hardware_keyboard";
    public static final String PREF_GESTURE_PREVIEW_TRAIL = "gesture_preview_trail";
    public static final String PREF_GESTURE_FLOATING_PREVIEW_TEXT = "gesture_floating_preview_text";
    public static final String PREF_GESTURE_FLOATING_PREVIEW_DYNAMIC = "gesture_floating_preview_dynamic";
    public static final String PREF_GESTURE_DYNAMIC_PREVIEW_FOLLOW_SYSTEM = "gesture_dynamic_preview_follow_system";
    public static final String PREF_GESTURE_SPACE_AWARE = "gesture_space_aware";
    public static final String PREF_GESTURE_FAST_TYPING_COOLDOWN = "gesture_fast_typing_cooldown";
    public static final String PREF_GESTURE_TRAIL_FADEOUT_DURATION = "gesture_trail_fadeout_duration";
    public static final String PREF_SHOW_SETUP_WIZARD_ICON = "show_setup_wizard_icon";
    public static final String PREF_USE_CONTACTS = "use_contacts";
    public static final String PREF_USE_APPS = "use_apps";
    public static final String PREFS_LONG_PRESS_SYMBOLS_FOR_NUMPAD = "long_press_symbols_for_numpad";

    public static final String PREF_ONE_HANDED_MODE_PREFIX = "one_handed_mode_enabled";
    public static final String PREF_ONE_HANDED_GRAVITY_PREFIX = "one_handed_mode_gravity";
    public static final String PREF_ONE_HANDED_SCALE_PREFIX = "one_handed_mode_scale";

    public static final String PREF_SHOW_NUMBER_ROW = "show_number_row";
    public static final String PREF_SHOW_NUMBER_ROW_IN_SYMBOLS = "show_number_row_in_symbols";
    public static final String PREF_LOCALIZED_NUMBER_ROW = "localized_number_row";
    public static final String PREF_SHOW_NUMBER_ROW_HINTS = "show_number_row_hints";
    public static final String PREF_CUSTOM_CURRENCY_KEY = "custom_currency_key";

    public static final String PREF_SHOW_HINTS = "show_hints";
    public static final String PREF_POPUP_KEYS_ORDER = "popup_keys_order";
    public static final String PREF_POPUP_KEYS_LABELS_ORDER = "popup_keys_labels_order";
    public static final String PREF_SHOW_POPUP_HINTS = "show_popup_hints";
    public static final String PREF_MORE_POPUP_KEYS = "more_popup_keys";
    public static final String PREF_SHOW_TLD_POPUP_KEYS = "show_tld_popup_keys";

    public static final String PREF_SPACE_TO_CHANGE_LANG = "prefs_long_press_keyboard_to_change_lang";
    public static final String PREF_LANGUAGE_SWIPE_DISTANCE = "language_swipe_distance";

    public static final String PREF_ENABLE_CLIPBOARD_HISTORY = "enable_clipboard_history";
    public static final String PREF_CLIPBOARD_HISTORY_RETENTION_TIME = "clipboard_history_retention_time";
    public static final String PREF_CLIPBOARD_HISTORY_PINNED_FIRST = "clipboard_history_pinned_first";

    public static final String PREF_ADD_TO_PERSONAL_DICTIONARY = "add_to_personal_dictionary";
    public static final String PREF_NAVBAR_COLOR = "navbar_color";
    public static final String PREF_NARROW_KEY_GAPS = "narrow_key_gaps";
    public static final String PREF_ENABLED_SUBTYPES = "enabled_subtypes";
    public static final String PREF_SELECTED_SUBTYPE = "selected_subtype";
    public static final String PREF_URL_DETECTION = "url_detection";
    public static final String PREF_DONT_SHOW_MISSING_DICTIONARY_DIALOG = "dont_show_missing_dict_dialog";
    public static final String PREF_QUICK_PIN_TOOLBAR_KEYS = "quick_pin_toolbar_keys";
    public static final String PREF_PINNED_TOOLBAR_KEYS = "pinned_toolbar_keys";
    public static final String PREF_TOOLBAR_KEYS = "toolbar_keys";
    public static final String PREF_AUTO_SHOW_TOOLBAR = "auto_show_toolbar";
    public static final String PREF_AUTO_HIDE_TOOLBAR = "auto_hide_toolbar";
    public static final String PREF_CLIPBOARD_TOOLBAR_KEYS = "clipboard_toolbar_keys";
    public static final String PREF_ABC_AFTER_EMOJI = "abc_after_emoji";
    public static final String PREF_ABC_AFTER_CLIP = "abc_after_clip";
    public static final String PREF_ABC_AFTER_SYMBOL_SPACE = "abc_after_symbol_space";
    public static final String PREF_ABC_AFTER_NUMPAD_SPACE = "abc_after_numpad_space";
    public static final String PREF_REMOVE_REDUNDANT_POPUPS = "remove_redundant_popups";
    public static final String PREF_SPACE_BAR_TEXT = "space_bar_text";
    public static final String PREF_TIMESTAMP_FORMAT = "timestamp_format";
    public static final String PREF_TOOLBAR_MODE = "toolbar_mode";
    public static final String PREF_TOOLBAR_HIDING_GLOBAL = "toolbar_hiding_global";
    public static final String PREF_TOOLBAR_SWIPE_DOWN_TO_HIDE = "toolbar_swipe_down_to_hide";

    // Emoji
    public static final String PREF_EMOJI_MAX_SDK = "emoji_max_sdk";
    public static final String PREF_EMOJI_RECENT_KEYS = "emoji_recent_keys";
    public static final String PREF_LAST_SHOWN_EMOJI_CATEGORY_ID = "last_shown_emoji_category_id";
    public static final String PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID = "last_shown_emoji_category_page_id";

    public static final String PREF_VERSION_CODE = "version_code";
    public static final String PREF_LIBRARY_CHECKSUM = "lib_checksum";
    public static final String PREF_SAVE_SUBTYPE_PER_APP = "save_subtype_per_app";
    public static final String PREF_SAVED_APP_SUBTYPE_PREFIX = "saved_app_subtype_";

    private Context mContext;
    private SharedPreferences mPrefs;
    private SettingsValues mSettingsValues;
    private final ReentrantLock mSettingsValuesLock = new ReentrantLock();

    // static cache for background images to avoid potentially slow reload on every settings reload
    private final static Drawable[] sCachedBackgroundImages = new Drawable[4];
    private static Typeface sCachedTypeface;
    private static boolean sCustomTypefaceLoaded; // to avoid repeatedly checking custom typeface file when there is no custom typeface
    private static Typeface sCachedEmojiTypeface;
    private static boolean sCustomEmojiTypefaceLoaded;

    private static final Settings sInstance = new Settings();

    // preferences that are not used in SettingsValues and thus should not trigger reload when changed
    private static final HashSet<String> dontReloadOnChanged = new HashSet<>() {{
        add(PREF_LAST_SHOWN_EMOJI_CATEGORY_PAGE_ID);
        add(PREF_LAST_SHOWN_EMOJI_CATEGORY_ID);
        add(PREF_EMOJI_RECENT_KEYS);
        add(PREF_DONT_SHOW_MISSING_DICTIONARY_DIALOG);
        add(PREF_SELECTED_SUBTYPE);
    }};

    public static Settings getInstance() {
        return sInstance;
    }

    public static SettingsValues getValues() {
        return sInstance.mSettingsValues;
    }

    public static Context getCurrentContext() {
        return sInstance.mContext;
    }

    public static void init(final Context context) {
        sInstance.onCreate(context);
    }

    private Settings() {
        // Intentional empty constructor for singleton.
    }

    private void onCreate(final Context context) {
        mContext = context;
        mPrefs = KtxKt.prefs(context);
        mPrefs.registerOnSharedPreferenceChangeListener(this);
    }

    public void onDestroy() {
        mPrefs.unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(final SharedPreferences prefs, final String key) {
        if (dontReloadOnChanged.contains(key) || (key != null && key.startsWith(PREF_SAVED_APP_SUBTYPE_PREFIX)))
            return;
        mSettingsValuesLock.lock();
        try {
            if (mSettingsValues == null) {
                // TODO: Introduce a static function to register this class and ensure that
                // loadSettings must be called before "onSharedPreferenceChanged" is called.
                Log.w(TAG, "onSharedPreferenceChanged called before loadSettings.");
                return;
            }
            ToolbarUtilsKt.clearCustomToolbarKeyCodes();
            loadSettings(mContext, mSettingsValues.mLocale, mSettingsValues.mInputAttributes);
            StatsUtils.onLoadSettings(mSettingsValues);
        } finally {
            mSettingsValuesLock.unlock();
        }
        if (PREF_ADDITIONAL_SUBTYPES.equals(key)) {
            SubtypeSettings.INSTANCE.reloadEnabledSubtypes(mContext);
        }
    }

    /** convenience function for the rare situations where we need to load settings but may not have a keyboard */
    public void loadSettings(final Context context) {
        if (mSettingsValues != null) return;
        final Locale locale = ConfigurationCompatKt.locale(context.getResources().getConfiguration());
        final InputAttributes inputAttributes = new InputAttributes(new EditorInfo(), false, context.getPackageName());
        loadSettings(context, locale, inputAttributes);
    }

    public void loadSettings(final Context context, final Locale locale,
                             @NonNull final InputAttributes inputAttributes) {
        mSettingsValuesLock.lock();
        mContext = context;
        try {
            final SharedPreferences prefs = mPrefs;
            Log.i(TAG, "loadSettings");
            mSettingsValues = RunInLocaleKt.runInLocale(context, locale,
                    ctx -> new SettingsValues(ctx, prefs, ctx.getResources(), inputAttributes));
        } finally {
            mSettingsValuesLock.unlock();
        }
    }

    public void stopListener() {
        mPrefs.unregisterOnSharedPreferenceChangeListener(this);
    }

    public void startListener() {
        mPrefs.registerOnSharedPreferenceChangeListener(this);
    }

    // TODO: Remove this method and add proxy method to SettingsValues.
    public SettingsValues getCurrent() {
        return mSettingsValues;
    }

    public static int readScreenMetrics(final Resources res) {
        return res.getInteger(R.integer.config_screen_metrics);
    }

    public static boolean readVibrationEnabled(final SharedPreferences prefs) {
        return prefs.getBoolean(PREF_VIBRATE_ON, Defaults.PREF_VIBRATE_ON)
                && AudioAndHapticFeedbackManager.getInstance().hasVibrator();
    }

    public void toggleAutoCorrect() {
        final boolean oldValue = mPrefs.getBoolean(PREF_AUTO_CORRECTION, Defaults.PREF_AUTO_CORRECTION);
        mPrefs.edit().putBoolean(Settings.PREF_AUTO_CORRECTION, !oldValue).apply();
    }

    public static boolean readGestureDynamicPreviewEnabled(final SharedPreferences prefs) {
        final boolean followSystem = prefs.getBoolean(PREF_GESTURE_DYNAMIC_PREVIEW_FOLLOW_SYSTEM, Defaults.PREF_GESTURE_DYNAMIC_PREVIEW_FOLLOW_SYSTEM);
        final boolean defValue = Defaults.PREF_GESTURE_DYNAMIC_PREVIEW_FOLLOW_SYSTEM;
        final boolean curValue = prefs.getBoolean(Settings.PREF_GESTURE_FLOATING_PREVIEW_DYNAMIC, defValue);
        return followSystem ? defValue : curValue;
    }

    public static boolean readGestureDynamicPreviewDefault(final Context context) {
        // if transitions are disabled for the system (reduced motion), moving preview should be disabled
        return SettingsKt.getTransitionAnimationScale(context) != 0.0f;
    }

    public static int readDefaultGestureFastTypingCooldown(final Resources res) {
        return res.getInteger(R.integer.config_gesture_static_time_threshold_after_fast_typing);
    }

    public void toggleAlwaysIncognitoMode() {
        final boolean oldValue = mPrefs.getBoolean(Settings.PREF_ALWAYS_INCOGNITO_MODE, Defaults.PREF_ALWAYS_INCOGNITO_MODE);
        mPrefs.edit().putBoolean(Settings.PREF_ALWAYS_INCOGNITO_MODE, !oldValue).apply();
    }

    public static ToolbarMode readToolbarMode(final SharedPreferences prefs) {
        return ToolbarMode.valueOf(prefs.getString(PREF_TOOLBAR_MODE, Defaults.PREF_TOOLBAR_MODE));
    }

    public static int readHorizontalSpaceSwipe(final SharedPreferences prefs) {
        return switch (prefs.getString(PREF_SPACE_HORIZONTAL_SWIPE, Defaults.PREF_SPACE_HORIZONTAL_SWIPE)) {
            case "move_cursor" -> KeyboardActionListener.SWIPE_MOVE_CURSOR;
            case "switch_language" -> KeyboardActionListener.SWIPE_SWITCH_LANGUAGE;
            case "toggle_numpad" -> KeyboardActionListener.SWIPE_TOGGLE_NUMPAD;
            default -> KeyboardActionListener.SWIPE_NO_ACTION;
        };
    }

    public static int readVerticalSpaceSwipe(final SharedPreferences prefs) {
        return switch (prefs.getString(PREF_SPACE_VERTICAL_SWIPE, Defaults.PREF_SPACE_VERTICAL_SWIPE)) {
            case "move_cursor" -> KeyboardActionListener.SWIPE_MOVE_CURSOR;
            case "switch_language" -> KeyboardActionListener.SWIPE_SWITCH_LANGUAGE;
            case "toggle_numpad" -> KeyboardActionListener.SWIPE_TOGGLE_NUMPAD;
            case "hide_keyboard" -> KeyboardActionListener.SWIPE_HIDE_KEYBOARD;
            default -> KeyboardActionListener.SWIPE_NO_ACTION;
        };
    }

    public static boolean readFullscreenModeAllowed(final Resources res) {
        return res.getBoolean(R.bool.config_fullscreen_mode_allowed);
    }

    public static boolean readShowSetupWizardIcon(final SharedPreferences prefs,
                                                  final Context context) {
        if (!prefs.contains(PREF_SHOW_SETUP_WIZARD_ICON)) {
            final ApplicationInfo appInfo = context.getApplicationInfo();
            final boolean isApplicationInSystemImage =
                    (appInfo.flags & ApplicationInfo.FLAG_SYSTEM) != 0;
            // Default value
            return !isApplicationInSystemImage;
        }
        return prefs.getBoolean(PREF_SHOW_SETUP_WIZARD_ICON, Defaults.PREF_SHOW_SETUP_WIZARD_ICON);
    }

    public static boolean readOneHandedModeEnabled(final SharedPreferences prefs, final boolean landscape, final boolean split) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, split);
        final String key = SettingsKt.createPrefKeyForBooleanSettings(PREF_ONE_HANDED_MODE_PREFIX, index, 2);
        return prefs.getBoolean(key, Defaults.PREF_ONE_HANDED_MODE);
    }

    public void writeOneHandedModeEnabled(final boolean enabled) {
        final boolean landscape = mSettingsValues.mDisplayOrientation == Configuration.ORIENTATION_LANDSCAPE;
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, mSettingsValues.mIsSplitKeyboardEnabled);
        final String key = SettingsKt.createPrefKeyForBooleanSettings(PREF_ONE_HANDED_MODE_PREFIX, index, 2);
        mPrefs.edit().putBoolean(key, enabled).apply();
    }

    public static float readOneHandedModeScale(final SharedPreferences prefs, final boolean landscape, final boolean split) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, split);
        final String key = SettingsKt.createPrefKeyForBooleanSettings(PREF_ONE_HANDED_SCALE_PREFIX, index, 2);
        return prefs.getFloat(key, Defaults.PREF_ONE_HANDED_SCALE);
    }

    public void writeOneHandedModeScale(final Float scale) {
        final boolean landscape = mSettingsValues.mDisplayOrientation == Configuration.ORIENTATION_LANDSCAPE;
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, mSettingsValues.mIsSplitKeyboardEnabled);
        final String key = SettingsKt.createPrefKeyForBooleanSettings(PREF_ONE_HANDED_SCALE_PREFIX, index, 2);
        mPrefs.edit().putFloat(key, scale).apply();
    }

    public static int readOneHandedModeGravity(final SharedPreferences prefs, final boolean landscape, final boolean split) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, split);
        final String key = SettingsKt.createPrefKeyForBooleanSettings(PREF_ONE_HANDED_GRAVITY_PREFIX, index, 2);
        return prefs.getInt(key, Defaults.PREF_ONE_HANDED_GRAVITY);
    }

    public void writeOneHandedModeGravity(final int gravity) {
        final boolean landscape = mSettingsValues.mDisplayOrientation == Configuration.ORIENTATION_LANDSCAPE;
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, mSettingsValues.mIsSplitKeyboardEnabled);
        final String key = SettingsKt.createPrefKeyForBooleanSettings(PREF_ONE_HANDED_GRAVITY_PREFIX, index, 2);
        mPrefs.edit().putInt(key, gravity).apply();
    }

    public void writeSplitKeyboardEnabled(final boolean enabled, final boolean isLandscape) {
        final String pref = isLandscape ? PREF_ENABLE_SPLIT_KEYBOARD_LANDSCAPE : PREF_ENABLE_SPLIT_KEYBOARD;
        mPrefs.edit().putBoolean(pref, enabled).apply();
    }

    public static boolean readSplitKeyboardEnabled(final SharedPreferences prefs, final boolean isLandscape) {
        final String pref = isLandscape ? PREF_ENABLE_SPLIT_KEYBOARD_LANDSCAPE : PREF_ENABLE_SPLIT_KEYBOARD;
        return prefs.getBoolean(pref, isLandscape ? Defaults.PREF_ENABLE_SPLIT_KEYBOARD_LANDSCAPE : Defaults.PREF_ENABLE_SPLIT_KEYBOARD);
    }

    public static float readSplitSpacerScale(final SharedPreferences prefs, final boolean landscape) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape);
        final Float[] defaults = Defaults.PREF_SPLIT_SPACER_SCALE;
        final float defaultValue = defaults[index];
        return prefs.getFloat(SettingsKt.createPrefKeyForBooleanSettings(PREF_SPLIT_SPACER_SCALE_PREFIX, index, 1), defaultValue);
    }

    public static float readBottomPaddingScale(final SharedPreferences prefs, final boolean landscape) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape);
        final Float[] defaults = Defaults.PREF_BOTTOM_PADDING_SCALE;
        final float defaultValue = defaults[index];
        return prefs.getFloat(SettingsKt.createPrefKeyForBooleanSettings(PREF_BOTTOM_PADDING_SCALE_PREFIX, index, 1), defaultValue);
    }

    public static float readSidePaddingScale(final SharedPreferences prefs, final boolean landscape, final boolean split) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape, split);
        final Float[] defaults = Defaults.PREF_SIDE_PADDING_SCALE;
        final float defaultValue = defaults[index];
        return prefs.getFloat(SettingsKt.createPrefKeyForBooleanSettings(PREF_SIDE_PADDING_SCALE_PREFIX, index, 2), defaultValue);
    }

    public static float readHeightScale(final SharedPreferences prefs, final boolean landscape) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape);
        final Float[] defaults = Defaults.PREF_KEYBOARD_HEIGHT_SCALE;
        final float defaultValue = defaults[index];
        return prefs.getFloat(SettingsKt.createPrefKeyForBooleanSettings(PREF_KEYBOARD_HEIGHT_SCALE_PREFIX, index, 1), defaultValue);
    }

    public static float readBottomRowScale(final SharedPreferences prefs, final boolean landscape) {
        final int index = SettingsKt.findIndexOfDefaultSetting(landscape);
        final Float[] defaults = Defaults.PREF_BOTTOM_ROW_SCALE;
        final float defaultValue = defaults[index];
        return prefs.getFloat(SettingsKt.createPrefKeyForBooleanSettings(PREF_BOTTOM_ROW_SCALE_PREFIX, index, 1), defaultValue);
    }

    public static boolean readHasHardwareKeyboard(final Configuration conf) {
        // The standard way of finding out whether we have a hardware keyboard. This code is taken
        // from InputMethodService#onEvaluateInputShown, which canonically determines this.
        // In a nutshell, we have a keyboard if the configuration says the type of hardware keyboard
        // is NOKEYS and if it's not hidden (e.g. folded inside the device).
        return conf.keyboard != Configuration.KEYBOARD_NOKEYS
                && conf.hardKeyboardHidden != Configuration.HARDKEYBOARDHIDDEN_YES;
    }

    @Nullable public static Drawable readUserBackgroundImage(final Context context, final boolean night) {
        final boolean landscape = context.getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        final int index = (night ? 1 : 0) + (landscape ? 2 : 0);
        if (sCachedBackgroundImages[index] != null) return sCachedBackgroundImages[index];

        File image = getCustomBackgroundFile(context, night, landscape);
        if (!image.isFile() && landscape)
            image = getCustomBackgroundFile(context, night, false); // fall back to portrait image for historic reasons
        if (!image.isFile()) return null;
        try {
            sCachedBackgroundImages[index] = new BitmapDrawable(context.getResources(), BitmapFactory.decodeFile(image.getAbsolutePath()));
            return sCachedBackgroundImages[index];
        } catch (Exception e) {
            return null;
        }
    }

    public static File getCustomBackgroundFile(final Context context, final boolean night, final boolean landscape) {
        return new File(DeviceProtectedUtils.getFilesDir(context), "custom_background_image" + (landscape ? "_landscape" : "") + (night ? "_night" : ""));
    }

    public static void clearCachedBackgroundImages() {
        Arrays.fill(sCachedBackgroundImages, null);
    }

    public static Context getDayNightContext(final Context context, final boolean wantNight) {
        final boolean isNight = ResourceUtils.isNight(context.getResources());
        if (isNight == wantNight)
            return context;
        final Configuration config = new Configuration(context.getResources().getConfiguration());
        final int night = config.uiMode & Configuration.UI_MODE_NIGHT_MASK;
        final int uiModeWithNightBitsZero = config.uiMode - night;
        config.uiMode = uiModeWithNightBitsZero + (wantNight ? Configuration.UI_MODE_NIGHT_YES : Configuration.UI_MODE_NIGHT_NO);
        final ContextThemeWrapper wrapper = new ContextThemeWrapper(context, R.style.platformActivityTheme);
        wrapper.applyOverrideConfiguration(config);
        return wrapper;
    }

    public boolean isTablet() {
        return mContext.getResources().getInteger(R.integer.config_screen_metrics) >= 3;
    }

    @SuppressLint("DiscouragedApi")
    public int getStringResIdByName(final String name) {
        return mContext.getResources().getIdentifier(name, "string", mContext.getPackageName());
    }

    public String getInLocale(@StringRes final int resId, final Locale locale) {
        return RunInLocaleKt.runInLocale(mContext, locale, (ctx) -> ctx.getString(resId));
    }

    public String readCustomCurrencyKey() {
        return mPrefs.getString(PREF_CUSTOM_CURRENCY_KEY, Defaults.PREF_CUSTOM_CURRENCY_KEY);
    }

    public Integer getCustomToolbarKeyCode(ToolbarKey key) {
        return ToolbarUtilsKt.getCustomKeyCode(key, mPrefs);
    }

    public Integer getCustomToolbarLongpressCode(ToolbarKey key) {
        return ToolbarUtilsKt.getCustomLongpressKeyCode(key, mPrefs);
    }

    public static File getCustomFontFile(final Context context) {
        return new File(DeviceProtectedUtils.getFilesDir(context), "custom_font");
    }

    public static File getCustomEmojiFontFile(final Context context) {
        return new File(DeviceProtectedUtils.getFilesDir(context), "custom_emoji_font");
    }

    // "default" layout as in this is used if nothing else is specified in the subtype
    public static String readDefaultLayoutName(final LayoutType type, final SharedPreferences prefs) {
        return prefs.getString(PREF_LAYOUT_PREFIX + type.name(), Defaults.INSTANCE.getDefault(type));
    }

    public static void writeDefaultLayoutName(@Nullable final String name, final LayoutType type, final SharedPreferences prefs) {
        if (name == null) prefs.edit().remove(PREF_LAYOUT_PREFIX + type.name()).apply();
        else prefs.edit().putString(PREF_LAYOUT_PREFIX + type.name(), name).apply();
    }

    public void saveSubtypeForApp(RichInputMethodSubtype subtype, String packageName) {
        if (isSubtypePerApp() && ! StringUtils.isEmpty(packageName)) {
            mPrefs.edit().putString(PREF_SAVED_APP_SUBTYPE_PREFIX + packageName,
                                    SettingsSubtype.Companion.toSettingsSubtype(subtype.getRawSubtype()).toPref()).apply();
        }
    }

    public RichInputMethodSubtype getSubtypeForApp(String packageName) {
        if (! isSubtypePerApp() || StringUtils.isEmpty(packageName)) return null;
        var subtypePref = mPrefs.getString(PREF_SAVED_APP_SUBTYPE_PREFIX + packageName, null);
        if (subtypePref == null) return null;
        var settingsSubtype = SettingsSubtype.Companion.toSettingsSubtype(subtypePref);
        var subtype = settingsSubtype.toEnabledSubtype();
        if (subtype == null) subtype = RichInputMethodManager.getInstance().findSubtypeForHintLocale(settingsSubtype.getLocale());
        return subtype != null? RichInputMethodSubtype.Companion.get(subtype) : null;
    }

    private boolean isSubtypePerApp() {
        return mPrefs.getBoolean(PREF_SAVE_SUBTYPE_PER_APP, Defaults.PREF_SAVE_SUBTYPE_PER_APP);
    }

    @Nullable
    public Typeface getCustomTypeface() {
        if (!sCustomTypefaceLoaded) {
            try {
                sCachedTypeface = Typeface.createFromFile(getCustomFontFile(mContext));
            } catch (Exception ignored) { }
        }
        sCustomTypefaceLoaded = true;
        return sCachedTypeface;
    }

    @Nullable
    public Typeface getCustomEmojiTypeface() {
        if (!sCustomEmojiTypefaceLoaded) {
            try {
                sCachedEmojiTypeface = Typeface.createFromFile(getCustomEmojiFontFile(mContext));
            } catch (Exception ignored) { }
        }
        sCustomEmojiTypefaceLoaded = true;
        return sCachedEmojiTypeface;
    }

    public static void clearCachedTypeface() {
        sCachedTypeface = null;
        sCustomTypefaceLoaded = false;
        sCachedEmojiTypeface = null;
        sCustomEmojiTypefaceLoaded = false;
    }
}
