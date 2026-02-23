/*
 * Copyright (C) 2010 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.graphics.Rect;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.text.TextUtils;

import helium314.keyboard.keyboard.internal.KeyDrawParams;
import helium314.keyboard.keyboard.internal.KeySpecParser;
import helium314.keyboard.keyboard.internal.KeyVisualAttributes;
import helium314.keyboard.keyboard.internal.KeyboardIconsSet;
import helium314.keyboard.keyboard.internal.KeyboardParams;
import helium314.keyboard.keyboard.internal.PopupKeySpec;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.PopupSet;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.StringUtils;
import helium314.keyboard.latin.utils.PopupKeysUtilsKt;
import helium314.keyboard.latin.utils.ToolbarKey;
import helium314.keyboard.latin.utils.ToolbarUtilsKt;
import kotlin.collections.ArraysKt;

import java.util.Arrays;
import java.util.Locale;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Class for describing the position and characteristics of a single key in the keyboard.
 */
public class Key implements Comparable<Key> {
    /**
     * The key code (unicode or custom code) that this key generates.
     */
    private final int mCode;

    /** Label to display */
    private final String mLabel;
    /** Hint label to display on the key in conjunction with the label */
    private final String mHintLabel;
    /** Flags of the label */
    private final int mLabelFlags;
    public static final int LABEL_FLAGS_ALIGN_HINT_LABEL_TO_BOTTOM = 0x02;
    public static final int LABEL_FLAGS_ALIGN_ICON_TO_BOTTOM = 0x04;
    public static final int LABEL_FLAGS_ALIGN_LABEL_OFF_CENTER = 0x08;
    // Font typeface specification.
    private static final int LABEL_FLAGS_FONT_MASK = 0x30;
    public static final int LABEL_FLAGS_FONT_NORMAL = 0x10;
    public static final int LABEL_FLAGS_FONT_MONO_SPACE = 0x20;
    public static final int LABEL_FLAGS_FONT_DEFAULT = 0x30;
    // Start of key text ratio enum values
    private static final int LABEL_FLAGS_FOLLOW_KEY_TEXT_RATIO_MASK = 0x1C0;
    public static final int LABEL_FLAGS_FOLLOW_KEY_LARGE_LETTER_RATIO = 0x40;
    public static final int LABEL_FLAGS_FOLLOW_KEY_LETTER_RATIO = 0x80;
    public static final int LABEL_FLAGS_FOLLOW_KEY_LABEL_RATIO = 0xC0;
    public static final int LABEL_FLAGS_FOLLOW_KEY_HINT_LABEL_RATIO = 0x140;
    // End of key text ratio mask enum values
    public static final int LABEL_FLAGS_HAS_POPUP_HINT = 0x200;
    public static final int LABEL_FLAGS_HAS_SHIFTED_LETTER_HINT = 0x400;
    public static final int LABEL_FLAGS_HAS_HINT_LABEL = 0x800;
    // The bit to calculate the ratio of key label width against key width. If autoXScale bit is on
    // and autoYScale bit is off, the key label may be shrunk only for X-direction.
    // If both autoXScale and autoYScale bits are on, the key label text size may be auto scaled.
    public static final int LABEL_FLAGS_AUTO_X_SCALE = 0x4000;
    public static final int LABEL_FLAGS_AUTO_Y_SCALE = 0x8000;
    public static final int LABEL_FLAGS_AUTO_SCALE = LABEL_FLAGS_AUTO_X_SCALE
            | LABEL_FLAGS_AUTO_Y_SCALE;
    public static final int LABEL_FLAGS_PRESERVE_CASE = 0x10000;
    public static final int LABEL_FLAGS_SHIFTED_LETTER_ACTIVATED = 0x20000;
    public static final int LABEL_FLAGS_FROM_CUSTOM_ACTION_LABEL = 0x40000;
    public static final int LABEL_FLAGS_FOLLOW_FUNCTIONAL_TEXT_COLOR = 0x80000;
    public static final int LABEL_FLAGS_KEEP_BACKGROUND_ASPECT_RATIO = 0x100000;
    public static final int LABEL_FLAGS_DISABLE_HINT_LABEL = 0x40000000;
    public static final int LABEL_FLAGS_DISABLE_ADDITIONAL_POPUP_KEYS = 0x80000000;

    /** Icon to display instead of a label. Icon takes precedence over a label */
    @Nullable private final String mIconName;

    /** Width of the key, excluding the gap */
    private final int mWidth;
    /** Height of the key, excluding the gap */
    private final int mHeight;
    /**
     * The combined width in pixels of the horizontal gaps belonging to this key, both to the left
     * and to the right. I.e., mWidth + mHorizontalGap = total width belonging to the key.
     */
    private final int mHorizontalGap;
    /**
     * The combined height in pixels of the vertical gaps belonging to this key, both above and
     * below. I.e., mHeight + mVerticalGap = total height belonging to the key.
     */
    private final int mVerticalGap;
    /** X coordinate of the top-left corner of the key in the keyboard layout, excluding the gap. */
    private final int mX;
    /** Y coordinate of the top-left corner of the key in the keyboard layout, excluding the gap. */
    private final int mY;
    /** Hit bounding box of the key */
    @NonNull
    private final Rect mHitBox = new Rect();

    /** Popup keys. It is guaranteed that this is null or an array of one or more elements */
    @Nullable
    private final PopupKeySpec[] mPopupKeys;
    /** Popup keys column number and flags */
    private final int mPopupKeysColumnAndFlags;
    private static final int POPUP_KEYS_COLUMN_NUMBER_MASK = 0x000000ff;
    // If this flag is specified, popup keys keyboard should have the specified number of columns.
    // Otherwise popup keys keyboard should have less than or equal to the specified maximum number
    // of columns.
    private static final int POPUP_KEYS_FLAGS_FIXED_COLUMN = 0x00000100;
    // If this flag is specified, the order of popup keys is determined by the order in the popup
    // keys' specification. Otherwise the order of popup keys is automatically determined.
    private static final int POPUP_KEYS_FLAGS_FIXED_ORDER = 0x00000200;
    private static final int POPUP_KEYS_MODE_MAX_COLUMN_WITH_AUTO_ORDER = 0;
    private static final int POPUP_KEYS_MODE_FIXED_COLUMN_WITH_AUTO_ORDER = POPUP_KEYS_FLAGS_FIXED_COLUMN;
    private static final int POPUP_KEYS_MODE_FIXED_COLUMN_WITH_FIXED_ORDER = (POPUP_KEYS_FLAGS_FIXED_COLUMN | POPUP_KEYS_FLAGS_FIXED_ORDER);
    private static final int POPUP_KEYS_FLAGS_HAS_LABELS = 0x40000000;
    private static final int POPUP_KEYS_FLAGS_NEEDS_DIVIDERS = 0x20000000;
    private static final int POPUP_KEYS_FLAGS_NO_PANEL_AUTO_POPUP_KEY = 0x10000000;
    // TODO: Rename these specifiers to !autoOrder! and !fixedOrder! respectively.
    public static final String POPUP_KEYS_AUTO_COLUMN_ORDER = "!autoColumnOrder!";
    public static final String POPUP_KEYS_FIXED_COLUMN_ORDER = "!fixedColumnOrder!";
    public static final String POPUP_KEYS_HAS_LABELS = "!hasLabels!";
    private static final String POPUP_KEYS_NEEDS_DIVIDERS = "!needsDividers!";
    private static final String POPUP_KEYS_NO_PANEL_AUTO_POPUP_KEY = "!noPanelAutoPopupKey!";

    /** Background type that represents different key background visual than normal one. */
    private final int mBackgroundType;
    public static final int BACKGROUND_TYPE_EMPTY = 0;
    public static final int BACKGROUND_TYPE_NORMAL = 1;
    public static final int BACKGROUND_TYPE_FUNCTIONAL = 2;
    public static final int BACKGROUND_TYPE_ACTION = 3;
    public static final int BACKGROUND_TYPE_SPACEBAR = 4;

    private final int mActionFlags;
    private static final int ACTION_FLAGS_IS_REPEATABLE = 0x01;
    private static final int ACTION_FLAGS_NO_KEY_PREVIEW = 0x02;
    private static final int ACTION_FLAGS_ALT_CODE_WHILE_TYPING = 0x04;
    private static final int ACTION_FLAGS_ENABLE_LONG_PRESS = 0x08;

    @Nullable
    private final KeyVisualAttributes mKeyVisualAttributes;
    @Nullable
    private final OptionalAttributes mOptionalAttributes;

    private static final class OptionalAttributes {
        /** Text to output when pressed. This can be multiple characters, like ".com" */
        public final String mOutputText;
        public final int mAltCode;
        /** Icon for disabled state */
        @Nullable public final String mDisabledIconName;
        /** The visual insets */
        public final int mVisualInsetsLeft;
        public final int mVisualInsetsRight;

        private OptionalAttributes(final String outputText, final int altCode, @Nullable final String disabledIconName,
                                   final int visualInsetsLeft, final int visualInsetsRight) {
            mOutputText = outputText;
            mAltCode = altCode;
            mDisabledIconName = disabledIconName;
            mVisualInsetsLeft = visualInsetsLeft;
            mVisualInsetsRight = visualInsetsRight;
        }

        @Nullable
        public static OptionalAttributes newInstance(final String outputText, final int altCode,
                 @Nullable final String disabledIconName, final int visualInsetsLeft, final int visualInsetsRight) {
            if (outputText == null && altCode == KeyCode.NOT_SPECIFIED
                    && disabledIconName == null && visualInsetsLeft == 0
                    && visualInsetsRight == 0) {
                return null;
            }
            return new OptionalAttributes(outputText, altCode, disabledIconName, visualInsetsLeft,
                    visualInsetsRight);
        }
    }

    private final int mHashCode;

    /** The current pressed state of this key */
    private boolean mPressed;
    /** Key is enabled and responds on press */
    private boolean mEnabled;
    /** Key is locked (appears permanently pressed) */
    private boolean mLocked = false;
    /**
     * Constructor for a key on <code>PopupKeyKeyboard</code> and on <code>MoreSuggestions</code>.
     */
    public Key(@Nullable final String label, @Nullable final String iconName, final int code,
            @Nullable final String outputText, @Nullable final String hintLabel,
            final int labelFlags, final int backgroundType, final int x, final int y,
            final int width, final int height, final int horizontalGap, final int verticalGap) {
        mWidth = width - horizontalGap;
        mHeight = height - verticalGap;
        mHorizontalGap = horizontalGap;
        mVerticalGap = verticalGap;
        mHintLabel = hintLabel;
        mLabelFlags = labelFlags;
        mBackgroundType = backgroundType;
        // TODO: Pass keyActionFlags as an argument.
        mActionFlags = ACTION_FLAGS_NO_KEY_PREVIEW;
        mPopupKeys = null;
        mPopupKeysColumnAndFlags = 0;
        mLabel = label;
        mCode = code;
        mEnabled = (code != KeyCode.NOT_SPECIFIED);
        mIconName = iconName;
        mOptionalAttributes = OptionalAttributes.newInstance(outputText, KeyCode.NOT_SPECIFIED,
                mIconName == null ? null : getDisabledIconName(mIconName), 0, 0);
        // Horizontal gap is divided equally to both sides of the key.
        mX = x + mHorizontalGap / 2;
        mY = y;
        mHitBox.set(x, y, x + width + 1, y + height);
        mKeyVisualAttributes = null;

        mHashCode = computeHashCode(this);
    }

    /**
     * Copy constructor for DynamicGridKeyboard.GridKey.
     *
     * @param key the original key.
     * @param popupKeys the popup keys that should be assigned to this key.
     * @param labelHint the label hint that should be assigned to this key.
     * @param backgroundType the background type that should be assigned to this key.
     */
    protected Key(@NonNull final Key key, @Nullable final PopupKeySpec[] popupKeys,
                @Nullable final String labelHint, final int backgroundType) {
        // Final attributes.
        mCode = key.mCode;
        mLabel = key.mLabel;
        mHintLabel = labelHint;
        mLabelFlags = key.mLabelFlags;
        mIconName = key.mIconName;
        mWidth = key.mWidth;
        mHeight = key.mHeight;
        mHorizontalGap = key.mHorizontalGap;
        mVerticalGap = key.mVerticalGap;
        mX = key.mX;
        mY = key.mY;
        mHitBox.set(key.mHitBox);
        mPopupKeys = popupKeys;
        mPopupKeysColumnAndFlags = key.mPopupKeysColumnAndFlags;
        mBackgroundType = backgroundType;
        mActionFlags = key.mActionFlags;
        mKeyVisualAttributes = key.mKeyVisualAttributes;
        mOptionalAttributes = key.mOptionalAttributes;
        mHashCode = key.mHashCode;
        // Key state.
        mPressed = key.mPressed;
        mEnabled = key.mEnabled;
    }

    /** constructor for creating emoji recent keys when there is no keyboard to take keys from */
    public Key(@NonNull final Key key, @Nullable final PopupKeySpec[] popupKeys,
             @Nullable final String labelHint, final int backgroundType, final int code, @Nullable final String outputText) {
        // Final attributes.
        mCode = outputText == null ? code : KeyCode.MULTIPLE_CODE_POINTS;
        mLabel = outputText == null ? StringUtils.newSingleCodePointString(code) : outputText;
        mHintLabel = labelHint;
        mLabelFlags = key.mLabelFlags;
        mIconName = key.mIconName;
        mWidth = key.mWidth;
        mHeight = key.mHeight;
        mHorizontalGap = key.mHorizontalGap;
        mVerticalGap = key.mVerticalGap;
        mX = key.mX;
        mY = key.mY;
        mHitBox.set(key.mHitBox);
        mPopupKeys = popupKeys;
        mPopupKeysColumnAndFlags = key.mPopupKeysColumnAndFlags;
        mBackgroundType = backgroundType;
        mActionFlags = key.mActionFlags;
        mKeyVisualAttributes = key.mKeyVisualAttributes;
        mOptionalAttributes = outputText == null ? null
                : Key.OptionalAttributes.newInstance(outputText, KeyCode.NOT_SPECIFIED, null, 0, 0);
        mHashCode = key.mHashCode;
        // Key state.
        mPressed = key.mPressed;
        mEnabled = key.mEnabled;
    }

    /** constructor from KeyParams */
    private Key(KeyParams keyParams) {
        // stuff to copy
        mCode = keyParams.mCode;
        mLabel = keyParams.mLabel;
        mHintLabel = keyParams.mHintLabel;
        mLabelFlags = keyParams.mLabelFlags;
        mIconName = keyParams.mIconName;
        mPopupKeys = keyParams.mPopupKeys;
        mPopupKeysColumnAndFlags = keyParams.mPopupKeysColumnAndFlags;
        mBackgroundType = keyParams.mBackgroundType;
        mActionFlags = keyParams.mActionFlags;
        mKeyVisualAttributes = keyParams.mKeyVisualAttributes;
        mOptionalAttributes = keyParams.mOptionalAttributes;
        mEnabled = keyParams.mEnabled;

        // stuff to create

        // get the "correct" float gap: may shift keys by one pixel, but results in more uniform gaps between keys
        final float horizontalGapFloat = isSpacer() ? 0 : (keyParams.mKeyboardParams.mRelativeHorizontalGap * keyParams.mKeyboardParams.mOccupiedWidth);
        mHorizontalGap = Math.round(horizontalGapFloat);
        mVerticalGap = Math.round(keyParams.mKeyboardParams.mRelativeVerticalGap * keyParams.mKeyboardParams.mOccupiedHeight);
        mWidth = Math.round(keyParams.mAbsoluteWidth - horizontalGapFloat);
        // height is always rounded down, because rounding up may make the keyboard too high to fit, leading to issues
        mHeight = (int) (keyParams.mAbsoluteHeight - keyParams.mKeyboardParams.mVerticalGap);
        if (!isSpacer() && (mWidth == 0 || mHeight == 0)) {
            throw new IllegalStateException("key needs positive width and height");
        }
        // Horizontal gap is divided equally to both sides of the key.
        mX = Math.round(keyParams.xPos + horizontalGapFloat / 2);
        mY = Math.round(keyParams.yPos);
        mHitBox.set(Math.round(keyParams.xPos), Math.round(keyParams.yPos), Math.round(keyParams.xPos + keyParams.mAbsoluteWidth) + 1,
                Math.round(keyParams.yPos + keyParams.mAbsoluteHeight));
        mHashCode = computeHashCode(this);
    }

    private Key(@NonNull final Key key, @Nullable final PopupKeySpec[] popupKeys) {
        // Final attributes.
        mCode = key.mCode;
        mLabel = key.mLabel;
        mHintLabel = PopopUtilKt.findPopupHintLabel(popupKeys, key.mHintLabel);
        mLabelFlags = key.mLabelFlags;
        mIconName = key.mIconName;
        mWidth = key.mWidth;
        mHeight = key.mHeight;
        mHorizontalGap = key.mHorizontalGap;
        mVerticalGap = key.mVerticalGap;
        mX = key.mX;
        mY = key.mY;
        mHitBox.set(key.mHitBox);
        mPopupKeys = popupKeys;
        mPopupKeysColumnAndFlags = key.mPopupKeysColumnAndFlags;
        mBackgroundType = key.mBackgroundType;
        if (popupKeys == null && mCode > Constants.CODE_SPACE && (key.mActionFlags & ACTION_FLAGS_ENABLE_LONG_PRESS) != 0)
            mActionFlags = key.mActionFlags - ACTION_FLAGS_ENABLE_LONG_PRESS;
        else
            mActionFlags = key.mActionFlags;
        mKeyVisualAttributes = key.mKeyVisualAttributes;
        mOptionalAttributes = key.mOptionalAttributes;
        mHashCode = key.mHashCode;
        // Key state.
        mPressed = key.mPressed;
        mEnabled = key.mEnabled;
    }

    @NonNull
    public static Key removeRedundantPopupKeys(@NonNull final Key key,
            @NonNull final PopupKeySpec.LettersOnBaseLayout lettersOnBaseLayout) {
        if ((key.mPopupKeysColumnAndFlags & POPUP_KEYS_FLAGS_FIXED_COLUMN) != 0)
            return key; // don't remove anything for fixed column popup keys
        final PopupKeySpec[] popupKeys = key.getPopupKeys();
        final PopupKeySpec[] filteredPopupKeys = PopupKeySpec.removeRedundantPopupKeys(
                popupKeys, lettersOnBaseLayout);
        return (filteredPopupKeys == popupKeys) ? key : new Key(key, filteredPopupKeys);
    }

    private static boolean needsToUpcase(final int labelFlags, final int keyboardElementId) {
        if ((labelFlags & LABEL_FLAGS_PRESERVE_CASE) != 0) return false;
        return switch (keyboardElementId) {
            case KeyboardId.ELEMENT_ALPHABET_MANUAL_SHIFTED, KeyboardId.ELEMENT_ALPHABET_AUTOMATIC_SHIFTED,
                    KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCKED, KeyboardId.ELEMENT_ALPHABET_SHIFT_LOCK_SHIFTED -> true;
            default -> false;
        };
    }

    private static int computeHashCode(final Key key) {
        return Arrays.hashCode(new Object[] {
                key.mX,
                key.mY,
                key.mWidth,
                key.mHeight,
                key.mCode,
                key.mLabel,
                key.mHintLabel,
                key.mIconName,
                key.mBackgroundType,
                Arrays.hashCode(key.mPopupKeys),
                key.getOutputText(),
                key.mActionFlags,
                key.mLabelFlags,
                // Key can be distinguishable without the following members.
                // key.mOptionalAttributes.mAltCode,
                // key.mOptionalAttributes.mDisabledIconId,
                // key.mOptionalAttributes.mPreviewIconId,
                // key.mHorizontalGap,
                // key.mVerticalGap,
                // key.mOptionalAttributes.mVisualInsetLeft,
                // key.mOptionalAttributes.mVisualInsetRight,
                // key.mMaxPopupKeysColumn,
        });
    }

    private boolean equalsInternal(final Key o) {
        if (this == o) return true;
        return o.mX == mX
                && o.mY == mY
                && o.mWidth == mWidth
                && o.mHeight == mHeight
                && o.mCode == mCode
                && TextUtils.equals(o.mLabel, mLabel)
                && TextUtils.equals(o.mHintLabel, mHintLabel)
                && TextUtils.equals(o.mIconName, mIconName)
                && o.mBackgroundType == mBackgroundType
                && Arrays.equals(o.mPopupKeys, mPopupKeys)
                && TextUtils.equals(o.getOutputText(), getOutputText())
                && o.mActionFlags == mActionFlags
                && o.mLabelFlags == mLabelFlags;
    }

    @Override
    public int compareTo(Key o) {
        if (equalsInternal(o)) return 0;
        if (mHashCode > o.mHashCode) return 1;
        return -1;
    }

    @Override
    public int hashCode() {
        return mHashCode;
    }

    @Override
    public boolean equals(final Object o) {
        return o instanceof Key && equalsInternal((Key)o);
    }

    @NonNull
    @Override
    public String toString() {
        return toShortString() + " " + getX() + "," + getY() + " " + getWidth() + "x" + getHeight();
    }

    public String toShortString() {
        final int code = getCode();
        if (code == KeyCode.MULTIPLE_CODE_POINTS) {
            return getOutputText();
        }
        return Constants.printableCode(code);
    }

    public String toLongString() {
        final String iconName = getIconName();
        final String topVisual = (iconName != null)
                ? KeyboardIconsSet.PREFIX_ICON + iconName : getLabel();
        final String hintLabel = getHintLabel();
        final String visual = (hintLabel == null) ? topVisual : topVisual + "^" + hintLabel;
        return toString() + " " + visual + "/" + backgroundName(mBackgroundType);
    }

    private static String backgroundName(final int backgroundType) {
        return switch (backgroundType) {
            case BACKGROUND_TYPE_EMPTY -> "empty";
            case BACKGROUND_TYPE_NORMAL -> "normal";
            case BACKGROUND_TYPE_FUNCTIONAL -> "functional";
            case BACKGROUND_TYPE_ACTION -> "action";
            case BACKGROUND_TYPE_SPACEBAR -> "spacebar";
            default -> null;
        };
    }

    public int getCode() {
        return mCode;
    }

    @Nullable
    public String getLabel() {
        return mLabel;
    }

    @Nullable
    public String getHintLabel() {
        return mHintLabel;
    }

    @Nullable
    public PopupKeySpec[] getPopupKeys() {
        return mPopupKeys;
    }

    public void markAsLeftEdge(final KeyboardParams params) {
        mHitBox.left = params.mLeftPadding;
    }

    public void markAsRightEdge(final KeyboardParams params) {
        mHitBox.right = params.mOccupiedWidth - params.mRightPadding;
    }

    public void markAsTopEdge(final KeyboardParams params) {
        mHitBox.top = params.mTopPadding;
    }

    public void markAsBottomEdge(final KeyboardParams params) {
        mHitBox.bottom = params.mOccupiedHeight + params.mBottomPadding;
    }

    public final boolean isSpacer() {
        return this instanceof Spacer;
    }

    public final boolean hasActionKeyBackground() {
        return mBackgroundType == BACKGROUND_TYPE_ACTION;
    }

    public final boolean isShift() {
        return mCode == KeyCode.SHIFT;
    }

    public final boolean isModifier() {
        return KeyCode.INSTANCE.isModifier(mCode);
    }

    public final boolean isRepeatable() {
        return (mActionFlags & ACTION_FLAGS_IS_REPEATABLE) != 0;
    }

    public final boolean hasPreview() {
        return (mActionFlags & ACTION_FLAGS_NO_KEY_PREVIEW) == 0;
    }

    /**
     *  altCodeWhileTyping is a weird thing.
     *  When user pressed a typing key less than ignoreAltCodeKeyTimeout (config_ignore_alt_code_key_timeout / 350 ms) ago,
     *  this code will be used instead. There is no documentation, but it appears the purpose is to avoid unintentional layout switches.
     *  Assuming this is true, the key still is used now if pressed near the center, where we assume it's less likely to be accidental.
     *  See PointerTracker.isClearlyInsideKey
     */
    public final boolean altCodeWhileTyping() {
        return (mActionFlags & ACTION_FLAGS_ALT_CODE_WHILE_TYPING) != 0;
    }

    public final boolean isLongPressEnabled() {
        // We need not start long press timer on the key which has activated shifted letter.
        return (mActionFlags & ACTION_FLAGS_ENABLE_LONG_PRESS) != 0
                && (mLabelFlags & LABEL_FLAGS_SHIFTED_LETTER_ACTIVATED) == 0;
    }

    public KeyVisualAttributes getVisualAttributes() {
        return mKeyVisualAttributes;
    }

    @NonNull
    public final Typeface selectTypeface(final KeyDrawParams params) {
        return switch (mLabelFlags & LABEL_FLAGS_FONT_MASK) {
            case LABEL_FLAGS_FONT_NORMAL -> Typeface.DEFAULT;
            case LABEL_FLAGS_FONT_MONO_SPACE -> Typeface.MONOSPACE;
            default -> params.mTypeface; // The type-face is specified by keyTypeface attribute.
        };
    }

    public final int selectTextSize(final KeyDrawParams params) {
        return switch (mLabelFlags & LABEL_FLAGS_FOLLOW_KEY_TEXT_RATIO_MASK) {
            case LABEL_FLAGS_FOLLOW_KEY_LETTER_RATIO -> params.mLetterSize;
            case LABEL_FLAGS_FOLLOW_KEY_LARGE_LETTER_RATIO -> params.mLargeLetterSize;
            case LABEL_FLAGS_FOLLOW_KEY_LABEL_RATIO -> params.mLabelSize;
            case LABEL_FLAGS_FOLLOW_KEY_HINT_LABEL_RATIO -> params.mHintLabelSize;
            // No follow key ratio flag specified.
            default -> StringUtils.codePointCount(mLabel) == 1 ? params.mLetterSize : params.mLabelSize;
        };
    }

    public final int selectTextColor(final KeyDrawParams params) {
        if ((mLabelFlags & LABEL_FLAGS_FOLLOW_FUNCTIONAL_TEXT_COLOR) != 0) {
            return params.mFunctionalTextColor;
        }
        return isShiftedLetterActivated() ? params.mTextInactivatedColor : params.mTextColor;
    }

    public final int selectHintTextSize(final KeyDrawParams params) {
        if (hasHintLabel()) {
            return params.mHintLabelSize;
        }
        if (hasShiftedLetterHint()) {
            return params.mShiftedLetterHintSize;
        }
        return params.mHintLetterSize;
    }

    public final int selectHintTextColor(final KeyDrawParams params) {
        if (hasHintLabel()) {
            return params.mHintLabelColor;
        }
        if (hasShiftedLetterHint()) {
            return isShiftedLetterActivated() ? params.mShiftedLetterHintActivatedColor
                    : params.mShiftedLetterHintInactivatedColor;
        }
        return params.mHintLetterColor;
    }

    public final int selectPopupKeyTextSize(final KeyDrawParams params) {
        return hasLabelsInPopupKeys() ? params.mLabelSize : params.mLetterSize;
    }

    public final String getPreviewLabel() {
        return isShiftedLetterActivated() ? mHintLabel : mLabel;
    }

    private boolean previewHasLetterSize() {
        return (mLabelFlags & LABEL_FLAGS_FOLLOW_KEY_LETTER_RATIO) != 0
                || StringUtils.codePointCount(getPreviewLabel()) == 1;
    }

    public final int selectPreviewTextSize(final KeyDrawParams params) {
        if (previewHasLetterSize()) {
            return params.mPreviewTextSize;
        }
        return params.mLetterSize;
    }

    @NonNull
    public Typeface selectPreviewTypeface(final KeyDrawParams params) {
        if (previewHasLetterSize()) {
            return selectTypeface(params);
        }
        return Typeface.DEFAULT_BOLD;
    }

    public final boolean isAlignHintLabelToBottom(final int defaultFlags) {
        return ((mLabelFlags | defaultFlags) & LABEL_FLAGS_ALIGN_HINT_LABEL_TO_BOTTOM) != 0;
    }

    public final boolean isAlignIconToBottom() {
        return (mLabelFlags & LABEL_FLAGS_ALIGN_ICON_TO_BOTTOM) != 0;
    }

    public final boolean isAlignLabelOffCenter() {
        return (mLabelFlags & LABEL_FLAGS_ALIGN_LABEL_OFF_CENTER) != 0;
    }

    public final boolean hasPopupHint() {
        return (mLabelFlags & LABEL_FLAGS_HAS_POPUP_HINT) != 0;
    }

    public final boolean hasShiftedLetterHint() {
        return (mLabelFlags & LABEL_FLAGS_HAS_SHIFTED_LETTER_HINT) != 0
                && !TextUtils.isEmpty(mHintLabel);
    }

    public final boolean hasHintLabel() {
        return (mLabelFlags & LABEL_FLAGS_HAS_HINT_LABEL) != 0;
    }

    public final boolean needsAutoXScale() {
        return (mLabelFlags & LABEL_FLAGS_AUTO_X_SCALE) != 0;
    }

    public final boolean needsAutoScale() {
        return (mLabelFlags & LABEL_FLAGS_AUTO_SCALE) == LABEL_FLAGS_AUTO_SCALE;
    }

    public final boolean needsToKeepBackgroundAspectRatio(final int defaultFlags) {
        return ((mLabelFlags | defaultFlags) & LABEL_FLAGS_KEEP_BACKGROUND_ASPECT_RATIO) != 0;
    }

    public final boolean hasCustomActionLabel() {
        return (mLabelFlags & LABEL_FLAGS_FROM_CUSTOM_ACTION_LABEL) != 0;
    }

    private boolean isShiftedLetterActivated() {
        return (mLabelFlags & LABEL_FLAGS_SHIFTED_LETTER_ACTIVATED) != 0
                && !TextUtils.isEmpty(mHintLabel);
    }

    public final int getPopupKeysColumnNumber() {
        return mPopupKeysColumnAndFlags & POPUP_KEYS_COLUMN_NUMBER_MASK;
    }

    public final boolean isPopupKeysFixedColumn() {
        return (mPopupKeysColumnAndFlags & POPUP_KEYS_FLAGS_FIXED_COLUMN) != 0;
    }

    public final boolean isPopupKeysFixedOrder() {
        return (mPopupKeysColumnAndFlags & POPUP_KEYS_FLAGS_FIXED_ORDER) != 0;
    }

    public final boolean hasLabelsInPopupKeys() {
        return (mPopupKeysColumnAndFlags & POPUP_KEYS_FLAGS_HAS_LABELS) != 0;
    }

    public final int getPopupKeyLabelFlags() {
        final int labelSizeFlag = hasLabelsInPopupKeys()
                ? LABEL_FLAGS_FOLLOW_KEY_LABEL_RATIO
                : LABEL_FLAGS_FOLLOW_KEY_LETTER_RATIO;
        return labelSizeFlag | LABEL_FLAGS_AUTO_X_SCALE;
    }

    public final boolean needsDividersInPopupKeys() {
        return (mPopupKeysColumnAndFlags & POPUP_KEYS_FLAGS_NEEDS_DIVIDERS) != 0;
    }

    public final boolean hasNoPanelAutoPopupKey() {
        return (mPopupKeysColumnAndFlags & POPUP_KEYS_FLAGS_NO_PANEL_AUTO_POPUP_KEY) != 0;
    }

    @Nullable
    public final String getOutputText() {
        final OptionalAttributes attrs = mOptionalAttributes;
        return (attrs != null) ? attrs.mOutputText : null;
    }

    public final int getAltCode() {
        final OptionalAttributes attrs = mOptionalAttributes;
        return (attrs != null) ? attrs.mAltCode : KeyCode.NOT_SPECIFIED;
    }

    @Nullable
    public String getIconName() {
        return mIconName;
    }

    @Nullable
    public Drawable getIcon(final KeyboardIconsSet iconSet, final int alpha) {
        final OptionalAttributes attrs = mOptionalAttributes;
        final String iconName = mEnabled ? getIconName() : ((attrs != null) ? attrs.mDisabledIconName : null);
        final Drawable icon = iconSet.getIconDrawable(iconName);
        if (icon != null) {
            icon.setAlpha(alpha);
        }
        return icon;
    }

    @Nullable
    public Drawable getPreviewIcon(final KeyboardIconsSet iconSet) {
        return iconSet.getIconDrawable(getIconName());
    }

    /**
     * Gets the background type of this key.
     * @return Background type.
     * @see Key#BACKGROUND_TYPE_EMPTY
     * @see Key#BACKGROUND_TYPE_NORMAL
     * @see Key#BACKGROUND_TYPE_FUNCTIONAL
     * @see Key#BACKGROUND_TYPE_ACTION
     * @see Key#BACKGROUND_TYPE_SPACEBAR
     */
    public int getBackgroundType() {
        return mBackgroundType;
    }

    /**
     * Gets the width of the key in pixels, excluding the gap.
     * @return The width of the key in pixels, excluding the gap.
     */
    public int getWidth() {
        return mWidth;
    }

    /**
     * Gets the height of the key in pixels, excluding the gap.
     * @return The height of the key in pixels, excluding the gap.
     */
    public int getHeight() {
        return mHeight;
    }

    /**
     * The combined width in pixels of the horizontal gaps belonging to this key, both above and
     * below. I.e., getWidth() + getHorizontalGap() = total width belonging to the key.
     * @return Horizontal gap belonging to this key.
     */
    public int getHorizontalGap() {
        return mHorizontalGap;
    }

    /**
     * The combined height in pixels of the vertical gaps belonging to this key, both above and
     * below. I.e., getHeight() + getVerticalGap() = total height belonging to the key.
     * @return Vertical gap belonging to this key.
     */
    public int getVerticalGap() {
        return mVerticalGap;
    }

    /**
     * Gets the x-coordinate of the top-left corner of the key in pixels, excluding the gap.
     * @return The x-coordinate of the top-left corner of the key in pixels, excluding the gap.
     */
    public int getX() {
        return mX;
    }

    /**
     * Gets the y-coordinate of the top-left corner of the key in pixels, excluding the gap.
     * @return The y-coordinate of the top-left corner of the key in pixels, excluding the gap.
     */
    public int getY() {
        return mY;
    }

    public final int getDrawX() {
        final int x = getX();
        final OptionalAttributes attrs = mOptionalAttributes;
        return (attrs == null) ? x : x + attrs.mVisualInsetsLeft;
    }

    public final int getDrawWidth() {
        final OptionalAttributes attrs = mOptionalAttributes;
        return (attrs == null) ? mWidth
                : mWidth - attrs.mVisualInsetsLeft - attrs.mVisualInsetsRight;
    }

    /**
     * Informs the key that it has been pressed, in case it needs to change its appearance or
     * state.
     * @see #onReleased()
     */
    public void onPressed() {
        mPressed = true;
    }

    /**
     * Informs the key that it has been released, in case it needs to change its appearance or
     * state.
     * @see #onPressed()
     */
    public void onReleased() {
        mPressed = false;
    }

    public final boolean isEnabled() {
        return mEnabled;
    }

    public void setEnabled(final boolean enabled) {
        mEnabled = enabled;
    }

    public void setLocked(final boolean locked) {
        mLocked = locked;
    }

    @NonNull
    public Rect getHitBox() {
        return mHitBox;
    }

    /**
     * Detects if a point falls on this key.
     * @param x the x-coordinate of the point
     * @param y the y-coordinate of the point
     * @return whether or not the point falls on the key. If the key is attached to an edge, it
     * will assume that all points between the key and the edge are considered to be on the key.
     * @see #markAsLeftEdge(KeyboardParams) etc.
     */
    public boolean isOnKey(final int x, final int y) {
        return mHitBox.contains(x, y);
    }

    /**
     * Returns the square of the distance to the nearest edge of the key and the given point.
     * @param x the x-coordinate of the point
     * @param y the y-coordinate of the point
     * @return the square of the distance of the point from the nearest edge of the key
     */
    public int squaredDistanceToEdge(final int x, final int y) {
        final int left = getX();
        final int right = left + mWidth;
        final int top = getY();
        final int bottom = top + mHeight;
        final int edgeX = x < left ? left : Math.min(x, right);
        final int edgeY = y < top ? top : Math.min(y, bottom);
        final int dx = x - edgeX;
        final int dy = y - edgeY;
        return dx * dx + dy * dy;
    }

    static class KeyBackgroundState {
        private final int[] mReleasedState;
        private final int[] mPressedState;

        private KeyBackgroundState(final int ... attrs) {
            mReleasedState = attrs;
            mPressedState = Arrays.copyOf(attrs, attrs.length + 1);
            mPressedState[attrs.length] = android.R.attr.state_pressed;
        }

        public int[] getState(final boolean pressed) {
            return pressed ? mPressedState : mReleasedState;
        }

        public static final KeyBackgroundState[] STATES = {
            // 0: BACKGROUND_TYPE_EMPTY
            new KeyBackgroundState(android.R.attr.state_empty),
            // 1: BACKGROUND_TYPE_NORMAL
            new KeyBackgroundState(),
            // 2: BACKGROUND_TYPE_FUNCTIONAL
            new KeyBackgroundState(),
            // 3: BACKGROUND_TYPE_ACTION
            new KeyBackgroundState(android.R.attr.state_active),
            // 4: BACKGROUND_TYPE_SPACEBAR
            new KeyBackgroundState(),
        };
    }

    /**
     * Returns the background drawable for the key, based on the current state and type of the key.
     * @return the background drawable of the key.
     * @see android.graphics.drawable.StateListDrawable#setState(int[])
     */
    @NonNull
    public final Drawable selectBackgroundDrawable(@NonNull final Drawable keyBackground,
            @NonNull final Drawable functionalKeyBackground,
            @NonNull final Drawable spacebarBackground,
            @NonNull final Drawable actionKeyBackground) {
        final Drawable background;
        if (hasActionKeyBackground()) {
            background = actionKeyBackground;
        } else if (hasFunctionalBackground()) {
            background = functionalKeyBackground;
        } else if (mBackgroundType == BACKGROUND_TYPE_SPACEBAR) {
            background = spacebarBackground;
        } else {
            background = keyBackground;
        }
        final int[] state = KeyBackgroundState.STATES[mBackgroundType].getState(mPressed || mLocked);
        background.setState(state);
        return background;
    }

    public final boolean hasActionKeyPopups() {
        if (!hasActionKeyBackground()) return false;
        // only use the special action key popups for action colored keys, and only for icon popups
        return ArraysKt.none(getPopupKeys(), (key) -> key.mIconName == null);
    }

    public boolean hasFunctionalBackground() {
        return mBackgroundType == BACKGROUND_TYPE_FUNCTIONAL;
    }

    @Nullable private static String getDisabledIconName(@NonNull final String iconName) {
        if (iconName.equals(ToolbarUtilsKt.getToolbarKeyStrings().get(ToolbarKey.VOICE)))
            return KeyboardIconsSet.NAME_SHORTCUT_KEY_DISABLED;
        return null;
    }

    public static class Spacer extends Key {
        private Spacer(KeyParams keyParams) {
            super(keyParams);
        }

        /**
         * This constructor is being used only for divider in popup keys keyboard.
         */
        protected Spacer(final KeyboardParams params, final int x, final int y, final int width,
                final int height) {
            super(null, null, KeyCode.NOT_SPECIFIED, null,
                    null, 0, BACKGROUND_TYPE_EMPTY, x, y, width,
                    height, params.mHorizontalGap, params.mVerticalGap);
        }
    }

    // for creating keys that might get modified later
    public static class KeyParams {
        // params for building
        public boolean isSpacer;
        private final KeyboardParams mKeyboardParams; // for reading gaps and keyboard width / height
        public float mWidth;
        public float mHeight; // also should allow negative values, indicating absolute height is defined

        // params that may change
        public float mAbsoluteWidth;
        public float mAbsoluteHeight;
        public float xPos;
        public float yPos;

        // params that remains constant
        public final int mCode;
        @Nullable public final String mLabel;
        @Nullable public final String mHintLabel;
        public final int mLabelFlags;
        @Nullable public final String mIconName;
        @Nullable public final PopupKeySpec[] mPopupKeys;
        public final int mPopupKeysColumnAndFlags;
        public final int mBackgroundType;
        public final int mActionFlags;
        @Nullable public final KeyVisualAttributes mKeyVisualAttributes;
        @Nullable final OptionalAttributes mOptionalAttributes;
        public final boolean mEnabled;

        public static KeyParams newSpacer(final KeyboardParams params, final float width) {
            final KeyParams spacer = new KeyParams(params);
            spacer.mWidth = width;
            spacer.mHeight = params.mDefaultRowHeight;
            return spacer;
        }

        public Key createKey() {
            if (isSpacer) return new Spacer(this);
            return new Key(this);
        }

        public void setAbsoluteDimensions(final float newX, final float newY) {
            if (mHeight == 0)
                mHeight = mKeyboardParams.mDefaultRowHeight;
            if (!isSpacer && mWidth == 0)
                throw new IllegalStateException("width = 0 should have been evaluated already");
            if (mHeight < 0)
                // todo (later): deal with it properly when it needs to be adjusted, i.e. when changing popupKeys or moreSuggestions
                throw new IllegalStateException("can't (yet) deal with absolute height");
            xPos = newX;
            yPos = newY;
            mAbsoluteWidth = mWidth * mKeyboardParams.mBaseWidth;
            mAbsoluteHeight = mHeight * mKeyboardParams.mBaseHeight;
        }

        private static int getPopupKeysColumnAndFlagsAndSetNullInArray(final KeyboardParams params, final String[] popupKeys) {
            // Get maximum column order number and set a relevant mode value.
            int popupKeysColumnAndFlags = POPUP_KEYS_MODE_MAX_COLUMN_WITH_AUTO_ORDER | params.mMaxPopupKeysKeyboardColumn;
            int value;
            if ((value = PopupKeySpec.getIntValue(popupKeys, POPUP_KEYS_AUTO_COLUMN_ORDER, -1)) > 0) {
                // Override with fixed column order number and set a relevant mode value.
                popupKeysColumnAndFlags = POPUP_KEYS_MODE_FIXED_COLUMN_WITH_AUTO_ORDER | (value & POPUP_KEYS_COLUMN_NUMBER_MASK);
            }
            if ((value = PopupKeySpec.getIntValue(popupKeys, POPUP_KEYS_FIXED_COLUMN_ORDER, -1)) > 0) {
                // Override with fixed column order number and set a relevant mode value.
                popupKeysColumnAndFlags = POPUP_KEYS_MODE_FIXED_COLUMN_WITH_FIXED_ORDER | (value & POPUP_KEYS_COLUMN_NUMBER_MASK);
            }
            if (PopupKeySpec.getBooleanValue(popupKeys, POPUP_KEYS_HAS_LABELS)) {
                popupKeysColumnAndFlags |= POPUP_KEYS_FLAGS_HAS_LABELS;
            }
            if (PopupKeySpec.getBooleanValue(popupKeys, POPUP_KEYS_NEEDS_DIVIDERS)) {
                popupKeysColumnAndFlags |= POPUP_KEYS_FLAGS_NEEDS_DIVIDERS;
            }
            if (PopupKeySpec.getBooleanValue(popupKeys, POPUP_KEYS_NO_PANEL_AUTO_POPUP_KEY)) {
                popupKeysColumnAndFlags |= POPUP_KEYS_FLAGS_NO_PANEL_AUTO_POPUP_KEY;
            }
            return popupKeysColumnAndFlags;
        }

        public String getOutputText() {
            return mOptionalAttributes == null ? null : mOptionalAttributes.mOutputText;
        }

        public KeyParams(
                @NonNull final String keySpec,
                @NonNull final KeyboardParams params,
                final float relativeWidth,
                final int labelFlags,
                final int backgroundType,
                @Nullable final PopupSet<?> popupSet
        ) {
            this(keySpec, KeySpecParser.getCode(keySpec), params, relativeWidth, labelFlags, backgroundType, popupSet);
        }

        /**
         *  constructor that does not require attrs, style or absolute key dimension / position
         *  setDimensionsFromRelativeSize needs to be called before creating the key
         */
        public KeyParams(
                // todo (much later): replace keySpec? these encoded icons and codes are not really great
                @NonNull final String keySpec, // key text or some special string for KeySpecParser, e.g. "!icon/shift_key|!code/key_shift" (avoid using !text, should be removed)
                final int code,
                @NonNull final KeyboardParams params,
                final float width,
                final int labelFlags,
                final int backgroundType,
                @Nullable final PopupSet<?> popupSet
        ) {
            mKeyboardParams = params;
            mBackgroundType = backgroundType;
            mLabelFlags = labelFlags;
            mWidth = width;
            mHeight = params.mDefaultRowHeight;
            mIconName = KeySpecParser.getIconName(keySpec);

            final boolean needsToUpcase = needsToUpcase(mLabelFlags, params.mId.mElementId);
            final Locale localeForUpcasing = params.mId.getLocale();
            int actionFlags = 0;
            if (params.mId.isNumberLayout())
                actionFlags = ACTION_FLAGS_NO_KEY_PREVIEW;

            // label
            String label = null;
            if ((mLabelFlags & LABEL_FLAGS_FROM_CUSTOM_ACTION_LABEL) != 0) {
                mLabel = params.mId.mCustomActionLabel;
            } else if (code >= Character.MIN_SUPPLEMENTARY_CODE_POINT) {
                // This is a workaround to have a key that has a supplementary code point in its label.
                // Because we can put a string in resource neither as a XML entity of a supplementary
                // code point nor as a surrogate pair.
                mLabel = new StringBuilder().appendCodePoint(code).toString();
            } else {
                label = KeySpecParser.getLabel(keySpec);
                mLabel = needsToUpcase
                        ? StringUtils.toTitleCaseOfKeyLabel(label, localeForUpcasing)
                        : label;
            }

            // popupKeys
            final String[] popupKeys = PopupKeysUtilsKt.createPopupKeysArray(popupSet, mKeyboardParams, label != null ? label : keySpec);
            mPopupKeysColumnAndFlags = getPopupKeysColumnAndFlagsAndSetNullInArray(params, popupKeys);
            final String[] finalPopupKeys = popupKeys == null ? null : PopupKeySpec.filterOutEmptyString(popupKeys);
            if (finalPopupKeys != null) {
                actionFlags |= ACTION_FLAGS_ENABLE_LONG_PRESS;
                mPopupKeys = new PopupKeySpec[finalPopupKeys.length];
                for (int i = 0; i < finalPopupKeys.length; i++) {
                    mPopupKeys[i] = new PopupKeySpec(finalPopupKeys[i], needsToUpcase, localeForUpcasing);
                }
            } else {
                mPopupKeys = null;
            }

            // hint label
            if ((mLabelFlags & LABEL_FLAGS_DISABLE_HINT_LABEL) != 0) {
                mHintLabel = null;
            } else {
                // maybe also always null for comma and period keys
                final String hintLabel = PopupKeysUtilsKt.getHintLabel(popupSet, params, keySpec);
                mHintLabel = needsToUpcase
                        ? StringUtils.toTitleCaseOfKeyLabel(hintLabel, localeForUpcasing)
                        : hintLabel;
            }

            String outputText = KeySpecParser.getOutputText(keySpec, code);
            if (needsToUpcase) {
                outputText = StringUtils.toTitleCaseOfKeyLabel(outputText, localeForUpcasing);
            }
            // Choose the first letter of the label as primary code if not specified.
            if (code == KeyCode.NOT_SPECIFIED && TextUtils.isEmpty(outputText) && !TextUtils.isEmpty(mLabel)) {
                if (StringUtils.codePointCount(mLabel) == 1) {
                    // Use the first letter of the hint label if shiftedLetterActivated flag is
                    // specified.
                    if ((mLabelFlags & LABEL_FLAGS_HAS_SHIFTED_LETTER_HINT) != 0 && (mLabelFlags & LABEL_FLAGS_SHIFTED_LETTER_ACTIVATED) != 0
                            && !TextUtils.isEmpty(mHintLabel)) {
                        mCode = mHintLabel.codePointAt(0);
                    } else {
                        mCode = mLabel.codePointAt(0);
                    }
                } else {
                    // In some locale and case, the character might be represented by multiple code
                    // points, such as upper case Eszett of German alphabet.
                    outputText = mLabel;
                    mCode = KeyCode.MULTIPLE_CODE_POINTS;
                }
            } else if (code == KeyCode.NOT_SPECIFIED && outputText != null) {
                if (StringUtils.codePointCount(outputText) == 1) {
                    mCode = outputText.codePointAt(0);
                    outputText = null;
                } else {
                    mCode = KeyCode.MULTIPLE_CODE_POINTS;
                }
            } else {
                mCode = needsToUpcase ? StringUtils.toTitleCaseOfKeyCode(code, localeForUpcasing) : code;
            }

            // action flags don't need to be specified, they can be deduced from the key
            if (mCode == Constants.CODE_SPACE
                    || mCode == KeyCode.LANGUAGE_SWITCH
                    || (mCode == KeyCode.SYMBOL_ALPHA && !params.mId.isAlphabetKeyboard())
            )
                actionFlags |= ACTION_FLAGS_ENABLE_LONG_PRESS;
            if (mCode <= Constants.CODE_SPACE && mCode != KeyCode.MULTIPLE_CODE_POINTS && mIconName == null)
                actionFlags |= ACTION_FLAGS_NO_KEY_PREVIEW;
            switch (mCode) {
            case KeyCode.DELETE, KeyCode.ARROW_LEFT, KeyCode.ARROW_RIGHT, KeyCode.ARROW_UP, KeyCode.ARROW_DOWN,
                    KeyCode.WORD_LEFT, KeyCode.WORD_RIGHT, KeyCode.PAGE_UP, KeyCode.PAGE_DOWN:
                // repeating is disabled if a key is configured with pop-ups
                if (mPopupKeys == null)
                    actionFlags |= ACTION_FLAGS_IS_REPEATABLE;
                // fallthrough
            case KeyCode.SHIFT, Constants.CODE_ENTER, KeyCode.SHIFT_ENTER, KeyCode.ALPHA, Constants.CODE_SPACE, KeyCode.NUMPAD,
                    KeyCode.SYMBOL, KeyCode.SYMBOL_ALPHA, KeyCode.LANGUAGE_SWITCH, KeyCode.EMOJI, KeyCode.CLIPBOARD,
                    KeyCode.MOVE_START_OF_LINE, KeyCode.MOVE_END_OF_LINE, KeyCode.MOVE_START_OF_PAGE, KeyCode.MOVE_END_OF_PAGE,
                    KeyCode.EMOJI_SEARCH:
                actionFlags |= ACTION_FLAGS_NO_KEY_PREVIEW; // no preview even if icon!
            }
            if (mCode == KeyCode.SETTINGS || mCode == KeyCode.LANGUAGE_SWITCH)
                actionFlags |= ACTION_FLAGS_ALT_CODE_WHILE_TYPING;
            mActionFlags = actionFlags;

            final int altCodeInAttr; // settings and language switch keys have alt code space, all others nothing
            if (mCode == KeyCode.SETTINGS || mCode == KeyCode.LANGUAGE_SWITCH || mCode == KeyCode.EMOJI || mCode == KeyCode.CLIPBOARD)
                altCodeInAttr = Constants.CODE_SPACE;
            else
                altCodeInAttr = KeyCode.NOT_SPECIFIED;
            final int altCode = needsToUpcase
                    ? StringUtils.toTitleCaseOfKeyCode(altCodeInAttr, localeForUpcasing)
                    : altCodeInAttr;
            mOptionalAttributes = OptionalAttributes.newInstance(outputText, altCode,
                    // disabled icon only shortcut / voice key, visual insets can be replaced with spacer
                    mIconName == null ? null : getDisabledIconName(mIconName), 0, 0);
            // KeyVisualAttributes for a key essentially are what the theme has, but on a per-key base
            // could be used e.g. for having a color gradient on key color
            mKeyVisualAttributes = null;
            mEnabled = true;
        }

        /** constructor for emoji parser */
        public KeyParams(@Nullable final String label, final int code, @Nullable final String hintLabel,
                   @Nullable final String popupKeySpecs, final int labelFlags, final KeyboardParams params) {
            mKeyboardParams = params;
            mHintLabel = hintLabel;
            mLabelFlags = labelFlags;
            mBackgroundType = BACKGROUND_TYPE_EMPTY;

            if (popupKeySpecs != null) {
                String[] popupKeys = PopupKeySpec.splitKeySpecs(popupKeySpecs);
                mPopupKeysColumnAndFlags = getPopupKeysColumnAndFlagsAndSetNullInArray(params, popupKeys);

                popupKeys = PopupKeySpec.insertAdditionalPopupKeys(popupKeys, null);
                int actionFlags = 0;
                if (popupKeys != null) {
                    actionFlags |= ACTION_FLAGS_ENABLE_LONG_PRESS;
                    mPopupKeys = new PopupKeySpec[popupKeys.length];
                    for (int i = 0; i < popupKeys.length; i++) {
                        mPopupKeys[i] = new PopupKeySpec(popupKeys[i], false, Locale.getDefault());
                    }
                } else {
                    mPopupKeys = null;
                }
                mActionFlags = actionFlags;
            } else {
                // TODO: Pass keyActionFlags as an argument.
                mActionFlags = ACTION_FLAGS_NO_KEY_PREVIEW;
                mPopupKeys = null;
                mPopupKeysColumnAndFlags = 0;
            }

            mLabel = label;
            mOptionalAttributes = code == KeyCode.MULTIPLE_CODE_POINTS
                    ? OptionalAttributes.newInstance(label, KeyCode.NOT_SPECIFIED, null, 0, 0)
                    : null;
            mCode = code;
            mEnabled = (code != KeyCode.NOT_SPECIFIED);
            mIconName = null;
            mKeyVisualAttributes = null;
        }

        /** constructor for a spacer whose size MUST be determined using setDimensionsFromRelativeSize */
        private KeyParams(final KeyboardParams params) {
            isSpacer = true; // this is only for spacer!
            mKeyboardParams = params;

            mCode = KeyCode.NOT_SPECIFIED;
            mLabel = null;
            mHintLabel = null;
            mKeyVisualAttributes = null;
            mOptionalAttributes = null;
            mIconName = null;
            mBackgroundType = BACKGROUND_TYPE_NORMAL;
            mActionFlags = ACTION_FLAGS_NO_KEY_PREVIEW;
            mPopupKeys = null;
            mPopupKeysColumnAndFlags = 0;
            mLabelFlags = LABEL_FLAGS_FONT_NORMAL;
            mEnabled = true;
        }

        public KeyParams(final KeyParams keyParams) {
            xPos = keyParams.xPos;
            yPos = keyParams.yPos;
            mWidth = keyParams.mWidth;
            mHeight = keyParams.mHeight;
            isSpacer = keyParams.isSpacer;
            mKeyboardParams = keyParams.mKeyboardParams;
            mEnabled = keyParams.mEnabled;

            mCode = keyParams.mCode;
            mLabel = keyParams.mLabel;
            mHintLabel = keyParams.mHintLabel;
            mLabelFlags = keyParams.mLabelFlags;
            mIconName = keyParams.mIconName;
            mAbsoluteWidth = keyParams.mAbsoluteWidth;
            mAbsoluteHeight = keyParams.mAbsoluteHeight;
            mPopupKeys = keyParams.mPopupKeys;
            mPopupKeysColumnAndFlags = keyParams.mPopupKeysColumnAndFlags;
            mBackgroundType = keyParams.mBackgroundType;
            mActionFlags = keyParams.mActionFlags;
            mKeyVisualAttributes = keyParams.mKeyVisualAttributes;
            mOptionalAttributes = keyParams.mOptionalAttributes;
        }
    }
}
