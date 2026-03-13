/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard;

import android.animation.AnimatorInflater;
import android.animation.ObjectAnimator;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Align;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.ContextThemeWrapper;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import helium314.keyboard.accessibility.AccessibilityUtils;
import helium314.keyboard.accessibility.MainKeyboardAccessibilityDelegate;
import helium314.keyboard.compat.ConfigurationCompatKt;
import helium314.keyboard.keyboard.internal.DrawingPreviewPlacerView;
import helium314.keyboard.keyboard.internal.DrawingProxy;
import helium314.keyboard.keyboard.internal.GestureFloatingTextDrawingPreview;
import helium314.keyboard.keyboard.internal.GestureTrailsDrawingPreview;
import helium314.keyboard.keyboard.internal.KeyDrawParams;
import helium314.keyboard.keyboard.internal.KeyPreviewChoreographer;
import helium314.keyboard.keyboard.internal.KeyPreviewDrawParams;
import helium314.keyboard.keyboard.internal.KeyPreviewView;
import helium314.keyboard.keyboard.internal.PopupKeySpec;
import helium314.keyboard.keyboard.internal.NonDistinctMultitouchHelper;
import helium314.keyboard.keyboard.internal.SlidingKeyInputDrawingPreview;
import helium314.keyboard.keyboard.internal.TimerHandler;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.RichInputMethodSubtype;
import helium314.keyboard.latin.SuggestedWords;
import helium314.keyboard.latin.common.ColorType;
import helium314.keyboard.latin.common.Colors;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.common.CoordinateUtils;
import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.settings.DebugSettings;
import helium314.keyboard.latin.settings.Defaults;
import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.KtxKt;
import helium314.keyboard.latin.utils.LanguageOnSpacebarUtils;
import helium314.keyboard.latin.utils.Log;
import helium314.keyboard.latin.utils.TypefaceUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.WeakHashMap;

/** A view that is responsible for detecting key presses and touch movements. */
public final class MainKeyboardView extends KeyboardView implements DrawingProxy,
        PopupKeysPanel.Controller {
    private static final String TAG = MainKeyboardView.class.getSimpleName();

    /** Listener for {@link KeyboardActionListener}. */
    private KeyboardActionListener mKeyboardActionListener;

    /* Space key and its icon and background. */
    private Key mSpaceKey;
    // Stuff to draw language name on spacebar.
    private final int mLanguageOnSpacebarFinalAlpha;
    private final ObjectAnimator mLanguageOnSpacebarFadeoutAnimator;
    private int mLanguageOnSpacebarFormatType;
    private boolean mHasMultipleEnabledIMEsOrSubtypes;
    private int mLanguageOnSpacebarAnimAlpha = Constants.Color.ALPHA_OPAQUE;
    private final float mLanguageOnSpacebarTextRatio;
    private float mLanguageOnSpacebarTextSize;
    private final int mLanguageOnSpacebarTextColor;
    private final float mLanguageOnSpacebarTextShadowRadius;
    private final int mLanguageOnSpacebarTextShadowColor;
    private static final float LANGUAGE_ON_SPACEBAR_TEXT_SHADOW_RADIUS_DISABLED = -1.0f;
    // The minimum x-scale to fit the language name on spacebar.
    private static final float MINIMUM_XSCALE_OF_LANGUAGE_NAME = 0.8f;

    // Stuff to draw altCodeWhileTyping keys.
    private final ObjectAnimator mAltCodeKeyWhileTypingFadeoutAnimator;
    private final ObjectAnimator mAltCodeKeyWhileTypingFadeinAnimator;

    // Drawing preview placer view
    private final DrawingPreviewPlacerView mDrawingPreviewPlacerView;
    private final int[] mOriginCoords = CoordinateUtils.newInstance();
    private final GestureFloatingTextDrawingPreview mGestureFloatingTextDrawingPreview;
    private final GestureTrailsDrawingPreview mGestureTrailsDrawingPreview;
    private final SlidingKeyInputDrawingPreview mSlidingKeyInputDrawingPreview;

    // Key preview
    private final KeyPreviewDrawParams mKeyPreviewDrawParams;
    private final KeyPreviewChoreographer mKeyPreviewChoreographer;

    // More keys keyboard
    private final Paint mBackgroundDimAlphaPaint = new Paint(); // todo: not used at all
    private final View mPopupKeysKeyboardContainer;
    private final View mPopupKeysKeyboardForActionContainer;
    private final WeakHashMap<Key, Keyboard> mPopupKeysKeyboardCache = new WeakHashMap<>();
    private final boolean mConfigShowPopupKeysKeyboardAtTouchedPoint;
    // More keys panel (used by both popup keys keyboard and more suggestions view)
    // TODO: Consider extending to support multiple popup keys panels
    private PopupKeysPanel mPopupKeysPanel;

    // Gesture floating preview text
    private final int mGestureFloatingPreviewTextLingerTimeout;

    private final KeyDetector mKeyDetector;
    private final NonDistinctMultitouchHelper mNonDistinctMultitouchHelper;

    private final TimerHandler mTimerHandler;
    private final int mLanguageOnSpacebarHorizontalMargin;

    private MainKeyboardAccessibilityDelegate mAccessibilityDelegate;

    public MainKeyboardView(final Context context, final AttributeSet attrs) {
        this(context, attrs, R.attr.mainKeyboardViewStyle);
    }

    public MainKeyboardView(final Context context, final AttributeSet attrs, final int defStyle) {
        super(context, attrs, defStyle);

        final DrawingPreviewPlacerView drawingPreviewPlacerView =
                new DrawingPreviewPlacerView(new ContextThemeWrapper(context, R.style.platformActivityTheme), attrs);

        final TypedArray mainKeyboardViewAttr = context.obtainStyledAttributes(
                attrs, R.styleable.MainKeyboardView, defStyle, R.style.MainKeyboardView);
        final int ignoreAltCodeKeyTimeout = mainKeyboardViewAttr.getInt(
                R.styleable.MainKeyboardView_ignoreAltCodeKeyTimeout, 0);
        final int gestureRecognitionUpdateTime = mainKeyboardViewAttr.getInt(
                R.styleable.MainKeyboardView_gestureRecognitionUpdateTime, 0);
        mTimerHandler = new TimerHandler(this, ignoreAltCodeKeyTimeout, gestureRecognitionUpdateTime);

        final float keyHysteresisDistance = mainKeyboardViewAttr.getDimension(
                R.styleable.MainKeyboardView_keyHysteresisDistance, 0.0f);
        final float keyHysteresisDistanceForSlidingModifier = mainKeyboardViewAttr.getDimension(
                R.styleable.MainKeyboardView_keyHysteresisDistanceForSlidingModifier, 0.0f);
        mKeyDetector = new KeyDetector(keyHysteresisDistance, keyHysteresisDistanceForSlidingModifier);

        PointerTracker.init(mainKeyboardViewAttr, mTimerHandler, this /* DrawingProxy */);

        final SharedPreferences prefs = KtxKt.prefs(context);
        final boolean forceNonDistinctMultitouch = prefs.getBoolean(
                DebugSettings.PREF_FORCE_NON_DISTINCT_MULTITOUCH, Defaults.PREF_FORCE_NON_DISTINCT_MULTITOUCH);
        final boolean hasDistinctMultitouch = context.getPackageManager()
                .hasSystemFeature(PackageManager.FEATURE_TOUCHSCREEN_MULTITOUCH_DISTINCT)
                && !forceNonDistinctMultitouch;
        mNonDistinctMultitouchHelper = hasDistinctMultitouch ? null : new NonDistinctMultitouchHelper();

        final int backgroundDimAlpha = mainKeyboardViewAttr.getInt(
                R.styleable.MainKeyboardView_backgroundDimAlpha, 0);
        mBackgroundDimAlphaPaint.setColor(Color.BLACK);
        mBackgroundDimAlphaPaint.setAlpha(backgroundDimAlpha);
        mLanguageOnSpacebarTextRatio = mainKeyboardViewAttr.getFraction(
                R.styleable.MainKeyboardView_languageOnSpacebarTextRatio, 1, 1, 1.0f)
                * Settings.getValues().mFontSizeMultiplier;
        final Colors colors = Settings.getValues().mColors;
        mLanguageOnSpacebarTextColor = colors.get(ColorType.SPACE_BAR_TEXT);
        mLanguageOnSpacebarTextShadowRadius = mainKeyboardViewAttr.getFloat(
                R.styleable.MainKeyboardView_languageOnSpacebarTextShadowRadius,
                LANGUAGE_ON_SPACEBAR_TEXT_SHADOW_RADIUS_DISABLED);
        mLanguageOnSpacebarTextShadowColor = mainKeyboardViewAttr.getColor(
                R.styleable.MainKeyboardView_languageOnSpacebarTextShadowColor, 0);
        mLanguageOnSpacebarFinalAlpha = Color.alpha(mLanguageOnSpacebarTextColor);
        final int languageOnSpacebarFadeoutAnimatorResId = mainKeyboardViewAttr.getResourceId(
                R.styleable.MainKeyboardView_languageOnSpacebarFadeoutAnimator, 0);
        final int altCodeKeyWhileTypingFadeoutAnimatorResId = mainKeyboardViewAttr.getResourceId(
                R.styleable.MainKeyboardView_altCodeKeyWhileTypingFadeoutAnimator, 0);
        final int altCodeKeyWhileTypingFadeinAnimatorResId = mainKeyboardViewAttr.getResourceId(
                R.styleable.MainKeyboardView_altCodeKeyWhileTypingFadeinAnimator, 0);

        mKeyPreviewDrawParams = new KeyPreviewDrawParams(mainKeyboardViewAttr);
        mKeyPreviewChoreographer = new KeyPreviewChoreographer(mKeyPreviewDrawParams);

        final int popupKeysKeyboardLayoutId = mainKeyboardViewAttr.getResourceId(
                R.styleable.MainKeyboardView_popupKeysKeyboardLayout, 0);
        final int popupKeysKeyboardForActionLayoutId = mainKeyboardViewAttr.getResourceId(
                R.styleable.MainKeyboardView_popupKeysKeyboardForActionLayout,
                popupKeysKeyboardLayoutId);
        mConfigShowPopupKeysKeyboardAtTouchedPoint = mainKeyboardViewAttr.getBoolean(
                R.styleable.MainKeyboardView_showPopupKeysKeyboardAtTouchedPoint, false);

        final int gestureTrailFadeoutDuration = Settings.getValues().mGestureTrailFadeoutDuration;
        mGestureFloatingPreviewTextLingerTimeout = gestureTrailFadeoutDuration / 4;

        mGestureFloatingTextDrawingPreview = new GestureFloatingTextDrawingPreview(mainKeyboardViewAttr);
        mGestureFloatingTextDrawingPreview.setDrawingView(drawingPreviewPlacerView);

        mGestureTrailsDrawingPreview = new GestureTrailsDrawingPreview(mainKeyboardViewAttr);
        mGestureTrailsDrawingPreview.setDrawingView(drawingPreviewPlacerView);

        mSlidingKeyInputDrawingPreview = new SlidingKeyInputDrawingPreview(mainKeyboardViewAttr);
        mSlidingKeyInputDrawingPreview.setDrawingView(drawingPreviewPlacerView);
        mainKeyboardViewAttr.recycle();

        mDrawingPreviewPlacerView = drawingPreviewPlacerView;

        final LayoutInflater inflater = LayoutInflater.from(getContext());
        mPopupKeysKeyboardContainer = inflater.inflate(popupKeysKeyboardLayoutId, null);
        mPopupKeysKeyboardForActionContainer = inflater.inflate(popupKeysKeyboardForActionLayoutId, null);
        mLanguageOnSpacebarFadeoutAnimator = loadObjectAnimator(languageOnSpacebarFadeoutAnimatorResId, this);
        if (mLanguageOnSpacebarFadeoutAnimator != null)
            mLanguageOnSpacebarFadeoutAnimator.setIntValues(255, mLanguageOnSpacebarFinalAlpha);
        mAltCodeKeyWhileTypingFadeoutAnimator = loadObjectAnimator(altCodeKeyWhileTypingFadeoutAnimatorResId, this);
        mAltCodeKeyWhileTypingFadeinAnimator = loadObjectAnimator(altCodeKeyWhileTypingFadeinAnimatorResId, this);

        mKeyboardActionListener = KeyboardActionListener.EMPTY_LISTENER;

        mLanguageOnSpacebarHorizontalMargin = (int) getResources().getDimension(
                R.dimen.config_language_on_spacebar_horizontal_margin);
    }

    @Override
    public void setHardwareAcceleratedDrawingEnabled(final boolean enabled) {
        super.setHardwareAcceleratedDrawingEnabled(enabled);
        mDrawingPreviewPlacerView.setHardwareAcceleratedDrawingEnabled(enabled);
    }

    private ObjectAnimator loadObjectAnimator(final int resId, final Object target) {
        if (resId == 0) {
            // TODO: Stop returning null.
            return null;
        }
        final ObjectAnimator animator = (ObjectAnimator)AnimatorInflater.loadAnimator(
                getContext(), resId);
        if (animator != null) {
            animator.setTarget(target);
        }
        return animator;
    }

    private static void cancelAndStartAnimators(final ObjectAnimator animatorToCancel,
            final ObjectAnimator animatorToStart) {
        if (animatorToCancel == null || animatorToStart == null) {
            // TODO: Stop using null as a no-operation animator.
            return;
        }
        float startFraction = 0.0f;
        if (animatorToCancel.isStarted()) {
            animatorToCancel.cancel();
            startFraction = 1.0f - animatorToCancel.getAnimatedFraction();
        }
        final long startTime = (long)(animatorToStart.getDuration() * startFraction);
        animatorToStart.start();
        animatorToStart.setCurrentPlayTime(startTime);
    }

    // Implements {@link DrawingProxy#startWhileTypingAnimation(int)}.
    /**
     * Called when a while-typing-animation should be started.
     * @param fadeInOrOut {@link DrawingProxy#FADE_IN} starts while-typing-fade-in animation.
     * {@link DrawingProxy#FADE_OUT} starts while-typing-fade-out animation.
     */
    @Override
    public void startWhileTypingAnimation(final int fadeInOrOut) {
        switch (fadeInOrOut) {
            case DrawingProxy.FADE_IN -> cancelAndStartAnimators(
                    mAltCodeKeyWhileTypingFadeoutAnimator, mAltCodeKeyWhileTypingFadeinAnimator);
            case DrawingProxy.FADE_OUT -> cancelAndStartAnimators(
                    mAltCodeKeyWhileTypingFadeinAnimator, mAltCodeKeyWhileTypingFadeoutAnimator);
        }
    }

    public void setLanguageOnSpacebarAnimAlpha(final int alpha) {
        mLanguageOnSpacebarAnimAlpha = alpha;
        invalidateKey(mSpaceKey);
    }

    public void setKeyboardActionListener(final KeyboardActionListener listener) {
        mKeyboardActionListener = listener;
        PointerTracker.setKeyboardActionListener(listener);
    }

    // TODO: We should reconsider which coordinate system should be used to represent keyboard event.
    public int getKeyX(final int x) {
        return Constants.isValidCoordinate(x) ? mKeyDetector.getTouchX(x) : x;
    }

    // TODO: We should reconsider which coordinate system should be used to represent keyboard event.
    public int getKeyY(final int y) {
        return Constants.isValidCoordinate(y) ? mKeyDetector.getTouchY(y) : y;
    }

    /**
     * Attaches a keyboard to this view. The keyboard can be switched at any time and the
     * view will re-layout itself to accommodate the keyboard.
     * @see Keyboard
     * @see #getKeyboard()
     * @param keyboard the keyboard to display in this view
     */
    @Override
    public void setKeyboard(@NonNull final Keyboard keyboard) {
        // Remove any pending messages, except dismissing preview and key repeat.
        mTimerHandler.cancelLongPressTimers();
        super.setKeyboard(keyboard);
        mKeyDetector.setKeyboard(
                keyboard, -getPaddingLeft(), -getPaddingTop() + getVerticalCorrection());
        PointerTracker.setKeyDetector(mKeyDetector);
        mPopupKeysKeyboardCache.clear();

        mSpaceKey = keyboard.getKey(Constants.CODE_SPACE);
        final int keyHeight = keyboard.mMostCommonKeyHeight - keyboard.mVerticalGap;
        mLanguageOnSpacebarTextSize = keyHeight * mLanguageOnSpacebarTextRatio;

        if (AccessibilityUtils.Companion.getInstance().isAccessibilityEnabled()) {
            if (mAccessibilityDelegate == null) {
                mAccessibilityDelegate = new MainKeyboardAccessibilityDelegate(this, mKeyDetector);
            }
            mAccessibilityDelegate.setKeyboard(keyboard);
        } else {
            mAccessibilityDelegate = null;
        }
    }

    /**
     * Enables or disables the key preview popup. This is a popup that shows a magnified
     * version of the depressed key. By default the preview is enabled.
     * @param previewEnabled whether or not to enable the key feedback preview
     */
    public void setKeyPreviewPopupEnabled(final boolean previewEnabled) {
        mKeyPreviewDrawParams.setPopupEnabled(previewEnabled);
    }

    private void locatePreviewPlacerView() {
        getLocationInWindow(mOriginCoords);
        mDrawingPreviewPlacerView.setKeyboardViewGeometry(mOriginCoords, getWidth(), getHeight());
    }

    private void installPreviewPlacerView() {
        final View rootView = getRootView();
        if (rootView == null) {
            Log.w(TAG, "Cannot find root view");
            return;
        }
        final ViewGroup windowContentView = rootView.findViewById(android.R.id.content);
        // Note: It'd be very weird if we get null by android.R.id.content.
        if (windowContentView == null) {
            Log.w(TAG, "Cannot find android.R.id.content view to add DrawingPreviewPlacerView");
            return;
        }
        windowContentView.addView(mDrawingPreviewPlacerView);
    }

    // Implements {@link DrawingProxy#onKeyPressed(Key,boolean)}.
    @Override
    public void onKeyPressed(@NonNull final Key key, final boolean withPreview) {
        key.onPressed();
        invalidateKey(key);

        final Keyboard keyboard = getKeyboard();
        if (keyboard == null) {
            return;
        }
        mKeyPreviewDrawParams.setVisibleOffset(-keyboard.mVerticalGap);
        if (withPreview && key.hasPreview() && mKeyPreviewDrawParams.isPopupEnabled()) {
            showKeyPreview(key);
        }
    }

    private void showKeyPreview(@NonNull final Key key) {
        locatePreviewPlacerView();
        getLocationInWindow(mOriginCoords);
        mKeyPreviewChoreographer.placeAndShowKeyPreview(key, getKeyboard().mIconsSet, getKeyDrawParams(),
                KeyboardSwitcher.getInstance().getWrapperView().getWidth(), mOriginCoords, mDrawingPreviewPlacerView);
    }

    private void dismissKeyPreviewWithoutDelay(@NonNull final Key key) {
        mKeyPreviewChoreographer.dismissKeyPreview(key);
        invalidateKey(key);
    }

    // Implements {@link DrawingProxy#onKeyReleased(Key,boolean)}.
    @Override
    public void onKeyReleased(@NonNull final Key key, final boolean withAnimation) {
        key.onReleased();
        invalidateKey(key);
        if (key.hasPreview()) {
            if (withAnimation) {
                dismissKeyPreview(key);
            } else {
                dismissKeyPreviewWithoutDelay(key);
            }
        }
    }

    private void dismissKeyPreview(@NonNull final Key key) {
        if (isHardwareAccelerated()) {
            mKeyPreviewChoreographer.dismissKeyPreview(key);
        }
    }

    public void setSlidingKeyInputPreviewEnabled(final boolean enabled) {
        mSlidingKeyInputDrawingPreview.setPreviewEnabled(enabled);
    }

    @Override
    public void showSlidingKeyInputPreview(@Nullable final PointerTracker tracker) {
        locatePreviewPlacerView();
        if (tracker != null) {
            mSlidingKeyInputDrawingPreview.setPreviewPosition(tracker);
        } else {
            mSlidingKeyInputDrawingPreview.dismissSlidingKeyInputPreview();
        }
    }

    private void setGesturePreviewMode(final boolean isGestureTrailEnabled,
            final boolean isGestureFloatingPreviewTextEnabled) {
        mGestureFloatingTextDrawingPreview.setPreviewEnabled(isGestureFloatingPreviewTextEnabled);
        mGestureTrailsDrawingPreview.setPreviewEnabled(isGestureTrailEnabled);
    }

    public void showGestureFloatingPreviewText(@NonNull final SuggestedWords suggestedWords,
            final boolean dismissDelayed) {
        locatePreviewPlacerView();
        mGestureFloatingTextDrawingPreview.setSuggestedWords(suggestedWords);
        if (dismissDelayed) {
            mTimerHandler.postDismissGestureFloatingPreviewText(
                    mGestureFloatingPreviewTextLingerTimeout);
        }
    }

    // Implements {@link DrawingProxy#dismissGestureFloatingPreviewTextWithoutDelay()}.
    @Override
    public void dismissGestureFloatingPreviewTextWithoutDelay() {
        mGestureFloatingTextDrawingPreview.dismissGestureFloatingPreviewText();
    }

    @Override
    public void showGestureTrail(@NonNull final PointerTracker tracker,
            final boolean showsFloatingPreviewText) {
        locatePreviewPlacerView();
        if (showsFloatingPreviewText) {
            mGestureFloatingTextDrawingPreview.setPreviewPosition(tracker);
        }
        mGestureTrailsDrawingPreview.setPreviewPosition(tracker);
    }

    // Note that this method is called from a non-UI thread.
    @SuppressWarnings("static-method")
    public void setMainDictionaryAvailability(final boolean mainDictionaryAvailable) {
        PointerTracker.setMainDictionaryAvailability(mainDictionaryAvailable);
    }

    public void setGestureHandlingEnabledByUser(final boolean isGestureHandlingEnabledByUser,
            final boolean isGestureTrailEnabled,
            final boolean isGestureFloatingPreviewTextEnabled) {
        PointerTracker.setGestureHandlingEnabledByUser(isGestureHandlingEnabledByUser);
        setGesturePreviewMode(isGestureHandlingEnabledByUser && isGestureTrailEnabled,
                isGestureHandlingEnabledByUser && isGestureFloatingPreviewTextEnabled);
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        installPreviewPlacerView();
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        mDrawingPreviewPlacerView.removeAllViews();
    }

    // Implements {@link DrawingProxy@showPopupKeysKeyboard(Key,PointerTracker)}.
    @Override
    @Nullable
    public PopupKeysPanel showPopupKeysKeyboard(@NonNull final Key key,
                                                @NonNull final PointerTracker tracker) {
        final PopupKeySpec[] popupKeys = key.getPopupKeys();
        if (popupKeys == null) {
            return null;
        }
        Keyboard popupKeysKeyboard = mPopupKeysKeyboardCache.get(key);
        if (popupKeysKeyboard == null) {
            // {@link KeyPreviewDrawParams#mPreviewVisibleWidth} should have been set at
            // {@link KeyPreviewChoreographer#placeKeyPreview(Key,TextView,KeyboardIconsSet,KeyDrawParams,int,int[]},
            // though there may be some chances that the value is zero. <code>width == 0</code>
            // will cause zero-division error at
            // {@link PopupKeysKeyboardParams#setParameters(int,int,int,int,int,int,boolean,int)}.
            final boolean isSinglePopupKeyWithPreview = mKeyPreviewDrawParams.isPopupEnabled()
                    && key.hasPreview() && popupKeys.length == 1
                    && mKeyPreviewDrawParams.getVisibleWidth() > 0;
            final PopupKeysKeyboard.Builder builder = new PopupKeysKeyboard.Builder(
                    getContext(), key, getKeyboard(), isSinglePopupKeyWithPreview,
                    mKeyPreviewDrawParams.getVisibleWidth(),
                    mKeyPreviewDrawParams.getVisibleHeight(), newLabelPaint(key));
            popupKeysKeyboard = builder.build();
            mPopupKeysKeyboardCache.put(key, popupKeysKeyboard);
        }

        final View container = key.hasActionKeyPopups() ? mPopupKeysKeyboardForActionContainer
                : mPopupKeysKeyboardContainer;
        final PopupKeysKeyboardView popupKeysKeyboardView =
                container.findViewById(R.id.popup_keys_keyboard_view);
        popupKeysKeyboardView.setKeyboard(popupKeysKeyboard);
        container.measure(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);

        final int[] lastCoords = CoordinateUtils.newInstance();
        tracker.getLastCoordinates(lastCoords);
        final boolean keyPreviewEnabled = mKeyPreviewDrawParams.isPopupEnabled() && key.hasPreview();
        // The popup keys keyboard is usually horizontally aligned with the center of the parent key.
        // If showPopupKeysKeyboardAtTouchedPoint is true and the key preview is disabled, the more
        // keys keyboard is placed at the touch point of the parent key.
        final int pointX = (mConfigShowPopupKeysKeyboardAtTouchedPoint && !keyPreviewEnabled)
                ? CoordinateUtils.x(lastCoords)
                : key.getX() + key.getWidth() / 2;
        // The popup keys keyboard is usually vertically aligned with the top edge of the parent key
        // (plus vertical gap). If the key preview is enabled, the popup keys keyboard is vertically
        // aligned with the bottom edge of the visible part of the key preview.
        // {@code mPreviewVisibleOffset} has been set appropriately in
        // {@link KeyboardView#showKeyPreview(PointerTracker)}.
        final int pointY = key.getY() + mKeyPreviewDrawParams.getVisibleOffset();
        popupKeysKeyboardView.showPopupKeysPanel(this, this, pointX, pointY, mKeyboardActionListener);
        return popupKeysKeyboardView;
    }

    public boolean isInDraggingFinger() {
        if (isShowingPopupKeysPanel()) {
            return true;
        }
        return PointerTracker.isAnyInDraggingFinger();
    }

    @Override
    public void onShowPopupKeysPanel(final PopupKeysPanel panel) {
        locatePreviewPlacerView();
        // Dismiss another {@link PopupKeysPanel} that may be being showed.
        onDismissPopupKeysPanel();
        // Dismiss all key previews that may be being showed.
        PointerTracker.setReleasedKeyGraphicsToAllKeys();
        // Dismiss sliding key input preview that may be being showed.
        mSlidingKeyInputDrawingPreview.dismissSlidingKeyInputPreview();
        panel.showInParent(mDrawingPreviewPlacerView);
        mPopupKeysPanel = panel;
    }

    public boolean isShowingPopupKeysPanel() {
        return mPopupKeysPanel != null && mPopupKeysPanel.isShowingInParent();
    }

    @Override
    public void onCancelPopupKeysPanel() {
        PointerTracker.dismissAllPopupKeysPanels();
    }

    @Override
    public void onDismissPopupKeysPanel() {
        if (isShowingPopupKeysPanel()) {
            mPopupKeysPanel.removeFromParent();
            mPopupKeysPanel = null;
        }
    }

    public void startDoubleTapShiftKeyTimer() {
        mTimerHandler.startDoubleTapShiftKeyTimer();
    }

    public void cancelDoubleTapShiftKeyTimer() {
        mTimerHandler.cancelDoubleTapShiftKeyTimer();
    }

    public boolean isInDoubleTapShiftKeyTimeout() {
        return mTimerHandler.isInDoubleTapShiftKeyTimeout();
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(final MotionEvent event) {
        if (getKeyboard() == null) {
            return false;
        }
        if (mNonDistinctMultitouchHelper != null) {
            if (event.getPointerCount() > 1 && mTimerHandler.isInKeyRepeat()) {
                // Key repeating timer will be canceled if 2 or popup keys are in action.
                mTimerHandler.cancelKeyRepeatTimers();
            }
            // Non distinct multitouch screen support
            mNonDistinctMultitouchHelper.processMotionEvent(event, mKeyDetector);
            return true;
        }
        return processMotionEvent(event);
    }

    public boolean processMotionEvent(final MotionEvent event) {
        final int index = event.getActionIndex();
        final int id = event.getPointerId(index);
        final PointerTracker tracker = PointerTracker.getPointerTracker(id);
        // When a popup keys panel is showing, we should ignore other fingers' single touch events
        // other than the finger that is showing the popup keys panel.
        if (isShowingPopupKeysPanel() && !tracker.isShowingPopupKeysPanel()
                && PointerTracker.getActivePointerTrackerCount() == 1) {
            return true;
        }
        tracker.processMotionEvent(event, mKeyDetector);
        return true;
    }

    public void cancelAllOngoingEvents() {
        mTimerHandler.cancelAllMessages();
        PointerTracker.setReleasedKeyGraphicsToAllKeys();
        mGestureFloatingTextDrawingPreview.dismissGestureFloatingPreviewText();
        mSlidingKeyInputDrawingPreview.dismissSlidingKeyInputPreview();
        PointerTracker.dismissAllPopupKeysPanels();
        PointerTracker.cancelAllPointerTrackers();
    }

    public void closing() {
        cancelAllOngoingEvents();
        mPopupKeysKeyboardCache.clear();
    }

    public void onHideWindow() {
        onDismissPopupKeysPanel();
        final MainKeyboardAccessibilityDelegate accessibilityDelegate = mAccessibilityDelegate;
        if (accessibilityDelegate != null
                && AccessibilityUtils.Companion.getInstance().isAccessibilityEnabled()) {
            accessibilityDelegate.onHideWindow();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean onHoverEvent(final MotionEvent event) {
        final MainKeyboardAccessibilityDelegate accessibilityDelegate = mAccessibilityDelegate;
        if (accessibilityDelegate != null
                && AccessibilityUtils.Companion.getInstance().isTouchExplorationEnabled()) {
            return accessibilityDelegate.onHoverEvent(event);
        }
        return super.onHoverEvent(event);
    }

    public void updateShortcutKey(final boolean available) {
        final Keyboard keyboard = getKeyboard();
        if (keyboard == null) {
            return;
        }
        final Key shortcutKey = keyboard.getKey(KeyCode.VOICE_INPUT);
        if (shortcutKey == null) {
            return;
        }
        shortcutKey.setEnabled(available);
        invalidateKey(shortcutKey);
    }

    public void updateLockState(final int keyCode, final boolean locked) {
        final Keyboard keyboard = getKeyboard();
        if (keyboard == null) {
            return;
        }
        final Key lockKey = keyboard.getKey(keyCode);
        if (lockKey == null) {
            return;
        }
        lockKey.setLocked(locked);
        invalidateKey(lockKey);
    }

    // the whole language on spacebar thing could probably be simplified quite a bit
    public void startDisplayLanguageOnSpacebar(final boolean subtypeChanged,
            final int languageOnSpacebarFormatType,
            final boolean hasMultipleEnabledIMEsOrSubtypes) {
        if (subtypeChanged) {
            KeyPreviewView.clearTextCache();
        }
        mLanguageOnSpacebarFormatType = languageOnSpacebarFormatType;
        mHasMultipleEnabledIMEsOrSubtypes = hasMultipleEnabledIMEsOrSubtypes;
        final ObjectAnimator animator = mLanguageOnSpacebarFadeoutAnimator;
        if (animator == null) {
            mLanguageOnSpacebarFormatType = LanguageOnSpacebarUtils.FORMAT_TYPE_NONE;
        } else {
            if (subtypeChanged
                    && languageOnSpacebarFormatType != LanguageOnSpacebarUtils.FORMAT_TYPE_NONE) {
                setLanguageOnSpacebarAnimAlpha(Constants.Color.ALPHA_OPAQUE);
                if (animator.isStarted()) {
                    animator.cancel();
                }
                animator.start();
            } else {
                if (!animator.isStarted()) {
                    mLanguageOnSpacebarAnimAlpha = mLanguageOnSpacebarFinalAlpha;
                }
            }
        }
        invalidateKey(mSpaceKey);
    }

    @Override
    protected void onDrawKeyTopVisuals(@NonNull final Key key, @NonNull final Canvas canvas,
            @NonNull final Paint paint, @NonNull final KeyDrawParams params) {
        if (key.altCodeWhileTyping() && key.isEnabled()) {
            params.mAnimAlpha = Constants.Color.ALPHA_OPAQUE;
        }
        super.onDrawKeyTopVisuals(key, canvas, paint, params);
        final int code = key.getCode();
        if (code == Constants.CODE_SPACE) {
            // If input language are explicitly selected.
            if (mLanguageOnSpacebarFormatType != LanguageOnSpacebarUtils.FORMAT_TYPE_NONE) {
                drawLanguageOnSpacebar(key, canvas, paint);
            }
            // Whether space key needs to show the "..." popup hint for special purposes
            if (key.isLongPressEnabled() && mHasMultipleEnabledIMEsOrSubtypes && Settings.getValues().mSpaceForLangChange) {
                drawKeyPopupHint(key, canvas, paint, params);
            }
        } else if (code == KeyCode.LANGUAGE_SWITCH) {
            drawKeyPopupHint(key, canvas, paint, params);
        }
    }

    private boolean fitsTextIntoWidth(final int width, final String text, final Paint paint) {
        final int maxTextWidth = width - mLanguageOnSpacebarHorizontalMargin * 2;
        paint.setTextScaleX(1.0f);
        final float textWidth = TypefaceUtils.getStringWidth(text, paint);
        if (textWidth < width) {
            return true;
        }

        final float scaleX = maxTextWidth / textWidth;
        if (scaleX < MINIMUM_XSCALE_OF_LANGUAGE_NAME) {
            return false;
        }

        paint.setTextScaleX(scaleX);
        return TypefaceUtils.getStringWidth(text, paint) < maxTextWidth;
    }

    // Layout language name on spacebar.
    private String layoutLanguageOnSpacebar(final Paint paint,
            final RichInputMethodSubtype subtype, final int width) {
        // Choose appropriate language name to fit into the width.

        final List<Locale> secondaryLocales = Settings.getValues().mSecondaryLocales;
        // avoid showing same language twice
        final List<Locale> secondaryLocalesToUse = withoutDuplicateLanguages(secondaryLocales, subtype.getLocale().getLanguage());
        if (!secondaryLocalesToUse.isEmpty()) {
            StringBuilder sb = new StringBuilder(subtype.getMiddleDisplayName());
            final Locale displayLocale = ConfigurationCompatKt.locale(getResources().getConfiguration());
            for (Locale locale : secondaryLocales) {
                sb.append(" - ");
                sb.append(locale.getDisplayLanguage(displayLocale));
            }
            final String full = sb.toString();
            if (fitsTextIntoWidth(width, full, paint)) {
                return full;
            }
            sb.setLength(0);
            sb.append(subtype.getLocale().getLanguage().toUpperCase(displayLocale));
            for (Locale locale : secondaryLocales) {
                sb.append(" - ");
                sb.append(locale.getLanguage().toUpperCase(displayLocale));
            }
            final String middle = sb.toString();
            if (fitsTextIntoWidth(width, middle, paint)) {
                return middle;
            }
        }

        if (mLanguageOnSpacebarFormatType == LanguageOnSpacebarUtils.FORMAT_TYPE_FULL_LOCALE) {
            final String fullText = subtype.getFullDisplayName();
            if (fitsTextIntoWidth(width, fullText, paint)) {
                return fullText;
            }
        }

        final String middleText = subtype.getMiddleDisplayName();
        if (fitsTextIntoWidth(width, middleText, paint)) {
            return middleText;
        }

        return "";
    }

    private List<Locale> withoutDuplicateLanguages(List<Locale> locales, String mainLanguage) {
        ArrayList<String> languages = new ArrayList<String>() {{ add(mainLanguage); }};
        ArrayList<Locale> newLocales = new ArrayList<>();
        for (Locale locale : locales) {
            boolean keep = true;
            for (String language : languages) {
                if (locale.getLanguage().equals(language))
                    keep = false;
            }
            if (!keep)
                continue;
            languages.add(locale.getLanguage());
            newLocales.add(locale);
        }
        return newLocales;
    }

    private void drawLanguageOnSpacebar(final Key key, final Canvas canvas, final Paint paint) {
        final Keyboard keyboard = getKeyboard();
        if (keyboard == null) {
            return;
        }
        final int width = key.getWidth();
        final int height = key.getHeight();
        paint.setTextAlign(Align.CENTER);
        paint.setTypeface(mTypeface == null ? Typeface.DEFAULT : mTypeface);
        paint.setTextSize(mLanguageOnSpacebarTextSize);
        final String customText = Settings.getValues().mSpaceBarText;
        final String spaceText;
        if (!customText.isEmpty()) {
            spaceText = customText;
        } else if (DebugFlags.DEBUG_ENABLED) {
            final String l = KeyboardSwitcher.getInstance().getLocaleAndConfidenceInfo();
            spaceText = l != null ? l : layoutLanguageOnSpacebar(paint, keyboard.mId.mSubtype, width);
        }
        else
            spaceText = layoutLanguageOnSpacebar(paint, keyboard.mId.mSubtype, width);
        // Draw language text with shadow
        final float descent = paint.descent();
        final float textHeight = -paint.ascent() + descent;
        final float baseline = height / 2f + textHeight / 2;
        if (mLanguageOnSpacebarTextShadowRadius > 0.0f) {
            paint.setShadowLayer(mLanguageOnSpacebarTextShadowRadius, 0, 0,
                    mLanguageOnSpacebarTextShadowColor);
        } else {
            paint.clearShadowLayer();
        }
        paint.setColor(mLanguageOnSpacebarTextColor);
        paint.setAlpha(mLanguageOnSpacebarAnimAlpha);
        if (!fitsTextIntoWidth(width, spaceText, paint)) {
            final float textWidth = TypefaceUtils.getStringWidth(spaceText, paint);
            paint.setTextScaleX((width - mLanguageOnSpacebarHorizontalMargin * 2) / textWidth);
        }
        canvas.drawText(spaceText, width / 2f, baseline - descent, paint);
        paint.clearShadowLayer();
        paint.setTextScaleX(1.0f);
    }

    @Override
    public void deallocateMemory() {
        super.deallocateMemory();
        mDrawingPreviewPlacerView.deallocateMemory();
    }
}
