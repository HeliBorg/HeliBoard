/*
 * Copyright (C) 2008 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.keyboard

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.inputmethodservice.InputMethodService
import android.os.Build
import android.view.ContextThemeWrapper
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup.MarginLayoutParams
import android.view.animation.AnimationUtils
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodSubtype
import android.widget.HorizontalScrollView
import android.widget.Toast
import androidx.core.view.isGone
import androidx.core.view.isVisible
import helium314.keyboard.event.Event
import helium314.keyboard.keyboard.KeyboardLayoutSet.Companion.KeyboardLayoutSetException
import helium314.keyboard.keyboard.KeyboardTheme.Companion.getKeyboardTheme
import helium314.keyboard.keyboard.clipboard.ClipboardHistoryView
import helium314.keyboard.keyboard.emoji.EmojiPalettesView
import helium314.keyboard.keyboard.internal.KeyboardState
import helium314.keyboard.keyboard.internal.LayoutDirective
import helium314.keyboard.keyboard.internal.ShiftMode
import helium314.keyboard.keyboard.internal.keyboard_parser.loadEmojiDefaultVersionsAndPopupSpecs
import helium314.keyboard.latin.CapsMode
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.RichInputMethodSubtype
import helium314.keyboard.latin.common.InsetsOutlineProvider
import helium314.keyboard.latin.databinding.InputViewBinding
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValues
import helium314.keyboard.latin.settings.setFloatingKeyboardEnabled
import helium314.keyboard.latin.utils.CapsModeUtils
import helium314.keyboard.latin.utils.FloatingKeyboardUtils
import helium314.keyboard.latin.utils.FoldableUtils.isFolded
import helium314.keyboard.latin.utils.LanguageOnSpacebarUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.RecapitalizeMode
import helium314.keyboard.latin.utils.ResourceUtils
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional.createDefaultSubtype
import helium314.keyboard.latin.utils.ToolbarMode
import helium314.keyboard.latin.utils.dpToPx
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.updateSoftInputWindowLayoutParameters

class KeyboardSwitcher private constructor() {
    // created in onCreateInputView (called by LatinIME before starting input), i.e. might not exist initially
    // when using a hardware keyboard onCreateInputView might not get called at all
    private var inputView: InputViewBinding? = null

    private val insetsUpdater: InsetsOutlineProvider = InsetsOutlineProvider()

    // clipboard & emoji views are initially view stubs, and only created when needed
    // the inflated views are not accessible from the binding (really?), so we manage them separately
    private val InputViewBinding.clipboardHistoryView: ClipboardHistoryView get() {
        clipboardHistory?.let { return it }
        val view = mainKeyboardFrame.clipboardHistory.inflate() as ClipboardHistoryView
        view.setHardwareAcceleratedDrawingEnabled(isHardwareAcceleratedDrawingEnabled)
        view.keyboardActionListener = latinIME.mKeyboardActionListener
        latinIME.prefs().registerOnSharedPreferenceChangeListener(view)
        clipboardHistory = view
        return view
    }
    private var clipboardHistory: ClipboardHistoryView? = null
    private val InputViewBinding.emojiPalettesView: EmojiPalettesView get() {
        emojiPalettes?.let { return it }
        val view = mainKeyboardFrame.emojiPalettes.inflate() as EmojiPalettesView
        view.setHardwareAcceleratedDrawingEnabled(isHardwareAcceleratedDrawingEnabled)
        view.setKeyboardActionListener(latinIME.mKeyboardActionListener)
        emojiPalettes = view
        return view
    }
    private var emojiPalettes: EmojiPalettesView? = null

    // created on init, i.e. only available when set as active input method
    private lateinit var latinIME: LatinIME
    private lateinit var richImm: RichInputMethodManager
    private var isHardwareAcceleratedDrawingEnabled = false
    private lateinit var state: KeyboardState

    private lateinit var themeContext: Context // initialized together with inputView

    // currently used KeyboardLayoutSet, null until first call to loadKeyboard
    private var keyboardLayoutSet: KeyboardLayoutSet? = null

    val mainKeyboardView get() = inputView?.keyboardView
    val keyboard get() = mainKeyboardView?.keyboard
    val wrapperView get() = inputView?.wrapperView
    val stripContainer get() = inputView?.stripContainer
    val emojiTabStrip get() = inputView?.emojiTabStrip
    val clipboardStrip get() = inputView?.mainKeyboardFrame?.stripContainer?.clipboardStrip
    val isShowingEmojiPalettes get() = emojiPalettes?.isShown == true
    val isShowingClipboardHistory get() = clipboardHistory?.isShown == true
    val isShowingPopupKeysPanel get () = !isShowingEmojiPalettes && !isShowingClipboardHistory && mainKeyboardView?.isShowingPopupKeysPanel == true
    val visibleKeyboardView get() = when {
        isShowingEmojiPalettes -> inputView?.emojiPalettesView
        isShowingClipboardHistory -> inputView?.clipboardHistoryView
        else -> mainKeyboardView
    }
    val keyboardCapsMode get() = keyboard?.mId?.element?.capsMode ?: CapsMode.OFF
    val localeAndConfidenceInfo: String? get() = latinIME.localeAndConfidenceInfo // used for debug
    val keyboardSwitchState: KeyboardSwitchState get() = when {
        isShowingEmojiPalettes -> KeyboardSwitchState.EMOJI
        isShowingClipboardHistory -> KeyboardSwitchState.CLIPBOARD
        (keyboardLayoutSet == null || inputView?.keyboardView?.isShown != true) && stripContainer?.isShown != true -> KeyboardSwitchState.HIDDEN
        isShowingKeyboardId(KeyboardElement.SYMBOLS_SHIFTED) -> KeyboardSwitchState.SYMBOLS_SHIFTED
        else -> KeyboardSwitchState.OTHER
    }

    private fun initInternal(latinIme: LatinIME) {
        latinIME = latinIme
        richImm = RichInputMethodManager.getInstance()
        state = KeyboardState(SwitchActions())
        isHardwareAcceleratedDrawingEnabled = latinIME.enableHardwareAcceleration()
    }

    fun onCreateInputView(displayContext: Context, isHardwareAcceleratedDrawingEnabled: Boolean): View {
        Log.d(TAG, "create new input view")
        val prefs = displayContext.prefs()
        inputView?.keyboardView?.closing()
        clipboardHistory?.let { prefs.unregisterOnSharedPreferenceChangeListener(it) }
        inputView?.suggestionStripView?.let { prefs.unregisterOnSharedPreferenceChangeListener(it) }
        PointerTracker.clearOldViewData()
        if (themeNeedsReload) // necessary in some cases (e.g. theme switch) when themeNeedsReload is set before first keyboard load
            Settings.getInstance().loadSettings(displayContext, Settings.getValues().mLocale, Settings.getValues().mInputAttributes)

        updateKeyboardThemeAndContextThemeWrapper(displayContext, getKeyboardTheme(displayContext))
        val inputView = InputViewBinding.inflate(LayoutInflater.from(themeContext))
        this.inputView = inputView
        emojiPalettes = null
        clipboardHistory = null
        inputView.wrapperView.keyboardActionListener = latinIME.mKeyboardActionListener
        inputView.keyboardView.setHardwareAcceleratedDrawingEnabled(isHardwareAcceleratedDrawingEnabled)
        inputView.keyboardView.setKeyboardActionListener(latinIME.mKeyboardActionListener)
        inputView.root.outlineProvider = insetsUpdater

        prefs.registerOnSharedPreferenceChangeListener(inputView.suggestionStripView)
        PointerTracker.switchTo(inputView.keyboardView)
        return inputView.root
    }

    /** Marks the theme as outdated. The theme will be reloaded next time the keyboard is shown.
     * If the keyboard is currently showing, theme will be reloaded immediately.  */
    fun setThemeNeedsReload() {
        themeNeedsReload = true
        if (inputView == null || !latinIME.isInputViewShown) return  // will be reloaded right before showing IME

        // Hide and show IME, showing will trigger the reload.
        // Reloading while IME is shown is glitchy, and hiding / showing is so fast the user shouldn't notice.
        latinIME.hideWindow()
        try {
            latinIME.showWindow(true)
        } catch (e: IllegalStateException) {
            // in tests isInputViewShown returns true, but showWindow throws "IllegalStateException: Window token is not set yet."
        }
    }

    fun updateKeyboardTheme(displayContext: Context) {
        val themeUpdated = updateKeyboardThemeAndContextThemeWrapper(displayContext, getKeyboardTheme(displayContext))
        if (themeUpdated) {
            val settings = Settings.getInstance()
            settings.loadSettings(displayContext, settings.current.mLocale, settings.current.mInputAttributes)
            if (inputView != null) latinIME.setInputView(onCreateInputView(displayContext, isHardwareAcceleratedDrawingEnabled))
        } else if (inputView != null
            && (latinIME.hasSuggestionStripView() == (Settings.getValues().mToolbarMode == ToolbarMode.HIDDEN || latinIME.isEmojiSearch))) {
            latinIME.updateSuggestionStripView(inputView!!.root)
        }
    }

    // variables to determine whether a reload is necessary (usually because of theme switch)
    private var keyboardTheme: KeyboardTheme? = null
    private var currentUiMode = 0
    private var currentOrientation = 0
    private var currentDpi = 0
    private var themeNeedsReload = false
    private fun updateKeyboardThemeAndContextThemeWrapper(context: Context, theme: KeyboardTheme): Boolean {
        val res = context.resources
        val conf = res.configuration
        if (themeNeedsReload
            || theme != keyboardTheme
            || currentDpi != res.displayMetrics.densityDpi
            || currentOrientation != conf.orientation
            || (currentUiMode and Configuration.UI_MODE_NIGHT_MASK) != (conf.uiMode and Configuration.UI_MODE_NIGHT_MASK)
            || inputView == null
            || (themeContext.resources != res)
            || Settings.getValues().mColors.haveColorsChanged(context)
        ) {
            themeNeedsReload = false
            keyboardTheme = theme
            themeContext = ContextThemeWrapper(context, theme.mStyleId)
            currentUiMode = conf.uiMode
            currentOrientation = conf.orientation
            currentDpi = res.displayMetrics.densityDpi
            KeyboardLayoutSet.onKeyboardThemeChanged()
            return true
        }
        return false
    }

    fun updateInsets(outInsets: InputMethodService.Insets) {
        // This method may be called before onCreateInputView.
        val inputView = inputView ?: return

        val visibleKeyboardView = inputView.wrapperView
        val inputHeight = inputView.root.height
        if (isImeSuppressedByHardwareKeyboard(Settings.getValues(), keyboardSwitchState) && !visibleKeyboardView.isShown) {
            // If there is a hardware keyboard and a visible software keyboard view has been hidden,
            // no visual element will be shown on the screen.
            // for some reason setting contentTopInsets and visibleTopInsets broke somewhere along the
            // way from OpenBoard to HeliBoard (GH-702, GH-1455), but not setting anything seems to work
            insetsUpdater.setInsets(outInsets, inputView.root)
            return
        }
        val stripContainer = inputView.stripContainer
        val stripHeight = if (stripContainer.isShown) stripContainer.height else 0
        var visibleTopY = inputHeight - (if (visibleKeyboardView.isShown) visibleKeyboardView.height else 0) - stripHeight
        if (Settings.getValues().mIsFloatingKeyboard)
            visibleTopY = themeContext.resources.displayMetrics.heightPixels

        if (hasSuggestionStripView()) {
            inputView.suggestionStripView.setMoreSuggestionsHeight(visibleTopY)
        }


        // Need to set expanded touchable region only if a keyboard view is being shown.
        // When using a floating keyboard and the showToolbarOnly setting, the wrapper view may be hidden but we still have a view
        if (visibleKeyboardView.isShown || (Settings.getValues().mIsFloatingKeyboard && inputView.root.isShown)) {
            var touchLeft = 0
            var touchTop = if (isShowingPopupKeysPanel) 0 else visibleTopY
            var touchRight = visibleKeyboardView.width
            var touchBottom = inputHeight + EXTENDED_TOUCHABLE_REGION_HEIGHT // Extend touchable region below the keyboard.
            if (Settings.getValues().mIsFloatingKeyboard) {
                val xy = FloatingKeyboardUtils.readPosition(themeContext, Int.MAX_VALUE, Int.MAX_VALUE)
                touchLeft = xy.first
                touchTop = xy.second
                touchRight = touchLeft + Settings.getValues().mFloatingWidth
                touchBottom = touchTop + inputHeight
            }
            outInsets.touchableInsets = InputMethodService.Insets.TOUCHABLE_INSETS_REGION
            outInsets.touchableRegion.set(touchLeft, touchTop, touchRight, touchBottom)
        }


        // Has to be subtracted after calculating touchableRegion
        visibleTopY -= latinIME.emojiSearchActivityHeight

        outInsets.contentTopInsets = visibleTopY
        outInsets.visibleTopInsets = visibleTopY
        insetsUpdater.setInsets(outInsets, inputView.root)
    }

    private fun hasSuggestionStripView() = Settings.getValues().mToolbarMode != ToolbarMode.HIDDEN && !latinIME.isEmojiSearch

    fun loadKeyboard(
        editorInfo: EditorInfo?, settingsValues: SettingsValues, currentAutoCapsState: Int,
        currentRecapitalizeState: RecapitalizeMode?, internalAction: KeyboardLayoutSet.InternalAction?
    ) {
        val builder = KeyboardLayoutSet.Builder(themeContext, editorInfo, settingsValues)
        val keyboardWidth = ResourceUtils.getKeyboardWidth(themeContext, settingsValues)
        val keyboardHeight = ResourceUtils.getKeyboardHeight(themeContext.resources, settingsValues)
        keyboardLayoutSet = builder.setKeyboardGeometry(keyboardWidth, keyboardHeight)
            .setSubtype(richImm.currentSubtype)
            .setInternalAction(internalAction)
            .build()
        try {
            state.onLoadKeyboard(currentAutoCapsState, currentRecapitalizeState, settingsValues.mOneHandedModeEnabled)
        } catch (e: KeyboardLayoutSetException) {
            Log.e(TAG, "loading keyboard failed: ${e.keyboardId}", e.cause)
            try {
                keyboardLayoutSet = builder.setKeyboardGeometry(keyboardWidth, keyboardHeight)
                    .setSubtype(RichInputMethodSubtype.get(createDefaultSubtype(richImm.currentSubtypeLocale)))
                    .build()
                state.onLoadKeyboard(currentAutoCapsState, currentRecapitalizeState, false)
                showToast("error loading the keyboard, falling back to defaults", false)
            } catch (e2: KeyboardLayoutSetException) {
                Log.e(TAG, "even fallback to defaults failed: ${e2.keyboardId}", e2.cause)
            }
        }
    }

    fun onHideWindow() {
        inputView?.keyboardView?.onHideWindow()
    }

    fun onWindowShown() {
        if (Settings.getValues().mIsFloatingKeyboard)
            inputView?.let { FloatingKeyboardUtils.setFloating(it.root) }
    }

    fun updateSoftInputWindowLayoutParameters() = latinIME.updateSoftInputWindowLayoutParameters(inputView?.root)

    private fun setMainKeyboardFrame(settingsValues: SettingsValues, toggleState: KeyboardSwitchState) {
        val inputView = inputView ?: return
        val isGone = isImeSuppressedByHardwareKeyboard(settingsValues, toggleState)
        inputView.stripContainer.isVisible = latinIME.hasSuggestionStripView()
        PointerTracker.switchTo(inputView.keyboardView)
        inputView.keyboardView.isGone = isGone
        inputView.keyboardFrame.isGone = isGone
        inputView.wrapperView.isGone = Settings.getInstance().readShowToolbarOnly() && toggleState == KeyboardSwitchState.OTHER
        emojiPalettes?.isGone = true
        emojiPalettes?.stopEmojiPalettes()
        inputView.emojiTabStrip.isGone = true
        inputView.clipboardStripScrollView.isGone = true
        inputView.suggestionStripView.isVisible = latinIME.hasSuggestionStripView()
        clipboardHistory?.isGone = true
        clipboardHistory?.stopClipboardHistory()
    }

    fun onToggleKeyboard(toggleState: KeyboardSwitchState) {
        val currentState = keyboardSwitchState
        Log.w(TAG, "onToggleKeyboard() : Current = $currentState : Toggle = $toggleState")
        if (currentState == toggleState) {
            if (!Settings.getInstance().readShowToolbarOnly()) {
                latinIME.stopShowingInputView()
                latinIME.hideWindow()
            }
            resetKeyboardStateToAlphabet()
        } else {
            latinIME.startShowingInputView(true)
            if (toggleState == KeyboardSwitchState.EMOJI) {
                setEmojiKeyboard()
            } else if (toggleState == KeyboardSwitchState.CLIPBOARD) {
                setClipboardKeyboard()
            } else {
                emojiPalettes?.stopEmojiPalettes()
                emojiPalettes?.isGone = true

                clipboardHistory?.stopClipboardHistory()
                clipboardHistory?.isGone = true

                inputView?.keyboardFrame?.isGone = false
                inputView?.keyboardView?.isGone = false
                if (toggleState == KeyboardSwitchState.SYMBOLS_SHIFTED)
                    // possible other states OTHER and HIDDEN have keyboardElement null, which we just ignore
                    // might need to be adjusted when functionality is extended
                    state.setLayout(LayoutDirective.Utility.SYMBOLS_SHIFTED)
            }
        }
    }

    fun setOneHandedModeEnabled(enabled: Boolean, force: Boolean) {
        if (inputView == null) return
        if (!force && inputView?.wrapperView?.oneHandedModeEnabled == enabled) {
            return
        }
        val settings = Settings.getInstance()
        inputView?.wrapperView?.oneHandedModeEnabled = enabled
        inputView?.wrapperView?.oneHandedGravity = settings.current.mOneHandedModeGravity

        // oneHandeMode is always disabled when floating, and we shouldn't mess up the setting
        if (enabled != settings.current.mOneHandedModeEnabled) settings.writeOneHandedModeEnabled(enabled)
        reloadKeyboard()
    }

    fun toggleSplitKeyboardMode() {
        val settings = Settings.getInstance()
        settings.writeSplitKeyboardEnabled(
            !settings.current.mIsSplitKeyboardEnabled,
            currentOrientation == Configuration.ORIENTATION_LANDSCAPE,
            isFolded
        )
        setOneHandedModeEnabled(settings.current.mOneHandedModeEnabled, true)
        reloadKeyboard()
    }

    fun reloadKeyboard() {
        if (inputView == null) return
        emojiPalettes?.clearKeyboardCache()
        reloadMainKeyboard()
    }

    fun reloadMainKeyboard() {
        // Reload the entire keyboard, and switch to the previous layout
        val wasEmoji = isShowingEmojiPalettes
        val wasClipboard = isShowingClipboardHistory
        loadKeyboard(
            latinIME.currentInputEditorInfo, Settings.getValues(),
            latinIME.currentAutoCapsState, latinIME.currentRecapitalizeState, null
        )
        if (wasEmoji) {
            setEmojiKeyboard()
        } else if (wasClipboard) {
            setClipboardKeyboard()
        }
    }

    /**
     * Displays a toast message.
     *
     * @param text The text to display in the toast message.
     * @param briefToast If true, the toast duration will be short; otherwise, it will last longer.
     */
    fun showToast(text: String, briefToast: Boolean) {
        // In API 32 and below, toasts can be shown without a notification permission.
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
            val toastLength = if (briefToast) Toast.LENGTH_SHORT else Toast.LENGTH_LONG
            val toast = Toast.makeText(latinIME, text, toastLength)
            toast.setGravity(Gravity.CENTER, 0, 0)
            toast.show()
        } else {
            val toastLength = if (briefToast) 2000 else 3500
            showFakeToast(text, toastLength)
        }
    }

    // Displays a toast-like message with the provided text for a specified duration.
    private fun showFakeToast(text: String, timeMillis: Int) {
        val fakeToast = inputView?.fakeToast?.root ?: return
        if (fakeToast.isVisible) return

        val appIcon = fakeToast.compoundDrawables[0]
        if (appIcon != null) {
            val bound = fakeToast.lineHeight
            appIcon.setBounds(0, 0, bound, bound)
            fakeToast.setCompoundDrawables(appIcon, null, null, null)
        }
        fakeToast.text = text
        fakeToast.isGone = false
        fakeToast.bringToFront()
        fakeToast.startAnimation(AnimationUtils.loadAnimation(latinIME, R.anim.fade_in))

        fakeToast.postDelayed({
            fakeToast.startAnimation(AnimationUtils.loadAnimation(latinIME, R.anim.fade_out))
            fakeToast.isGone = true
        }, timeMillis.toLong())
    }

    fun setBackgroundGatheringIndicator(enabled: Boolean, hasData: Boolean, saving: Boolean) {
        val backgroundGatheringIndicator = inputView?.mainKeyboardFrame?.backgroundGatheringIndicator ?: return
        backgroundGatheringIndicator.isVisible = enabled
        if (!enabled) return
        backgroundGatheringIndicator.setImageResource(if (hasData) R.drawable.btn_keyboard_key_action_normal_lxx_base else R.drawable.ring)
        setBackgroundGatheringIndicatorPosition()
        if (!saving) return
        backgroundGatheringIndicator.imageTintList = ColorStateList.valueOf(-0xff6000)
        backgroundGatheringIndicator.postDelayed(
            { backgroundGatheringIndicator.imageTintList = ColorStateList.valueOf(-0x600000) },
            1500
        )
    }

    private fun setBackgroundGatheringIndicatorPosition() {
        val backgroundGatheringIndicator = inputView?.mainKeyboardFrame?.backgroundGatheringIndicator ?: return
        if (!backgroundGatheringIndicator.isVisible) return
        val margin = backgroundGatheringIndicator.layoutParams as? MarginLayoutParams ?: return
            val kb = keyboard
            if (kb != null) margin.topMargin = kb.mOccupiedHeight - 16.dpToPx(backgroundGatheringIndicator.resources)
            backgroundGatheringIndicator.layoutParams = backgroundGatheringIndicator.layoutParams
    }

    fun isShowingKeyboardId(vararg keyboardElements: KeyboardElement): Boolean {
        if (mainKeyboardView?.isShown != true) {
            return false
        }
        val activeKeyboardElement = keyboard?.mId?.element ?: return false // may happen when using hardware keyboard
        return keyboardElements.contains(activeKeyboardElement)
    }

    fun deallocateMemory() {
        val inputView = inputView ?: return
        inputView.keyboardView.cancelAllOngoingEvents()
        inputView.keyboardView.deallocateMemory()
        emojiPalettes?.stopEmojiPalettes()
        clipboardHistory?.stopClipboardHistory()
    }

    fun trimMemory() {
        emojiPalettes?.clearKeyboardCache()
    }

    fun switchToSubtype(subtype: InputMethodSubtype?) {
        latinIME.switchToSubtype(subtype)
    }

    // TODO: Remove this method. Come up with a more comprehensive way to reset the keyboard layout
    // when a keyboard layout set doesn't get reloaded in LatinIME.onStartInputViewInternal().
    fun resetKeyboardStateToAlphabet() =
        state.onResetKeyboardStateToAlphabet(latinIME.currentAutoCapsState, latinIME.currentRecapitalizeState)

    fun onPressKey(code: Int, pointerCount: Int, currentAutoCapsState: Int, currentRecapitalizeState: RecapitalizeMode?) =
        state.onPressKey(code, pointerCount, currentAutoCapsState, currentRecapitalizeState)

    fun onReleaseKey(code: Int, withSliding: Boolean, currentAutoCapsState: Int, currentRecapitalizeState: RecapitalizeMode?) =
        state.onReleaseKey(code, withSliding, currentAutoCapsState, currentRecapitalizeState)

    fun onFinishSlidingInput(currentAutoCapsState: Int, currentRecapitalizeState: RecapitalizeMode?) =
        state.onFinishSlidingInput(currentAutoCapsState, currentRecapitalizeState)

    fun setEmojiKeyboard() = state.setLayout(LayoutDirective.Utility.EMOJI)

    fun setClipboardKeyboard() = state.setLayout(LayoutDirective.Utility.CLIPBOARD)

    fun toggleLayout(layout: LayoutDirective.Utility, autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) =
        state.toggleLayout(layout, autoCapsFlags, recapitalizeMode)

    fun onLongPressAlphaSymbolForNumpad() {
        if (KeyboardState.SwitchActions.DEBUG_ACTION) {
            Log.d(TAG, "onLongPressAlphaSymbol")
        }
        state.onLongPressAlphaSymbolForNumpad()
    }

    fun updateShiftState(autoCapsFlags: Int, recapitalizeMode: RecapitalizeMode?) {
        if (KeyboardState.SwitchActions.DEBUG_ACTION) {
            Log.d(TAG, "updateShiftState: autoCapsFlags=${CapsModeUtils.flagsToString(autoCapsFlags)} recapitalizeMode=$recapitalizeMode")
        }
        state.onUpdateShiftState(autoCapsFlags, recapitalizeMode)
    }

    fun saveKeyboardState() {
        if (keyboard != null || isShowingEmojiPalettes || isShowingClipboardHistory) {
            state.onSaveKeyboardState()
        }
    }

    /**
     * Updates state machine to figure out when to automatically switch back to the previous mode.
     */
    fun onEvent(event: Event, currentAutoCapsState: Int, currentRecapitalizeState: RecapitalizeMode?) {
        state.onEvent(event, currentAutoCapsState, currentRecapitalizeState)
    }

    // private SwitchActions implementation so e.g. setEmojiKeyboard can only be called via KeyboardState (avoid inconsistencies!)
    private inner class SwitchActions : KeyboardState.SwitchActions {
        override fun setAlphabetKeyboard(shiftMode: ShiftMode) {
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setAlphabetKeyboard")
            }
            setKeyboard(shiftMode.element, KeyboardSwitchState.OTHER)
        }

        override fun setSymbolsKeyboard() {
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setSymbolsKeyboard")
            }
            setKeyboard(KeyboardElement.SYMBOLS, KeyboardSwitchState.OTHER)
        }

        override fun setSymbolsShiftedKeyboard() {
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setSymbolsShiftedKeyboard")
            }
            setKeyboard(KeyboardElement.SYMBOLS_SHIFTED, KeyboardSwitchState.SYMBOLS_SHIFTED)
        }

        override fun setEmojiKeyboard() {
            val inputView = inputView ?: return
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setEmojiKeyboard")
            }
            inputView.wrapperView.isGone = false
            inputView.keyboardFrame.isGone = false
            inputView.keyboardView.isGone = true
            inputView.suggestionStripView.isGone = true
            inputView.stripContainer.isVisible = Settings.getValues().isSecondaryStripVisible
            inputView.clipboardStripScrollView.isGone = true
            inputView.emojiTabStrip.isGone = false
            clipboardHistory?.isGone = true
            inputView.emojiPalettesView.startEmojiPalettes(
                inputView.keyboardView.keyVisualAttribute,
                latinIME.currentInputEditorInfo, latinIME.mKeyboardActionListener
            )
            inputView.emojiPalettesView.isGone = false
        }

        override fun setClipboardKeyboard() {
            val inputView = inputView ?: return
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setClipboardKeyboard")
            }
            inputView.wrapperView.isGone = false
            inputView.keyboardFrame.isGone = false
            inputView.keyboardView.isGone = true
            inputView.emojiTabStrip.isGone = true
            inputView.suggestionStripView.isGone = true
            inputView.stripContainer.isVisible = Settings.getValues().isSecondaryStripVisible
            inputView.clipboardStripScrollView.post { inputView.clipboardStripScrollView.fullScroll(HorizontalScrollView.FOCUS_RIGHT) }
            inputView.clipboardStripScrollView.isGone = false
            emojiPalettes?.isGone = true
            inputView.clipboardHistoryView.startClipboardHistory(
                latinIME.clipboardHistoryManager, inputView.keyboardView.keyVisualAttribute,
                latinIME.currentInputEditorInfo, latinIME.mKeyboardActionListener
            )
            inputView.clipboardHistoryView.isGone = false
        }

        override fun setNumpadKeyboard() {
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setNumpadKeyboard")
            }
            setKeyboard(KeyboardElement.NUMPAD, KeyboardSwitchState.OTHER)
        }

        override fun setDpadKeyboard() {
            if (KeyboardState.SwitchActions.DEBUG_ACTION) {
                Log.d(TAG, "setDpadKeyboard")
            }
            setKeyboard(KeyboardElement.DPAD, KeyboardSwitchState.OTHER)
        }

        override fun startDoubleTapShiftKeyTimer() {
            if (KeyboardState.SwitchActions.DEBUG_TIMER_ACTION) {
                Log.d(TAG, "startDoubleTapShiftKeyTimer")
            }
            mainKeyboardView?.startDoubleTapShiftKeyTimer()
        }

        override fun cancelDoubleTapShiftKeyTimer() {
            if (KeyboardState.SwitchActions.DEBUG_TIMER_ACTION) {
                Log.d(TAG, "cancelDoubleTapShiftKeyTimer")
            }
            mainKeyboardView?.cancelDoubleTapShiftKeyTimer()
        }

        override fun setOneHandedModeEnabled(enabled: Boolean) {
            this@KeyboardSwitcher.setOneHandedModeEnabled(enabled, false)
        }

        override fun switchOneHandedMode() {
            val wrapperView = inputView?.wrapperView ?: return
            wrapperView.switchOneHandedModeSide()
            Settings.getInstance().writeOneHandedModeGravity(wrapperView.oneHandedGravity)
        }

        override fun setFloatingKeyboardEnabled(enabled: Boolean) {
            val inputView = inputView ?: return
            if (enabled != Settings.getValues().mIsFloatingKeyboard)
                // mIsFloatingKeyboard is always disabled when device is locked, and we shouldn't mess up the setting
                setFloatingKeyboardEnabled(themeContext, enabled)
            if (enabled) FloatingKeyboardUtils.setFloating(inputView.root)
            else FloatingKeyboardUtils.disableFloating(inputView.root)
            setBackgroundGatheringIndicatorPosition()
        }

        override fun popDoubleTapShiftKeyTimer(): Boolean {
            if (KeyboardState.SwitchActions.DEBUG_TIMER_ACTION) {
                Log.d(TAG, "isInDoubleTapShiftKeyTimeout")
            }
            return mainKeyboardView?.popDoubleTapShiftKeyTimer() ?: false
        }

        // not a SwitchAction, but should only be called from a SwitchAction to avoid inconsistent state / actual layout
        fun setKeyboard(keyboardElement: KeyboardElement, toggleState: KeyboardSwitchState) {
            // with a hardware keyboard we might get here without ever calling onCreateInputView, so don't crash
            val inputView = inputView ?: return

            // Make {@link MainKeyboardView} visible and hide {@link EmojiPalettesView}.
            val currentSettingsValues = Settings.getValues()
            setMainKeyboardFrame(currentSettingsValues, toggleState)
            // TODO: pass this object to setKeyboard instead of getting the current values.
            val oldKeyboard = inputView.keyboardView.keyboard
            val newKeyboard = keyboardLayoutSet!!.getKeyboard(keyboardElement)
            inputView.keyboardView.setKeyboard(newKeyboard)
            inputView.root.setKeyboardTopPadding(newKeyboard.mTopPadding)
            inputView.keyboardView.setKeyPreviewPopupEnabled(currentSettingsValues.mKeyPreviewPopupOn)
            inputView.keyboardView.updateShortcutKey(richImm.isShortcutImeReady)
            val subtypeChanged = (oldKeyboard == null) || newKeyboard.mId.subtype != oldKeyboard.mId.subtype
            val languageOnSpacebarFormatType = LanguageOnSpacebarUtils.getLanguageOnSpacebarFormatType(newKeyboard.mId.subtype)
            val hasMultipleEnabledIMEsOrSubtypes = richImm.hasMultipleEnabledIMEsOrSubtypes(true)
            inputView.keyboardView.startDisplayLanguageOnSpacebar(subtypeChanged, languageOnSpacebarFormatType, hasMultipleEnabledIMEsOrSubtypes)

            if (currentSettingsValues.needsToLookupSuggestions()
                && (currentSettingsValues.mInlineEmojiSearch || currentSettingsValues.mSuggestEmojis)
            ) {
                loadEmojiDefaultVersionsAndPopupSpecs(themeContext)
            }
        }
    }

    enum class KeyboardSwitchState(val keyboardElement: KeyboardElement?) {
        HIDDEN(null),
        SYMBOLS_SHIFTED(KeyboardElement.SYMBOLS_SHIFTED),
        EMOJI(KeyboardElement.EMOJI_RECENTS),
        CLIPBOARD(KeyboardElement.CLIPBOARD),
        OTHER(null)
    }

    companion object {
        private val TAG = KeyboardSwitcher::class.java.simpleName
        private const val EXTENDED_TOUCHABLE_REGION_HEIGHT = 100

        @SuppressLint("StaticFieldLeak") // todo: actually we would want the instance to go down with LatinIME
        private val instance = KeyboardSwitcher()

        @JvmStatic
        fun getInstance() = instance

        @JvmStatic
        fun init(latinIme: LatinIME) {
            instance.initInternal(latinIme)
        }

        @JvmStatic
        fun isImeSuppressedByHardwareKeyboard(settingsValues: SettingsValues, toggleState: KeyboardSwitchState) =
            settingsValues.mHasHardwareKeyboard && toggleState == KeyboardSwitchState.HIDDEN

        private val InputViewBinding.wrapperView get() = mainKeyboardFrame.keyboardViewWrapper
        private val InputViewBinding.keyboardFrame get() = mainKeyboardFrame.root
        private val InputViewBinding.keyboardView get() = mainKeyboardFrame.keyboardView
        private val InputViewBinding.emojiTabStrip get() = mainKeyboardFrame.stripContainer.emojiTabStrip
        private val InputViewBinding.clipboardStripScrollView get() = mainKeyboardFrame.stripContainer.clipboardStripScrollView
        private val InputViewBinding.suggestionStripView get() = mainKeyboardFrame.stripContainer.suggestionStripView
        private val InputViewBinding.stripContainer get() = mainKeyboardFrame.stripContainer.root
    }
}
