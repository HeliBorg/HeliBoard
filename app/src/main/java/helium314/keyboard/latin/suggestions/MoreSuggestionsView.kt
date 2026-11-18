/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.suggestions

import android.content.Context
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import helium314.keyboard.accessibility.AccessibilityUtils
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardActionListener
import helium314.keyboard.keyboard.MainKeyboardView
import helium314.keyboard.keyboard.PopupKeysKeyboardView
import helium314.keyboard.keyboard.PopupKeysPanel
import helium314.keyboard.latin.R
import helium314.keyboard.latin.suggestions.MoreSuggestions.MoreSuggestionKey
import helium314.keyboard.latin.utils.Log
import kotlin.math.abs

/**
 * A view that renders a virtual [MoreSuggestions]. It handles rendering of keys and detecting
 * key presses and touch movements.
 */
class MoreSuggestionsView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet?,
    defStyle: Int = R.attr.popupKeysKeyboardViewStyle
) : PopupKeysKeyboardView(context, attrs, defStyle) {

    private val moreSuggestionsListener = object : KeyboardActionListener.Adapter() {
        override fun onCancelInput() {
            dismissPopupKeysPanel()
        }
    }

    private val moreSuggestionsController: PopupKeysPanel.Controller = object : PopupKeysPanel.Controller {
        override fun onDismissPopupKeysPanel() {
            mainKeyboardView.onDismissPopupKeysPanel()
        }

        override fun onShowPopupKeysPanel(panel: PopupKeysPanel) {
            mainKeyboardView.onShowPopupKeysPanel(panel)
        }

        override fun onCancelPopupKeysPanel() {
            dismissPopupKeysPanel()
        }
    }

    lateinit var listener: SuggestionStripView.Listener
    lateinit var mainKeyboardView: MainKeyboardView

    private val moreSuggestionsModalTolerance = context.resources.getDimensionPixelOffset(R.dimen.config_more_suggestions_modal_tolerance)
    private val moreSuggestionsBuilder by lazy { MoreSuggestions.Builder(context, this) }

    lateinit var gestureDetector: GestureDetector
    private var isInModalMode = false

    // Working variables for onInterceptTouchEvent(MotionEvent) and onTouchEvent(MotionEvent).
    private var needsToTransformTouchEventToHoverEvent = false
    private var isDispatchingHoverEventToMoreSuggestions = false
    private var lastX = 0
    private var lastY = 0
    private var originX = 0
    private var originY = 0

    override fun setKeyboard(keyboard: Keyboard) {
        super.setKeyboard(keyboard)
        isInModalMode = false
        // With accessibility mode off, mAccessibilityDelegate is set to null at the above PopupKeysKeyboardView#setKeyboard call.
        // With accessibility mode on, mAccessibilityDelegate is set to a PopupKeysKeyboardAccessibilityDelegate object at the above
        // PopupKeysKeyboardView#setKeyboard call.
        if (mAccessibilityDelegate != null) {
            mAccessibilityDelegate.setOpenAnnounce(R.string.spoken_open_more_suggestions)
            mAccessibilityDelegate.setCloseAnnounce(R.string.spoken_close_more_suggestions)
        }
    }

    override fun getDefaultCoordX() = (keyboard as MoreSuggestions).mOccupiedWidth / 2

    fun updateKeyboardGeometry(keyHeight: Int) {
        updateKeyDrawParams(keyHeight)
    }

    private fun setModalMode() {
        isInModalMode = true
        // Set vertical correction to zero (Reset popup keys keyboard sliding allowance R.dimen.config_popup_keys_keyboard_slide_allowance).
        mKeyDetector.setKeyboard(keyboard, -paddingLeft.toFloat(), -paddingTop.toFloat())
    }

    override fun onKeyInput(key: Key, x: Int, y: Int) {
        if (key !is MoreSuggestionKey) {
            Log.e(TAG, "Expected key is MoreSuggestionKey, but found ${key.javaClass.name}")
            return
        }
        val keyboard = keyboard
        if (keyboard !is MoreSuggestions) {
            Log.e(TAG, "Expected keyboard is MoreSuggestions, but found ${keyboard?.javaClass?.name}")
            return
        }
        val suggestedWords = keyboard.mSuggestedWords
        val index = key.mSuggestedWordIndex
        if (index < 0 || index >= suggestedWords.size) {
            Log.e(TAG, "Selected suggestion has an illegal index: $index")
            return
        }
        listener.pickSuggestionManually(suggestedWords.getInfo(index))
        dismissPopupKeysPanel()
    }

    internal fun show(
        suggestedWords: SuggestedWords, fromIndex: Int, container: View,
        layoutHelper: SuggestionStripLayoutHelper, parentView: View
    ): Boolean {
        val maxWidth = parentView.width - container.paddingLeft - container.paddingRight
        val parentKeyboard = mainKeyboardView.keyboard ?: return false
        val keyboard = moreSuggestionsBuilder.layout(
            suggestedWords, fromIndex, maxWidth,
            (maxWidth * layoutHelper.mMinMoreSuggestionsWidth).toInt(),
            layoutHelper.maxMoreSuggestionsRow, parentKeyboard
        ).build()
        setKeyboard(keyboard)
        container.measure(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        val pointX = parentView.width / 2
        val pointY = -layoutHelper.mMoreSuggestionsBottomGap
        showPopupKeysPanel(parentView, moreSuggestionsController, pointX, pointY, moreSuggestionsListener)
        originX = lastX
        originY = lastY
        return true
    }

    fun shouldInterceptTouchEvent(motionEvent: MotionEvent): Boolean {
        if (!isShowingInParent) {
            lastX = motionEvent.x.toInt()
            lastY = motionEvent.y.toInt()
            return gestureDetector.onTouchEvent(motionEvent)
        }
        if (isInModalMode) {
            return false
        }

        val index = motionEvent.actionIndex
        if (abs((motionEvent.getX(index).toInt() - originX).toDouble()) >= moreSuggestionsModalTolerance
            || originY - motionEvent.getY(index).toInt() >= moreSuggestionsModalTolerance
        ) {
            // Decided to be in the sliding suggestion mode only when the touch point has been moved
            // upward. Further MotionEvents will be delivered to SuggestionStripView.onTouchEvent.
            needsToTransformTouchEventToHoverEvent = AccessibilityUtils.instance.isTouchExplorationEnabled
            isDispatchingHoverEventToMoreSuggestions = false
            return true
        }

        if (motionEvent.action == MotionEvent.ACTION_UP || motionEvent.action == MotionEvent.ACTION_POINTER_UP) {
            // Decided to be in the modal input mode.
            setModalMode()
        }
        return false
    }

    fun touchEvent(motionEvent: MotionEvent) {
        if (!isShowingInParent) {
            return // Ignore any touch event while more suggestions panel hasn't been shown.
        }
        // In the sliding input mode. MotionEvent should be forwarded to MoreSuggestionsView.
        val index = motionEvent.actionIndex
        val x = translateX(motionEvent.getX(index).toInt())
        val y = translateY(motionEvent.getY(index).toInt())
        motionEvent.setLocation(x.toFloat(), y.toFloat())
        if (!needsToTransformTouchEventToHoverEvent) {
            onTouchEvent(motionEvent)
            return
        }
        // In sliding suggestion mode with accessibility mode on, a touch event should be transformed to a hover event.
        val onMoreSuggestions = x in 0..<width && y in 0..<height
        if (!onMoreSuggestions && !isDispatchingHoverEventToMoreSuggestions) {
            // Just drop this touch event because dispatching hover event isn't started yet and
            // the touch event isn't on MoreSuggestionsView.
            return
        }
        val hoverAction: Int
        if (onMoreSuggestions && !isDispatchingHoverEventToMoreSuggestions) {
            // Transform this touch event to a hover enter event and start dispatching a hover event to MoreSuggestionsView.
            isDispatchingHoverEventToMoreSuggestions = true
            hoverAction = MotionEvent.ACTION_HOVER_ENTER
        } else if (motionEvent.actionMasked == MotionEvent.ACTION_UP) {
            // Transform this touch event to a hover exit event and stop dispatching a hover event after this.
            isDispatchingHoverEventToMoreSuggestions = false
            needsToTransformTouchEventToHoverEvent = false
            hoverAction = MotionEvent.ACTION_HOVER_EXIT
        } else {
            // Transform this touch event to a hover move event.
            hoverAction = MotionEvent.ACTION_HOVER_MOVE
        }
        motionEvent.action = hoverAction
        onHoverEvent(motionEvent)
    }

    companion object {
        private val TAG = MoreSuggestionsView::class.java.simpleName
    }
}
