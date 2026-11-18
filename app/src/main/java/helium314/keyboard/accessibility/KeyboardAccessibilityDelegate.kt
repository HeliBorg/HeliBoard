/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.accessibility

import android.os.SystemClock
import helium314.keyboard.latin.utils.Log
import android.view.MotionEvent
import android.view.View
import android.view.accessibility.AccessibilityEvent
import androidx.core.view.AccessibilityDelegateCompat
import androidx.core.view.ViewCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.KeyDetector
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardView

/**
 * This class represents a delegate that can be registered in a class that extends
 * [KeyboardView] to enhance accessibility support via composition rather via inheritance.
 *
 * To implement accessibility mode, the target keyboard view has to:
 *
 *
 * - Call [.setKeyboard] when a new keyboard is set to the keyboard view.
 * - Dispatch a hover event by calling [.onHoverEnter].
 *
 * @param <KV> The keyboard view class type.
</KV> */
open class KeyboardAccessibilityDelegate<KV : KeyboardView>(
    protected val mKeyboardView: KV,
    protected val mKeyDetector: KeyDetector
    ) : AccessibilityDelegateCompat() {
    private var mKeyboard: Keyboard? = null
    private var mAccessibilityNodeProvider: KeyboardAccessibilityNodeProvider<KV>? = null
    private var mLastHoverKey: Key? = null

    protected open var lastHoverKey: Key?
        get() = mLastHoverKey
        set(key) {
            mLastHoverKey = key
        }
    /**
     * Called when the keyboard layout changes.
     *
     * **Note:** This method will be called even if accessibility is not
     * enabled.
     * [keyboard]: The keyboard that is being set to the wrapping view.
     */
    open var keyboard: Keyboard?
        get() = mKeyboard
        set(keyboard) {
            if (keyboard == null) {
                return
            }
            mAccessibilityNodeProvider?.setKeyboard(keyboard)
            mKeyboard = keyboard
        }

    /**
     * Sends a window state change event with the specified string resource id.
     *
     * @param resId The string resource id of the text to send with the event.
     */
    protected fun sendWindowStateChanged(resId: Int) {
        if (resId == 0) {
            return
        }
        val context = mKeyboardView.context
        sendWindowStateChanged(context.getString(resId))
    }

    /**
     * Sends a window state change event with the specified text.
     *
     * @param text The text to send with the event.
     */
    protected fun sendWindowStateChanged(text: String?) {
        val stateChange = AccessibilityUtils.obtainEvent(AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED)
        mKeyboardView.onInitializeAccessibilityEvent(stateChange)
        stateChange.text.add(text)
        stateChange.contentDescription = null
        val parent = mKeyboardView.parent
        parent?.requestSendAccessibilityEvent(mKeyboardView, stateChange)
    }

    /**
     * Delegate method for View.getAccessibilityNodeProvider(). This method is called in SDK
     * version 15 (Build.VERSION_CODES.ICE_CREAM_SANDWICH_MR1) and higher to obtain the virtual
     * node hierarchy provider.
     *
     * @param host The host view for the provider.
     * @return The accessibility node provider for the current keyboard.
     */
    override fun getAccessibilityNodeProvider(host: View): KeyboardAccessibilityNodeProvider<KV> {
        return accessibilityNodeProvider
    }
    // Instantiate the provide only when requested. Since the system
    // will call this method multiple times it is a good practice to
    // cache the provider instance.

    /**
     * @return A lazily-instantiated node provider for this view delegate.
     */
    protected val accessibilityNodeProvider: KeyboardAccessibilityNodeProvider<KV>
        get() {
            // Instantiate the provide only when requested. Since the system
            // will call this method multiple times it is a good practice to
            // cache the provider instance.
            return mAccessibilityNodeProvider ?: KeyboardAccessibilityNodeProvider(mKeyboardView, this)
        }

    /**
     * Get a key that a hover event is on.
     *
     * @param event The hover event.
     * @return key The key that the `event` is on.
     */
    protected fun getHoverKeyOf(event: MotionEvent): Key? {
        val actionIndex = event.actionIndex
        val x = event.getX(actionIndex).toInt()
        val y = event.getY(actionIndex).toInt()
        return mKeyDetector.detectHitKey(x, y)
    }

    /**
     * Receives hover events when touch exploration is turned on in SDK versions ICS and higher.
     *
     * @param event The hover event.
     * @return `true` if the event is handled.
     */
    fun onHoverEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_HOVER_ENTER -> onHoverEnter(event)
            MotionEvent.ACTION_HOVER_MOVE -> onHoverMove(event)
            MotionEvent.ACTION_HOVER_EXIT -> onHoverExit(event)
            else -> Log.w(javaClass.simpleName, "Unknown hover event: $event")
        }
        return true
    }

    /**
     * Process [MotionEvent.ACTION_HOVER_ENTER] event.
     *
     * @param event A hover enter event.
     */
    protected open fun onHoverEnter(event: MotionEvent) {
        val key = getHoverKeyOf(event)
        if (DEBUG_HOVER) {
            Log.d(TAG, "onHoverEnter: key=$key")
        }
        key?.let { onHoverEnterTo(it) }
        mLastHoverKey = key
    }

    /**
     * Process [MotionEvent.ACTION_HOVER_MOVE] event.
     *
     * @param event A hover move event.
     */
    protected open fun onHoverMove(event: MotionEvent) {
        val lastKey = mLastHoverKey
        val key = getHoverKeyOf(event)
        if (key !== lastKey) {
            lastKey?.let { onHoverExitFrom(it) }
            key?.let { onHoverEnterTo(it) }
        }
        key?.let { onHoverMoveWithin(it) }
        mLastHoverKey = key
    }

    /**
     * Process [MotionEvent.ACTION_HOVER_EXIT] event.
     *
     * @param event A hover exit event.
     */
    protected open fun onHoverExit(event: MotionEvent) {
        val lastKey = mLastHoverKey
        if (DEBUG_HOVER) {
            Log.d(TAG, "onHoverExit: key=" + getHoverKeyOf(event) + " last=" + lastKey)
        }
        lastKey?.let { onHoverExitFrom(it) }
        val key = getHoverKeyOf(event)
        // Make sure we're not getting an EXIT event because the user slid
        // off the keyboard area, then force a key press.
        key?.let { performClickOn(it)
        onHoverExitFrom(it) }
        mLastHoverKey = null
    }

    /**
     * Perform click on a key.
     *
     * @param key A key to be registered.
     */
    open fun performClickOn(key: Key) {
        if (DEBUG_HOVER) {
            Log.d(TAG, "performClickOn: key=$key")
        }
        simulateTouchEvent(MotionEvent.ACTION_DOWN, key)
        simulateTouchEvent(MotionEvent.ACTION_UP, key)
    }

    /**
     * Simulating a touch event by injecting a synthesized touch event into [KeyboardView].
     *
     * @param touchAction The action of the synthesizing touch event.
     * @param key The key that a synthesized touch event is on.
     */
    private fun simulateTouchEvent(touchAction: Int, key: Key) {
        val x = key.hitBox.centerX()
        val y = key.hitBox.centerY()
        val eventTime = SystemClock.uptimeMillis()
        val touchEvent = MotionEvent.obtain(
                eventTime, eventTime, touchAction, x.toFloat(), y.toFloat(), 0 /* metaState */)
        mKeyboardView.onTouchEvent(touchEvent)
        touchEvent.recycle()
    }

    /**
     * Handles a hover enter event on a key.
     *
     * @param key The currently hovered key.
     */
    protected open fun onHoverEnterTo(key: Key) {
        if (DEBUG_HOVER) {
            Log.d(TAG, "onHoverEnterTo: key=$key")
        }
        key.onPressed()
        mKeyboardView.invalidateKey(key)
        val provider = accessibilityNodeProvider
        provider.onHoverEnterTo(key)
        provider.performActionForKey(key, AccessibilityNodeInfoCompat.ACTION_ACCESSIBILITY_FOCUS)
    }

    /**
     * Handles a hover move event on a key.
     *
     * @param key The currently hovered key.
     */
    protected fun onHoverMoveWithin(key: Key?) {}

    /**
     * Handles a hover exit event on a key.
     *
     * @param key The currently hovered key.
     */
    protected open fun onHoverExitFrom(key: Key) {
        if (DEBUG_HOVER) {
            Log.d(TAG, "onHoverExitFrom: key=$key")
        }
        key.onReleased()
        mKeyboardView.invalidateKey(key)
        val provider = accessibilityNodeProvider
        provider.onHoverExitFrom(key)
    }

    /**
     * Perform long click on a key.
     *
     * @param key A key to be long pressed on.
     */
    open fun performLongClickOn(key: Key) {
        // A extended class should override this method to implement long press.
    }

    companion object {
        private val TAG = KeyboardAccessibilityDelegate::class.java.simpleName
        const val DEBUG_HOVER = false
        const val HOVER_EVENT_POINTER_ID = 0
    }

    init {
        // Ensure that the view has an accessibility delegate.
        ViewCompat.setAccessibilityDelegate(mKeyboardView, this)
    }
}
