/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.accessibility

import android.graphics.Rect
import android.os.Bundle
import helium314.keyboard.latin.utils.Log
import android.view.View
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityRecord
import androidx.core.view.ViewCompat
import androidx.core.view.accessibility.AccessibilityEventCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat
import androidx.core.view.accessibility.AccessibilityNodeProviderCompat
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardView
import helium314.keyboard.latin.common.CoordinateUtils
import helium314.keyboard.latin.settings.Settings

/**
 * Exposes a virtual view sub-tree for [KeyboardView] and generates
 * [AccessibilityEvent]s for individual [Key]s.
 *
 *
 * A virtual sub-tree is composed of imaginary [View]s that are reported
 * as a part of the view hierarchy for accessibility purposes. This enables
 * custom views that draw complex content to report them selves as a tree of
 * virtual views, thus conveying their logical structure.
 *
 */
class KeyboardAccessibilityNodeProvider<KV : KeyboardView>(
    /** The keyboard view to provide an accessibility node info.  */
    private val mKeyboardView: KV,
    /** The accessibility delegate.  */
    private val mDelegate: KeyboardAccessibilityDelegate<KV>
) : AccessibilityNodeProviderCompat() {
    private val mKeyCodeDescriptionMapper: KeyCodeDescriptionMapper = KeyCodeDescriptionMapper.instance
    private val mAccessibilityUtils: AccessibilityUtils = AccessibilityUtils.instance
    /** Temporary rect used to calculate in-screen bounds.  */
    private val mTempBoundsInScreen = Rect()
    /** The parent view's cached on-screen location.  */
    private val mParentLocation = CoordinateUtils.newInstance()
    /** The virtual view identifier for the focused node.  */
    private var mAccessibilityFocusedView = UNDEFINED
    /** The virtual view identifier for the hovering node.  */
    private var mHoveringNodeId = UNDEFINED
    /** The current keyboard.  */
    private var mKeyboard: Keyboard? = mKeyboardView.keyboard

    /**
     * Sets the keyboard represented by this node provider.
     *
     * @param keyboard The keyboard that is being set to the keyboard view.
     */
    fun setKeyboard(keyboard: Keyboard?) {
        mKeyboard = keyboard
    }

    private fun getKeyOf(virtualViewId: Int): Key? {
        val keyboard = mKeyboard ?: return null
        val sortedKeys = keyboard.sortedKeys
        // Use a virtual view id as an index of the sorted keys list.
        return if (virtualViewId >= 0 && virtualViewId < sortedKeys.size) {
            sortedKeys[virtualViewId]
        } else null
    }

    private fun getVirtualViewIdOf(key: Key): Int {
        val keyboard = mKeyboard ?: return View.NO_ID
        val sortedKeys = keyboard.sortedKeys
        val size = sortedKeys.size
        for (index in 0 until size) {
            if (sortedKeys[index] === key) { // Use an index of the sorted keys list as a virtual view id.
                return index
            }
        }
        return View.NO_ID
    }

    /**
     * Creates and populates an [AccessibilityEvent] for the specified key
     * and event type.
     *
     * @param key A key on the host keyboard view.
     * @param eventType The event type to create.
     * @return A populated [AccessibilityEvent] for the key.
     * @see AccessibilityEvent
     */
    fun createAccessibilityEvent(key: Key, eventType: Int): AccessibilityEvent {
        val virtualViewId = getVirtualViewIdOf(key)
        val keyDescription = getKeyDescription(key)
        val event = AccessibilityUtils.obtainEvent(eventType)
        event.packageName = mKeyboardView.context.packageName
        event.className = key.javaClass.name
        event.contentDescription = keyDescription
        event.isEnabled = true
        val record: AccessibilityRecord = event
        record.setSource(mKeyboardView, virtualViewId)
        return event
    }

    fun onHoverEnterTo(key: Key) {
        val id = getVirtualViewIdOf(key)
        if (id == View.NO_ID) {
            return
        }
        // Start hovering on the key. Because our accessibility model is lift-to-type, we should
        // report the node info without click and long click actions to avoid unnecessary
        // announcements.
        mHoveringNodeId = id
        // Invalidate the node info of the key.
        sendAccessibilityEventForKey(key, AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED)
        sendAccessibilityEventForKey(key, AccessibilityEvent.TYPE_VIEW_HOVER_ENTER)
    }

    fun onHoverExitFrom(key: Key) {
        mHoveringNodeId = UNDEFINED
        // Invalidate the node info of the key to be able to revert the change we have done
        // in {@link #onHoverEnterTo(Key)}.
        sendAccessibilityEventForKey(key, AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED)
        sendAccessibilityEventForKey(key, AccessibilityEvent.TYPE_VIEW_HOVER_EXIT)
    }

    /**
     * Returns an [AccessibilityNodeInfoCompat] representing a virtual
     * view, i.e. a descendant of the host View, with the given `virtualViewId` or
     * the host View itself if `virtualViewId` equals to [View.NO_ID].
     *
     *
     * A virtual descendant is an imaginary View that is reported as a part of
     * the view hierarchy for accessibility purposes. This enables custom views
     * that draw complex content to report them selves as a tree of virtual
     * views, thus conveying their logical structure.
     *
     *
     *
     * The implementer is responsible for obtaining an accessibility node info
     * from the pool of reusable instances and setting the desired properties of
     * the node info before returning it.
     *
     *
     * @param virtualViewId A client defined virtual view id.
     * @return A populated [AccessibilityNodeInfoCompat] for a virtual descendant or the host
     * View.
     * @see AccessibilityNodeInfoCompat
     */
    override fun createAccessibilityNodeInfo(virtualViewId: Int): AccessibilityNodeInfoCompat? {
        if (virtualViewId == UNDEFINED) {
            return null
        }
        val keyboard = mKeyboard ?: return null
        if (virtualViewId == View.NO_ID) {
            // We are requested to create an AccessibilityNodeInfo describing
            // this View, i.e. the root of the virtual sub-tree.
            val rootInfo = AccessibilityNodeInfoCompat.obtain(mKeyboardView)
            ViewCompat.onInitializeAccessibilityNodeInfo(mKeyboardView, rootInfo)
            updateParentLocation()
            // Add the virtual children of the root View.
            val sortedKeys = keyboard.sortedKeys
            val size = sortedKeys.size
            for (index in 0 until size) {
                val key = sortedKeys[index]
                if (key.isSpacer) {
                    continue
                }
                // Use an index of the sorted keys list as a virtual view id.
                rootInfo.addChild(mKeyboardView, index)
            }
            return rootInfo
        }
        // Find the key that corresponds to the given virtual view id.
        val key = getKeyOf(virtualViewId)
        if (key == null) {
            Log.e(TAG, "Invalid virtual view ID: $virtualViewId")
            return null
        }
        val keyDescription = getKeyDescription(key)
        val boundsInParent = key.hitBox
        // Calculate the key's in-screen bounds.
        mTempBoundsInScreen.set(boundsInParent)
        mTempBoundsInScreen.offset(CoordinateUtils.x(mParentLocation), CoordinateUtils.y(mParentLocation))
        val boundsInScreen = mTempBoundsInScreen
        // Obtain and initialize an AccessibilityNodeInfo with information about the virtual view.
        val info = AccessibilityNodeInfoCompat.obtain()
        info.packageName = mKeyboardView.context.packageName
        info.className = key.javaClass.name
        info.contentDescription = keyDescription
        info.setBoundsInParent(boundsInParent)
        info.setBoundsInScreen(boundsInScreen)
        info.setParent(mKeyboardView)
        info.setSource(mKeyboardView, virtualViewId)
        info.isEnabled = key.isEnabled
        info.isVisibleToUser = true
        // Don't add ACTION_CLICK and ACTION_LONG_CLOCK actions while hovering on the key.
        // See {@link #onHoverEnterTo(Key)} and {@link #onHoverExitFrom(Key)}.
        if (virtualViewId != mHoveringNodeId) {
            info.addAction(AccessibilityNodeInfoCompat.ACTION_CLICK)
            if (key.isLongPressEnabled) {
                info.addAction(AccessibilityNodeInfoCompat.ACTION_LONG_CLICK)
            }
        }
        if (mAccessibilityFocusedView == virtualViewId) {
            info.addAction(AccessibilityNodeInfoCompat.ACTION_CLEAR_ACCESSIBILITY_FOCUS)
        } else {
            info.addAction(AccessibilityNodeInfoCompat.ACTION_ACCESSIBILITY_FOCUS)
        }
        return info
    }

    override fun performAction(virtualViewId: Int, action: Int,
                               arguments: Bundle?): Boolean {
        val key = getKeyOf(virtualViewId) ?: return false
        return performActionForKey(key, action)
    }

    /**
     * Performs the specified accessibility action for the given key.
     *
     * @param key The on which to perform the action.
     * @param action The action to perform.
     * @return The result of performing the action, or false if the action is not supported.
     */
    fun performActionForKey(key: Key, action: Int): Boolean {
        return when (action) {
            AccessibilityNodeInfoCompat.ACTION_ACCESSIBILITY_FOCUS -> {
                mAccessibilityFocusedView = getVirtualViewIdOf(key)
                sendAccessibilityEventForKey(key, AccessibilityEventCompat.TYPE_VIEW_ACCESSIBILITY_FOCUSED)
                true
            }
            AccessibilityNodeInfoCompat.ACTION_CLEAR_ACCESSIBILITY_FOCUS -> {
                mAccessibilityFocusedView = UNDEFINED
                sendAccessibilityEventForKey(key, AccessibilityEventCompat.TYPE_VIEW_ACCESSIBILITY_FOCUS_CLEARED)
                true
            }
            AccessibilityNodeInfoCompat.ACTION_CLICK -> {
                sendAccessibilityEventForKey(key, AccessibilityEvent.TYPE_VIEW_CLICKED)
                mDelegate.performClickOn(key)
                true
            }
            AccessibilityNodeInfoCompat.ACTION_LONG_CLICK -> {
                sendAccessibilityEventForKey(key, AccessibilityEvent.TYPE_VIEW_LONG_CLICKED)
                mDelegate.performLongClickOn(key)
                true
            }
            else -> false
        }
    }

    /**
     * Sends an accessibility event for the given [Key].
     *
     * @param key The key that's sending the event.
     * @param eventType The type of event to send.
     */
    fun sendAccessibilityEventForKey(key: Key, eventType: Int) {
        val event = createAccessibilityEvent(key, eventType)
        mAccessibilityUtils.requestSendAccessibilityEvent(event)
    }

    /**
     * Returns the context-specific description for a [Key].
     *
     * @param key The key to describe.
     * @return The context-specific description of the key.
     */
    private fun getKeyDescription(key: Key): String? {
        val shouldObscure = mKeyboard?.mId?.inputType?.let { mAccessibilityUtils.shouldObscureInput(it) } ?: false
        val currentSettings = Settings.getValues()
        val keyCodeDescription = mKeyCodeDescriptionMapper.getDescriptionForKey(
                mKeyboardView.context, mKeyboard, key, shouldObscure)
        return if (currentSettings.isWordSeparator(key.code)) {
            mAccessibilityUtils.getAutoCorrectionDescription(keyCodeDescription, shouldObscure)
        } else keyCodeDescription
    }

    /**
     * Updates the parent's on-screen location.
     */
    private fun updateParentLocation() {
        mKeyboardView.getLocationOnScreen(mParentLocation)
    }

    companion object {
        private val TAG = KeyboardAccessibilityNodeProvider::class.java.simpleName
        // From {@link android.view.accessibility.AccessibilityNodeInfo#UNDEFINED_ITEM_ID}.
        private const val UNDEFINED = Int.MAX_VALUE
    }

}
