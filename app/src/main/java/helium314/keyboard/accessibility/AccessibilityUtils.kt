/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.accessibility

import android.content.Context
import android.media.AudioDeviceInfo.*
import android.media.AudioManager
import android.os.Build
import android.os.SystemClock
import android.provider.Settings
import android.text.TextUtils
import helium314.keyboard.latin.utils.Log
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityManager
import android.view.inputmethod.EditorInfo
import helium314.keyboard.latin.R
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.utils.InputTypeUtils

class AccessibilityUtils private constructor() {
    private lateinit var mContext: Context
    private lateinit var mAccessibilityManager: AccessibilityManager
    private lateinit var mAudioManager: AudioManager
    /** The most recent auto-correction.  */
    private var mAutoCorrectionWord: String? = null
    /** The most recent typed word for auto-correction.  */
    private var mTypedWord: String? = null

    private fun initInternal(context: Context) {
        mContext = context
        mAccessibilityManager = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
        mAudioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    }

    /**
     * Returns `true` if accessibility is enabled. Currently, this means
     * that the kill switch is off and system accessibility is turned on.
     *
     * @return `true` if accessibility is enabled.
     */
    val isAccessibilityEnabled: Boolean
        get() = ENABLE_ACCESSIBILITY && mAccessibilityManager.isEnabled

    /**
     * Returns `true` if touch exploration is enabled. Currently, this
     * means that the kill switch is off, the device supports touch exploration,
     * and system accessibility is turned on.
     *
     * @return `true` if touch exploration is enabled.
     */
    val isTouchExplorationEnabled: Boolean
        get() = isAccessibilityEnabled && mAccessibilityManager.isTouchExplorationEnabled

    /**
     * Returns whether the device should obscure typed password characters.
     * Typically this means speaking "dot" in place of non-control characters.
     *
     * @return `true` if the device should obscure password characters.
     */
    fun shouldObscureInput(inputType: Int): Boolean {
        // The user can optionally force speaking passwords.
        if (Settings.Secure.ACCESSIBILITY_SPEAK_PASSWORD != null) {
            val speakPassword = Settings.Secure.getInt(mContext.contentResolver,
                    Settings.Secure.ACCESSIBILITY_SPEAK_PASSWORD, 0) != 0
            if (speakPassword) return false
        }
        // Always speak if the user is listening through headphones.
        val listeningThroughHeadphones = if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            @Suppress("deprecation") // no replacement available
            mAudioManager.isWiredHeadsetOn || mAudioManager.isBluetoothA2dpOn
        } else {
            // try the same as the deprecated thing above, for what we can assume to be headphones
            mAudioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS).any {
                when (it.type) {
                    TYPE_WIRED_HEADSET, TYPE_WIRED_HEADPHONES, TYPE_BLUETOOTH_SCO, TYPE_BLUETOOTH_A2DP, TYPE_USB_HEADSET, TYPE_HEARING_AID, TYPE_BLE_HEADSET -> true
                    else -> false
                }
            }
        }
        return if (listeningThroughHeadphones) false
        else InputTypeUtils.isPasswordInputType(inputType)
        // Don't speak if the IME is connected to a password field.
    }

    /**
     * Sets the current auto-correction word and typed word. These may be used
     * to provide the user with a spoken description of what auto-correction
     * will occur when a key is typed.
     *
     * @param suggestedWords the list of suggested auto-correction words
     */
    fun setAutoCorrection(suggestedWords: SuggestedWords) {
        if (suggestedWords.mWillAutoCorrect) {
            mAutoCorrectionWord = suggestedWords.getWord(SuggestedWords.INDEX_OF_AUTO_CORRECTION)
            val typedWordInfo = suggestedWords.mTypedWordInfo
            mTypedWord = typedWordInfo?.mWord
        } else {
            mAutoCorrectionWord = null
            mTypedWord = null
        }
    }

    /**
     * Obtains a description for an auto-correction key, taking into account the
     * currently typed word and auto-correction.
     *
     * @param keyCodeDescription spoken description of the key that will insert
     * an auto-correction
     * @param shouldObscure whether the key should be obscured
     * @return a description including a description of the auto-correction, if
     * needed
     */
    fun getAutoCorrectionDescription(
            keyCodeDescription: String?, shouldObscure: Boolean): String? {
        if (!TextUtils.isEmpty(mAutoCorrectionWord)) {
            if (!TextUtils.equals(mAutoCorrectionWord, mTypedWord)) {
                return if (shouldObscure) { // This should never happen, but just in case...
                    mContext.getString(R.string.spoken_auto_correct_obscured,
                            keyCodeDescription)
                } else mContext.getString(R.string.spoken_auto_correct, keyCodeDescription,
                        mTypedWord, mAutoCorrectionWord)
            }
        }
        return keyCodeDescription
    }

    /**
     * Sends the specified text to the [AccessibilityManager] to be
     * spoken.
     *
     * @param view The source view.
     * @param text The text to speak.
     */
    fun announceForAccessibility(view: View, text: CharSequence?) {
        if (!mAccessibilityManager.isEnabled) {
            Log.e(TAG, "Attempted to speak when accessibility was disabled!")
            return
        }
        // The following is a hack to avoid using the heavy-weight TextToSpeech
        // class. Instead, we're just forcing a fake AccessibilityEvent into
        // the screen reader to make it speak.
        val event = obtainEvent()
        event.packageName = PACKAGE
        event.className = CLASS
        event.eventTime = SystemClock.uptimeMillis()
        event.isEnabled = true
        event.text.add(text)
        // Platforms starting at SDK version 16 (Build.VERSION_CODES.JELLY_BEAN) should use
        // announce events.
        event.eventType = AccessibilityEvent.TYPE_ANNOUNCEMENT
        val viewParent = view.parent
        if (viewParent == null || viewParent !is ViewGroup) {
            Log.e(TAG, "Failed to obtain ViewParent in announceForAccessibility")
            return
        }
        viewParent.requestSendAccessibilityEvent(view, event)
    }

    /**
     * Handles speaking the "connect a headset to hear passwords" notification
     * when connecting to a password field.
     *
     * @param view The source view.
     * @param editorInfo The input connection's editor info attribute.
     * @param restarting Whether the connection is being restarted.
     */
    fun onStartInputViewInternal(view: View, editorInfo: EditorInfo, restarting: Boolean) {
        if (shouldObscureInput(editorInfo.inputType)) {
            val text = mContext.getText(R.string.spoken_use_headphones)
            announceForAccessibility(view, text)
        }
    }

    /**
     * Sends the specified [AccessibilityEvent] if accessibility is
     * enabled. No operation if accessibility is disabled.
     *
     * @param event The event to send.
     */
    fun requestSendAccessibilityEvent(event: AccessibilityEvent?) {
        if (mAccessibilityManager.isEnabled) {
            mAccessibilityManager.sendAccessibilityEvent(event)
        }
    }

    companion object {
        private val TAG = AccessibilityUtils::class.java.simpleName
        private val CLASS = AccessibilityUtils::class.java.name
        private val PACKAGE = AccessibilityUtils::class.java.getPackage()!!.name
        val instance = AccessibilityUtils()
        /*
         * Setting this constant to {@code false} will disable all keyboard
         * accessibility code, regardless of whether Accessibility is turned on in
         * the system settings. It should ONLY be used in the event of an emergency.
         */
        private const val ENABLE_ACCESSIBILITY = true

        @JvmStatic
        fun init(context: Context) {
            if (!ENABLE_ACCESSIBILITY) return
            // These only need to be initialized if the kill switch is off.
            instance.initInternal(context)
        }

        /**
         * Returns {@true} if the provided event is a touch exploration (e.g. hover)
         * event. This is used to determine whether the event should be processed by
         * the touch exploration code within the keyboard.
         *
         * @param event The event to check.
         * @return {@true} is the event is a touch exploration event
         */
        fun isTouchExplorationEvent(event: MotionEvent): Boolean {
            val action = event.action
            return action == MotionEvent.ACTION_HOVER_ENTER || action == MotionEvent.ACTION_HOVER_EXIT || action == MotionEvent.ACTION_HOVER_MOVE
        }

        fun obtainEvent(eventType: Int): AccessibilityEvent =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                AccessibilityEvent(eventType)
            } else {
                @Suppress("deprecation")
                AccessibilityEvent.obtain(eventType)
            }

        fun obtainEvent(): AccessibilityEvent =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                AccessibilityEvent()
            } else {
                @Suppress("deprecation")
                AccessibilityEvent.obtain()
            }
    }
}
