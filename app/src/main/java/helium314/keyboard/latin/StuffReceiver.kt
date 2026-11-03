// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.database.ContentObserver
import android.media.AudioManager
import android.net.Uri
import android.os.Build
import android.os.Process
import android.provider.ContactsContract
import android.provider.UserDictionary.Words
import androidx.core.content.ContextCompat
import helium314.keyboard.dictionarypack.DictionaryPackConstants
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.emoji.EmojiSearchActivity
import helium314.keyboard.latin.dictionary.ExpandableDictionaryUtils
import helium314.keyboard.latin.dictionary.UserBinaryDictionary
import helium314.keyboard.latin.dictionary.contacts.ContactsUtils
import helium314.keyboard.latin.permissions.PermissionsUtil
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.AppsUtils
import helium314.keyboard.latin.utils.FoldableUtils
import helium314.keyboard.latin.utils.FoldableUtils.FoldableObserver
import helium314.keyboard.latin.utils.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class StuffReceiver(latinIME: LatinIME) {
    private val dictionaryChangeReceiver: BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != DictionaryPackConstants.NEW_DICTIONARY_INTENT_ACTION) return
            Log.i(TAG, "dictionary changed")
            latinIME.resetSuggestMainDict()
        }
    }

    private val emojiSearchReceiver: BroadcastReceiver = object : BroadcastReceiver() {
        private fun isImeClosed(intent: Intent) = intent.getBooleanExtra(EmojiSearchActivity.IME_CLOSED_KEY, false)

        private fun getSelectedEmoji(intent: Intent) = intent.getStringExtra(EmojiSearchActivity.EMOJI_KEY)

        private fun isEmojiSearch() = EmojiSearchActivity.decodePrivateImeOptions(latinIME.currentInputEditorInfo).height > 0

        override fun onReceive(context: Context?, intent: Intent?) {
            Log.d(
                TAG, "after activity closing. isEmojiSearch: ${isEmojiSearch()}. Intent: $intent" +
                    intent?.let { ". imeClosed: ${isImeClosed(intent)}. selected emoji: ${getSelectedEmoji(intent)}" }
            )
            if (intent?.action != EmojiSearchActivity.EMOJI_SEARCH_DONE_ACTION || isEmojiSearch()) return
            if (isImeClosed(intent)) {
                latinIME.requestHideSelf(0)
            } else {
                latinIME.mHandler.postDelayed({ KeyboardSwitcher.getInstance().setEmojiKeyboard() }, 100)
                if (intent.hasExtra(EmojiSearchActivity.EMOJI_KEY)) {
                    latinIME.onTextInput(intent.getStringExtra(EmojiSearchActivity.EMOJI_KEY))
                }
            }
        }
    }

    private var appsReceiver: BroadcastReceiver? = null
    private var foldableObserver: FoldableObserver? = null
    private var personalDictObserver: ContentObserver? = null
    private var contactsObserver: ContentObserver? = null

    fun register(context: Context) {
        context.registerReceiver(ringerModeChangeReceiver, IntentFilter(AudioManager.RINGER_MODE_CHANGED_ACTION))

        val dictFilter = IntentFilter(DictionaryPackConstants.NEW_DICTIONARY_INTENT_ACTION)
        // RECEIVER_EXPORTED is necessary because apparently Android 15 (and others?) don't recognize
        // if the sender and receiver are the same app, see https://github.com/HeliBorg/HeliBoard/pull/1756
        ContextCompat.registerReceiver(context, dictionaryChangeReceiver, dictFilter, ContextCompat.RECEIVER_EXPORTED)

        ContextCompat.registerReceiver(context, emojiSearchReceiver, IntentFilter(EmojiSearchActivity.EMOJI_SEARCH_DONE_ACTION), ContextCompat.RECEIVER_NOT_EXPORTED)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N)
            context.registerReceiver(restartAfterDeviceUnlockReceiver, IntentFilter(Intent.ACTION_USER_UNLOCKED))

        if (FoldableUtils.isFoldable)
            foldableObserver = FoldableObserver(context)

        updateDictionaryObservers(context)
    }

    fun unregisterAll(context: Context) {
        context.unregisterReceiver(ringerModeChangeReceiver)
        foldableObserver?.unregister(context)
        foldableObserver = null
        context.unregisterReceiver(dictionaryChangeReceiver)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N)
            context.unregisterReceiver(restartAfterDeviceUnlockReceiver)
        context.unregisterReceiver(emojiSearchReceiver)
        unregisterAppsReceiver(context)
        unregisterPersonalDictObserver(context)
        unregisterContactsObserver(context)
    }

    fun updateDictionaryObservers(context: Context) {
        if (Settings.getValues().mUseAppsDictionary) registerAppsReceiver(context)
        else unregisterAppsReceiver(context)

        if (Settings.getValues().mUsePersonalizedDicts) registerPersonalDictObserver(context)
        else unregisterPersonalDictObserver(context)

        if (Settings.getValues().mUseContactsDictionary) registerContactsObserver(context)
        else unregisterContactsObserver(context)
    }

    private fun registerAppsReceiver(context: Context) {
        if (appsReceiver != null) return
        val appsFilter = IntentFilter(Intent.ACTION_PACKAGE_ADDED)
        appsFilter.addAction(Intent.ACTION_PACKAGE_REMOVED)
        appsFilter.addDataScheme(SCHEME_PACKAGE)
        appsReceiver = Companion.appsReceiver
        context.registerReceiver(appsReceiver, appsFilter)
    }

    private fun unregisterAppsReceiver(context: Context) {
        if (appsReceiver == null) return
        context.unregisterReceiver(appsReceiver)
        appsReceiver = null
    }

    private fun registerPersonalDictObserver(context: Context) {
        if (personalDictObserver != null) return
        personalDictObserver = Companion.personalDictObserver
        context.contentResolver.registerContentObserver(Words.CONTENT_URI, true, Companion.personalDictObserver)
    }

    private fun unregisterPersonalDictObserver(context: Context) {
        if (personalDictObserver == null) return
        context.contentResolver.unregisterContentObserver(personalDictObserver!!)
        personalDictObserver = null
    }

    private fun registerContactsObserver(context: Context) {
        if (contactsObserver != null || !PermissionsUtil.checkAllPermissionsGranted(context, Manifest.permission.READ_CONTACTS))
            return
        contactsObserver = ContactsObserver(context)
        context.contentResolver.registerContentObserver(ContactsContract.Contacts.CONTENT_URI, true, contactsObserver!!)
    }

    private fun unregisterContactsObserver(context: Context) {
        if (contactsObserver == null) return
        context.contentResolver.unregisterContentObserver(contactsObserver!!)
        contactsObserver = null
    }

    private inner class ContactsObserver(val context: Context): ContentObserver(null) {
        override fun onChange(self: Boolean, uri: Uri?) {
            scope.launch {
                if (PermissionsUtil.checkAllPermissionsGranted(context, Manifest.permission.READ_CONTACTS))
                    ContactsUtils.onContactsChanged(context)
                else unregisterContactsObserver(context)
            }
        }
    }

    companion object {
        private val TAG = StuffReceiver::class.java.simpleName
        // The name of the scheme used by the Package Manager to warn of a new package installation, replacement or removal.
        private const val SCHEME_PACKAGE: String = "package"
        private val scope = CoroutineScope(Dispatchers.IO)

        private val ringerModeChangeReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val action: String? = intent.action
                if (AudioManager.RINGER_MODE_CHANGED_ACTION == action) {
                    var dnd: Boolean
                    try {
                        dnd = android.provider.Settings.Global.getInt(context.contentResolver, "zen_mode") != 0
                    } catch (_: android.provider.Settings.SettingNotFoundException) {
                        dnd = false
                        Log.w(TAG, "zen_mode setting not found, assuming disabled")
                    }
                    Log.i(TAG, "ringer mode changed, zen_mode on: $dnd")
                    AudioAndHapticFeedbackManager.getInstance().onRingerModeChanged(dnd)
                }
            }
        }

        private val restartAfterDeviceUnlockReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent) {
                val action = intent.action
                // Restart the keyboard if credential encrypted storage is unlocked. This reloads the
                // dictionary and other data from credential-encrypted storage (with the onCreate()
                // method).
                if (Intent.ACTION_USER_UNLOCKED == action) {
                    val myPid = Process.myPid()
                    Log.i(TAG, "Killing my process: pid=$myPid")
                    Process.killProcess(myPid)
                } else {
                    Log.e(TAG, "Unexpected intent $intent")
                }
            }
        }

        private val appsReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent?) {
                if (intent?.getBooleanExtra(Intent.EXTRA_REPLACING, false) != true)
                    scope.launch { AppsUtils.onAppsChanged(context.packageManager) }
            }
        }

        private val personalDictObserver = object : ContentObserver(null) {
            override fun onChange(self: Boolean, uri: Uri?) {
                // just always reload
                ExpandableDictionaryUtils.setNeedsUpdate(UserBinaryDictionary.NAME)
            }
        }
    }
}
