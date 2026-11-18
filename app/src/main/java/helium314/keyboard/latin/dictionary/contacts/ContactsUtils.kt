/*
 * Copyright (C) 2014 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.dictionary.contacts

import android.Manifest
import android.content.ContentResolver
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteException
import android.net.Uri
import android.os.SystemClock
import android.provider.BaseColumns
import android.provider.ContactsContract
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.dictionary.ExpandableDictionaryUtils
import helium314.keyboard.latin.dictionary.contacts.ContactsUtils.MAX_CONTACT_NAMES
import helium314.keyboard.latin.permissions.PermissionsUtil
import helium314.keyboard.latin.utils.Log
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

object ContactsUtils {
    fun onContactsChanged(context: Context) {
        // reload, check if actually changed
        if (!isUpdating.compareAndSet(false, true)) {
            if (DebugFlags.DEBUG_ENABLED) {
                Log.d(TAG, "onContactsChanged: Already running")
            }
            return
        }
        try {
            if (reloadNamesIfNecessary(context))
                ExpandableDictionaryUtils.setNeedsUpdate(ContactsBinaryDictionary.NAME)
        } catch (e: Exception) {
            Log.e(TAG, "onContactsChanged: Error reloading names", e)
        }
        isUpdating.set(false)
    }

    /** checks if contacts need a reload, retuns if anything has changed */
    private fun reloadNamesIfNecessary(context: Context): Boolean {
        if (!PermissionsUtil.checkAllPermissionsGranted(context, Manifest.permission.READ_CONTACTS)) {
            Log.i(TAG, "No permission to read contacts. Marking contacts as not changed.")
            return false
        }

        val resolver = context.contentResolver
        val startTime = SystemClock.uptimeMillis()
        val contactCount = getContactCount(resolver)
        if (contactCount > MAX_CONTACTS_PROVIDER_QUERY_LIMIT) {
            // If there are too many contacts then return false. In this rare case it is impossible
            // to include all of them anyway and the cost of rebuilding the dictionary is too high.
            return false
        }
        val oldHashCode = currentNames.hashCode()
        reloadNames(resolver)
        if (contactCount != contactCountAtLastRebuild) {
            if (DebugFlags.DEBUG_ENABLED) {
                Log.d(TAG, "Contacts count changed from $contactCountAtLastRebuild to $contactCount")
            }
            contactCountAtLastRebuild = contactCount
            return true
        }
        if (currentNames.hashCode() != oldHashCode) {
            return true
        }
        if (DebugFlags.DEBUG_ENABLED) {
            Log.d(TAG, "No contacts change detected in ${SystemClock.uptimeMillis() - startTime} ms")
        }
        return false
    }

    private fun getContactCount(contentResolver: ContentResolver): Int {
        // TODO: consider switching to a rawQuery("select count(*)...") on the database if performance is a bottleneck.
        try {
            contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI, PROJECTION_ID_ONLY, null, null, null
            ).use { cursor ->
                if (null == cursor) return 0
                return cursor.count
            }
        } catch (e: SQLiteException) {
            Log.e(TAG, "SQLiteException in the remote Contacts process.", e)
        }
        return 0
    }

    private fun reloadNames(resolver: ContentResolver) {
        currentNames = getValidNames(ContactsContract.Contacts.CONTENT_URI, resolver) + getValidNames(ContactsContract.Profile.CONTENT_URI, resolver)
    }

    /**
     * Returns up to [MAX_CONTACT_NAMES] valid names in the Contacts DB.
     * <p>
     * These names are sorted by their affinity to the user, with favorite
     * contacts appearing first.
     */
    private fun getValidNames(uri: Uri, contentResolver: ContentResolver): Collection<String> {
        // Check all contacts since it's not possible to find out which names have changed.
        // This is needed because it's possible to receive extraneous onChange events even when no
        // name has changed.
        val cursor = contentResolver.query(uri, PROJECTION, null, null, null)
            ?: return emptyList()
        val contacts = ArrayList<RankedContact>()
        var maxTimesContacted = 0
        cursor.use { cursor ->
            if (cursor.moveToFirst()) {
                while (!cursor.isAfterLast) {
                    val name = cursor.getString(NAME_INDEX)
                    if (isValidName(name)) {
                        val timesContacted = cursor.getInt(TIMES_CONTACTED_INDEX)
                        if (timesContacted > maxTimesContacted) {
                            maxTimesContacted = timesContacted
                        }
                        contacts.add(RankedContact(cursor))
                    }
                    cursor.moveToNext()
                }
            }
        }
        val currentTime = System.currentTimeMillis()
        for (contact in contacts) {
            contact.computeAffinity(maxTimesContacted, currentTime)
        }
        contacts.sortWith { contact1, contact2 -> contact2.affinity.compareTo(contact1.affinity) }
        val names = HashSet<String>()
        var i = 0
        while (i < contacts.size && names.size < MAX_CONTACT_NAMES) {
            names.add(contacts[i].mName)
            ++i
        }
        return names
    }

    private fun isValidName(name: String?): Boolean {
        if (name.isNullOrEmpty() || name.indexOf(Constants.CODE_COMMERCIAL_AT.toChar()) != -1) {
            return false
        }
        val hasSpace = name.indexOf(Constants.CODE_SPACE.toChar()) != -1
        if (!hasSpace) {
            // Only allow an isolated word if it does not contain a hyphen.
            // This helps to filter out mailing lists.
            return name.indexOf(Constants.CODE_DASH.toChar()) == -1
        }
        return true
    }

    private val isUpdating = AtomicBoolean(false)
    private var contactCountAtLastRebuild = 0
    private var currentNames = listOf<String>()
    fun getNames(context: Context): List<String> {
        if (currentNames.isEmpty())
            onContactsChanged(context)
        return currentNames
    }

    // Frequency for contacts information into the dictionary
    const val FREQUENCY_FOR_CONTACTS = 100 // much increased from original frequency because contacts were barely suggested
    const val FREQUENCY_FOR_CONTACTS_BIGRAM = 200 // todo: seems broken, how to actually get bigrams?

    private val TAG = ContactsUtils::class.java.simpleName
    private const val MAX_CONTACT_NAMES = 500
    private val PROJECTION: Array<String> = arrayOf(
        BaseColumns._ID, ContactsContract.Contacts.DISPLAY_NAME,
        ContactsContract.Contacts.TIMES_CONTACTED, ContactsContract.Contacts.LAST_TIME_CONTACTED, ContactsContract.Contacts.IN_VISIBLE_GROUP
    )
    private val PROJECTION_ID_ONLY: Array<String> = arrayOf(BaseColumns._ID)

    // Do not attempt to query contacts if there are more than this many entries.
    private const val MAX_CONTACTS_PROVIDER_QUERY_LIMIT = 10000

    // indices for columns in Contacts & ContactsContract.Profile
    private const val NAME_INDEX = 1
    private const val TIMES_CONTACTED_INDEX = 2
    private const val LAST_TIME_CONTACTED_INDEX = 3
    private const val IN_VISIBLE_GROUP_INDEX = 4

    private class RankedContact(cursor: Cursor) {
        val mName: String = cursor.getString(NAME_INDEX)
        val mLastContactedTime = cursor.getLong(LAST_TIME_CONTACTED_INDEX)
        val mTimesContacted = cursor.getInt(TIMES_CONTACTED_INDEX)
        val mInVisibleGroup = cursor.getInt(IN_VISIBLE_GROUP_INDEX) == 1

        var affinity: Float = 0.0f
            private set

        /**
         * Calculates the affinity with the contact based on:
         * - How many times it has been contacted
         * - How long since the last contact.
         * - Whether the contact is in the visible group (i.e., Contacts list).
         *
         * Note: This affinity is limited by the fact that some apps currently do not update the
         * LAST_TIME_CONTACTED or TIMES_CONTACTED counters. As a result, a frequently messaged
         * contact may still have 0 affinity.
         */
        fun computeAffinity(maxTimesContacted: Int, currentTime: Long) {
            val timesWeight = (mTimesContacted.toFloat() + 1) / (maxTimesContacted + 1)
            val timeSinceLastContact = min(
                max(0, currentTime - mLastContactedTime),
                TimeUnit.MILLISECONDS.convert(180, TimeUnit.DAYS)
            )
            val lastTimeWeight = 0.5.pow((timeSinceLastContact / (TimeUnit.MILLISECONDS.convert(10, TimeUnit.DAYS))).toDouble())
            val visibleWeight = if (mInVisibleGroup) 1.0f else 0.0f
            affinity = (timesWeight + lastTimeWeight.toFloat() + visibleWeight) / 3
        }
    }
}
