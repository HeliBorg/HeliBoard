// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Context
import android.content.SharedPreferences
import android.os.SystemClock
import android.view.inputmethod.EditorInfo
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.content.edit
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.settings.SettingsDestination
import helium314.keyboard.settings.dialogs.ThreeButtonAlertDialog
import helium314.keyboard.settings.screens.gesturedata.END_DATE_EPOCH_MILLIS
import helium314.keyboard.settings.screens.gesturedata.TWO_WEEKS_IN_MILLIS
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.text.DateFormat
import java.util.Date
import java.util.SortedSet
import kotlin.random.Random

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

object GestureDataGatheringSettings {
    private const val PREF_WORD_EXCLUSIONS = "gesture_data_word_exclusions"
    private const val PREF_APP_EXCLUSIONS = "gesture_data_app_exclusions"
    private const val PREF_APP_EXCLUSIONS_INCLUDE_BY_DEFAULT = "gesture_data_app_exclusions_ignore_by_default"
    private const val PREF_DELETED_ACTIVE = "gesture_data_deleted_active_words"
    private const val PREF_DELETED_BACKGROUND = "gesture_data_deleted_background_words"
    private const val PREF_BACKGROUND_NOTIFY_COUNT = "gesture_data_background_notify_count"
    const val PREF_BACKGROUND_GATHERING_ENABLED = "gesture_data_background_gathering_enabled"
    private const val PREF_BACKGROUND_MANUAL_SAVE = "gesture_data_background_gathering_manual_save"
    const val PREF_BACKGROUND_DISABLED_BEFORE_TIME_MILLIS = "gesture_data_background_gathering_disabled_before"
    private const val PREF_END_NOTIFICATION_LAST_SHOWN = "gesture_data_end_notification_shown"
    private const val PREF_SHOW_PROMOTION_DIALOG_NEXT = "gesture_data_show_promotion_dialog_next_time"
    private const val PREF_SHOW_REMINDER_DIALOG_NEXT = "gesture_data_show_reminder_dialog_next_time"


    fun isInActiveGatheringMode(editorInfo: EditorInfo) =
        dictTestImeOption == editorInfo.privateImeOptions && gestureDataActiveFacilitator != null

    fun isBackgroundGatheringEnabled(prefs: SharedPreferences): Boolean {
        if (!prefs.getBoolean(PREF_BACKGROUND_GATHERING_ENABLED, false)) return false
        val disabledBefore = prefs.getLong(PREF_BACKGROUND_DISABLED_BEFORE_TIME_MILLIS, 0L)
        if (disabledBefore > SystemClock.elapsedRealtime() + 5 * 60 * 1000L) {
            // elapsedRealtime decreased -> phone was rebooted -> reset
            prefs.edit { remove(PREF_BACKGROUND_DISABLED_BEFORE_TIME_MILLIS) }
            return true
        }
        return SystemClock.elapsedRealtime() > disabledBefore
    }

    fun setBackgroundGatheringEnabled(prefs: SharedPreferences, enabled: Boolean) = prefs.edit {
        putBoolean(PREF_BACKGROUND_GATHERING_ENABLED, enabled)
        remove(PREF_BACKGROUND_DISABLED_BEFORE_TIME_MILLIS)
    }

    fun tempDisableBackgroundGathering(prefs: SharedPreferences) {
        // disable for 5 min
        prefs.edit { putLong(PREF_BACKGROUND_DISABLED_BEFORE_TIME_MILLIS, SystemClock.elapsedRealtime() + 5 * 60 * 1000L) }
    }

    fun String.filterBackgroundGatheringToolbarKeys(prefs: SharedPreferences) = split(Separators.ENTRY).filter {
        if (hasBackgroundGatheringPref(prefs)) true
        // only show keys if background gathering was enabled at some point
        else ToolbarKey.BACKGROUND_GATHERING.name !in it
    }.joinToString(Separators.ENTRY)

    fun hasBackgroundGatheringPref(prefs: SharedPreferences) = prefs.contains(PREF_BACKGROUND_GATHERING_ENABLED)

    fun addWordExclusion(context: Context, exclusion: String) {
        setWordExclusions(context, getWordExclusions(context) + exclusion)
    }

    fun setWordExclusions(context: Context, list: Collection<String>) {
        excludedWords = null
        val json = Json.encodeToString(list)
        context.prefs().edit { putString(PREF_WORD_EXCLUSIONS, json) }
        GlobalScope.launch { GestureDataDao.getInstance(context)?.deleteBackgroundWords(list) }
    }

    fun getWordExclusions(context: Context): Set<String> {
        excludedWords?.let { return it }
        val json = context.prefs().getString(PREF_WORD_EXCLUSIONS, "[]") ?: "[]"
        excludedWords = if (json.isEmpty()) sortedSetOf()
        else Json.decodeFromString<List<String?>>(json).toSortedSet(compareBy(String.CASE_INSENSITIVE_ORDER) { it ?: "" })
        return excludedWords!!
    }

    private var excludedWords: SortedSet<String>? = null

    private var excludedApps: Set<String>? = null

    fun setAppExclusions(context: Context, list: Collection<String>) {
        context.prefs().edit { putString(PREF_APP_EXCLUSIONS, list.joinToString(",")) }
        excludedApps = null
    }

    fun getAppExclusions(context: Context): Collection<String> {
        excludedApps?.let { return it }
        val string = context.prefs().getString(PREF_APP_EXCLUSIONS, "") ?: ""
        excludedApps = string.split(",").filterNot { it.isEmpty() }.toHashSet()
        return excludedApps!!
    }

    fun setAppIncludeByDefault(context: Context, value: Boolean) =
        context.prefs().edit { putBoolean(PREF_APP_EXCLUSIONS_INCLUDE_BY_DEFAULT, value) }

    fun getAppIncludeByDefault(context: Context) =
        context.prefs().getBoolean(PREF_APP_EXCLUSIONS_INCLUDE_BY_DEFAULT, false)

    fun isDiscardByDefault(context: Context) =
        context.prefs().getBoolean(PREF_BACKGROUND_MANUAL_SAVE, false)

    fun setDiscardByDefault(context: Context, value: Boolean) =
        context.prefs().edit { putBoolean(PREF_BACKGROUND_MANUAL_SAVE, value) }

    fun isForbiddenForDataGathering(packageName: String?, context: Context): Boolean {
        val exclusions = getAppExclusions(context)
        return if (getAppIncludeByDefault(context)) packageName in exclusions
        else packageName !in exclusions
    }

    fun addExportedActiveDeletionCount(context: Context, count: Int) {
        val oldCount = getExportedActiveDeletionCount(context)
        context.prefs().edit { putInt(PREF_DELETED_ACTIVE, oldCount + count) }
    }

    fun getExportedActiveDeletionCount(context: Context) = context.prefs().getInt(PREF_DELETED_ACTIVE, 0)

    fun addExportedBackgroundDeletionCount(context: Context, count: Int) {
        val oldCount = getExportedBackgroundDeletionCount(context)
        context.prefs().edit { putInt(PREF_DELETED_BACKGROUND, oldCount + count) }
    }

    fun getExportedBackgroundDeletionCount(context: Context) = context.prefs().getInt(PREF_DELETED_BACKGROUND, 0)

    fun onTrySaveData(prefs: SharedPreferences) {
        if (prefs.getLong(PREF_SHOW_PROMOTION_DIALOG_NEXT, 0) < Long.MAX_VALUE)
            prefs.edit { putLong(PREF_SHOW_REMINDER_DIALOG_NEXT, System.currentTimeMillis() + TWO_WEEKS_IN_MILLIS) }
    }

    fun onExported(context: Context) {
        val dao = GestureDataDao.getInstance(context) ?: return
        if (dao.count(exported = false, activeMode = false) < context.prefs().getInt(PREF_BACKGROUND_NOTIFY_COUNT, 0))
            context.prefs().edit { remove(PREF_BACKGROUND_NOTIFY_COUNT) } // reset if we exported background data
    }

    // show a toast every 5k words, to avoid having to upload multiple files at a time because they are over their email attachment size limit
    // but don't check on every word, because getting count from DB is not free
    fun informAboutTooManyBackgroundModeWords(context: Context, dao: GestureDataDao) {
        if (Random.nextInt() % 20 != 0) return
        val count = dao.count(exported = false, activeMode = false)
        val nextNotifyCount = context.prefs().getInt(PREF_BACKGROUND_NOTIFY_COUNT, 5000)
        if (count <= nextNotifyCount) return
        val approxCount = (count / 1000) * 1000
        // show a toast
        KeyboardSwitcher.getInstance().showToast(context.getString(R.string.gesture_data_many_not_shared_words, approxCount.toString()), true)
        context.prefs().edit { putInt(PREF_BACKGROUND_NOTIFY_COUNT, approxCount + 5000) }
    }

    /** shows dialog promoting contribution of gesture data, or ask to do again if last contribution was more than 2 weeks ago */
    @Composable fun GestureDataPromotionReminderDialog() {
        val ctx = LocalContext.current
        val promotionShowNext = ctx.prefs().getLong(PREF_SHOW_PROMOTION_DIALOG_NEXT, 0)
        val reminderShowNext = ctx.prefs().getLong(PREF_SHOW_REMINDER_DIALOG_NEXT, 0)
        val neverShow = promotionShowNext == Long.MAX_VALUE || reminderShowNext == Long.MAX_VALUE // user selected "don't show again"
            // we only show the dialog if the use actively loaded the gesture typing library (as opposed to having the lib in the system and HeliBoard as a system app)
            || ctx.protectedPrefs().getString(Settings.PREF_LIBRARY_CHECKSUM, "").isNullOrEmpty() || !JniUtils.sHaveGestureLib
        var shouldShowReminder by remember { mutableStateOf(
            !neverShow && reminderShowNext < System.currentTimeMillis() && reminderShowNext > 0L
        ) }
        var shouldShowPromotion by remember { mutableStateOf(
            // only show if the user never contributed data
            !neverShow && promotionShowNext < System.currentTimeMillis() && reminderShowNext == 0L
        ) }
        if (shouldShowPromotion) {
            ThreeButtonAlertDialog(
                cancelButtonText = stringResource(R.string.ask_later),
                onDismissRequest = {
                    ctx.prefs().edit { putLong(PREF_SHOW_PROMOTION_DIALOG_NEXT, System.currentTimeMillis() + 30L * 60 * 60 * 1000) }
                    shouldShowPromotion = false
                },
                title = { Text(stringResource(R.string.gesture_data_screen)) },
                content = { Text(stringResource(R.string.gesture_data_promotion_message)) },
                confirmButtonText = stringResource(R.string.gesture_data_take_me_there),
                onConfirmed = { SettingsDestination.navigateTo(SettingsDestination.DataGathering) },
                neutralButtonText = stringResource(R.string.no_dictionary_dont_show_again_button),
                onNeutral = {
                    ctx.prefs().edit { putLong(PREF_SHOW_PROMOTION_DIALOG_NEXT, Long.MAX_VALUE) }
                    shouldShowPromotion = false
                },
            )
        }
        if (shouldShowReminder) {
            ThreeButtonAlertDialog(
                cancelButtonText = stringResource(R.string.ask_later),
                onDismissRequest = {
                    ctx.prefs().edit { putLong(PREF_SHOW_REMINDER_DIALOG_NEXT, System.currentTimeMillis() + 30L * 60 * 60 * 1000) }
                    shouldShowReminder = false
                },
                title = { Text(stringResource(R.string.gesture_data_screen)) },
                content = { Text(stringResource(R.string.gesture_data_reminder_message)) },
                confirmButtonText = stringResource(R.string.gesture_data_take_me_there),
                onConfirmed = { SettingsDestination.navigateTo(SettingsDestination.DataGathering) },
                neutralButtonText = stringResource(R.string.no_dictionary_dont_show_again_button),
                onNeutral = {
                    ctx.prefs().edit { putLong(PREF_SHOW_REMINDER_DIALOG_NEXT, Long.MAX_VALUE) }
                    shouldShowReminder = false
                },
            )
        }
    }

    /** shows a toast notification if we're close to the end of the data gathering phase (at most once per 24 hours, only if there is non-exported data) */
    fun showEndNotificationIfNecessary(context: Context) {
        val now = System.currentTimeMillis()
        if (now < END_DATE_EPOCH_MILLIS - TWO_WEEKS_IN_MILLIS) return
        val lastShown = context.prefs().getLong(PREF_END_NOTIFICATION_LAST_SHOWN, 0)
        if (lastShown > now - 24L * 60 * 60 * 1000) return // show at most once per 24 hours
        context.prefs().edit { putLong(PREF_END_NOTIFICATION_LAST_SHOWN, now) } // set even if we have nothing to tell
        val notExported = GestureDataDao.getInstance(context)?.count(exported = false) ?: 0
        if (notExported == 0) return // nothing to export

        // show a toast
        val endDate = DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(END_DATE_EPOCH_MILLIS))
        KeyboardSwitcher.getInstance().showToast(context.getString(R.string.gesture_data_ends_at, endDate), false)
    }
}
