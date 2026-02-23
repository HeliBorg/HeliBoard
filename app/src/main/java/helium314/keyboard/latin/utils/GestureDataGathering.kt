// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.ContentValues
import android.content.Context
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.core.content.edit
import com.android.inputmethod.latin.BinaryDictionary
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.InputAttributes
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.R
import helium314.keyboard.latin.SingleDictionaryFacilitator
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.database.Database
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.dictionary.ReadOnlyBinaryDictionary
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.settings.screens.gesturedata.END_DATE_EPOCH_MILLIS
import helium314.keyboard.settings.screens.gesturedata.TWO_WEEKS_IN_MILLIS
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.text.DateFormat
import java.util.Date
import kotlin.random.Random

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

fun isInActiveGatheringMode(editorInfo: EditorInfo) =
    dictTestImeOption == editorInfo.privateImeOptions && gestureDataActiveFacilitator != null

fun setWordIgnoreList(context: Context, list: Collection<String>) {
    val json = Json.encodeToString(list)
    context.prefs().edit { putString(PREF_WORD_EXCLUSIONS, json) }
}

fun getWordIgnoreList(context: Context): Set<String> {
    val json = context.prefs().getString(PREF_WORD_EXCLUSIONS, "[]") ?: "[]"
    if (json.isEmpty()) return sortedSetOf()
    return Json.decodeFromString<List<String>>(json).toSortedSet(compareBy(String.CASE_INSENSITIVE_ORDER) { it })
}

fun setAppIgnoreList(context: Context, list: Collection<String>) {
    context.prefs().edit { putString(PREF_APP_EXCLUSIONS, list.joinToString(",")) }
}

fun getAppIgnoreList(context: Context): List<String> {
    val string = context.prefs().getString(PREF_APP_EXCLUSIONS, "") ?: ""
    return string.split(",").filterNot { it.isEmpty() }
}

fun addExportedActiveDeletionCount(context: Context, count: Int) {
    val oldCount = getExportedActiveDeletionCount(context)
    context.prefs().edit { putInt(PREF_DELETED_ACTIVE, oldCount + count) }
}

fun getExportedActiveDeletionCount(context: Context) = context.prefs().getInt(PREF_DELETED_ACTIVE, 0)

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

private const val PREF_WORD_EXCLUSIONS = "gesture_data_word_exclusions"
private const val PREF_APP_EXCLUSIONS = "gesture_data_app_exclusions"
private const val PREF_DELETED_ACTIVE = "gesture_data_deleted_active_words"
private const val PREF_PASSIVE_NOTIFY_COUNT = "gesture_data_passive_notify_count"
private const val PREF_END_NOTIFICATION_LAST_SHOWN = "gesture_data_end_notification_shown"

const val dictTestImeOption = "useTestDictionaryFacilitator,${BuildConfig.APPLICATION_ID}.${Constants.ImeOption.NO_FLOATING_GESTURE_PREVIEW}"

var gestureDataActiveFacilitator: SingleDictionaryFacilitator? = null

// class for storing relevant information
class WordData(
    val targetWord: String,
    val suggestions: SuggestionResults,
    val composedData: ComposedData,
    val ngramContext: NgramContext,
    val keyboard: Keyboard,
    val inputStyle: Int,
    val activeMode: Boolean,
) {
    // keyboard is not immutable, so better store potentially relevant information immediately
    private val keys = keyboard.sortedKeys
    private val height = keyboard.mOccupiedHeight
    private val width = keyboard.mOccupiedWidth

    // if contacts dict is used we keep this information
    private val dictionariesInSuggestions = suggestions.map { it.mSourceDict }.toSet()
    private val packageName = keyboard.mId.mEditorInfo.packageName

    private val timestamp = System.currentTimeMillis()

    fun save(context: Context) {
        if (!isSavingOk(context))
            return
        val dao = GestureDataDao.getInstance(context) ?: return

        val keyboardInfo = KeyboardInfo(
            width, // baseHeight is without padding, but coordinates include padding
            height,
            keys.map {
                KeyInfo(
                    it.x, it.width, it.y, it.height,
                    it.outputText ?: if (it.code > 0) StringUtils.newSingleCodePointString(it.code) else "",
                    it.popupKeys.orEmpty().map { popup ->
                        popup.mOutputText ?: if (popup.mCode > 0) StringUtils.newSingleCodePointString(popup.mCode) else ""
                    }
                )
            }
        )
        val filteredSuggestions = mutableListOf<SuggestedWords.SuggestedWordInfo>()
        for (word in suggestions) { // suggestions are sorted with highest score first
            if (word.mSourceDict.mDictType == Dictionary.TYPE_CONTACTS
                || suggestions.any { it.mWord == word.mWord && it.mSourceDict.mDictType == Dictionary.TYPE_CONTACTS })
                continue // never store contacts (might be in user history too)
            // for the personal dictionary we rely on the ignore list
            if (word.mScore < 0 && filteredSuggestions.size > 5)
                continue // no need to add bad matches
            if (filteredSuggestions.any { it.mWord == word.mWord })
                continue // only one occurrence per word
            if (filteredSuggestions.size > 12)
                continue // should be enough
            filteredSuggestions.add(word)
        }
        val data = GestureData(
            context.getString(R.string.english_ime_name) + " " + BuildConfig.VERSION_NAME,
            if (!context.protectedPrefs().contains(Settings.PREF_LIBRARY_CHECKSUM)) null
                else context.protectedPrefs().getString(Settings.PREF_LIBRARY_CHECKSUM, "") == JniUtils.expectedDefaultChecksum(),
            targetWord,
            dictionariesInSuggestions.map {
                val hash = (it as? BinaryDictionary)?.hash ?: (it as? ReadOnlyBinaryDictionary)?.hash
                DictInfo(hash, it.mDictType, it.mLocale?.toLanguageTag())
            },
            filteredSuggestions.map { Suggestion(it.mWord, it.mScore) },
            PointerData.fromPointers(composedData.mInputPointers),
            keyboardInfo,
            activeMode,
            null
        )
        dao.add(data, timestamp)
        informAboutTooManyPassiveModeWords(context, dao)
    }

    // show a toast every 5k words, to avoid having to upload multiple files at a time because they are over their email attachment size limit
    // but don't check on every word, because getting count from DB is not free
    private fun informAboutTooManyPassiveModeWords(context: Context, dao: GestureDataDao) {
        if (!activeMode || Random.nextInt() % 20 != 0) return
        val count = dao.count(exported = false, activeMode = false)
        val nextNotifyCount = context.prefs().getInt(PREF_PASSIVE_NOTIFY_COUNT, 5000)
        if (count <= nextNotifyCount) return
        val approxCount = (count / 1000) * 1000
        // show a toast
        KeyboardSwitcher.getInstance().showToast(context.getString(R.string.gesture_data_many_not_shared_words, approxCount.toString()), true)
        context.prefs().edit { putInt(PREF_PASSIVE_NOTIFY_COUNT, approxCount + 5000) }
    }

    // find when we should NOT save
    private fun isSavingOk(context: Context): Boolean {
        if (inputStyle != SuggestedWords.INPUT_STYLE_TAIL_BATCH)
            return false
        if (activeMode && dictionariesInSuggestions.size == 1)
            return true // active mode should be fine, the size check is just an addition in case there is a bug that sets the wrong mode or dictionary facilitator
        if (Settings.getValues().mIncognitoModeEnabled)
            return false // don't save in incognito mode
        if (packageName in getAppIgnoreList(context))
            return false // package ignored
        val inputAttributes = InputAttributes(keyboard.mId.mEditorInfo, false, "")
        val isEmailField = InputTypeUtils.isEmailVariation(inputAttributes.mInputType and InputType.TYPE_MASK_VARIATION)
        if (inputAttributes.mIsPasswordField || inputAttributes.mNoLearning || isEmailField)
            return false // probably some more inputAttributes to consider
        val ignoreWords = getWordIgnoreList(context)
        // how to deal with the ignore list?
        // check targetWord and first 5 suggestions?
        // or check only what is in the actually saved suggestions?
        if (targetWord in ignoreWords || suggestions.take(5).any { it.word in ignoreWords })
            return false
        if (suggestions.first().mSourceDict.mDictType == Dictionary.TYPE_CONTACTS)
            return false
        return true
    }
}

data class GestureDataInfo(val id: Long, val targetWord: String, val timestamp: Long, val exported: Boolean, val activeMode: Boolean)

@Serializable
data class GestureData(
    val application: String,
    val knownLibrary: Boolean?,
    val targetWord: String?, // this will be tricky for active gathering if user corrects the word
    val dictionaries: List<DictInfo>,
    val suggestions: List<Suggestion>,
    val gesture: List<PointerData>,
    val keyboardInfo: KeyboardInfo,
    val activeMode: Boolean,
    val uuid: String?
)

// hash is only available for dictionaries from .dict files
// language can be null (but should not be)
@Serializable
data class DictInfo(val hash: String?, val type: String, val language: String?)

@Serializable
data class Suggestion(val word: String, val score: Int)

@Serializable
data class PointerData(val id: Int, val x: Int, val y: Int, val millis: Int) {
    companion object {
        fun fromPointers(pointers: InputPointers): List<PointerData> {
            val result = mutableListOf<PointerData>()
            for (i in 0..<pointers.pointerSize) {
                result.add(PointerData(
                    pointers.pointerIds[i],
                    pointers.xCoordinates[i],
                    pointers.yCoordinates[i],
                    pointers.times[i]
                ))
            }
            return result
        }
    }
}

// the old gesture typing library only works with code, not with arbitrary text
// but we take the output text (usually still a single codepoint) because we'd like to change this
@Serializable
data class KeyInfo(val left: Int, val width: Int, val top: Int, val height: Int, val value: String, val alts: List<String>)

@Serializable
data class KeyboardInfo(val width: Int, val height: Int, val keys: List<KeyInfo>)

class GestureDataDao(val db: Database) {
    fun add(data: GestureData, timestamp: Long) {
        require(data.uuid == null)
        val jsonString = Json.encodeToString(data)
        // if uuid in the resulting string is replaced with null, we should be able to reproduce it
        val dataWithId = data.copy(uuid = ChecksumCalculator.checksum(jsonString.byteInputStream()))
        val cv = ContentValues(3)
        cv.put(COLUMN_TIMESTAMP, timestamp)
        cv.put(COLUMN_WORD, data.targetWord)
        if (data.activeMode)
            cv.put(COLUMN_SOURCE_ACTIVE, 1)
        cv.put(COLUMN_DATA, Json.encodeToString(dataWithId))
        db.writableDatabase.insert(TABLE, null, cv)
    }

    fun filterInfos(
        word: String? = null,
        begin: Long? = null,
        end: Long? = null,
        exported: Boolean? = null,
        activeMode: Boolean? = null,
        limit: Int? = null
    ): List<GestureDataInfo> {
        val result = mutableListOf<GestureDataInfo>()
        val query = mutableListOf<String>()
        if (word != null) query.add("LOWER($COLUMN_WORD) like ?||'%'")
        if (begin != null) query.add("$COLUMN_TIMESTAMP >= $begin")
        if (end != null) query.add("$COLUMN_TIMESTAMP <= $end")
        if (exported != null) query.add("$COLUMN_EXPORTED = ${if (exported) 1 else 0}")
        if (activeMode != null) query.add("$COLUMN_SOURCE_ACTIVE = ${if (activeMode) 1 else 0}")
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_ID, COLUMN_WORD, COLUMN_TIMESTAMP, COLUMN_EXPORTED, COLUMN_SOURCE_ACTIVE),
            query.joinToString(" AND "),
            word?.let { arrayOf(it.lowercase()) },
            null,
            null,
            null,
            limit?.toString()
        ).use {
            while (it.moveToNext()) {
                result.add(GestureDataInfo(
                    it.getLong(0),
                    it.getString(1),
                    it.getLong(2),
                    it.getInt(3) != 0,
                    it.getInt(4) != 0
                ))
            }
        }
        return result
    }

    fun getJsonData(ids: List<Long>): List<String> {
        val result = mutableListOf<String>()
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_DATA),
            "$COLUMN_ID IN (${ids.joinToString(",")})",
            null,
            null,
            null,
            null
        ).use {
            while (it.moveToNext()) {
                result.add(it.getString(0))
            }
        }
        return result
    }

    fun getAllJsonData(): List<String> {
        val result = mutableListOf<String>()
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_DATA),
            null,
            null,
            null,
            null,
            null
        ).use {
            while (it.moveToNext()) {
                result.add(it.getString(0))
            }
        }
        return result
    }

    fun markAsExported(ids: List<Long>, context: Context) {
        if (ids.isEmpty()) return
        val cv = ContentValues(1)
        cv.put(COLUMN_EXPORTED, 1)
        db.writableDatabase.update(TABLE, cv, "$COLUMN_ID IN (${ids.joinToString(",")})", null)
        if (count(exported = false, activeMode = false) < context.prefs().getInt(PREF_PASSIVE_NOTIFY_COUNT, 0))
            context.prefs().edit { remove(PREF_PASSIVE_NOTIFY_COUNT) } // reset if we exported passive data
    }

    fun delete(ids: List<Long>, onlyExported: Boolean, context: Context): Int {
        if (ids.isEmpty()) return 0
        val where = "$COLUMN_ID IN (${ids.joinToString(",")})"
        val whereExported = " AND $COLUMN_EXPORTED <> 0"
        val count: Int
        if (onlyExported) {
            count = db.writableDatabase.delete(TABLE, where + whereExported, null)
            addExportedActiveDeletionCount(context, count) // actually we could also have a counter in the db
        } else {
            val exportedCount = db.readableDatabase.rawQuery("SELECT COUNT(1) FROM $TABLE WHERE $where$whereExported", null).use {
                it.moveToFirst()
                it.getInt(0)
            }
            count = db.writableDatabase.delete(TABLE, where, null)
            addExportedActiveDeletionCount(context, exportedCount)
        }
        return count
    }

    fun deleteAll() {
        db.writableDatabase.delete(TABLE, null, null)
    }

    fun deletePassiveWords(words: Collection<String>) {
        val wordsString = words.joinToString("','") { it.lowercase() }
        db.writableDatabase.delete(
            TABLE,
            "$COLUMN_SOURCE_ACTIVE <> 0 AND LOWER($COLUMN_WORD) in (?)",
            arrayOf(wordsString)
        )
    }

    fun count(exported: Boolean? = null, activeMode: Boolean? = null): Int {
        val where = mutableListOf<String>()
        if (exported != null)
            where.add("$COLUMN_EXPORTED ${if (exported) "<>" else "="} 0")
        if (activeMode != null)
            where.add("$COLUMN_SOURCE_ACTIVE ${if (activeMode) "<>" else "="} 0")
        val whereString = if (where.isEmpty()) "" else "WHERE ${where.joinToString(" AND ")}"
        return db.readableDatabase.rawQuery("SELECT COUNT(1) FROM $TABLE $whereString", null).use {
            it.moveToFirst()
            it.getInt(0)
        }
    }

    fun isEmpty(): Boolean {
        db.readableDatabase.rawQuery("SELECT EXISTS (SELECT 1 FROM $TABLE)", null).use {
            it.moveToFirst()
            return it.getInt(0) == 0
        }
    }

    companion object {
        private const val TAG = "GestureDataDao"

        private const val TABLE = "GESTURE_DATA"
        private const val COLUMN_ID = "ID"
        private const val COLUMN_TIMESTAMP = "TIMESTAMP"
        private const val COLUMN_WORD = "WORD"
        private const val COLUMN_EXPORTED = "EXPORTED"
        private const val COLUMN_SOURCE_ACTIVE = "SOURCE_ACTIVE"
        private const val COLUMN_DATA = "DATA" // data is text, blob with zip is slower to store, and probably not worth the saved space

        const val CREATE_TABLE = """
            CREATE TABLE IF NOT EXISTS $TABLE (
                $COLUMN_ID INTEGER PRIMARY KEY,
                $COLUMN_TIMESTAMP INTEGER NOT NULL,
                $COLUMN_WORD TEXT NOT NULL,
                $COLUMN_EXPORTED TINYINT NOT NULL DEFAULT 0,
                $COLUMN_SOURCE_ACTIVE TINYINT NOT NULL DEFAULT 0,
                $COLUMN_DATA TEXT
            )
        """

        private var instance: GestureDataDao? = null

        /** Returns the instance or creates a new one. Returns null if instance can't be created (e.g. no access to db due to device being locked) */
        fun getInstance(context: Context): GestureDataDao? {
            if (instance == null)
                try {
                    instance = GestureDataDao(Database.getInstance(context))
                } catch (e: Throwable) {
                    Log.e(TAG, "can't create ClipboardDao", e)
                }
            return instance
        }
    }
}
