// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.ContentValues
import android.content.Context
import helium314.keyboard.latin.database.Database
import helium314.keyboard.latin.utils.GestureData.Companion.toJsonWithChecksum
import kotlinx.serialization.json.Json

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

class GestureDataDao(val db: Database) {
    fun add(data: GestureData, word: String?, timestamp: Long) {
        require(data.uuid == null)
        val jsonString = data.toJsonWithChecksum()
        val cv = ContentValues(3)
        cv.put(COLUMN_TIMESTAMP, timestamp)
        cv.put(COLUMN_WORD, word)
        if (data.activeMode)
            cv.put(COLUMN_SOURCE_ACTIVE, 1)
        cv.put(COLUMN_DATA, jsonString)
        synchronized(this) {
            db.writableDatabase.insert(TABLE, null, cv)
        }
    }

    fun filterInfos(
        word: String? = null,
        begin: Long? = null,
        end: Long? = null,
        exported: Boolean? = null,
        activeMode: Boolean? = null,
        limit: Int? = null
    ): List<GestureDataInfo> = synchronized(this) {
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

    fun getJsonData(ids: List<Long>, context: Context): Sequence<String> = synchronized(this) { sequence {
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_DATA),
            "$COLUMN_ID IN (${ids.joinToString(",")})",
            null,
            null,
            null,
            "RANDOM()"
        ).use {
            val exclusions = GestureDataGatheringSettings.getWordExclusions(context)
            while (it.moveToNext()) {
                yield(it.getString(0).filterExcludedSuggestions(exclusions))
            }
        }
    }}

    fun getAllJsonData(context: Context): Sequence<String> = synchronized(this) { sequence {
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_DATA),
            null,
            null,
            null,
            null,
            "RANDOM()"
        ).use {
            val exclusions = GestureDataGatheringSettings.getWordExclusions(context)
            while (it.moveToNext()) {
                yield(it.getString(0).filterExcludedSuggestions(exclusions))
            }
        }
    }}

    fun markAsExported(ids: List<Long>, context: Context) = synchronized(this) {
        if (ids.isEmpty()) return@synchronized
        val cv = ContentValues(1)
        cv.put(COLUMN_EXPORTED, 1)
        db.writableDatabase.update(TABLE, cv, "$COLUMN_ID IN (${ids.joinToString(",")})", null)
        GestureDataGatheringSettings.onExported(context)
    }

    fun delete(ids: List<Long>, onlyExported: Boolean, context: Context): Int = synchronized(this) {
        if (ids.isEmpty()) return 0
        val where = "$COLUMN_ID IN (${ids.joinToString(",")})"
        val whereExported = " AND $COLUMN_EXPORTED <> 0"
        val active = " AND $COLUMN_SOURCE_ACTIVE <> 0"
        val background = " AND $COLUMN_SOURCE_ACTIVE = 0"
        val exportedActiveCount = db.readableDatabase.rawQuery("SELECT COUNT(1) FROM $TABLE WHERE $where$whereExported$active", null).use {
            it.moveToFirst()
            it.getInt(0)
        }
        val exportedBackgroundCount = db.readableDatabase.rawQuery("SELECT COUNT(1) FROM $TABLE WHERE $where$whereExported$background", null).use {
            it.moveToFirst()
            it.getInt(0)
        }
        val count = if (onlyExported) // this is just to be sure we don't delete anything that somehow accidentally came here
            db.writableDatabase.delete(TABLE, where + whereExported, null)
        else
            db.writableDatabase.delete(TABLE, where, null)
        GestureDataGatheringSettings.addExportedActiveDeletionCount(context, exportedActiveCount)
        GestureDataGatheringSettings.addExportedBackgroundDeletionCount(context, exportedBackgroundCount)
        return count
    }

    fun deleteAll() = synchronized(this) {
        db.writableDatabase.delete(TABLE, null, null)
    }

    fun deleteBackgroundWords(words: Collection<String>) = synchronized(this) {
        if (words.isEmpty()) return@synchronized
        val questions = "?,".repeat(words.size)
        db.writableDatabase.delete(
            TABLE,
            "$COLUMN_SOURCE_ACTIVE = 0 AND LOWER($COLUMN_WORD) IN (${questions.take(questions.length - 1)})",
            words.map { it.lowercase() }.toTypedArray()
        )
    }

    fun count(exported: Boolean? = null, activeMode: Boolean? = null): Int = synchronized(this) {
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

    fun isEmpty(): Boolean = synchronized(this) {
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
                    val dao = GestureDataDao(Database.getInstance(context))
                    dao.db.writableDatabase // try access so we get null if db can't be created
                    instance = dao
                } catch (e: Throwable) {
                    Log.e(TAG, "can't create GestureDataDao", e)
                }
            return instance
        }

        // when excluding a word, it's only removed from db by first suggestion / target word
        // so we should clean the other suggestions here
        // actually this should not be necessary anymore as we redact suggestions, but keep in case we want to change redacting suggestions
        private fun String.filterExcludedSuggestions(exclusions: Collection<String>): String {
            if (exclusions.isEmpty()) return this
            var result = this
            exclusions.forEach { excludedWord ->
                if (!result.contains(excludedWord, true)) return@forEach
                runCatching {
                    val data = Json.decodeFromString<GestureData>(result)
                    val newData = data.copy(suggestions = data.suggestions.filterNot { excludedWord.equals(it.word, true) })
                    result = newData.toJsonWithChecksum()
                }
            }
            return result
        }
    }
}
