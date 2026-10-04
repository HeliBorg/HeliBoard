// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.database

import android.content.ClipDescription
import android.content.ContentValues
import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.os.SystemClock
import android.webkit.MimeTypeMap
import androidx.core.content.FileProvider
import androidx.core.database.getStringOrNull
import helium314.keyboard.latin.ClipboardHistoryEntry
import helium314.keyboard.latin.common.FileUtils
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ChecksumCalculator
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.prefs
import java.io.File
import kotlin.collections.joinToString

/** Class providing cached access to the clipboard table */
// currently we should not need to worry about synchronizing access (though maybe we could addClip in a coroutine, then it might be relevant)
class ClipboardDao private constructor(private val db: Database) {
    interface Listener {
        fun onClipInserted(position: Int)
        fun onClipsRemoved(position: Int, count: Int)
        fun onClipMoved(oldPosition: Int, newPosition: Int)
    }

    var listener: Listener? = null

    // we clean up old clips when a new clip is added, but not too frequently
    private var lastClearOldClips = 0L

    // cache is loaded at start and never dropped
    private val cache = mutableListOf<ClipboardHistoryEntry>().apply {
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_ID, COLUMN_TIMESTAMP, COLUMN_PINNED, COLUMN_TEXT, COLUMN_FILE, COLUMN_MIME_TYPE),
            null,
            null,
            null,
            null,
            "$COLUMN_PINNED, $COLUMN_TIMESTAMP DESC" // was only relevant in the initial approach of using a cursor instead of a cache
        ).use {
            while (it.moveToNext()) {
                add(ClipboardHistoryEntry(
                    it.getLong(0),
                    it.getLong(1),
                    it.getInt(2) != 0,
                    it.getStringOrNull(3),
                    it.getStringOrNull(4),
                    it.getStringOrNull(5)?.split('§')?.filter { it.isNotEmpty() },
                ))
            }
        }
        sort()
    }

    fun addClip(timestamp: Long, pinned: Boolean, text: String) = synchronized(this) {
        clearOldClips()
        val existingIndex = cache.indexOfFirst { it.text == text }
        if (existingIndex >= 0 && cache[existingIndex].timeStamp == timestamp)
            return@synchronized // nothing to do
        if (existingIndex >= 0) {
            updateTimestampAt(existingIndex, timestamp)
            return@synchronized
        }
        insertNewEntry(timestamp, pinned, text, null, null, null)
    }

    fun addClipUri(timestamp: Long, pinned: Boolean, uri: Uri, description: ClipDescription, context: Context) = synchronized(this) {
        clearOldClips()
        val extension = if (description.mimeTypeCount == 0) ""
            else ".${MimeTypeMap.getSingleton().getExtensionFromMimeType(description.getMimeType(0))}"
        val tempFile = File(context.filesDir, "temp_clip")
        tempFile.delete()
        runCatching { FileUtils.copyContentUriToNewFile(uri, context, tempFile) }.onFailure { return@synchronized }

        // we set the file name to the sha256 of the content to have virtually unique names and an easy way to find duplicates
        val sha256 = ChecksumCalculator.checksum(tempFile)
        val file = File(clipFilesDir, sha256 + extension)

        val existingIndex = cache.indexOfFirst { it.filename == file.name }
        if (existingIndex >= 0) {
            if (cache[existingIndex].timeStamp != timestamp)
                updateTimestampAt(existingIndex, timestamp)
            tempFile.delete()
            return@synchronized
        }
        tempFile.renameTo(file)
        // we could try getting a thumbnail using context.contentResolver.loadThumbnail(uri, Size(a, b), null)
        // but currently we don't cache them anyway, so no use for that
        insertNewEntry(timestamp, pinned, description.label?.toString(), file.name, description.getMimeTypes(), context)
    }

    // keep pinned and the first non-pinned, others can be deleted
    private fun deleteIfSizeExceeded(prefs: SharedPreferences) {
        val sizeLimit = prefs.getInt(Settings.PREF_CLIPBOARD_FILES_SIZE_LIMIT, Defaults.PREF_CLIPBOARD_FILES_SIZE_LIMIT) * 1000000
        var size = 0L
        var keepMin = 1
        val toRemove = mutableListOf<ClipboardHistoryEntry>()
        cache.forEach {
            if (it.filename == null) return@forEach
            val file = File(clipFilesDir, it.filename)
            size += file.length()
            if (it.isPinned)
                return@forEach
            if (size > sizeLimit) {
                if (keepMin > 0) --keepMin
                else toRemove.add(it)
            }
        }
        delete(toRemove)
    }

    /** only public for restoring backups */
    fun insertNewEntry(timestamp: Long, pinned: Boolean, text: String?, filename: String?, mimeTypes: List<String>?, context: Context?) {
        val cv = ContentValues(5)
        cv.put(COLUMN_TIMESTAMP, timestamp)
        cv.put(COLUMN_PINNED, pinned)
        cv.put(COLUMN_TEXT, text)
        cv.put(COLUMN_FILE, filename)
        // § should be a safe separator, not allowed in mime types: https://datatracker.ietf.org/doc/html/rfc6838#section-4.2
        cv.put(COLUMN_MIME_TYPE, mimeTypes?.joinToString("§"))
        val rowId = db.writableDatabase.insert(TABLE, null, cv)

        val entry = ClipboardHistoryEntry(rowId, timestamp, pinned, text, filename, mimeTypes)
        if (filename != null && context != null)
            deleteIfSizeExceeded(context.prefs())
        cache.add(entry)
        cache.sort()
        listener?.onClipInserted(cache.indexOf(entry))
    }

    private fun updateTimestampAt(index: Int, timestamp: Long) {
        val entry = cache[index]
        entry.timeStamp = timestamp
        cache.sort()
        listener?.onClipMoved(index, cache.indexOf(entry))
        val cv = ContentValues(1)
        cv.put(COLUMN_TIMESTAMP, timestamp)
        db.writableDatabase.update(TABLE, cv, "$COLUMN_ID = ${entry.id}", null)
    }

    fun isPinned(index: Int) = cache[index].isPinned

    fun getAt(index: Int) = cache[index]

    fun get(id: Long) = cache.first { it.id == id }

    fun getAll(): List<ClipboardHistoryEntry> = cache

    fun count() = cache.size

    fun sort() = cache.sort()

    fun togglePinned(id: Long) = synchronized(this) {
        val entry = cache.first { it.id == id }
        entry.isPinned = !entry.isPinned
        entry.timeStamp = System.currentTimeMillis()
        if (listener != null) {
            val oldPos = cache.indexOf(entry)
            cache.sort()
            val newPos = cache.indexOf(entry)
            listener?.onClipMoved(oldPos, newPos)
        } else {
            cache.sort()
        }
        val cv = ContentValues(2)
        cv.put(COLUMN_PINNED, entry.isPinned)
        cv.put(COLUMN_TIMESTAMP, entry.timeStamp)
        db.writableDatabase.update(TABLE, cv, "$COLUMN_ID = ${entry.id}", null)
    }

    // RecyclerView initiates this, so we don't call listener (or we'll get an IndexOutOfRangeException from RecyclerView)
    fun deleteClipAt(index: Int) {
        delete(listOf(cache[index]))
    }

    private fun delete(entries: List<ClipboardHistoryEntry>) = synchronized(this) {
        if (entries.isEmpty()) return@synchronized
        cache.removeAll(entries)
        db.writableDatabase.delete(TABLE, "$COLUMN_ID IN (${entries.joinToString(",") { it.id.toString() }})", null)
        entries.forEach { if (it.filename != null) File(clipFilesDir, it.filename).delete() }
    }

    fun clearOldClips(now: Boolean = false) {
        if (listener != null)
            return // never clear when clipboard is visible
        if (!now && lastClearOldClips > SystemClock.elapsedRealtime() - 5 * 1000)
            return

        lastClearOldClips = SystemClock.elapsedRealtime()
        val retentionTime = Settings.getValues()?.mClipboardHistoryRetentionTime ?: 121L
        if (retentionTime > 120) return
        val minTime = System.currentTimeMillis() - retentionTime * 60 * 1000L
        val toRemove = cache.filter { it.timeStamp < minTime && !it.isPinned }
        delete(toRemove)
    }

    fun clearNonPinned() {
        val indicesToRemove = mutableListOf<Int>()
        cache.forEachIndexed { idx, clip ->
            if (!clip.isPinned)
                indicesToRemove.add(idx)
        }
        if (indicesToRemove.isEmpty())
            return // nothing to remove
        delete(cache.filter { !it.isPinned })
        listener?.onClipsRemoved(indicesToRemove[0], indicesToRemove.size)
    }

    fun clear() {
        if (count() == 0) return
        cache.clear()
        listener?.onClipsRemoved(0, count())
        db.writableDatabase.delete(TABLE, null, null)
    }

    fun cleanupFiles(prefs: SharedPreferences) {
        if (!prefs.getBoolean(Settings.PREF_CLIPBOARD_USE_FILES, Defaults.PREF_CLIPBOARD_USE_FILES)) {
            delete(cache.filter { it.filename != null && !it.isPinned })
            return
        }

        val files = clipFilesDir.listFiles()?.toMutableList() ?: return
        val fnames = files.mapTo(HashSet()) { it.name }
        val entries = cache.filter { it.filename != null }
        val enames = entries.mapTo(HashSet()) { it.filename }

        val filesToRemove = files.filter { it.name !in enames }
        val entriesToRemove = entries.filter { it.filename!! !in fnames }
        if (filesToRemove.isEmpty() && entriesToRemove.isEmpty()) {
            deleteIfSizeExceeded(prefs)
            return
        }

        Log.w(TAG, "deleting ${filesToRemove.size} files and ${entriesToRemove.size} clipboard entries")
        filesToRemove.forEach { it.delete() }
        delete(entriesToRemove)

        deleteIfSizeExceeded(prefs)
    }

    companion object {
        private const val TAG = "ClipboardDao"

        private const val TABLE = "CLIPBOARD"
        // it's possible timestamp is not unique, so we use a separate ID
        // ID is generated and returned on insert, see https://sqlite.org/rowidtable.html
        private const val COLUMN_ID = "ID"
        private const val COLUMN_TIMESTAMP = "TIMESTAMP"
        private const val COLUMN_PINNED = "PINNED"
        private const val COLUMN_TEXT = "TEXT" // we could enforce unique text, but that's only necessary if we can drop the cache (later)
        private const val COLUMN_FILE = "FILE" // path relative to files dir
        private const val COLUMN_MIME_TYPE = "MIME_TYPE" // for files, actually a list of mime types according to clipboard description
        const val CREATE_TABLE = """
            CREATE TABLE $TABLE (
                $COLUMN_ID INTEGER PRIMARY KEY,
                $COLUMN_TIMESTAMP INTEGER NOT NULL,
                $COLUMN_PINNED TINYINT NOT NULL,
                $COLUMN_TEXT TEXT
            )
        """

        const val ADD_FILE_COLUMN = "ALTER TABLE $TABLE ADD COLUMN $COLUMN_FILE TEXT"
        const val ADD_MIME_TYPE_COLUMN = "ALTER TABLE $TABLE ADD COLUMN $COLUMN_MIME_TYPE TEXT"

        private var instance: ClipboardDao? = null
        lateinit var clipFilesDir: File
            private set

        /** Returns the instance or creates a new one. Returns null if instance can't be created (e.g. no access to db due to device being locked) */
        fun getInstance(context: Context): ClipboardDao? {
            if (instance == null)
                try {
                    instance = ClipboardDao(Database.getInstance(context))
                    clipFilesDir = File(context.filesDir, "clipboard")
                    clipFilesDir.mkdirs()
                    instance?.cleanupFiles(context.prefs())
                } catch (e: Throwable) {
                    Log.e(TAG, "can't create ClipboardDao", e)
                }
            return instance
        }

        private fun ClipDescription.getMimeTypes(): List<String> {
            val types = mutableListOf<String>()
            for (i in 0..<mimeTypeCount) {
                types.add(getMimeType(i))
            }
            if (types.isEmpty())
                types.add("*/*")
            return types
        }
    }
}

class ClipboardContentProvider : FileProvider()
