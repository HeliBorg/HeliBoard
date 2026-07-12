// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Context
import android.text.InputType
import android.view.inputmethod.EditorInfo
import com.android.inputmethod.latin.BinaryDictionary
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.InputAttributes
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.SingleDictionaryFacilitator
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_SHORTCUT
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.common.getTouchedWordRange
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.dictionary.ReadOnlyBinaryDictionary
import helium314.keyboard.latin.settings.Settings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

// functionality for gesture data gathering as part of the NLNet Project https://nlnet.nl/project/GestureTyping/
// will be removed once the project is finished

object BackgroundGatheringCache {
    private val cachedWords = mutableListOf<WordData>()
    private const val TAG = "BackgroundGathering"
    private const val DEBUG = false // hardcoded debug flag because data should not be logged even in normal debug mode
    private val scope = CoroutineScope(Dispatchers.IO)

    private fun updateIcon(save: Boolean = false) {
        scope.launch(Dispatchers.Main) { // on main thread because it's touching views
            KeyboardSwitcher.getInstance().setBackgroundGatheringIndicator(useBackgroundGathering, cachedWords.isNotEmpty(), save)
        }
    }

    fun addWord(word: WordData) {
        if (KeyboardSwitcher.getInstance().keyboard?.mId?.internalAction?.code == KeyCode.INLINE_EMOJI_SEARCH_DONE) {
            if (DEBUG) Log.i(TAG, "inline emoji search, not adding anything")
            return
        }
        if (DEBUG) Log.i(TAG, "adding ${word.topSuggestion}")
        cachedWords.add(word)
        updateIcon()
    }

    // used when pressing backspace or entering inline emoji search, because in this case the word is added before the internalAction is set
    fun removeLast(word: String) {
        if (cachedWords.lastOrNull()?.topSuggestion?.word?.equals(word, true) != true) return
        if (DEBUG) Log.i(TAG, "removing $word")
        cachedWords.removeAt(cachedWords.lastIndex)
        updateIcon()
    }

    fun onPickSuggestionAfterGesturing(suggestion: SuggestedWords.SuggestedWordInfo, originalWord: String) {
        // replace the latest entry in cache, but do a sanity check
        if (DEBUG) Log.i(TAG, "picked ${suggestion.word} instead of $originalWord after gesturing")
        val lastEntry = cachedWords.lastOrNull()
        if (lastEntry?.topSuggestion?.word?.equals(originalWord, true) != true) {
            if (DEBUG) Log.w(TAG, "...but our last word is ${lastEntry?.topSuggestion}, not $originalWord")
            return
        }
        lastEntry.targetWord = lastEntry.suggestions.firstOrNull { it.mWord == suggestion.mWord }?.mWord ?:
            // consider that suggestion might be capitalized, but prefer exact match
            lastEntry.suggestions.firstOrNull { it.mWord.equals(suggestion.mWord, true) }?.mWord
        if (DEBUG) Log.i(TAG, "setting target word to ${lastEntry.targetWord}")
    }

    fun onPickSuggestion(suggestion: SuggestedWords.SuggestedWordInfo, originalWord: String) {
        // this happens after tap-typing (new word or corrected gesture word), or when moving the cursor and then selecting a different suggestion
        // don't update anything if we have the word more than once
        val word = cachedWords.singleOrNull { it.topSuggestion?.word?.equals(originalWord, true) == true } ?: return
        word.targetWord = word.suggestions.firstOrNull { it.mWord.equals(suggestion.mWord, true) }?.mWord
        if (DEBUG) Log.i(TAG, "picked ${word.targetWord} instead of $originalWord")
    }

    fun onRejectedSuggestion(suggestion: String) {
        if (DEBUG) Log.i(TAG, "rejected $suggestion")
        if (cachedWords.lastOrNull()?.topSuggestion?.word?.equals(suggestion, true) != true) {
            if (DEBUG) Log.w(TAG, "...but last word is ${cachedWords.lastOrNull()?.topSuggestion?.word}")
            return
        }
        cachedWords.removeAt(cachedWords.lastIndex)
        updateIcon()
    }

    fun onUndo(lastComposedWord: CharSequence) {
        if (DEBUG) Log.i(TAG, "undo after committing $lastComposedWord")
        if (cachedWords.lastOrNull()?.topSuggestion == lastComposedWord || cachedWords.lastOrNull()?.targetWord == lastComposedWord) {
            if (DEBUG) Log.i(TAG, "removing $lastComposedWord")
            cachedWords.removeAt(cachedWords.lastIndex)
        }
        updateIcon()
    }

    fun onEditWord(word: String) {
        // this is pretty aggressive, because repeated backspace might remove different words
        // but better remove a few % of the words instead of having potentially bad data
        if (DEBUG) Log.i(TAG, "edit something in $word")
        cachedWords.removeAll { it.topSuggestion?.word?.equals(word, true) == true || it.targetWord?.equals(word, true) == true }
        updateIcon()
    }

    fun onEditSelection(selection: CharSequence?, before: CharSequence?, after: CharSequence?) {
        // null should only occur in very rare cases when there are problems communicating with the text field
        if (selection == null || before == null || after == null) return

        if (DEBUG) Log.i(TAG, "replace selection \"$selection\", before: \"$before\", after: \"$after\"")
        val script = RichInputMethodManager.getInstance().currentSubtypeLocale.script
        val spacingAndPunctuations = Settings.getValues().mSpacingAndPunctuations
        val wordAtStart = getTouchedWordRange(before, "$selection$after", script, spacingAndPunctuations)
        val wordAtEnd = getTouchedWordRange("$before$selection", after, script, spacingAndPunctuations)
        if (DEBUG) Log.i(TAG, "at start \"${wordAtStart.mWord}\", at end \"${wordAtEnd.mWord}\"")
        val trimmed = selection.trim()
        if (
            (wordAtEnd.mWord == wordAtStart.mWord && selection in wordAtStart.mWord)
            || (trimmed != selection && (wordAtStart == trimmed || wordAtEnd == trimmed)) // treat word + space like word
        ) {
            if (DEBUG) Log.i(TAG, "word or part of word selected, removing word")
            cachedWords.removeAll { it.topSuggestion == wordAtStart.mWord || it.targetWord == wordAtStart.mWord }
        } else {
            // more than one word selected, we do nothing because deleting much is unlikely to happen because of bad gesture typing
        }
        updateIcon()
    }

    @JvmStatic
    fun saveOrClear(context: Context) {
        if (useBackgroundGathering && !GestureDataGatheringSettings.isDiscardByDefault(context))
            save(context)
        else clear()
    }

    fun save(context: Context) {
        // save all words and clear cache
        val words = cachedWords.toList()
        if (DEBUG) Log.i(TAG, "save cached data")
        cachedWords.clear()
        updateIcon(words.isNotEmpty())
        scope.launch { words.forEach { it.save(context) } }
    }

    fun clear() {
        // just clear it without saving
        if (DEBUG) Log.i(TAG, "clear cache")
        cachedWords.clear()
        updateIcon()
    }

    val isEmpty get() = cachedWords.isEmpty()
}

@JvmField
var useBackgroundGathering = false

fun setUseBackgroundGathering(context: Context, editorInfo: EditorInfo): Boolean {
    useBackgroundGathering = isBackgroundGatheringUsed(context, editorInfo)
    if (!useBackgroundGathering)
        BackgroundGatheringCache.clear()
    return useBackgroundGathering
}

private fun isBackgroundGatheringUsed(context: Context, editorInfo: EditorInfo): Boolean {
    if (!JniUtils.sHaveGestureLib) return false
    if (!GestureDataGatheringSettings.isBackgroundGatheringEnabled(context.prefs())) return false
    if (Settings.getValues().mIncognitoModeEnabled) return false
    val inputAttributes = InputAttributes(editorInfo, false, "")
    if (inputAttributes.mInputType and InputType.TYPE_CLASS_TEXT == 0)
        return false // undefined (e.g. terminal apps) type should work, but will likely not allow to track corrections
    val isEmailField = InputTypeUtils.isEmailVariation(inputAttributes.mInputType and InputType.TYPE_MASK_VARIATION)
    if (inputAttributes.mIsPasswordField || inputAttributes.mNoLearning || isEmailField) return false
    if (GestureDataGatheringSettings.isForbiddenForDataGathering(editorInfo.packageName, context)) return false
    if (editorInfo.privateImeOptions == "noBackground") return false // meant for review screen
    // we might not have a known dictionary, they are informed about this when enabling background gathering
    return true
}

const val dictTestImeOption = "useTestDictionaryFacilitator,${BuildConfig.APPLICATION_ID}.${Constants.ImeOption.NO_FLOATING_GESTURE_PREVIEW}"

var gestureDataActiveFacilitator: SingleDictionaryFacilitator? = null

private val scope = CoroutineScope(Dispatchers.IO)

// class for storing relevant information
class WordData(
    var targetWord: String?, // might be adjusted when using background gathering
    val suggestions: SuggestionResults,
    val composedData: ComposedData,
    val ngramContext: NgramContext, // actually we don't use it
    val keyboard: Keyboard,
    val inputStyle: Int,
    val activeMode: Boolean,
    // first suggestion in background gathering, used to track later changes
    // may not be in SuggestionResults due to processing
    val topSuggestion: SuggestedWords.SuggestedWordInfo? = null
) {
    // keyboard is not immutable, so better store potentially relevant information immediately
    private val keys = keyboard.sortedKeys
    private val height = keyboard.mOccupiedHeight
    private val width = keyboard.mOccupiedWidth

    private val packageName = keyboard.mId.editorInfo.packageName
    private val pointerData = PointerData.fromPointers(composedData.mInputPointers)

    private val timestamp = System.currentTimeMillis()

    fun save(context: Context) {
        GestureDataGatheringSettings.onTrySaveData(context.prefs())
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
        if (!activeMode && targetWord != null && suggestions.none { it.mWord == targetWord }) {
            // change target word to the first case insensitive match if there is no case sensitive match (actually this should not be necessary?)
            val match = suggestions.firstOrNull { it.mWord.equals(targetWord, true) }
            if (match != null) targetWord = match.mWord
        }

        val topWord = targetWord ?: topSuggestion?.word
        val filteredSuggestions = filterSuggestions(GestureDataGatheringSettings.getWordExclusions(context))
        // we want to store which dictionaries are used, and a dict index (in used dict list) for each suggestion
        var dictCount = 0
        val dictionariesInUsedSuggestions = LinkedHashMap<Dictionary, Int>().apply { // linked because we need the order (well, not any more...)
            filteredSuggestions.forEach { if (!containsKey(it.mSourceDict)) put(it.mSourceDict, dictCount++) }
            // we always want all main known dictionaries
            suggestions.forEach { if (!containsKey(it.mSourceDict) && it.isFromKnownMainDict(context)) put(it.mSourceDict, dictCount++) }
        }

        val data = GestureData(
            context.getString(R.string.english_ime_name) + " " + BuildConfig.VERSION_NAME,
            if (!context.protectedPrefs().contains(Settings.PREF_LIBRARY_CHECKSUM)) null
                else context.protectedPrefs().getString(Settings.PREF_LIBRARY_CHECKSUM, "") == JniUtils.expectedDefaultChecksum(),
            targetWord,
            dictionariesInUsedSuggestions.map { (dict, _) ->
                val hash = (dict as? BinaryDictionary)?.hash ?: (dict as? ReadOnlyBinaryDictionary)?.hash
                DictInfo(hash, dict.mDictType, dict.mLocale?.toLanguageTag())
            },
            // index not needed any more
            filteredSuggestions.map { Suggestion(it.mWord, it.mOriginalScore /*, dictionariesInUsedSuggestions[it.mSourceDict]*/) },
            pointerData,
            keyboardInfo,
            activeMode,
            null
        )
        scope.launch { dao.add(data, topWord, timestamp) }
        if (!activeMode)
            scope.launch(Dispatchers.Main) { GestureDataGatheringSettings.informAboutTooManyBackgroundModeWords(context, dao) }
    }

    fun filterSuggestions(blockedWords: Collection<String>): List<SuggestedWords.SuggestedWordInfo> {
        val filteredSuggestions = mutableListOf<SuggestedWords.SuggestedWordInfo>()
        for (word in suggestions.sortedByDescending { it.mOriginalScore }) {
            if (activeMode && word.mWord == targetWord) {
                // always add the targetWord if we have it
                filteredSuggestions.add(word)
                continue
            }
            if (filteredSuggestions.any { it.mWord == word.mWord })
                continue // only first occurrence word
            if (filteredSuggestions.size > 18) // user sees 18 suggestions at most
                continue
            if (word.mOriginalScore < 0 && filteredSuggestions.size > 5)
                continue // no need to add bad matches
            if (activeMode) {
                filteredSuggestions.add(word)
                continue
            }
            if (word.mWord in blockedWords)
                continue // we should never come here, but better check twice
            filteredSuggestions.add(word)
            if (word.mWord == (targetWord ?: topSuggestion?.word))
                break // no use for suggestions after that
        }
        // redact words that don't match the top suggestion / target word
        if (!activeMode) {
            for (i in filteredSuggestions.indices) {
                if (filteredSuggestions[i].mWord != (targetWord ?: topSuggestion?.word))
                    filteredSuggestions[i] = filteredSuggestions[i].redact()
            }
        }
        return filteredSuggestions
    }

    // find when we should NOT save
    fun isSavingOk(context: Context): Boolean {
        if (inputStyle != SuggestedWords.INPUT_STYLE_TAIL_BATCH)
            return false
        if (activeMode)
            // active mode should be fine, the check is just an addition in case there is a bug that sets the wrong mode or dictionary facilitator
            return suggestions.all { it.mSourceDict == suggestions.first().mSourceDict }
        if (Settings.getValues().mIncognitoModeEnabled)
            return false // don't save in incognito mode
        if (!GestureDataGatheringSettings.isBackgroundGatheringEnabled(context.prefs()))
            return false
        if ((targetWord ?: topSuggestion?.word)?.contains(' ') == true) // no support for SPACE_AWARE_GESTURE
            return false
        if (GestureDataGatheringSettings.isForbiddenForDataGathering(packageName, context))
            return false // package ignored (we should never come here for blocked apps, but better be safe)
        val inputAttributes = InputAttributes(keyboard.mId.editorInfo, false, "")
        val isEmailField = InputTypeUtils.isEmailVariation(inputAttributes.mInputType and InputType.TYPE_MASK_VARIATION)
        if (inputAttributes.mIsPasswordField || inputAttributes.mNoLearning || isEmailField)
            return false // background gathering should not even be enabled, but better have this backup

        val matchingSuggestions = suggestions.filter { it.mWord.equals(targetWord ?: topSuggestion?.word, true) }
        if (matchingSuggestions.all { (it.mKindAndFlags and 0xFF) == KIND_SHORTCUT })
            return false // we want at least one non-shortcut

        // word and dict-based filtering
        if (matchingSuggestions.none { it.isFromKnownMainDict(context) })
            return false // we have no use if not in main dictionary, also potentially sensitive
        if (matchingSuggestions.any { it.mSourceDict.mDictType == Dictionary.TYPE_CONTACTS })
            return false // if there is a suggestion from contacts -> never use it
        val ignoreWords = GestureDataGatheringSettings.getWordExclusions(context)
        // don't store if target word / first suggestion is blocked, the other suggestions will get redacted anyway
        if ((targetWord ?: topSuggestion?.mWord) in ignoreWords)
            return false
        return true
    }

    private fun SuggestedWords.SuggestedWordInfo.redact() =
        SuggestedWords.SuggestedWordInfo("", "", Int.MIN_VALUE, 0, mSourceDict, 0, 0)

    private fun SuggestedWords.SuggestedWordInfo.isFromKnownMainDict(context: Context): Boolean {
        val hash = (mSourceDict as? BinaryDictionary)?.hash ?: (mSourceDict as? ReadOnlyBinaryDictionary)?.hash ?: return false
        return hash in getKnownDictHashes(context)
    }
}

private var knownHashes: Set<String>? = null
fun getKnownDictHashes(context: Context): Set<String> {
    if (knownHashes == null)
        knownHashes = context.assets.open("known_dict_hashes.txt")
            .use { it.reader().readLines() }.filterNot { it.isBlank() || it.startsWith("#") }
            .toHashSet()
    return knownHashes!!
}

data class GestureDataInfo(val id: Long, val targetWord: String, val timestamp: Long, val exported: Boolean, val activeMode: Boolean)

@Serializable
data class GestureData(
    val application: String,
    val knownLibrary: Boolean?,
    val targetWord: String?,
    val dictionaries: List<DictInfo>,
    val suggestions: List<Suggestion>,
    val gesture: List<PointerData>,
    val keyboardInfo: KeyboardInfo,
    val activeMode: Boolean,
    val uuid: String?
) {
    companion object {
        fun GestureData.toJsonWithChecksum(): String {
            val jsonString = Json.encodeToString(this.copy(uuid = null))
            // if uuid in the resulting string is replaced with null, we should be able to reproduce it
            val dataWithId = copy(uuid = ChecksumCalculator.checksum(jsonString.byteInputStream()))
            return Json.encodeToString(dataWithId)
        }
    }
}

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
