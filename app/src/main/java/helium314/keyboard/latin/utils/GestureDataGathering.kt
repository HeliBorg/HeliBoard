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

// todo: remove logging, it may contain sensitive data!
object PassiveGatheringCache {
    private val cachedWords = mutableListOf<WordData>()
    private const val TAG = "PassiveGathering"
    private val scope = CoroutineScope(Dispatchers.IO)

    private fun updateIcon(save: Boolean = false) {
        scope.launch(Dispatchers.Main) { // on main thread because it's touching views
            KeyboardSwitcher.getInstance().setPassiveGatheringIndicator(usePassiveGathering, cachedWords.isNotEmpty(), save)
        }
    }

    fun addWord(word: WordData) {
        if (KeyboardSwitcher.getInstance().keyboard.mId.mInternalAction?.code == KeyCode.INLINE_EMOJI_SEARCH_DONE) {
            Log.i(TAG, "inline emoji search, not adding anything")
            return
        }
        if (!word.isSavingOk(Settings.getCurrentContext())) {
            Log.i(TAG, "not adding ${word.usedWord} because it's not allowed")
            return
        }
        Log.i(TAG, "adding ${word.usedWord}")
        cachedWords.add(word)
        updateIcon()
    }

    // only used when entering inline emoji search, because in this case the word is added before the internalAction is set
    fun removeLast(word: String) {
        if (cachedWords.last().usedWord != word) return
        Log.i(TAG, "removing $word")
        cachedWords.removeAt(cachedWords.lastIndex)
        updateIcon()
    }

    fun onPickSuggestionAfterGesturing(suggestion: SuggestedWords.SuggestedWordInfo, originalWord: String) {
        // replace the latest entry in cache, but do a sanity check
        Log.i(TAG, "picked ${suggestion.word} instead of $originalWord after gesturing")
        val lastEntry = cachedWords.lastOrNull()
        if (lastEntry?.usedWord != originalWord) {
            Log.w(TAG, "...but our last word is ${lastEntry?.usedWord}, not $originalWord")
            return
        }
        lastEntry.usedWord = suggestion.mWord
        lastEntry.targetWord = suggestion.mWord
    }

    fun onPickSuggestion(suggestion: SuggestedWords.SuggestedWordInfo, originalWord: String) {
        Log.i(TAG, "picked ${suggestion.word} instead of $originalWord")
        // this happens after tap-typing (new word or corrected gesture word), or when moving the cursor and then selecting a different suggestion
        // don't update anything if we have the word more than once
        val word = cachedWords.singleOrNull { it.usedWord == originalWord } ?: return
        word.usedWord = suggestion.mWord
        word.targetWord = suggestion.mWord
    }

    fun onRejectedSuggestion(suggestion: String) {
        Log.i(TAG, "rejected $suggestion")
        if (cachedWords.lastOrNull()?.usedWord != suggestion) {
            Log.w(TAG, "...but last word is ${cachedWords.lastOrNull()?.usedWord}")
            return
        }
        cachedWords.removeAt(cachedWords.lastIndex)
        updateIcon()
    }

    fun onUndo(lastComposedWord: CharSequence) {
        Log.i(TAG, "undo after committing $lastComposedWord")
        if (cachedWords.lastOrNull()?.usedWord == lastComposedWord) {
            Log.i(TAG, "removing $lastComposedWord")
            cachedWords.removeAt(cachedWords.lastIndex)
        }
        updateIcon()
    }

    fun onEditWord(word: String) {
        // this is pretty aggressive, because repeated backspace might remove different words
        // but better remove a few % of the words instead of having potentially bad data
        Log.i(TAG, "edit something in $word")
        cachedWords.removeAll { it.usedWord == word }
        updateIcon()
    }

    fun onEditSelection(selection: CharSequence?, before: CharSequence?, after: CharSequence?) {
        // null should only occur in very rare cases when there are problems communicating with the text field
        if (selection == null || before == null || after == null) return

        Log.i(TAG, "replace selection \"$selection\", before: \"$before\", after: \"$after\"")
        val script = RichInputMethodManager.getInstance().currentSubtypeLocale.script
        val spacingAndPunctuations = Settings.getValues().mSpacingAndPunctuations
        val wordAtStart = getTouchedWordRange(before, "$selection$after", script, spacingAndPunctuations)
        val wordAtEnd = getTouchedWordRange("$before$selection", after, script, spacingAndPunctuations)
        Log.i(TAG, "at start \"${wordAtStart.mWord}\", at end \"${wordAtEnd.mWord}\"")
        val trimmed = selection.trim()
        if (
            (wordAtEnd.mWord == wordAtStart.mWord && selection in wordAtStart.mWord)
            || (trimmed != selection && (wordAtStart == trimmed || wordAtEnd == trimmed)) // treat word + space like word
        ) {
            Log.i(TAG, "word or part of word selected, removing word")
            cachedWords.removeAll { it.usedWord == wordAtStart.mWord }
        } else {
            // more than one word selected, we do nothing because deleting much is unlikely to happen because of bad gesture typing
        }
        updateIcon()
    }

    @JvmStatic
    fun saveOrClear(context: Context) {
        if (usePassiveGathering && !GestureDataGatheringSettings.isOptInMode(context))
            save(context)
        else clear()
    }

    fun save(context: Context) {
        // save all words and clear cache
        val words = cachedWords.toList()
        Log.i(TAG, "save cached data")
        cachedWords.clear()
        updateIcon(words.isNotEmpty())
        scope.launch { words.forEach { it.save(context) } }
    }

    fun clear() {
        // just clear it without saving
        Log.i(TAG, "clear cache")
        cachedWords.clear()
        updateIcon()
    }
}

@JvmField
var usePassiveGathering = false

fun setUsePassiveGathering(context: Context, editorInfo: EditorInfo): Boolean {
    usePassiveGathering = isPassiveGatheringUsed(context, editorInfo)
    if (!usePassiveGathering)
        PassiveGatheringCache.clear()
    return usePassiveGathering
}

private fun isPassiveGatheringUsed(context: Context, editorInfo: EditorInfo): Boolean {
    if (!JniUtils.sHaveGestureLib) return false
    if (!GestureDataGatheringSettings.isPassiveGatheringEnabled(context.prefs())) return false
    if (Settings.getValues().mIncognitoModeEnabled) return false
    val inputAttributes = InputAttributes(editorInfo, false, "")
    if (inputAttributes.mInputType and InputType.TYPE_CLASS_TEXT == 0)
        return false // undefined (e.g. terminal apps) type should work, but will likely not allow to track corrections
    val isEmailField = InputTypeUtils.isEmailVariation(inputAttributes.mInputType and InputType.TYPE_MASK_VARIATION)
    if (inputAttributes.mIsPasswordField || inputAttributes.mNoLearning || isEmailField) return false
    if (GestureDataGatheringSettings.isForbiddenForDataGathering(editorInfo.packageName, context)) return false
    if (editorInfo.privateImeOptions == "noPassive") return false // meant for review screen
    // we might not have a known dictionary, but I guess that's acceptable
    return true
}

const val dictTestImeOption = "useTestDictionaryFacilitator,${BuildConfig.APPLICATION_ID}.${Constants.ImeOption.NO_FLOATING_GESTURE_PREVIEW}"

var gestureDataActiveFacilitator: SingleDictionaryFacilitator? = null

private val scope = CoroutineScope(Dispatchers.IO)

// class for storing relevant information
class WordData(
    var targetWord: String?, // might be adjusted when using passive gathering
    val suggestions: SuggestionResults,
    val composedData: ComposedData,
    val ngramContext: NgramContext,
    val keyboard: Keyboard,
    val inputStyle: Int,
    val activeMode: Boolean,
    // first suggestion in passive gathering, used to track later changes (not saved)
    // may have different capitalization compared to suggestions
    var usedWord: String? = null
) {
    // keyboard is not immutable, so better store potentially relevant information immediately
    private val keys = keyboard.sortedKeys
    private val height = keyboard.mOccupiedHeight
    private val width = keyboard.mOccupiedWidth

    private val packageName = keyboard.mId.mEditorInfo.packageName

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
            // change target word to the first case insensitive match if there is no case sensitive match
            val match = suggestions.firstOrNull { it.mWord.equals(targetWord, true) }
            if (match != null) targetWord = match.mWord
        }
        val filteredSuggestions = mutableListOf<SuggestedWords.SuggestedWordInfo>()
        for (word in suggestions.sortedByDescending { it.mOriginalScore }) {
            if (word.mWord == targetWord) {
                // always add the targetWord if we have it
                filteredSuggestions.add(word)
                continue
            }
            if (word.mSourceDict.mDictType == Dictionary.TYPE_CONTACTS
                || suggestions.any { it.mWord == word.mWord && it.mSourceDict.mDictType == Dictionary.TYPE_CONTACTS })
                continue // never store contacts (might be in user history too)
            // for the personal dictionary we rely on the ignore list
            if (word.mOriginalScore < 0 && filteredSuggestions.size > 5)
                continue // no need to add bad matches
            if (filteredSuggestions.any { it.mWord == word.mWord })
                continue // only first occurrence word
            if (filteredSuggestions.size > 18) // user sees 18 suggestions at most
            // todo: some data contains more suggestions than allowed -> there is something broken here!
                continue // should be enough
            if (!activeMode && word.mWord in GestureDataGatheringSettings.getWordExclusions(context))
                continue // keep blocked suggestions in active mode because otherwise one might find out which word is blocked
            filteredSuggestions.add(word)
        }
        // we want to store which dictionaries are used, and a dict index (in used dict list) for each suggestion
        var dictCount = 0
        val dictionariesInUsedSuggestions = LinkedHashMap<Dictionary, Int>().apply { // linked because we need the order
            filteredSuggestions.forEach { if (!containsKey(it.mSourceDict)) put(it.mSourceDict, dictCount++) }
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
            filteredSuggestions.map { Suggestion(it.mWord, it.mOriginalScore, dictionariesInUsedSuggestions[it.mSourceDict]) },
            PointerData.fromPointers(composedData.mInputPointers),
            keyboardInfo,
            activeMode,
            null
        )
        scope.launch { dao.add(data, targetWord ?: usedWord, timestamp) }
        GestureDataGatheringSettings.informAboutTooManyPassiveModeWords(activeMode, context, dao)
    }

    // find when we should NOT save
    fun isSavingOk(context: Context): Boolean {
        if (inputStyle != SuggestedWords.INPUT_STYLE_TAIL_BATCH)
            return false
        if (activeMode && suggestions.all { it.mSourceDict == suggestions.first().mSourceDict })
            return true // active mode should be fine, the size check is just an addition in case there is a bug that sets the wrong mode or dictionary facilitator
        if (Settings.getValues().mIncognitoModeEnabled)
            return false // don't save in incognito mode
        if (!activeMode && !GestureDataGatheringSettings.isPassiveGatheringEnabled(context.prefs()))
            return false
        if ((targetWord ?: usedWord)?.contains(' ') == true) // no support for SPACE_AWARE_GESTURE
            return false
        if (!activeMode && GestureDataGatheringSettings.isForbiddenForDataGathering(packageName, context))
            return false // package ignored (we should never come here for blocked apps, but better be safe)
        val inputAttributes = InputAttributes(keyboard.mId.mEditorInfo, false, "")
        val isEmailField = InputTypeUtils.isEmailVariation(inputAttributes.mInputType and InputType.TYPE_MASK_VARIATION)
        if (inputAttributes.mIsPasswordField || inputAttributes.mNoLearning || isEmailField)
            return false // probably some more inputAttributes to consider
        if (suggestions.firstOrNull()?.mSourceDict?.mDictType == Dictionary.TYPE_CONTACTS)
            return false
        val matchingSuggestions = suggestions.filter { it.mWord.equals(targetWord ?: usedWord, true) }
        if (matchingSuggestions.all { (it.mKindAndFlags and 0xFF) == KIND_SHORTCUT })
            return false // we want at least one non-shortcut
        val ignoreWords = GestureDataGatheringSettings.getWordExclusions(context)
        // how to deal with the ignore list?
        // check targetWord and first 5 suggestions?
        // or check only what is in the actually saved suggestions?
        if (usedWord in ignoreWords || suggestions.take(5).any { it.word in ignoreWords })
            return false
        return true
    }
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
data class Suggestion(val word: String, val score: Int, val dictIndex: Int? = null)

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
