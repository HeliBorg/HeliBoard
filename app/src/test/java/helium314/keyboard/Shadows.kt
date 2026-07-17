// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard

import android.content.Context
import android.inputmethodservice.InputMethodService
import android.os.Bundle
import android.os.Handler
import android.text.InputType
import android.view.KeyEvent
import android.view.inputmethod.CompletionInfo
import android.view.inputmethod.CorrectionInfo
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.ExtractedText
import android.view.inputmethod.ExtractedTextRequest
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputContentInfo
import android.view.inputmethod.InputMethodInfo
import android.view.inputmethod.InputMethodManager
import android.view.inputmethod.InputMethodSubtype
import androidx.core.app.LocaleManagerCompat
import androidx.core.os.LocaleListCompat
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.DictionaryFacilitatorImpl
import helium314.keyboard.latin.common.StringUtils
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import org.robolectric.shadows.ShadowInputMethodManager
import java.util.*

@Implements(LocaleManagerCompat::class)
object ShadowLocaleManagerCompat {
    @Implementation
    @JvmStatic
    fun getSystemLocales(context: Context) = LocaleListCompat.create(Locale.ENGLISH)
}

// why doesn't the original ShadowInputMethodManager simply work?
@Implements(InputMethodManager::class)
class ShadowInputMethodManager2 : ShadowInputMethodManager() {
    @Implementation
    override fun getInputMethodList() = listOf(
        if (BuildConfig.BUILD_TYPE == "debug" || BuildConfig.BUILD_TYPE == "debugNoMinify")
            InputMethodInfo("helium314.keyboard.debug", "LatinIME", "HeliBoard debug", null)
        else InputMethodInfo("helium314.keyboard", "LatinIME", "HeliBoard", null),
    )
    @Implementation
    fun getShortcutInputMethodsAndSubtypes() = emptyMap<InputMethodInfo, List<InputMethodSubtype>>()
    @Implementation
    override fun getCurrentInputMethodSubtype() = InputMethodSubtype.InputMethodSubtypeBuilder()
        .setSubtypeLocale("en")
        .build()
}

@Implements(BinaryDictionaryUtils::class)
object ShadowBinaryDictionaryUtils {
    @Implementation
    @JvmStatic
    fun calcNormalizedScore(beforeString: String, afterString: String, score: Int): Float {
        val before = StringUtils.toCodePointArray(beforeString)
        val after = StringUtils.toCodePointArray(afterString)
        val distance = editDistance(beforeString, afterString)
        val beforeLength = before.size
        val afterLength = after.size
        if (0 == beforeLength || 0 == afterLength) return 0.0f
        var spaceCount = 0
        for (j: Int in after) {
            if (j == KeyEvent.KEYCODE_SPACE) ++spaceCount
        }
        if (spaceCount == afterLength) return 0.0f
        if (score <= 0 || distance >= afterLength) {
            // normalizedScore must be 0.0f (the minimum value) if the score is less than or equal to 0,
            // or if the edit distance is larger than or equal to afterLength.
            return 0.0f
        }
        // add a weight based on edit distance.
        val weight = 1.0f - distance.toFloat() / afterLength.toFloat()
        return score.toFloat() / 1000000.0f * weight
    }

    private fun editDistance(x: String, y: String): Int {
        val dp = Array(x.length + 1) {
            IntArray(
                y.length + 1
            )
        }
        for (i in 0..x.length) {
            for (j in 0..y.length) {
                if (i == 0) {
                    dp[i][j] = j
                } else if (j == 0) {
                    dp[i][j] = i
                } else {
                    dp[i][j] = min(
                        dp[i - 1][j - 1]
                                + costOfSubstitution(x[i - 1], y[j - 1]),
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    )
                }
            }
        }
        return dp[x.length][y.length]
    }

    private fun min(vararg numbers: Int): Int {
        var min = Int.MAX_VALUE
        for (n: Int in numbers) {
            if (n < min) min = n
        }
        return min
    }

    private fun costOfSubstitution(a: Char, b: Char): Int {
        return if (a == b) 0 else 1
    }
}

@Implements(DictionaryFacilitatorImpl::class)
class ShadowDictionaryFacilitatorImpl {
    @Implementation
    fun hasAtLeastOneInitializedMainDictionary() = true
}

// could also extend LatinIME, it's not final anyway
@Implements(InputMethodService::class)
class ShadowInputMethodService {
    companion object {
        var batchEdit = 0
        var text = ""
        var selectionStart = 0
        var selectionEnd = 0
        var composingStart = -1
        var composingEnd = -1
        var currentInputType = InputType.TYPE_CLASS_TEXT

        // convenience for access
        val textBeforeCursor get() = text.substring(0, selectionStart)
        val textAfterCursor get() = text.substring(selectionEnd)
        val selectedText get() = text.substring(selectionStart, selectionEnd)
        val cursor get() = if (selectionStart == selectionEnd) selectionStart else -1

        // composingText should return everything, but RichInputConnection.mComposingText only returns up to cursor
        val composingText get() = if (composingStart == -1 || composingEnd == -1) ""
        else text.substring(composingStart, composingEnd)

        fun reset() {
            batchEdit = 0
            text = ""
            selectionStart = 0
            selectionEnd = 0
            composingStart = -1
            composingEnd = -1
            currentInputType = InputType.TYPE_CLASS_TEXT
        }
    }

    @Implementation
    fun getCurrentInputEditorInfo() = EditorInfo().apply {
        inputType = currentInputType
        // anything else?
    }
    @Implementation
    fun getCurrentInputConnection() = ic
    @Implementation
    fun isInputViewShown() = true // otherwise selection updates will do nothing

    // essentially this is the text field we're editing in
    private val ic = object : InputConnection {
        // pretty clear (though this may be slow depending on the editor)
        // bad return value here is likely the cause for that weird bug improved/fixed by fixIncorrectLength
        override fun getTextBeforeCursor(p0: Int, p1: Int): CharSequence = textBeforeCursor.take(p0)
        // pretty clear (though this may be slow depending on the editor)
        override fun getTextAfterCursor(p0: Int, p1: Int): CharSequence = textAfterCursor.take(p0)
        // pretty clear
        override fun getSelectedText(p0: Int): CharSequence? = if (selectionStart == selectionEnd) null
        else text.substring(selectionStart, selectionEnd)
        // inserts text at cursor (right?), and sets it as composing text
        // this REPLACES currently composing text (even if at a different position)
        // moves the cursor: positive means relative to composing text start, negative means relative to start
        override fun setComposingText(newText: CharSequence, cursor: Int): Boolean {
            // first remove the composing text if any
            if (composingStart != -1 && composingEnd != -1)
                text = text.substring(0, composingStart) + text.substring(composingEnd)
            else // no composing span active, we should remove selected text
                if (selectionStart != selectionEnd) {
                    text = textBeforeCursor + textAfterCursor
                    selectionEnd = selectionStart
                }
            // then set the new text at old composing start
            // if no composing start, set it at cursor position
            val insertStart = if (composingStart == -1) selectionStart else composingStart
            text = text.substring(0, insertStart) + newText + text.substring(insertStart)
            composingStart = insertStart
            composingEnd = insertStart + newText.length
            // the cursor -1 is not clear in documentation, but
            // "So a value of 1 will always advance you to the position after the full text being inserted"
            // means that 1 must be composingEnd
            selectionStart = if (cursor > 0) composingEnd + cursor - 1
            else -cursor
            selectionEnd = selectionStart
            // todo: this should call InputMethodManager#updateSelection(View, int, int, int, int)
            //  but only after batch edit has ended
            //  this is not used in RichInputMethodManager, but probably ends up in LatinIME.onUpdateSelection
            //  -> DO IT (though it will likely only trigger that belatedSelectionUpdate thing, it might be relevant)
            return true
        }
        override fun setComposingRegion(p0: Int, p1: Int): Boolean {
            println("setComposingRegion, $p0, $p1")
            composingStart = p0
            composingEnd = p1
            return true // never checked
        }
        // sets composing text empty, but doesn't change actual text
        override fun finishComposingText(): Boolean {
            composingStart = -1
            composingEnd = -1
            return true // always true
        }
        // as per documentation: "This behaves like calling setComposingText(text, newCursorPosition) then finishComposingText()"
        override fun commitText(p0: CharSequence, p1: Int): Boolean {
            setComposingText(p0, p1)
            finishComposingText()
            return true // whether we added the text
        }
        // just tells the text field that we add many updated, and that the editor should not
        // send status updates until batch edit ended (not actually used for this simulation)
        override fun beginBatchEdit(): Boolean {
            ++batchEdit
            return true // always true
        }
        // end a batch edit, but maybe there are multiple batch edits happening
        override fun endBatchEdit(): Boolean {
            if (batchEdit > 0)
                return --batchEdit == 0
            return false // returns true if there is still a batch edit ongoing
        }
        // should notify about cursor info containing composing text, selection, ...
        // todo: maybe that could be interesting, implement it?
        override fun requestCursorUpdates(p0: Int): Boolean {
            // we call this, but don't have onUpdateCursorAnchorInfo overridden in latinIME, so it does nothing
            // also currently we don't care about the return value
            return false
        }
        override fun setSelection(p0: Int, p1: Int): Boolean {
            selectionStart = p0
            selectionEnd = p1
            // todo: call InputMethodService.onUpdateSelection(int, int, int, int, int, int), but only after batch edit is done!
            return true
        }
        // delete beforeLength before cursor position, and afterLength after cursor position
        // chars, not codepoints or glyphs
        // todo: may delete only one half of a surrogate pair, but this should be avoided by RichInputConnection (maybe throw error)
        override fun deleteSurroundingText(beforeLength: Int, afterLength: Int): Boolean {
            // delete only before or after selection
            text = textBeforeCursor.substring(0, textBeforeCursor.length - beforeLength) +
                text.substring(selectionStart, selectionEnd) +
                textAfterCursor.substring(afterLength)

            // if parts of the composing span are deleted, shorten the span (set end to shorter)
            if (selectionStart <= composingStart) {
                composingStart -= beforeLength // is this correct?
                composingEnd -= beforeLength
            } else if (selectionStart <= composingEnd) {
                composingEnd -= beforeLength // is this correct?
            }
            if (selectionEnd <= composingStart) {
                composingStart -= afterLength
                composingEnd -= afterLength
            } else if (selectionEnd <= composingEnd) {
                composingEnd -= afterLength
            }
            // update selection
            selectionStart -= beforeLength
            selectionEnd -= beforeLength
            return true
        }
        override fun sendKeyEvent(p0: KeyEvent): Boolean {
            if (p0.action != KeyEvent.ACTION_DOWN) return true // only change the text on key down, like RichInputConnection does
            if (p0.keyCode == KeyEvent.KEYCODE_DEL) {
                if (selectionEnd == 0) return true // nothing to delete
                if (selectedText.isEmpty()) {
                    text = text.substring(0, selectionStart - 1) + text.substring(selectionEnd)
                    selectionStart -= 1
                } else {
                    text = text.substring(0, selectionStart) + text.substring(selectionEnd)
                }
                selectionEnd = selectionStart
                return true
            }
            val textToAdd = when (p0.keyCode) {
                KeyEvent.KEYCODE_ENTER -> "\n"
                KeyEvent.KEYCODE_DEL -> null
                KeyEvent.KEYCODE_UNKNOWN -> p0.characters
                else -> StringUtils.newSingleCodePointString(p0.unicodeChar)
            }
            if (textToAdd != null && textToAdd.singleOrNull()?.code != 0) {
                text = text.substring(0, selectionStart) + textToAdd + text.substring(selectionEnd)
                selectionStart += textToAdd.length
                selectionEnd = selectionStart
                composingStart = -1
                composingEnd = -1
            }
            return true
        }
        // implementation is only to work with getTextBeforeCursorAndDetectLaggyConnection
        override fun getExtractedText(p0: ExtractedTextRequest?, p1: Int): ExtractedText {
            return ExtractedText().also {
                it.startOffset = 0
                it.selectionStart = selectionStart
                it.selectionEnd = selectionEnd
            }
        }
        // only effect is flashing, so whatever...
        override fun commitCorrection(p0: CorrectionInfo?): Boolean = true
        // implement only when necessary
        override fun getCursorCapsMode(p0: Int): Int = TODO("Not yet implemented")
        override fun deleteSurroundingTextInCodePoints(p0: Int, p1: Int): Boolean = TODO("Not yet implemented")
        override fun commitCompletion(p0: CompletionInfo?): Boolean = TODO("Not yet implemented")
        override fun performEditorAction(p0: Int): Boolean = TODO("Not yet implemented")
        override fun performContextMenuAction(p0: Int): Boolean = TODO("Not yet implemented")
        override fun clearMetaKeyStates(p0: Int): Boolean = TODO("Not yet implemented")
        override fun reportFullscreenMode(p0: Boolean): Boolean = TODO("Not yet implemented")
        override fun performPrivateCommand(p0: String?, p1: Bundle?): Boolean = TODO("Not yet implemented")
        override fun getHandler(): Handler = TODO("Not yet implemented")
        override fun closeConnection() = TODO("Not yet implemented")
        override fun commitContent(p0: InputContentInfo, p1: Int, p2: Bundle?): Boolean = TODO("Not yet implemented")
    }
}
