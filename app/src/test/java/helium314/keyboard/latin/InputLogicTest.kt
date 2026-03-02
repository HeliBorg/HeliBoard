// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import android.inputmethodservice.InputMethodService
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.text.InputType
import android.view.KeyEvent
import android.view.inputmethod.*
import androidx.core.content.edit
import helium314.keyboard.ShadowInputMethodManager2
import helium314.keyboard.ShadowLocaleManagerCompat
import helium314.keyboard.event.Event
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.MainKeyboardView
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.ShadowFacilitator2.Companion.lastAddedWord
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.inputlogic.InputLogic
import helium314.keyboard.latin.inputlogic.SpaceState
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ScriptUtils
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.getTimestampFormatter
import helium314.keyboard.latin.utils.prefs
import org.junit.runner.RunWith
import org.mockito.Mockito
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import org.robolectric.shadows.ShadowLog
import java.util.*
import kotlin.math.min
import kotlin.streams.asSequence
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

@RunWith(RobolectricTestRunner::class)
@Config(shadows = [
    ShadowLocaleManagerCompat::class,
    ShadowInputMethodManager2::class,
    ShadowInputMethodService::class,
    ShadowKeyboardSwitcher::class,
    ShadowHandler::class,
    ShadowFacilitator2::class,
])
class InputLogicTest {
    private lateinit var latinIME: LatinIME
    private val settingsValues get() = Settings.getValues()
    private val inputLogic get() = latinIME.mInputLogic
    private val connection: RichInputConnection get() = inputLogic.mConnection
    private val composerReader = InputLogic::class.java.getDeclaredField("mWordComposer").apply { isAccessible = true }
    private val composer get() = composerReader.get(inputLogic) as WordComposer
    private val spaceStateReader = InputLogic::class.java.getDeclaredField("mSpaceState").apply { isAccessible = true }
    private val spaceState get() = spaceStateReader.get(inputLogic) as Int
    private val beforeComposingReader = RichInputConnection::class.java.getDeclaredField("mCommittedTextBeforeComposingText").apply { isAccessible = true }
    private val connectionTextBeforeComposingText get() = (beforeComposingReader.get(connection) as CharSequence).toString()
    private val composingReader = RichInputConnection::class.java.getDeclaredField("mComposingText").apply { isAccessible = true }
    private val connectionComposingText get() = (composingReader.get(connection) as CharSequence).toString()

    @BeforeTest
    fun setUp() {
        latinIME = Robolectric.setupService(LatinIME::class.java)
        // start logging only after latinIME is created, avoids showing the stack traces if library is not found
        ShadowLog.setupLogging()
        ShadowLog.stream = System.out
    }

    @Test fun inputCode() {
        reset()
        input('c')
        assertEquals("c", textBeforeCursor)
        assertEquals("c", getText())
        assertEquals("", textAfterCursor)
        assertEquals("c", composingText)
        latinIME.mHandler.onFinishInput()
        assertEquals("", composingText)
    }

    @Test fun delete() {
        reset()
        setText("hello there ")
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hello there", text)
        assertEquals("there", composingText)
    }

    @Test fun deleteInsideWord() {
        reset()
        setText("hello you there")
        setCursorPosition(8) // after o in you
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hello yu there", text)
        assertEquals("yu", composingText)
    }

    @Test fun insertLetterIntoWord() {
        reset()
        setText("hello")
        setCursorPosition(3) // after first l
        input('i')
        assertEquals("helilo", getWordAtCursor())
        assertEquals("helilo", getText())
        assertEquals(4, getCursorPosition())
        assertEquals(4, cursor)
        assertEquals("", composingText)
    }

    @Test fun insertLetterIntoWordWithWeirdEditor() {
        reset()
        currentInputType = 180225 // should not change much, but just to be sure
        setText("hello")
        setCursorPosition(3, weirdTextField = true) // after first l
        input('i')
        assertEquals("helilo", getWordAtCursor())
        assertEquals("helilo", getText())
        assertEquals(4, getCursorPosition())
        assertEquals(4, cursor)
    }

    @Test fun insertLetterIntoOneOfSeveralWords() {
        reset()
        setText("hello my friend")
        setCursorPosition(7) // between m and y
        input('a')
        assertEquals("may", getWordAtCursor())
        assertEquals("hello may friend", getText())
        assertEquals(8, getCursorPosition())
        assertEquals(8, cursor)
    }

    @Test fun combineHangul() {
        reset()
        val ko = SubtypeSettings.getResourceSubtypesForLocale("ko".constructLocale()).first()
        latinIME.switchToSubtype(ko)
        chainInput("ㅂㄱㅑ")
        assertEquals("ㅂ갸", text)
    }

    // todo: make it work, but it might not be that simple because adding is done in combiner
    //  https://github.com/Helium314/HeliBoard/issues/214
    @Test fun insertLetterIntoWordHangulFails() {
        if (BuildConfig.BUILD_TYPE == "runTests") return
        reset()
        latinIME.switchToSubtype(SubtypeSettings.getResourceSubtypesForLocale("ko".constructLocale()).first())
        chainInput("ㅛㅎㄹㅎㅕㅛ")
        setCursorPosition(3)
        input('ㄲ') // fails, as expected from the hangul issue when processing the event in onCodeInput
        assertEquals("ㅛㅎㄹㄲ혀ㅛ", getWordAtCursor())
        assertEquals("ㅛㅎㄹㄲ혀ㅛ", getText())
        assertEquals("ㅛㅎㄹㄲ혀ㅛ", textBeforeCursor + textAfterCursor)
        assertEquals(4, getCursorPosition())
        assertEquals(4, cursor)
    }

    // see issue 1447
    @Test fun separatorAfterHangul() {
        reset()
        latinIME.switchToSubtype(SubtypeSettings.getResourceSubtypesForLocale("ko".constructLocale()).first())
        chainInput("ㅛ.")
        assertEquals("ㅛ.", text)
    }

    @Test fun deleteHangulInDebugMode() { // issue 1551, later only happened on phone
        reset()
        latinIME.switchToSubtype(SubtypeSettings.getResourceSubtypesForLocale("ko".constructLocale()).first())
        setText("ㅛㅛ ")
        functionalKeyPress(KeyCode.DELETE)
        functionalKeyPress(KeyCode.DELETE)
        functionalKeyPress(KeyCode.DELETE)
    }

    @Test fun separatorUnselectsWord() {
        reset()
        setText("hello")
        assertEquals("hello", composingText)
        input('.')
        assertEquals("", composingText)
    }

    @Test fun autospace() {
        reset()
        setText("hello")
        input('.')
        input('a')
        assertEquals("hello.a", textBeforeCursor)
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        setText("hello")
        input('.')
        input('a')
        assertEquals("hello. a", textBeforeCursor)
    }

    @Test fun autospaceButWithTextAfter() {
        reset()
        setText("hello there")
        setCursorPosition(5) // after hello
        input('.')
        input('a')
        assertEquals("hello.a", textBeforeCursor)
        assertEquals("hello.a there", text)
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        setText("hello there")
        setCursorPosition(5) // after hello
        input('.')
        input('a')
        assertEquals("hello. a", textBeforeCursor)
        assertEquals("hello. a there", text)
    }

    @Test fun noAutospaceInUrlField() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("example.net")
        assertEquals("example. net", text)
        lastAddedWord = ""
        setText("")
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        chainInput("example.net")
        assertEquals("", lastAddedWord)
        assertEquals("example.net", text)
        assertEquals("example.net", composingText)
    }

    @Test fun noAutospaceInUrlFieldWhenPickingSuggestion() {
        reset()
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        chainInput("exam")
        pickSuggestion("example")
        assertEquals("example", text)
        input('.')
        assertEquals("example.", text)
    }

    @Test fun noAutospaceForDetectedUrl() { // "light" version, should work without url detection
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("http://example.net")
        assertEquals("http://example.net", text)
        assertEquals("http", lastAddedWord)
        assertEquals("example.net", composingText)
    }

    @Test fun noAutospaceForDetectedEmail() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("mail@example.com")
        assertEquals("mail@example.com", text)
        assertEquals("mail@example", lastAddedWord) // todo: do we want this? not really nice, but don't want to be too aggressive with URL detection disabled
        assertEquals("com", composingText) // todo: maybe this should still see the whole address as a single word? or don't be too aggressive?
        setText("")
        lastAddedWord = ""
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("mail@example.com")
        assertEquals("", lastAddedWord)
        assertEquals("mail@example.com", composingText)
    }

    @Test fun urlDetectionThings() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("...h")
        assertEquals("...h", text)
        assertEquals("h", composingText)
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("bla..")
        assertEquals("bla..", text)
        assertEquals("", composingText)
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("bla.c")
        assertEquals("bla.c", text)
        assertEquals("bla.c", composingText)
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        latinIME.prefs().edit { putBoolean(Settings.PREF_SHIFT_REMOVES_AUTOSPACE, true) }
        input("bla")
        input('.')
        functionalKeyPress(KeyCode.SHIFT) // should remove the phantom space (in addition to normal effect)
        input('c')
        assertEquals("bla.c", text)
        assertEquals("bla.c", composingText)
    }

    @Test fun stripSeparatorsBeforeAddingToHistoryWithURLDetection() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("example.com.")
        assertEquals("example.com.", composingText)
        input(' ')
        assertEquals("example.com", lastAddedWord)
    }

    @Test fun dontSelectConsecutiveSeparatorsWithURLDetection() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("bla..")
        assertEquals("", composingText)
        assertEquals("bla..", text)
    }

    @Test fun selectDoesSelect() {
        reset()
        setText("this is some text")
        setCursorPosition(3, 8)
        assertEquals("s is ", text.substring(3, 8))
    }

    @Test fun noComposingForPasswordFields() {
        reset()
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD)
        input('a')
        input('b')
        assertEquals("", composingText)
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        input('.')
        input('c')
        assertEquals("", composingText)
    }

    @Test fun `don't select whole thing as composing word if URL detection disabled`() {
        reset()
        setText("http://example.com")
        setCursorPosition(13) // between l and e
        assertEquals("example", composingText)
    }

    @Test fun `select whole thing except http(s) as composing word if URL detection enabled and selecting`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setText("http://example.com")
        setCursorPosition(13) // between l and e
        assertEquals("example.com", composingText)
        setText("http://bla.com http://example.com ")
        setCursorPosition(29) // between l and e
        assertEquals("example.com", composingText)
    }

    @Test fun `select whole thing except http(s) as composing word if URL detection enabled and typing`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("http://example.com")
        assertEquals("example.com", composingText)
    }

    @Test fun `don't add partial URL to history`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setText("http:/") // just so lastAddedWord isn't set to http
        chainInput("/bla.com")
        assertEquals("", lastAddedWord)
    }

    @Test fun urlProperlySelected() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        setText("http://example.com/here")
        setCursorPosition(18) // after .com
        functionalKeyPress(KeyCode.DELETE)
        functionalKeyPress(KeyCode.DELETE)
        functionalKeyPress(KeyCode.DELETE) // delete com
        // todo: do we really want no composing text?
        //  probably not... try not to break composing
        assertEquals("", composingText)
        chainInput("net")
        assertEquals("example.net", composingText)
    }

    @Test fun urlProperlySelectedWhenNotDeletingFullTld() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setText("http://example.com/here")
        setCursorPosition(18) // after .com
        functionalKeyPress(KeyCode.DELETE)
        functionalKeyPress(KeyCode.DELETE) // delete om
        // todo: this is a weird difference to deleting the full TLD (see urlProperlySelected)
        //  what do we want here? (probably consistency)
        assertEquals("example.c/here", composingText)
        chainInput("z")
        assertEquals("", composingText) // todo: this is a weird difference to deleting the full TLD
//        assertEquals("example.cz", composingText) // fails, but probably would be better than above
    }

    @Test fun dontCommitPartialUrlBeforeFirstPeriod() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        // type http://bla. -> bla not selected, but clearly url, also means http://bla is committed which we probably don't want
        chainInput("http://bla.")
        assertEquals("bla.", composingText)
    }

    @Test fun `intermediate commits in text field without protocol`() {
        reset()
        chainInput("bla.")
        assertEquals("bla", lastAddedWord)
        chainInput("com/")
        assertEquals("com", lastAddedWord)
        chainInput("img.jpg")
        assertEquals("img", lastAddedWord)
        assertEquals("jpg", composingText)
    }

    @Test fun `intermediate commit in text field without protocol and with URL detection`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("bla.com/img.jpg")
        assertEquals("bla", lastAddedWord)
        assertEquals("bla.com/img.jpg", composingText)
    }

    @Test fun `only protocol commit in text field with protocol and URL detection`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("http://bla.com/img.jpg")
        assertEquals("http", lastAddedWord)
        assertEquals("bla.com/img.jpg", composingText)
    }

    @Test fun `no intermediate commit in URL field with protocol`() {
        reset()
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        chainInput("http://bla.com/img.jpg")
        assertEquals("http", lastAddedWord) // todo: somehow avoid?
        assertEquals("http://bla.com/img.jpg", text)
        assertEquals("bla.com/img.jpg", composingText)
    }

    @Test fun `no intermediate commit in URL field with protocol and URL detection`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        chainInput("http://bla.com/img.jpg")
        assertEquals("http", lastAddedWord) // todo: somehow avoid?
        assertEquals("http://bla.com/img.jpg", text)
        assertEquals("bla.com/img.jpg", composingText)
    }

    @Test fun `no intermediate commit in URL field without protocol`() {
        reset()
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        chainInput("bla.com/img.jpg")
        assertEquals("", lastAddedWord)
        assertEquals("bla.com/img.jpg", text)
        assertEquals("bla.com/img.jpg", composingText)
    }

    @Test fun `no intermediate commit in URL field without protocol and with URL detection`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        chainInput("bla.com/img.jpg")
        assertEquals("", lastAddedWord)
        assertEquals("bla.com/img.jpg", text)
        assertEquals("bla.com/img.jpg", composingText)
    }

    @Test fun `don't accidentally detect some other text fields as URI`() {
        // see comment in InputLogic.textBeforeCursorMayBeUrlOrSimilar
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_LONG_MESSAGE)
        chainInput("Hey,why")
        assertEquals("Hey, why", text)
    }

    @Test fun `URL detection does not trigger on non-words`() {
        // first make sure it works without URL detection
        reset()
        chainInput("15:50-17")
        assertEquals("15:50-17", text)
        assertEquals("", composingText)
        // then with URL detection
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        chainInput("15:50-17")
        assertEquals("15:50-17", text)
        assertEquals("", composingText)
    }

    @Test fun `autospace after selecting a suggestion`() {
        reset()
        pickSuggestion("this")
        input('b')
        assertEquals("this b", text)
        assertEquals("b", composingText)
    }

    @Test fun `autospace works in URL field when input isn't URL`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        pickSuggestion("this")
        input('b')
        assertEquals("this b", text)
        assertEquals("b", composingText)
    }

    // https://github.com/Helium314/HeliBoard/issues/215
    // https://github.com/Helium314/HeliBoard/issues/229
    @Test fun `autospace works in URL field when input isn't URL, also for multiple suggestions`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        pickSuggestion("this")
        pickSuggestion("is")
        assertEquals("this is", text)
        pickSuggestion("not")
        assertEquals("this is not", text)
        input('c')
        assertEquals("this is not c", text)
        assertEquals("c", composingText)
    }

    @Test fun `emoji is added to dictionary`() {
        // check both text and codepoint input
        reset()
        chainInput("hello ")
        input(0x1F36D)
        assertEquals(StringUtils.newSingleCodePointString(0x1F36D), lastAddedWord)
        reset()
        chainInput("hello ")
        input("🤗")
        assertEquals("\uD83E\uDD17", lastAddedWord)

        reset()
        chainInput("hello ")
        input("why 🤗 ") // not added because it's not only emoji (input can come from pasting)
        assertEquals("hello", lastAddedWord)
    }

    @Test fun `emoji uses phantom space`() {
        // check both text and codepoint input
        reset()
        pickSuggestion("hi")
        input("🤗")
        assertEquals("\uD83E\uDD17", lastAddedWord)
        assertEquals("hi \uD83E\uDD17", text)
        reset()
        pickSuggestion("hi")
        input(0x1F36D)
        assertEquals(StringUtils.newSingleCodePointString(0x1F36D), lastAddedWord)
        assertEquals("hi ${StringUtils.newSingleCodePointString(0x1F36D)}", text)
    }

    // https://github.com/Helium314/HeliBoard/issues/230
    @Test fun `no autospace after opening quotes`() {
        reset()
        chainInput("\"Hi\" \"h")
        assertEquals("\"Hi\" \"h", text)
        assertEquals("h", composingText)
        reset()
        chainInput("\"Hi\", \"h")
        assertEquals("\"Hi\", \"h", text)
        assertEquals("h", composingText)
    }

    @Test fun `autospace works in URL field when starting with quotes`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_URL_DETECTION, true) }
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
        input("\"")
        pickSuggestion("this")
        input("i")
        assertEquals("\"this i", text)
    }

    @Test fun `double space results in period and space, and delete removes the period`() {
        reset()
        chainInput("hello")
        input(' ')
        input(' ')
        assertEquals("hello. ", text)
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hello ", text)
    }

    @Test fun `no weird space inside multi-"`() {
        reset()
        chainInput("\"\"\"")
        assertEquals("\"\"\"", text)

        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("\"\"\"")
        assertEquals("\"\"\"", text)
    }

    @Test fun `autospace still happens after "`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("\"hello\"you")
        assertEquals("\"hello\" you", text)
    }

    @Test fun `autospace still happens after " if next word is in quotes`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("\"hello\"\"you\"")
        assertEquals("\"hello\" \"you\"", text)
    }

    @Test fun `autospace propagates over "`() {
        reset()
        input('"')
        pickSuggestion("hello")
        assertEquals(spaceState, SpaceState.PHANTOM) // picking a suggestion sets phantom space state
        chainInput("\"you")
        assertEquals("\"hello\" you", text)
    }

    @Test fun `autospace still happens after " if nex word is in " and after comma`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("\"hello\",\"you\"")
        assertEquals("\"hello\", \"you\"", text)
    }

    @Test fun `autospace in json editor`() {
        reset()
        latinIME.prefs().edit { putBoolean(Settings.PREF_AUTOSPACE_AFTER_PUNCTUATION, true) }
        chainInput("{\"label\":\"")
        assertEquals("{\"label\": \"", text)
        input('c')
        assertEquals("{\"label\": \"c", text)
    }

    @Test fun `text input and delete`() {
        reset()
        input("hello")
        assertEquals("hello", text)
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hell", text)

        reset()
        input("hello ")
        assertEquals("hello ", text)
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hello", text)
    }

    @Test fun `emoji text input and delete`() {
        reset()
        input("🕵🏼")
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("", text)

        reset()
        input("\uD83D\uDD75\uD83C\uDFFC")
        input(' ')
        assertEquals("🕵🏼 ", text)
        functionalKeyPress(KeyCode.DELETE)
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("", text)
    }

    // emoRegex update to unicode 16.0 was required, https://github.com/Helium314/HeliBoard/issues/1760
    @Test fun `emojis deleted one by one`() {
        reset()
        chainInput("\uD83E\uDEC6\uD83E\uDEC6\uD83E\uDEC6")
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("\uD83E\uDEC6\uD83E\uDEC6", text)
    }

    @Test fun `revert autocorrect on delete`() {
        reset()
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_AUTO_CORRECT)
        chainInput("hullo")
        getAutocorrectedWithSpaceAfter("hello", "hullo")
        assertEquals("hello ", text)
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hullo", text)

        reset()
        setInputType(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_AUTO_CORRECT)
        latinIME.prefs().edit { putBoolean(Settings.PREF_BACKSPACE_REVERTS_AUTOCORRECT, false) }
        chainInput("hullo")
        getAutocorrectedWithSpaceAfter("hello", "hullo")
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("hello", text)
    }

    @Test fun `remove glide typing word on delete`() {
        reset()
        glideTypingInput("hello")
        assertEquals("hello", text)
        functionalKeyPress(KeyCode.DELETE)
        assertEquals("", text)

        // todo: now we want some way to disable delete-all on backspace, either per setting or something else
        //  need to avoid getting into the mWordComposer.isBatchMode() part of handleBackspaceEvent
    }

    @Test fun timestamp() {
        reset()
        chainInput("hello")
        functionalKeyPress(KeyCode.TIMESTAMP)
        assertEquals(Calendar.getInstance().time.time.toDouble(),
            getTimestampFormatter(latinIME).parse(text.substring(5))!!.time.toDouble(), 1000.0)
    }

    @Test fun inlineEmojiSearchStart() {
        assertEquals(true, InputLogic.isStartOfInlineEmojiSearch('t'.code, ':'.code, ' '.code, settingsValues))
        assertEquals(false, InputLogic.isStartOfInlineEmojiSearch(' '.code, ':'.code, ' '.code, settingsValues))
        assertEquals(true, InputLogic.isStartOfInlineEmojiSearch('t'.code, ':'.code, '.'.code, settingsValues))
        assertEquals(true, InputLogic.isStartOfInlineEmojiSearch('t'.code, ':'.code, "🌍".codePoints().asSequence().last(), settingsValues))
        assertEquals(false, InputLogic.isStartOfInlineEmojiSearch('t'.code, ':'.code, 't'.code, settingsValues))
        assertEquals(false, InputLogic.isStartOfInlineEmojiSearch('t'.code, ':'.code, '3'.code, settingsValues))
    }

    @Test fun inlineEmojiSearchString() {
        assertEquals("test", InputLogic.getInlineEmojiSearchString(":test"))
        assertEquals(null, InputLogic.getInlineEmojiSearchString("test"))
        assertEquals("test", InputLogic.getInlineEmojiSearchString(" :test"))
        assertEquals(null, InputLogic.getInlineEmojiSearchString("t:test"))
        assertEquals(null, InputLogic.getInlineEmojiSearchString("6:test"))
        assertEquals("test", InputLogic.getInlineEmojiSearchString("🌍:test"))
        assertEquals("test", InputLogic.getInlineEmojiSearchString(",:test"))
        assertEquals(null, InputLogic.getInlineEmojiSearchString(":test\nt"))
        assertEquals("/48", InputLogic.getInlineEmojiSearchString("2606:127.0.0.1::/48")) // do we want this?
    }

    // ------- helper functions ---------

    // should be called before every test, so the same state is guaranteed
    private fun reset() {
        // reset input connection & facilitator
        currentScript = ScriptUtils.SCRIPT_LATIN
        text = ""
        batchEdit = 0
        currentInputType = InputType.TYPE_CLASS_TEXT
        lastAddedWord = ""

        // reset settings
        latinIME.prefs().edit { clear() }

        setText("") // (re)sets selection and composing word
    }

    private fun chainInput(text: String) = text.forEach { input(it.code) }

    private fun input(char: Char) = input(char.code)

    private fun input(codePoint: Int) {
        require(codePoint > 0) { "not a codePoint: $codePoint" }
        val oldBefore = textBeforeCursor
        val oldAfter = textAfterCursor
        val insert = StringUtils.newSingleCodePointString(codePoint)
        val phantomSpaceToInsert = if (spaceState == SpaceState.PHANTOM) " " else ""

        latinIME.onEvent(Event.createEventForCodePointFromUnknownSource(codePoint))
        handleMessages()

        if (!latinIME.prefs().getString(Settings.PREF_SELECTED_SUBTYPE, "")!!.contains("CombiningRules") // check fails if combiner merges symbols
            && !(codePoint == Constants.CODE_SPACE && oldBefore.lastOrNull() == ' ') // check fails when 2 spaces are converted into a period
            && !latinIME.mInputLogic.mSuggestedWords.mWillAutoCorrect // autocorrect obviously creates inconsistencies
            ) {
            if (phantomSpaceToInsert.isEmpty())
                assertEquals(oldBefore + insert, textBeforeCursor)
            else // in some cases autospace might be suppressed
                assert(oldBefore + phantomSpaceToInsert + insert == textBeforeCursor || oldBefore + insert == textBeforeCursor)
        }
        assertEquals(oldAfter, textAfterCursor)
        assertEquals(textBeforeCursor + textAfterCursor, getText())
        checkConnectionConsistency()
    }

    private fun functionalKeyPress(keyCode: Int) {
        require(keyCode < 0) { "not a functional key code: $keyCode" }
        latinIME.onEvent(Event.createSoftwareKeypressEvent(Event.NOT_A_CODE_POINT, keyCode, 0, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false))
        handleMessages()
        checkConnectionConsistency()
    }

    // almost the same as codePoint input, but calls different latinIME function
    private fun input(insert: String) {
        val oldBefore = textBeforeCursor
        val oldAfter = textAfterCursor
        val phantomSpaceToInsert = if (spaceState == SpaceState.PHANTOM) " " else ""

        latinIME.onTextInput(insert)
        handleMessages()

        if (phantomSpaceToInsert.isEmpty())
            assertEquals(oldBefore + insert, textBeforeCursor)
        else // in some cases autospace might be suppressed
            assert(oldBefore + phantomSpaceToInsert + insert == textBeforeCursor || oldBefore + insert == textBeforeCursor)
        assert(oldBefore + insert == textBeforeCursor || "$oldBefore $insert" == textBeforeCursor)
        assertEquals(oldAfter, textAfterCursor)
        assertEquals(textBeforeCursor + textAfterCursor, getText())
        checkConnectionConsistency()
    }

    private fun getWordAtCursor() = connection.getWordRangeAtCursor(settingsValues.mSpacingAndPunctuations, currentScript)?.mWord

    private fun setCursorPosition(start: Int, end: Int = start, weirdTextField: Boolean = false) {
        val ei = EditorInfo()
        ei.inputType = currentInputType
        ei.initialSelStart = start
        ei.initialSelEnd = end
        // imeOptions should not matter

        // adjust text in inputConnection first, otherwise fixLyingCursorPosition will move cursor
        // to the end of the text
        val fullText = textBeforeCursor + selectedText + textAfterCursor
        assertEquals(fullText, getText())

        // need to update ic before, otherwise when reloading text cache from ic, ric will load wrong text before cursor
        val oldStart = selectionStart
        val oldEnd = selectionEnd
        selectionStart = start
        selectionEnd = end
        assertEquals(fullText, textBeforeCursor + selectedText + textAfterCursor)

        latinIME.onUpdateSelection(oldStart, oldEnd, start, end, composingStart, composingEnd)
        handleMessages()

        if (weirdTextField) {
            latinIME.mHandler.onStartInput(ei, true) // essentially does nothing
            latinIME.mHandler.onStartInputView(ei, true) // does the thing
            handleMessages()
        }

        assertEquals(fullText, getText())
        assertEquals(start, selectionStart)
        assertEquals(end, selectionEnd)
        checkConnectionConsistency()
    }

    // assumes we have nothing selected
    private fun getCursorPosition(): Int {
        assertEquals(cursor, connection.expectedSelectionStart)
        assertEquals(cursor, connection.expectedSelectionEnd)
        return cursor
    }

    // just sets the text and starts input so connection it set up correctly
    private fun setText(newText: String) {
        text = newText
        selectionStart = newText.length
        selectionEnd = selectionStart
        composingStart = -1
        composingStart = -1

        // we need to start input to notify that something changed
        // restarting is false, so this is seen as a new text field
        val ei = EditorInfo()
        ei.inputType = currentInputType
        latinIME.mHandler.onStartInput(ei, false)
        latinIME.mHandler.onStartInputView(ei, false)
        handleMessages() // this is important so the composing span is set correctly
        checkConnectionConsistency()
    }

    // like selecting a suggestion from strip
    private fun pickSuggestion(suggestion: String) {
        val info = SuggestedWordInfo(suggestion, "", 0, 0, null, 0, 0)
        latinIME.pickSuggestionManually(info)
        checkConnectionConsistency()
    }

    // only works when autocorrect is on, separator after word is required
    private fun getAutocorrectedWithSpaceAfter(suggestion: String, typedWord: String?) {
        val info = SuggestedWordInfo(suggestion, "", 0, 0, null, 0, 0)
        val typedInfo = SuggestedWordInfo(typedWord, "", 0, 0, null, 0, 0)
        val sw = SuggestedWords(ArrayList(listOf(typedInfo, info)), null, typedInfo, false, true, false, 0, 0)
        latinIME.mInputLogic.setSuggestedWords(sw) // this prepares for autocorrect
        input(' ')
        checkConnectionConsistency()
    }

    private fun glideTypingInput(word: String) {
        val info = SuggestedWordInfo(word, "", 0, 0, null, 0, 0)
        val sw = SuggestedWords(ArrayList(listOf(info)), null, info, true, false, false, 0, 0)
        latinIME.mInputLogic.onUpdateTailBatchInputCompleted(settingsValues, sw, KeyboardSwitcher.getInstance())
    }

    private fun checkConnectionConsistency() {
        // RichInputConnection only has composing text up to cursor, but InputConnection has full composing text
        val expectedConnectionComposingText = if (composingStart == -1 || composingEnd == -1) ""
        else text.substring(composingStart, min(composingEnd, selectionEnd))
        assert(composingText.startsWith(expectedConnectionComposingText))
        // RichInputConnection only returns text up to cursor
        val textBeforeComposingText = if (composingStart == -1) textBeforeCursor else text.substring(0, composingStart)

        println("consistency: $selectionStart, ${connection.expectedSelectionStart}, $selectionEnd, ${connection.expectedSelectionEnd}, $textBeforeComposingText, " +
                "$connectionTextBeforeComposingText, $composingText, $connectionComposingText, $textBeforeCursor, ${connection.getTextBeforeCursor(textBeforeCursor.length, 0)}" +
                ", $textAfterCursor, ${connection.getTextAfterCursor(textAfterCursor.length, 0)}")
        assertEquals(selectionStart, connection.expectedSelectionStart)
        assertEquals(selectionEnd, connection.expectedSelectionEnd)
        assertEquals(textBeforeComposingText, connectionTextBeforeComposingText)
        assertEquals(expectedConnectionComposingText, connectionComposingText)
        assertEquals(textBeforeCursor, connection.getTextBeforeCursor(textBeforeCursor.length, 0).toString())
        assertEquals(textAfterCursor, connection.getTextAfterCursor(textAfterCursor.length, 0).toString())
    }

    private fun getText() =
        connection.getTextBeforeCursor(100, 0).toString() + (connection.getSelectedText(0) ?: "") + connection.getTextAfterCursor(100, 0)

    private fun setInputType(inputType: Int) {
        // set text to actually apply input type
        currentInputType = inputType
        setText(text)
    }

    // always need to handle messages for proper simulation
    private fun handleMessages() {
        while (messages.isNotEmpty()) {
            latinIME.mHandler.handleMessage(messages.first())
            messages.removeAt(0)
        }
        while (delayedMessages.isNotEmpty()) {
            val msg = delayedMessages.first()
            if (msg.what != 2) // MSG_UPDATE_SUGGESTION_STRIP, we want to ignore it because it's irrelevant and has a 500 ms timeout
                latinIME.mHandler.handleMessage(delayedMessages.first())
            delayedMessages.removeAt(0)
            // delayed messages may post further messages, handle before next delayed message
            while (messages.isNotEmpty()) {
                latinIME.mHandler.handleMessage(messages.first())
                messages.removeAt(0)
            }
        }
        assertEquals(0, messages.size)
        assertEquals(0, delayedMessages.size)
    }

}

private var currentInputType = InputType.TYPE_CLASS_TEXT
private var currentScript = ScriptUtils.SCRIPT_LATIN
private val messages = mutableListOf<Message>() // for latinIME / ShadowInputMethodService
private val delayedMessages = mutableListOf<Message>() // for latinIME / ShadowInputMethodService
// inputconnection stuff
private var batchEdit = 0
private var text = ""
private var selectionStart = 0
private var selectionEnd = 0
private var composingStart = -1
private var composingEnd = -1
// convenience for access
private val textBeforeCursor get() = text.substring(0, selectionStart)
private val textAfterCursor get() = text.substring(selectionEnd)
private val selectedText get() = text.substring(selectionStart, selectionEnd)
private val cursor get() = if (selectionStart == selectionEnd) selectionStart else -1

// composingText should return everything, but RichInputConnection.mComposingText only returns up to cursor
private val composingText get() = if (composingStart == -1 || composingEnd == -1) ""
    else text.substring(composingStart, composingEnd)

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
        if (textToAdd != null) {
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

// Shadows are handled by Robolectric. @Implementation overrides built-in functionality.
// This is used for avoiding crashes (LocaleManagerCompat, InputMethodManager, KeyboardSwitcher)
// and for simulating system stuff (InputMethodService for controlling the InputConnection, which
// more or less is the contents of the text field), and for setting the current script in
// KeyboardSwitcher without having to care about InputMethodSubtypes

// could also extend LatinIME, it's not final anyway
@Implements(InputMethodService::class)
class ShadowInputMethodService {
    @Implementation
    fun getCurrentInputEditorInfo() = EditorInfo().apply {
        inputType = currentInputType
        // anything else?
    }
    @Implementation
    fun getCurrentInputConnection() = ic
    @Implementation
    fun isInputViewShown() = true // otherwise selection updates will do nothing
}

@Implements(Handler::class)
class ShadowHandler {
    @Implementation
    fun sendMessage(message: Message) {
        messages.add(message)
    }
    @Implementation
    fun sendMessageDelayed(message: Message, delay: Long) {
        delayedMessages.add(message)
    }
}

@Implements(KeyboardSwitcher::class)
class ShadowKeyboardSwitcher {
    @Implementation
    // basically only needed for null check
    fun getMainKeyboardView(): MainKeyboardView = Mockito.mock(MainKeyboardView::class.java)
    @Implementation
    // only affects view
    fun setKeyboard(keyboardId: Int, toggleState: KeyboardSwitcher.KeyboardSwitchState) = Unit
    @Implementation
    // only affects view
    fun setOneHandedModeEnabled(enabled: Boolean) = Unit
    @Implementation
    fun getCurrentKeyboardScript() = currentScript
}

@Implements(DictionaryFacilitatorImpl::class)
class ShadowFacilitator2 {
    @Implementation
    fun addToUserHistory(suggestion: String, wasAutoCapitalized: Boolean,
                         ngramContext: NgramContext, timeStampInSeconds: Long,
                         blockPotentiallyOffensive: Boolean) {
        lastAddedWord = suggestion
    }
    companion object {
        var lastAddedWord = ""
    }
}
