// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.emoji

import android.R.string.cancel
import android.content.Context
import android.content.Intent
import android.content.res.Configuration
import android.os.Bundle
import android.os.Handler
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.exclude
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.LocalTextSelectionColors
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PlatformImeOptions
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.intl.Locale
import androidx.compose.ui.text.intl.LocaleList
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.KeyboardTheme
import helium314.keyboard.keyboard.internal.KeyboardBuilder
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.keyboard_parser.EMOJI_HINT_LABEL
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.keyboard.internal.keyboard_parser.getCode
import helium314.keyboard.keyboard.internal.keyboard_parser.getEmojiDefaultVersion
import helium314.keyboard.keyboard.internal.keyboard_parser.getEmojiKeyDimensions
import helium314.keyboard.keyboard.internal.keyboard_parser.getEmojiNeutralVersion
import helium314.keyboard.keyboard.internal.keyboard_parser.getEmojiPopupSpec
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.RichInputMethodSubtype
import helium314.keyboard.latin.SingleDictionaryFacilitator
import helium314.keyboard.latin.common.ColorType
import helium314.keyboard.latin.common.splitOnWhitespace
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.dictionary.DictionaryFactory
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.DictionaryInfoUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.ResourceUtils
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.CloseIcon
import helium314.keyboard.latin.utils.SearchIcon
import kotlin.properties.Delegates

private const val TAG = "emoji-search"

/**
 * This activity is displayed in a gap created for it above the keyboard and below the host app, and disables the host app.
 */
class EmojiSearchActivity : ComponentActivity() {
    private val colors = Settings.getValues().mColors
    private var imeOpened = false
    private var firstSearchDone = false
    private var screenHeight by Delegates.notNull<Int>()
    private lateinit var hintLocales: LocaleList
    private lateinit var emojiPageKeyboardView: EmojiPageKeyboardView
    private lateinit var keyboardParams: KeyboardParams
    private var keyWidth by Delegates.notNull<Float>()
    private var keyHeight by Delegates.notNull<Float>()
    private var firstKey: Key? = null
    private var pressedKey: Key? = null
    private var imeVisible = false
    private var imeClosed = false

    private val closer = Runnable {
        if (!imeVisible) {
            Log.d(TAG, "IME closed")
            imeClosed = true
            cancel()
        }
    }

    @OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        init()
        enableEdgeToEdge()
        setContent {
            LocalContext.current.setTheme(KeyboardTheme.getKeyboardTheme(this).mStyleId)
            Surface(modifier = Modifier.fillMaxSize(), color = Color(0x80000000)) {
                var heightDp by remember { mutableStateOf(0.dp) }
                Column(modifier = Modifier.fillMaxSize().clickable(onClick = { cancel() })
                    .windowInsetsPadding(WindowInsets.safeDrawing.exclude(WindowInsets(bottom = heightDp))),
                    verticalArrangement = Arrangement.Bottom
                ) {
                    val localDensity = LocalDensity.current
                    var heightPx by remember { mutableIntStateOf(0) }
                    Column(modifier = Modifier.wrapContentHeight().background(Color(colors.get(ColorType.MAIN_BACKGROUND)))
                        .clickable(false) {}.onGloballyPositioned {
                            val bottom = it.localToScreen(Offset(0f, it.size.height.toFloat())).y.toInt()
                            imeVisible = bottom < screenHeight - 100
                            Log.d(TAG, "imeVisible: $imeVisible, firstSearchDone: $firstSearchDone, imeOpened: $imeOpened, " +
                                "bottom: $bottom, keyboardState: ${KeyboardSwitcher.getInstance().keyboardSwitchState}")
                            if (imeOpened && !imeVisible) {
                                Handler(this@EmojiSearchActivity.mainLooper).postDelayed(closer, 200)
                            }
                            if (imeOpened && !isAlphaKeyboard()) {
                                cancel()
                                return@onGloballyPositioned
                            }
                            if (imeVisible && firstSearchDone && isAlphaKeyboard()) {
                                Log.d(TAG, "IME opened in onGloballyPositioned")
                                imeOpened = true
                                Handler(this@EmojiSearchActivity.mainLooper).removeCallbacks(closer)
                            }
                            heightPx = it.size.height
                            heightDp = with(localDensity) { it.size.height.toDp() }
                        }) {
                        Row(modifier = Modifier.fillMaxWidth().height(30.dp)) {
                            IconButton(onClick = { cancel() }) {
                                Icon(painter = painterResource(R.drawable.ic_arrow_back),
                                    stringResource(R.string.spoken_description_action_previous),
                                    tint = Color(colors.get(ColorType.EMOJI_KEY_TEXT)))
                            }
                            Text(text = stringResource(R.string.emoji_search_title), fontSize = 18.sp,
                                color = Color(colors.get(ColorType.EMOJI_KEY_TEXT)),
                                modifier = Modifier.fillMaxWidth().align(Alignment.CenterVertically))
                        }
                        key(emojiPageKeyboardView) {
                            AndroidView({ emojiPageKeyboardView }, modifier = Modifier.wrapContentHeight().fillMaxWidth())
                        }
                        val focusRequester = remember { FocusRequester() }
                        var text by remember { mutableStateOf(TextFieldValue(searchText, selection = TextRange(searchText.length))) }
                        val textFieldColors = TextFieldDefaults.colors().copy(
                            unfocusedContainerColor = Color(colors.get(ColorType.FUNCTIONAL_KEY_BACKGROUND)),
                            unfocusedTextColor = Color(colors.get(ColorType.FUNCTIONAL_KEY_TEXT)),
                            cursorColor = Color(colors.get(ColorType.FUNCTIONAL_KEY_TEXT)),
                            unfocusedLeadingIconColor = Color(colors.get(ColorType.FUNCTIONAL_KEY_TEXT)),
                            unfocusedTrailingIconColor = Color(colors.get(ColorType.FUNCTIONAL_KEY_TEXT)),
                            unfocusedPlaceholderColor = lerp(Color(colors.get(ColorType.FUNCTIONAL_KEY_BACKGROUND)),
                                Color(colors.get(ColorType.FUNCTIONAL_KEY_TEXT)), 0.5f))
                        CompositionLocalProvider(LocalTextSelectionColors provides textFieldColors.textSelectionColors) {
                            BasicTextField(
                                value = text,
                                modifier = Modifier.fillMaxWidth().heightIn(20.dp, 30.dp).focusRequester(focusRequester),
                                textStyle = TextStyle(textDirection = TextDirection.Content, color = textFieldColors.unfocusedTextColor),
                                onValueChange = {
                                    text = it
                                    search(it.text)
                                },
                                enabled = true,
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text, imeAction = ImeAction.Done,
                                    hintLocales = hintLocales,
                                    platformImeOptions = PlatformImeOptions(encodePrivateImeOptions(PrivateImeOptions(heightPx)))),
                                keyboardActions = KeyboardActions(onDone = {
                                    if (Settings.getValues().mAutoCorrectEnabled) pressedKey = firstKey
                                    finish()
                                }),
                                singleLine = true,
                                cursorBrush = SolidColor(textFieldColors.cursorColor)
                            ) {
                                TextFieldDefaults.DecorationBox(
                                    value = text.text,
                                    colors = textFieldColors,

                                    /**
                                     * This is the reason for not using [androidx.compose.material3.TextField],
                                     * which uses [TextFieldDefaults.contentPaddingWithoutLabel]
                                     */
                                    contentPadding = PaddingValues(2.dp),
                                    visualTransformation = VisualTransformation.None,
                                    innerTextField = it,
                                    placeholder = { Text(stringResource(R.string.search_field_placeholder)) },
                                    leadingIcon = { SearchIcon() },
                                    trailingIcon = {
                                        IconButton(onClick = {
                                            text = TextFieldValue()
                                            search("")
                                        }) { CloseIcon(cancel) }
                                    },
                                    singleLine = true,
                                    enabled = true,
                                    interactionSource = MutableInteractionSource(),
                                )
                            }
                        }
                        LaunchedEffect(Unit) { focusRequester.requestFocus() }
                    }
                }
            }
        }
    }

    override fun onEnterAnimationComplete() {
        Log.d(TAG, "onEnterAnimationComplete")
        search(searchText)
        Log.d(TAG, "initial search done")
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        init()
        imeVisible = false
        imeOpened = false
        firstSearchDone = false
        search(searchText)
    }

    override fun onStop() {
        val intent = Intent(this, LatinIME::class.java).setAction(EMOJI_SEARCH_DONE_ACTION)
            .putExtra(IME_CLOSED_KEY, imeClosed)
        pressedKey?.let {
            intent.putExtra(EMOJI_KEY, if (it.code == KeyCode.MULTIPLE_CODE_POINTS)
                it.getOutputText()
            else
                Character.toString(it.code))

            KeyboardSwitcher.getInstance().emojiPalettesView.addRecentKey(it)
        }
        startService(intent)
        super.onStop()
    }

    private fun init() {
        Log.d(TAG, "init start")
        @Suppress("DEPRECATION")
        screenHeight = windowManager.defaultDisplay.height
        Log.d(TAG, "screenHeight: $screenHeight")
        hintLocales = LocaleList(DictionaryInfoUtils.getLocalesWithEmojiDicts(this).map { Locale(it.toLanguageTag()) })
        val keyboardWidth = ResourceUtils.getKeyboardWidth(this, Settings.getValues())
        val layoutSet = KeyboardLayoutSet.Builder(this, null).setSubtype(RichInputMethodSubtype.emojiSubtype)
            .setKeyboardGeometry(keyboardWidth, EmojiLayoutParams(resources).emojiKeyboardHeight).build()

        // Initialize default versions and popup specs
        layoutSet.getKeyboard(KeyboardId.ELEMENT_EMOJI_CATEGORY2)

        val keyboard = DynamicGridKeyboard.ofRowCount(prefs(), layoutSet.getKeyboard(KeyboardId.ELEMENT_EMOJI_RECENTS),
            if (resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE) 1 else 2,
            KeyboardId.ELEMENT_EMOJI_CATEGORY16, keyboardWidth)
        val builder = KeyboardBuilder(this, KeyboardParams())
        builder.load(keyboard.mId)
        keyboardParams = builder.mParams
        val (width, height) = getEmojiKeyDimensions(keyboardParams, this)
        keyWidth = width
        keyHeight = height
        emojiPageKeyboardView = EmojiPageKeyboardView(this, null)
        emojiPageKeyboardView.setKeyboard(keyboard)
        emojiPageKeyboardView.layoutParams =
            ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        emojiPageKeyboardView.background = null
        colors.setBackground(emojiPageKeyboardView, ColorType.MAIN_BACKGROUND)
        emojiPageKeyboardView.setPadding(0, 10, 0, 10)

        emojiPageKeyboardView.setEmojiViewCallback(object : EmojiViewCallback {
            override fun onPressKey(key: Key) {
            }

            override fun onReleaseKey(key: Key) {
                pressedKey = key
                finish()
            }

            override fun getDescription(emoji: String): String? = if (Settings.getValues().mShowEmojiDescriptions)
                dictionaryFacilitator?.getWordProperty(getEmojiNeutralVersion(emoji))?.let {
                    if (it.mHasShortcuts) it.mShortcutTargets[0]?.mWord else null
                } else null
        })
        KeyboardSwitcher.getInstance().setAlphabetKeyboard()
        Log.d(TAG, "init end")
    }

    private fun isAlphaKeyboard() = KeyboardSwitcher.getInstance().keyboardSwitchState !in
        setOf(KeyboardSwitcher.KeyboardSwitchState.EMOJI, KeyboardSwitcher.KeyboardSwitchState.CLIPBOARD)

    private fun search(text: String) {
        initDictionaryFacilitator(this)
        if (dictionaryFacilitator == null) {
            cancel()
            return
        }

        if (firstSearchDone && text == searchText) {
            return
        }

        if (KeyboardSwitcher.getInstance().keyboard == null) {
            /** Avoid crash in [SingleDictionaryFacilitator.getSuggestions] */
            return
        }

        val keyboard = emojiPageKeyboardView.keyboard as DynamicGridKeyboard
        keyboard.removeAllKeys()
        firstKey = null
        pressedKey = null
        dictionaryFacilitator!!.getSuggestions(text.splitOnWhitespace()).filter { it.isEmoji }.forEach {
            val emoji = getEmojiDefaultVersion(it.word)
            val popupSpec = getEmojiPopupSpec(emoji)
            val keyParams = Key.KeyParams(emoji, emoji.getCode(), if (popupSpec != null) EMOJI_HINT_LABEL else null, popupSpec,
                Key.LABEL_FLAGS_FONT_NORMAL, keyboardParams)
            keyParams.mAbsoluteWidth = keyWidth
            keyParams.mAbsoluteHeight = keyHeight
            val key = keyParams.createKey()
            keyboard.addKeyLast(key)
            if (firstKey == null) firstKey = key
        }
        emojiPageKeyboardView.invalidate()

        searchText = text
        firstSearchDone = true
        if (imeVisible && !imeOpened) {
            Log.d(TAG, "IME opened in search")
            imeOpened = true
        }
    }

    private fun cancel() {
        finish()
    }

    @JvmRecord
    data class PrivateImeOptions(val height: Int)

    companion object {
        const val EMOJI_SEARCH_DONE_ACTION: String = "EMOJI_SEARCH_DONE"
        const val IME_CLOSED_KEY: String = "IME_CLOSED"
        const val EMOJI_KEY: String = "EMOJI"
        private const val PRIVATE_IME_OPTIONS_PREFIX: String = "helium314.keyboard.keyboard.emoji.search"
        private var dictionaryFacilitator: SingleDictionaryFacilitator? = null
        private var searchText: String = ""

        fun decodePrivateImeOptions(editorInfo: EditorInfo?): PrivateImeOptions = PrivateImeOptions(
            editorInfo?.privateImeOptions?.takeIf { it.startsWith(PRIVATE_IME_OPTIONS_PREFIX) }
                ?.let { it.substring(PRIVATE_IME_OPTIONS_PREFIX.length + 1, it.indexOf(',')) }?.toInt() ?: 0)

        fun closeDictionaryFacilitator() {
            dictionaryFacilitator?.closeDictionaries()
            dictionaryFacilitator = null
        }

        private fun encodePrivateImeOptions(privateImeOptions: PrivateImeOptions) =
            "$PRIVATE_IME_OPTIONS_PREFIX.${privateImeOptions.height},"

        private fun initDictionaryFacilitator(context: Context) {
            val locale = RichInputMethodManager.getInstance().currentSubtype.locale
            if (dictionaryFacilitator?.isForLocale(locale) != true) {
                dictionaryFacilitator?.closeDictionaries()
                dictionaryFacilitator = DictionaryInfoUtils.getCachedDictForLocaleAndType(locale, Dictionary.TYPE_EMOJI, context)
                    ?.let { DictionaryFactory.getDictionary(it, locale) }?.let { SingleDictionaryFacilitator(it) }
            }
        }
    }
}
