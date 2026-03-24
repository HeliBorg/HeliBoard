// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.keyboard

import android.content.Context
import android.graphics.Typeface
import android.widget.TextView
import androidx.compose.ui.text.font.FontFamily
import helium314.keyboard.latin.common.isEmoji
import helium314.keyboard.latin.settings.Settings

object KeyboardTypeface {
    private val lock = Any()

    private var cachedCustomTypeface: Typeface? = null
    private var cachedCustomFontFamily: FontFamily? = null
    @Volatile
    private var customTypefaceLoaded = false

    private var cachedEmojiTypeface: Typeface? = null
    @Volatile
    private var emojiTypefaceLoaded = false

    private fun loadCustomTypeface(context: Context): Typeface? {
        return runCatching {
            Typeface.createFromFile(Settings.getCustomFontFile(context))
        }.getOrNull()
    }

    private fun loadCustomEmojiTypeface(context: Context): Typeface? {
        return runCatching {
            Typeface.createFromFile(Settings.getCustomEmojiFontFile(context))
        }.getOrNull()
    }

    @JvmStatic
    fun customTypeface(): Typeface? {
        if (customTypefaceLoaded) return cachedCustomTypeface
        val context = Settings.getCurrentContext() ?: return null
        synchronized(lock) {
            if (!customTypefaceLoaded) {
                cachedCustomTypeface = loadCustomTypeface(context)
                cachedCustomFontFamily = cachedCustomTypeface?.let(::FontFamily)
                customTypefaceLoaded = true
            }
            return cachedCustomTypeface
        }
    }

    @JvmStatic
    fun emojiTypeface(): Typeface? {
        if (emojiTypefaceLoaded) return cachedEmojiTypeface
        val context = Settings.getCurrentContext() ?: return null
        synchronized(lock) {
            if (!emojiTypefaceLoaded) {
                cachedEmojiTypeface = loadCustomEmojiTypeface(context)
                emojiTypefaceLoaded = true
            }
            return cachedEmojiTypeface
        }
    }

    @JvmStatic
    fun customFontFamily(): FontFamily? {
        if (!customTypefaceLoaded) customTypeface()
        return cachedCustomFontFamily
    }

    @JvmStatic
    fun resolve(
        text: CharSequence?,
        customTypeface: Typeface? = customTypeface(),
        emojiTypeface: Typeface? = emojiTypeface(),
        defaultTypeface: Typeface = Typeface.DEFAULT,
    ): Typeface = if (emojiTypeface != null && text != null && isEmoji(text)) {
        emojiTypeface
    } else {
        customTypeface ?: defaultTypeface
    }

    @JvmStatic
    fun applyToTextView(
        textView: TextView,
        text: CharSequence? = textView.text,
        customTypeface: Typeface? = customTypeface(),
        emojiTypeface: Typeface? = emojiTypeface(),
        defaultTypeface: Typeface = Typeface.DEFAULT,
    ) {
        textView.typeface = resolve(text, customTypeface, emojiTypeface, defaultTypeface)
    }

    @JvmStatic
    fun applyToTextView(textView: TextView) {
        applyToTextView(textView, textView.text, customTypeface(), emojiTypeface(), Typeface.DEFAULT)
    }

    @JvmStatic
    fun applyToTextView(textView: TextView, text: CharSequence?, defaultTypeface: Typeface) {
        applyToTextView(textView, text, customTypeface(), emojiTypeface(), defaultTypeface)
    }

    @JvmStatic
    fun clearCache() {
        synchronized(lock) {
            cachedCustomTypeface = null
            cachedCustomFontFamily = null
            customTypefaceLoaded = false
            cachedEmojiTypeface = null
            emojiTypefaceLoaded = false
        }
    }
}
