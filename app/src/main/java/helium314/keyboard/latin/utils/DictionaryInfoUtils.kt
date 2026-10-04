/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.utils

import android.content.Context
import android.text.TextUtils
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.common.FileUtils
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.common.loopOverCodePoints
import helium314.keyboard.latin.define.DecoderSpecificConstants
import helium314.keyboard.latin.makedict.DictionaryHeader
import helium314.keyboard.latin.makedict.UnsupportedFormatException
import helium314.keyboard.latin.settings.SpacingAndPunctuations
import java.io.File
import java.io.IOException
import java.util.Locale

/** encapsulates the logic for the Latin-IME side of dictionary information management */
object DictionaryInfoUtils {
    private val TAG = DictionaryInfoUtils::class.java.simpleName
    const val DEFAULT_MAIN_DICT = "main"
    const val USER_DICTIONARY_SUFFIX = "user.dict"
    const val MAIN_DICT_PREFIX = DEFAULT_MAIN_DICT + "_"
    const val ASSETS_DICTIONARY_FOLDER = "dicts"
    const val MAIN_DICT_FILE_NAME = "$DEFAULT_MAIN_DICT.dict"
    private const val MAX_HEX_DIGITS_FOR_CODEPOINT = 6 // unicode is limited to 21 bits

    /**
     * Returns whether we may want to use this character as part of a file name.
     * This basically only accepts ascii letters and numbers, and rejects everything else.
     */
    private fun isFileNameCharacter(codePoint: Int): Boolean {
        if (codePoint in 0x30..0x39) return true // Digit
        if (codePoint in 0x41..0x5A) return true // Uppercase
        if (codePoint in 0x61..0x7A) return true // Lowercase
        return codePoint == '_'.code || codePoint == '-'.code
    }

    /**
     * Escapes a string for any characters that may be suspicious for a file or directory name.
     *
     * Concretely this does a sort of URL-encoding except it will encode everything that's not
     * alphanumeric or underscore. (true URL-encoding leaves alone characters like '*', which
     * we cannot allow here)
     */
    private fun replaceFileNameDangerousCharacters(name: String): String {
        // This assumes '%' is fully available as a non-separator, normal
        // character in a file name. This is probably true for all file systems.
        val sb = StringBuilder()
        loopOverCodePoints(name) { codePoint, _ ->
            if (isFileNameCharacter(codePoint)) {
                sb.appendCodePoint(codePoint)
            } else {
                sb.append(String.format(Locale.US, "%%%1$0" + MAX_HEX_DIGITS_FOR_CODEPOINT + "x", codePoint))
            }
            false
        }
        return sb.toString()
    }

    // we cache the extracted dictionaries in filesDir, because actual cache might be cleared at
    // any time, and we can't permanently check whether the dictionary still exists
    fun getWordListCacheDirectory(context: Context): String = context.filesDir?.toString() + File.separator + "dicts"

    /** Reverse escaping done by replaceFileNameDangerousCharacters. */
    fun getWordListIdFromFileName(fname: String): String {
        val sb = StringBuilder()
        val fnameLength = fname.length
        var i = 0
        while (i < fnameLength) {
            val codePoint = fname.codePointAt(i)
            if ('%'.code != codePoint) {
                sb.appendCodePoint(codePoint)
            } else {
                // + 1 to pass the % sign
                val encodedCodePoint = fname.substring(i + 1, i + 1 + MAX_HEX_DIGITS_FOR_CODEPOINT).toInt(16)
                i += MAX_HEX_DIGITS_FOR_CODEPOINT
                sb.appendCodePoint(encodedCodePoint)
            }
            i = fname.offsetByCodePoints(i, 1)
        }
        return sb.toString()
    }

    /** Helper method to the list of non-empty cache directories, one for each distinct locale. */
    fun getCacheDirectories(context: Context) = File(getWordListCacheDirectory(context)).listFiles()
        ?.filter { it.isDirectory && !it.list().isNullOrEmpty() }.orEmpty()

    /** Find out the cache directory associated with a specific locale. */
    fun getCacheDirectoryForLocale(locale: Locale, context: Context): String? {
        val relativeDirectoryName = replaceFileNameDangerousCharacters(locale.toLanguageTag())
        val absoluteDirectoryName = getWordListCacheDirectory(context) + File.separator + relativeDirectoryName
        val directory = File(absoluteDirectoryName)
        if (!directory.exists() && !directory.mkdirs()) {
            Log.e(TAG, "Could not create the directory for locale $locale")
            return null
        }
        return absoluteDirectoryName
    }

    @JvmStatic
    fun getLocalesWithEmojiDicts(context: Context): List<Locale> =
        SubtypeSettings.getEnabledSubtypes(true)
            .map { it.locale() }.filter { getCachedDictForLocaleAndType(it, Dictionary.TYPE_EMOJI, context) != null }

    @JvmStatic
    fun getCachedDictForLocaleAndType(locale: Locale, type: String, context: Context): File? =
        getCachedDictsForLocale(locale, context).firstOrNull { it.name.substringBefore("_") == type }

    fun getCachedDictsForLocale(locale: Locale, context: Context) =
        getCacheDirectoryForLocale(locale, context)?.let { File(it).listFiles() }.orEmpty()

    fun getDictionaryFileHeaderOrNull(file: File): DictionaryHeader? {
        return try {
            BinaryDictionaryUtils.getHeader(file)
        } catch (_: UnsupportedFormatException) {
            null
        } catch (_: IOException) {
            null
        }
    }

    /**
     * Returns the locale for a dictionary file name stored in assets.
     *
     * Assumes file name main_[locale].dict
     * Returns the locale, or null if file name does not match the pattern
     */
    fun extractLocaleFromAssetsDictionaryFile(dictionaryFileName: String): Locale {
        if (dictionaryFileName.contains('_') && !dictionaryFileName.contains('.'))
            throw IllegalStateException("invalid asset dictionary name $dictionaryFileName")
        return dictionaryFileName.substringAfter("_").substringBefore(".").constructLocale()
    }

    // actually we could extract assets dicts to unprotected storage
    fun extractAssetsDictionary(dictionaryFileName: String, locale: Locale, context: Context): File? {
        val cacheDir = getCacheDirectoryForLocale(locale, context) ?: return null
        val targetFile = File(cacheDir, "${dictionaryFileName.substringBefore("_")}.dict")
        try {
            FileUtils.copyStreamToNewFile(
                context.assets.open(ASSETS_DICTIONARY_FOLDER + File.separator + dictionaryFileName),
                targetFile
            )
        } catch (e: IOException) {
            Log.e(TAG, "Could not extract assets dictionary $dictionaryFileName", e)
            return null
        }
        return targetFile
    }

    fun getAssetsDictionaryList(context: Context): Array<String>? =
        runCatching { context.assets.list(ASSETS_DICTIONARY_FOLDER) }.getOrNull()

    @JvmStatic
    fun looksValidForDictionaryInsertion(text: CharSequence, spacingAndPunctuations: SpacingAndPunctuations): Boolean {
        if (TextUtils.isEmpty(text)) {
            return false
        }
        if (text.length > DecoderSpecificConstants.DICTIONARY_MAX_WORD_LENGTH) {
            return false
        }
        var digitCount = 0
        loopOverCodePoints(text) { codePoint, charCount ->
            if (Character.isDigit(codePoint)) {
                // Count digits: see below
                digitCount += charCount
                return@loopOverCodePoints false
            }
            if (!spacingAndPunctuations.isWordCodePoint(codePoint)) {
                return false
            }
            false
        }
        // We reject strings entirely comprised of digits to avoid using PIN codes or credit
        // card numbers. It would come in handy for word prediction though; a good example is
        // when writing one's address where the street number is usually quite discriminative,
        // as well as the postal code.
        return digitCount < text.length
    }
}
