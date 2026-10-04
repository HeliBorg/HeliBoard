/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.utils

import java.util.Locale

/**
 * A class to help with handling different writing scripts.
 */
object ScriptUtils {
    // Unicode scripts (ISO 15924), incomplete
    const val SCRIPT_UNKNOWN = "" // Used for hardware keyboards
    const val SCRIPT_ARABIC = "Arab"
    const val SCRIPT_ARMENIAN = "Armn"
    const val SCRIPT_BENGALI = "Beng"
    const val SCRIPT_CYRILLIC = "Cyrl"
    const val SCRIPT_DEVANAGARI = "Deva"
    const val SCRIPT_GEORGIAN = "Geor"
    const val SCRIPT_GREEK = "Grek"
    const val SCRIPT_HEBREW = "Hebr"
    const val SCRIPT_KANNADA = "Knda"
    const val SCRIPT_KHMER = "Khmr"
    const val SCRIPT_LAO = "Laoo"
    const val SCRIPT_LATIN = "Latn"
    const val SCRIPT_MALAYALAM = "Mlym"
    const val SCRIPT_MYANMAR = "Mymr"
    const val SCRIPT_SINHALA = "Sinh"
    const val SCRIPT_TAMIL = "Taml"
    const val SCRIPT_TELUGU = "Telu"
    const val SCRIPT_THAI = "Thai"
    const val SCRIPT_HANGUL = "Hang"
    const val SCRIPT_GUJARATI = "Gujr"

    @JvmStatic
    fun scriptSupportsUppercase(locale: Locale): Boolean {
        // only Latin, Cyrillic, Greek and Armenian have upper/lower case
        // https://unicode.org/faq/casemap_charprop.html#3
        return when (locale.script()) {
            SCRIPT_LATIN, SCRIPT_CYRILLIC, SCRIPT_GREEK, SCRIPT_ARMENIAN -> true
            else -> false
        }
    }

    /*
     * Returns whether the code point is a letter that makes sense for the specified
     * locale for this spell checker.
     * The dictionaries supported by Latin IME are described in res/xml/spellchecker.xml
     * and is limited to EFIGS languages and Russian.
     * Hence at the moment this explicitly tests for Cyrillic characters or Latin characters
     * as appropriate, and explicitly excludes CJK, Arabic and Hebrew characters.
     */
    @JvmStatic
    fun isLetterPartOfScript(codePoint: Int, script: String): Boolean {
        return when (script) {
            SCRIPT_ARABIC ->
                // Arabic letters can be in any of the following blocks:
                // Arabic U+0600..U+06FF
                // Arabic Supplement, Thaana U+0750..U+077F, U+0780..U+07BF
                // Arabic Extended-A U+08A0..U+08FF
                // Arabic Presentation Forms-A U+FB50..U+FDFF
                // Arabic Presentation Forms-B U+FE70..U+FEFF
                codePoint in 0x600..0x6FF
                        || codePoint in 0x750..0x7BF
                        || codePoint in 0x8A0..0x8FF
                        || codePoint in 0xFB50..0xFDFF
                        || codePoint in 0xFE70..0xFEFF
            SCRIPT_ARMENIAN ->
                // Armenian letters are in the Armenian unicode block, U+0530..U+058F and
                // Alphabetic Presentation Forms block, U+FB00..U+FB4F, but only in the Armenian part
                // of that block, which is U+FB13..U+FB17.
                codePoint in 0x530..0x58F || codePoint in 0xFB13..0xFB17
            SCRIPT_BENGALI ->
                // Bengali unicode block is U+0980..U+09FF
                codePoint in 0x980..0x9FF
            SCRIPT_CYRILLIC ->
                // All Cyrillic characters are in the 400~52F block. There are some in the upper
                // Unicode range, but they are archaic characters that are not used in modern
                // Russian and are not used by our dictionary.
                codePoint in 0x400..0x52F && Character.isLetter(codePoint)
            SCRIPT_DEVANAGARI ->
                // Devanagari unicode block is +0900..U+097F
                codePoint in 0x900..0x97F
            SCRIPT_GEORGIAN ->
                // Georgian letters are in the Georgian unicode block, U+10A0..U+10FF,
                // or Georgian supplement block, U+2D00..U+2D2F
                codePoint in 0x10A0..0x10FF || codePoint in 0x2D00..0x2D2F
            SCRIPT_GREEK ->
                // Greek letters are either in the 370~3FF range (Greek & Coptic), or in the
                // 1F00~1FFF range (Greek extended). Our dictionary contains both sort of characters.
                // Our dictionary also contains a few words with 0xF2; it would be best to check
                // if that's correct, but a web search does return results for these words so
                // they are probably okay.
                codePoint in 0x370..0x3FF || codePoint in 0x1F00..0x1FFF || codePoint == 0xF2
            SCRIPT_HEBREW ->
                // Hebrew letters are in the Hebrew unicode block, which spans from U+0590 to U+05FF,
                // or in the Alphabetic Presentation Forms block, U+FB00..U+FB4F, but only in the
                // Hebrew part of that block, which is U+FB1D..U+FB4F.
                codePoint in 0x590..0x5FF || codePoint in 0xFB1D..0xFB4F
            SCRIPT_KANNADA ->
                // Kannada unicode block is U+0C80..U+0CFF
                codePoint in 0xC80..0xCFF
            SCRIPT_KHMER ->
                // Khmer letters are in unicode block U+1780..U+17FF, and the Khmer symbols block
                // is U+19E0..U+19FF
                codePoint in 0x1780..0x17FF || codePoint in 0x19E0..0x19FF
            SCRIPT_LAO ->
                // The Lao block is U+0E80..U+0EFF
                codePoint in 0xE80..0xEFF
            SCRIPT_LATIN ->
                // Our supported latin script dictionaries (EFIGS) at the moment only include
                // characters in the C0, C1, Latin Extended A and B, IPA extensions unicode
                // blocks. As it happens, those are back-to-back in the code range 0x40 to 0x2AF,
                // so the below is a very efficient way to test for it. As for the 0-0x3F, it's
                // excluded from isLetter anyway.
                codePoint <= 0x2AF && Character.isLetter(codePoint)
            SCRIPT_MALAYALAM ->
                // Malayalam unicode block is U+0D00..U+0D7F
                codePoint in 0xD00..0xD7F
            SCRIPT_MYANMAR ->
                // Myanmar has three unicode blocks :
                // Myanmar U+1000..U+109F
                // Myanmar extended-A U+AA60..U+AA7F
                // Myanmar extended-B U+A9E0..U+A9FF
                codePoint in 0x1000..0x109F || codePoint in 0xAA60..0xAA7F || codePoint in 0xA9E0..0xA9FF
            SCRIPT_SINHALA ->
                // Sinhala unicode block is U+0D80..U+0DFF
                codePoint in 0xD80..0xDFF
            SCRIPT_TAMIL ->
                // Tamil unicode block is U+0B80..U+0BFF
                codePoint in 0xB80..0xBFF
            SCRIPT_TELUGU ->
                // Telugu unicode block is U+0C00..U+0C7F
                codePoint in 0xC00..0xC7F
            SCRIPT_THAI ->
                // Thai unicode block is U+0E00..U+0E7F
                codePoint in 0xE00..0xE7F
            SCRIPT_HANGUL -> codePoint in 0xAC00..0xD7A3
                    || codePoint in 0x3131..0x318E
                    || codePoint in 0x1100..0x11FF
                    || codePoint in 0xA960..0xA97C
                    || codePoint in 0xD7B0..0xD7C6
                    || codePoint in 0xD7CB..0xD7FB
            SCRIPT_GUJARATI ->
                // Gujarati unicode block is U+0A80..U+0AFF
                codePoint in 0xA80..0xAFF
            SCRIPT_UNKNOWN -> true
            else -> throw RuntimeException("Unknown value of script: $script")
        }
    }

    /**
     * returns the locale script with fallback to default scripts
     */
    @JvmStatic
    fun Locale.script(): String {
        if (script.isNotEmpty()) return script
        if (country.equals("ZZ", true)) {
            Log.w("ScriptUtils", "old _ZZ locale found: $this")
            return SCRIPT_LATIN
        }
        return when (language) {
            "ar", "ckb", "ur", "fa" -> SCRIPT_ARABIC
            "hy" -> SCRIPT_ARMENIAN
            "bn" -> SCRIPT_BENGALI
            "sr", "mk", "ru", "uk", "mn", "be", "kk", "ky", "bg", "xdq", "cv", "mhr", "mns", "dru" -> SCRIPT_CYRILLIC
            "ka" -> SCRIPT_GEORGIAN
            "el" -> SCRIPT_GREEK
            "iw", "he" -> SCRIPT_HEBREW
            "km" -> SCRIPT_KHMER
            "lo" -> SCRIPT_LAO
            "ml" -> SCRIPT_MALAYALAM
            "my" -> SCRIPT_MYANMAR
            "si" -> SCRIPT_SINHALA
            "ta" -> SCRIPT_TAMIL
            "te" -> SCRIPT_TELUGU
            "th" -> SCRIPT_THAI
            "ko" -> SCRIPT_HANGUL
            "hi", "mr", "ne" -> SCRIPT_DEVANAGARI
            "kn" -> SCRIPT_KANNADA
            "gu" -> SCRIPT_GUJARATI
            // "no language" is a latin keyboard.
            // setting to different script breaks editing existing keyboards, and hides the shift key on new custom layouts
            //"zz" -> SCRIPT_UNKNOWN
            else -> SCRIPT_LATIN // use as fallback
        }
    }

    @JvmStatic
    fun isScriptRtl(script: String): Boolean {
        return when (script) {
            SCRIPT_ARABIC, SCRIPT_HEBREW -> true
            else -> false
        }
    }
}
