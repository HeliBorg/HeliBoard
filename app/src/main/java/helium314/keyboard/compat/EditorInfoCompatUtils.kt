/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.compat

import android.os.Build
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.core.view.inputmethod.EditorInfoCompat
import helium314.keyboard.latin.utils.Log
import java.util.*

object EditorInfoCompatUtils {

    @JvmStatic
    fun imeActionName(imeOptions: Int): String {
        return when (val actionId = imeOptions and EditorInfo.IME_MASK_ACTION) {
            EditorInfo.IME_ACTION_UNSPECIFIED -> "actionUnspecified"
            EditorInfo.IME_ACTION_NONE -> "actionNone"
            EditorInfo.IME_ACTION_GO -> "actionGo"
            EditorInfo.IME_ACTION_SEARCH -> "actionSearch"
            EditorInfo.IME_ACTION_SEND -> "actionSend"
            EditorInfo.IME_ACTION_NEXT -> "actionNext"
            EditorInfo.IME_ACTION_DONE -> "actionDone"
            EditorInfo.IME_ACTION_PREVIOUS -> "actionPrevious"
            else -> "actionUnknown($actionId)"
        }
    }

    fun debugLog(editorInfo: EditorInfo, tag: String) {
        val format = HexFormat {
            upperCase = true
            number {
                prefix = "0x"
                minLength = 8
            }
        }
        Log.d(tag, "editorInfo: inputType: ${editorInfo.inputType.toHexString(format)}, imeOptions: ${editorInfo.imeOptions.toHexString(format)}")
        val allCaps = (editorInfo.inputType and InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS) != 0
        val sentenceCaps = (editorInfo.inputType and InputType.TYPE_TEXT_FLAG_CAP_SENTENCES) != 0
        val wordCaps = (editorInfo.inputType and InputType.TYPE_TEXT_FLAG_CAP_WORDS) != 0
        Log.d(tag, ("All caps: $allCaps, sentence caps: $sentenceCaps, word caps: $wordCaps"))
    }

    @JvmStatic
    fun isMimeTypeSupportedByEditor(editorInfo: EditorInfo?, mimeType: String): Boolean {
        if (editorInfo == null) return false
        val supportedMimeTypes = EditorInfoCompat.getContentMimeTypes(editorInfo)
        return supportedMimeTypes.any { supported ->
            supported == "*/*" || supported == mimeType
                    || (supported.endsWith("/*") && mimeType.startsWith(supported.removeSuffix("*")))
        }
    }

    @JvmStatic
    fun getHintLocales(editorInfo: EditorInfo?): List<Locale> {
        if (editorInfo == null || Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            return listOf()
        }
        val localeList = editorInfo.hintLocales ?: return listOf()
        val locales = ArrayList<Locale>(localeList.size())
        for (i in 0 until localeList.size()) {
            locales.add(localeList.get(i))
        }
        return locales
    }
}
