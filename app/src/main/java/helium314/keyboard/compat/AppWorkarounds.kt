// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.compat

import android.text.InputType
import android.view.inputmethod.EditorInfo
import helium314.keyboard.latin.utils.InputTypeUtils

object AppWorkarounds {
    fun adjustInputType(inputType: Int, packageName: String?): Int {
        return when (packageName) {
            "org.mozilla.fennec_fdroid", "org.mozilla.fenix", "org.mozilla.firefox_beta", "org.mozilla.focus",
            "org.mozilla.klar", "org.mozilla.firefox", "org.ironfoxoss.ironfox", "org.ironfoxoss.ironfox.nightly",
            "org.torproject.torbrowser", "org.torproject.torbrowser_alpha", "net.waterfox.android.release",
            "io.github.forkmaintainers.iceraven", "com.zen.web.tools.browser" -> {
                // Firefox and forks (assuming all of them) don't set these flags, so we want to force them for most text fields on websites
                // missing TYPE_TEXT_VARIATION_WEB_EDIT_TEXT is strange, considering all text fields on web pages should set it
                // missing TYPE_TEXT_FLAG_NO_SUGGESTIONS is horrible, because JS does not interact properly with composing region
                if (inputType and InputType.TYPE_MASK_CLASS != InputType.TYPE_CLASS_TEXT) return inputType
                if (inputType and InputType.TYPE_MASK_VARIATION != 0) return inputType // if any variation is specified we leave it (URL, email, password, ...)
                // looks like most (all?) non-password text fields on websites are either IME_MULTI_LINE or IME_MULTI_LINE + AUTO_CORRECT + CAP_SENTENCES
                if (inputType and InputType.TYPE_TEXT_FLAG_IME_MULTI_LINE == 0) return inputType
                // for the AUTO_CORRECT flag we assume suggestions are safe and only add WEB_EDIT_TEXT
                if (inputType and InputType.TYPE_TEXT_FLAG_AUTO_CORRECT == 0) return inputType or InputType.TYPE_TEXT_VARIATION_WEB_EDIT_TEXT
                // for all others we also add NO_SUGGESTIONS to avoid JS messing with the composing text
                inputType or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS or InputType.TYPE_TEXT_VARIATION_WEB_EDIT_TEXT
            }
            else -> inputType
        }
    }

    fun adjustImeOptions(imeOptions: Int, packageName: String?): Int {
        return when (packageName) {
            // Looks like Google decided to set inputType multiline and imeOptions no_enter_action
            // on their search bar in Pixel launcher, and all keyboards ignore the flags because otherwise
            // they would actually not perform the search action on action key. See https://github.com/Helium314/HeliBoard/issues/1989
            "com.google.android.apps.nexuslauncher" -> if (imeOptions and EditorInfo.IME_FLAG_NO_ENTER_ACTION != 0) imeOptions - EditorInfo.IME_FLAG_NO_ENTER_ACTION else imeOptions
            else -> imeOptions
        }
    }
}
