package helium314.keyboard.latin.utils

import android.content.SharedPreferences
import android.content.res.Resources
import android.os.Build
import android.view.inputmethod.InputMethodSubtype
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.RichInputMethodSubtype
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue
import helium314.keyboard.latin.common.Constants.Subtype.ExtraValue.KEYBOARD_LAYOUT_SET
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.common.LocaleUtils.isGoodMatch
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsSubtype.Companion.getExtraValueOf
import helium314.keyboard.latin.utils.SubtypeSettings.isEnabled
import org.xmlpull.v1.XmlPullParser
import java.util.Locale

fun InputMethodSubtype.locale(): Locale {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
        if (languageTag.isNotEmpty())
            return languageTag.constructLocale()
    }
    @Suppress("deprecation") return locale.constructLocale()
}

fun InputMethodSubtype.mainLayoutName(): String? {
    val map = LayoutType.getLayoutMap(getExtraValueOf(KEYBOARD_LAYOUT_SET) ?: "")
    return map[LayoutType.MAIN]
}

fun InputMethodSubtype.mainLayoutNameOrQwerty(): String = mainLayoutName() ?: SubtypeLocaleUtils.QWERTY

fun getResourceSubtypes(resources: Resources): List<InputMethodSubtype> {
    val subtypes = mutableListOf<InputMethodSubtype>()
    val xml = resources.getXml(R.xml.method)
    xml.setFeature(XmlPullParser.FEATURE_PROCESS_NAMESPACES, true)
    val namespace = "http://schemas.android.com/apk/res/android"
    var eventType = xml.eventType
    while (eventType != XmlPullParser.END_DOCUMENT) {
        if (eventType == XmlPullParser.START_TAG && xml.name == "subtype") {
            val icon = xml.getAttributeResourceValue(namespace, "icon", 0)
            val label = xml.getAttributeResourceValue(namespace, "label", 0)
            val subtypeId = xml.getAttributeIntValue(namespace, "subtypeId", 0)
            val localeString = xml.getAttributeValue(namespace, "imeSubtypeLocale").intern()
            val languageTag = xml.getAttributeValue(namespace, "languageTag").intern()
            val imeSubtypeMode = xml.getAttributeValue(namespace, "imeSubtypeMode")
            val imeSubtypeExtraValue = xml.getAttributeValue(namespace, "imeSubtypeExtraValue").intern()
            val isAsciiCapable = xml.getAttributeBooleanValue(namespace, "isAsciiCapable", false)
            val b = InputMethodSubtype.InputMethodSubtypeBuilder()
            b.setSubtypeIconResId(icon)
            b.setSubtypeNameResId(label)
            if (subtypeId != 0)
                b.setSubtypeId(subtypeId)
            b.setSubtypeLocale(localeString)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N)
                b.setLanguageTag(languageTag)
            b.setSubtypeMode(imeSubtypeMode)
            b.setSubtypeExtraValue(imeSubtypeExtraValue)
            b.setIsAsciiCapable(isAsciiCapable)
            subtypes.add(b.build())
        }
        eventType = xml.next()
    }
    return subtypes
}

fun getHasLocalizedNumberRow(subtype: InputMethodSubtype, prefs: SharedPreferences): Boolean =
    subtype.getExtraValueOf(ExtraValue.LOCALIZED_NUMBER_ROW)?.toBoolean()
        ?: prefs.getBoolean(Settings.PREF_LOCALIZED_NUMBER_ROW, Defaults.PREF_LOCALIZED_NUMBER_ROW)

fun getPopupKeyOrder(subtype: InputMethodSubtype, prefs: SharedPreferences): List<String> {
    val string = subtype.getExtraValueOf(ExtraValue.POPUP_ORDER)
        ?: prefs.getString(Settings.PREF_POPUP_KEYS_ORDER, Defaults.PREF_POPUP_KEYS_ORDER)!!
    return getEnabledPopupKeys(string)
}

fun getPopupKeyHintOrder(subtype: InputMethodSubtype, prefs: SharedPreferences): List<String> {
    val string = subtype.getExtraValueOf(ExtraValue.HINT_ORDER)
        ?: prefs.getString(Settings.PREF_POPUP_KEYS_HINT_ORDER, Defaults.PREF_POPUP_KEYS_HINT_ORDER)!!
    return getEnabledPopupKeys(string)
}

fun getMoreKeys(subtype: InputMethodSubtype, prefs: SharedPreferences, default: String): String =
    subtype.getExtraValueOf(ExtraValue.MORE_POPUPS)
        ?: prefs.getString(Settings.PREF_MORE_POPUP_KEYS, default)!!

fun getSecondaryLocales(extraValues: String): List<Locale> =
    extraValues.getExtraValueOf(ExtraValue.SECONDARY_LOCALES)
        ?.split(Separators.KV)?.map { it.constructLocale() }.orEmpty()

/** class for tracking subtype switching */
class SubtypeState(private val switchToSubtype: (InputMethodSubtype?) -> Unit) {
    // When HintLocales causes a subtype override, we store
    // the overridden subtype here in order to restore it when
    // we switch to another input context that has no HintLocales.
    private var overriddenByLocale: InputMethodSubtype? = null

    private var lastActiveSubtype: InputMethodSubtype? = null
    var currentSubtypeHasBeenUsed = true // starting with true avoids immediate switch
        private set

    fun setCurrentSubtypeHasBeenUsed() {
        currentSubtypeHasBeenUsed = true
    }

    // TextFields can provide locale/language hints that the IME should use via 'hintLocales',
    // and subtypes may be saved per app.
    // If a matching subtype is found, we temporarily switch to that subtype until
    // we return to a context that does not provide any hints or per-app saved subtype, or until the user
    // explicitly changes the language/subtype in use.
    // The steps are:
    // 1. if a subtype was saved for the app, and it matches any of the hint locales, we return it.
    // 2. if the current subtype matches any of the hint locales, we don't switch.
    // 3. for each hint locale, if any enabled subtype matches it, we return it.
    // 4. if a subtype was saved for the app, we fall back on it.
    // 5. we don't switch.
    fun getSubtypeForLocales(
        richImm: RichInputMethodManager,
        locales: MutableList<Locale>,
        subtypeForApp: RichInputMethodSubtype?
    ): InputMethodSubtype? {
        val overridden = overriddenByLocale
        if (locales.isEmpty() && subtypeForApp == null) {
            if (overridden != null) {
                // no locales provided, so switch back to
                // whatever subtype was used last time.
                overriddenByLocale = null

                return overridden
            }
            return null
        }

        val currentSubtype = richImm.currentSubtype

        // Try finding a subtype matching the hint languages and app saved subtype.
        var newSubtype: InputMethodSubtype? = null
        if (subtypeForApp != null) {
            if (locales.any { isGoodMatch(subtypeForApp, it) })
                newSubtype = subtypeForApp.rawSubtype // saved subtype is a good match
        }

        if (newSubtype == null) {
            if (locales.any { isGoodMatch(currentSubtype, it) }) {
                // current locales are already a good match, and we want to avoid unnecessary layout switches.
                return null
            }

            newSubtype = locales.firstNotNullOfOrNull { richImm.findSubtypeForHintLocale(it) }
        }

        if (newSubtype == null && subtypeForApp != null) {
            // fall back on saved subtype
            newSubtype = subtypeForApp.rawSubtype
        }

        if (newSubtype != null && newSubtype != currentSubtype.rawSubtype) {
            if (overridden == null) {
                // auto-switching based on hint locale, so store
                // whatever subtype was in use so we can switch back
                // to it later when there are no hint locales.
                overriddenByLocale = currentSubtype.rawSubtype
            }

            return newSubtype
        }

        return null
    }

    fun onSubtypeChanged(oldSubtype: InputMethodSubtype?, newSubtype: InputMethodSubtype?) {
        if (oldSubtype != overriddenByLocale) {
            // Whenever the subtype is changed, clear tracking
            // the subtype that is overridden by a HintLocale as
            // we no longer have a subtype to automatically switch back to.
            overriddenByLocale = null
        }
    }

    fun switchSubtype(richImm: RichInputMethodManager) {
        val currentSubtype = richImm.currentSubtype.rawSubtype
        val lastActiveSubtype = lastActiveSubtype
        val oldCurrentSubtypeHasBeenUsed = currentSubtypeHasBeenUsed
        if (oldCurrentSubtypeHasBeenUsed) {
            this@SubtypeState.lastActiveSubtype = currentSubtype
            currentSubtypeHasBeenUsed = false
        }
        if (oldCurrentSubtypeHasBeenUsed
            && isEnabled(lastActiveSubtype)
            && (currentSubtype != lastActiveSubtype)
        ) {
            switchToSubtype(lastActiveSubtype)
            return
        }
        // switchSubtype is called only for internal switching, so let's just switch to the next subtype
        switchToSubtype(richImm.getNextSubtypeInThisIme(true))
    }
}
