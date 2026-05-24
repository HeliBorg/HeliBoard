// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.utils

import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.content.res.Resources
import android.view.inputmethod.InputMethodSubtype
import android.widget.Toast
import androidx.core.app.LocaleManagerCompat
import androidx.core.content.edit
import helium314.keyboard.compat.locale
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.common.LocaleUtils
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsSubtype
import helium314.keyboard.latin.settings.SettingsSubtype.Companion.toSettingsSubtype
import helium314.keyboard.latin.utils.ScriptUtils.script
import java.util.Locale

object SubtypeSettings {
    /** @return enabled subtypes. If no subtypes are enabled, but a contextForFallback is provided,
     *  subtypes for system locales will be returned, or en-US if none found. */
    fun getEnabledSubtypes(fallback: Boolean = false): List<InputMethodSubtype> {
        val enabled = if (fallback && enabledSubtypes.isEmpty()) getDefaultEnabledSubtypes()
            else enabledSubtypes
        return if (Settings.getValues()?.mIsLocked != true) enabled // on app start SettingsValues are null
        else enabled + SettingsSubtype.fallbackSubtype.toAdditionalSubtype()
    }

    fun isEnabled(subtype: InputMethodSubtype?): Boolean = subtype in enabledSubtypes || subtype in getDefaultEnabledSubtypes()

    fun getAllAvailableSubtypes(): List<InputMethodSubtype> =
        resourceSubtypesByLocale.values.flatten() + additionalSubtypes

    fun getMatchingMainLayoutNameForLocale(locale: Locale): String {
        val subtypes = resourceSubtypesByLocale.values.flatten()
        val name = LocaleUtils.getBestMatch(locale, subtypes) { it.locale() }?.mainLayoutName()
        if (name != null) return name
        return when (locale.script()) {
            ScriptUtils.SCRIPT_LATIN -> "qwerty"
            ScriptUtils.SCRIPT_ARMENIAN -> "armenian_phonetic"
            ScriptUtils.SCRIPT_CYRILLIC -> "ru"
            ScriptUtils.SCRIPT_GREEK -> "greek"
            ScriptUtils.SCRIPT_HEBREW -> "hebrew"
            ScriptUtils.SCRIPT_GEORGIAN -> "georgian"
            ScriptUtils.SCRIPT_BENGALI -> "bengali_unijoy"
            else -> throw RuntimeException("Wrong script supplied: ${locale.script()}")
        }
    }

    fun addEnabledSubtype(prefs: SharedPreferences, newSubtype: InputMethodSubtype) {
        val subtype = newSubtype.toSettingsSubtype()
        val subtypes = createSettingsSubtypes(prefs.getString(Settings.PREF_ENABLED_SUBTYPES, Defaults.PREF_ENABLED_SUBTYPES)!!) + subtype
        val newString = createPrefSubtypes(subtypes)
        prefs.edit { putString(Settings.PREF_ENABLED_SUBTYPES, newString) }

        if (newSubtype !in enabledSubtypes) {
            enabledSubtypes.add(newSubtype)
            enabledSubtypes.sortBy { it.locale().toLanguageTag() } // for consistent order
            RichInputMethodManager.getInstance().refreshSubtypeCaches()
        }
    }

    /** @return whether subtype was actually removed */
    fun removeEnabledSubtype(context: Context, subtype: InputMethodSubtype): Boolean {
        val prefs = context.prefs()
        if (!removeEnabledSubtype(prefs, subtype.toSettingsSubtype())) {
            if (SubtypeUtilsAdditional.isAdditionalSubtype(subtype))
                return false
            // We want to disable a built-in subtype, but can't. This might be because it was changed in method.xml, and we definitely should disable it
            val enabledFromSettings = createSettingsSubtypes(context.prefs().getString(Settings.PREF_ENABLED_SUBTYPES, Defaults.PREF_ENABLED_SUBTYPES)!!)
            val match = enabledFromSettings.firstOrNull {
                !it.isAdditionalSubtype(context.prefs())
                    && it.locale == subtype.locale()
                    && it.mainLayoutName() == subtype.mainLayoutNameOrQwerty()
            }
            // the match is done on locale and main layout name, like in loadEnabledSubtypes
            if (match == null || !removeEnabledSubtype(prefs, match)) {
                Log.e(TAG, "tried to disable built-in ${subtype.toSettingsSubtype()}, but failed")
                return false
            }
            else {
                Log.w(TAG, "had to do some workaround to actually disable $match")
            }
        }
        if (!enabledSubtypes.remove(subtype)) reloadEnabledSubtypes(context)
        else RichInputMethodManager.getInstance().refreshSubtypeCaches()
        return true
    }

    fun getSelectedSubtype(prefs: SharedPreferences): InputMethodSubtype {
        val selectedSubtype = prefs.getString(Settings.PREF_SELECTED_SUBTYPE, Defaults.PREF_SELECTED_SUBTYPE)!!.toSettingsSubtype()
        if (selectedSubtype.isAdditionalSubtype(prefs) || selectedSubtype == SettingsSubtype.fallbackSubtype)
            return selectedSubtype.toAdditionalSubtype()
        // no additional subtype, must be a resource subtype

        val subtype = enabledSubtypes.firstOrNull { it.toSettingsSubtype() == selectedSubtype }
        if (subtype != null) {
            return subtype
        } else if (enabledSubtypes.isNotEmpty()) {
            Log.w(TAG, "selected subtype $selectedSubtype / ${prefs.getString(Settings.PREF_SELECTED_SUBTYPE, Defaults.PREF_SELECTED_SUBTYPE)} not found")
        }
        if (enabledSubtypes.isNotEmpty())
            return enabledSubtypes.first()
        val defaultSubtypes = getDefaultEnabledSubtypes()
        return defaultSubtypes.firstOrNull { it.locale() == selectedSubtype.locale && it.mainLayoutName() == it.mainLayoutName() }
            ?: defaultSubtypes.firstOrNull { it.locale().language == selectedSubtype.locale.language }
            ?: defaultSubtypes.first()
    }

    fun setSelectedSubtype(prefs: SharedPreferences, subtype: InputMethodSubtype) {
        val settingsSubtype = subtype.toSettingsSubtype()
        if (settingsSubtype.locale.toLanguageTag().isEmpty()) {
            Log.w(TAG, "tried to set subtype with empty locale: $settingsSubtype")
            return
        }
        prefs.edit { putString(Settings.PREF_SELECTED_SUBTYPE, settingsSubtype.toPref()) }
    }

    fun isAdditionalSubtype(subtype: InputMethodSubtype): Boolean = subtype in additionalSubtypes

    fun getAdditionalSubtypes(): List<InputMethodSubtype> = additionalSubtypes.toList()

    fun reloadSystemLocales(context: Context) {
        systemLocales.clear()
        try {
            val localeList = LocaleManagerCompat.getSystemLocales(context)
            (0 until localeList.size()).forEach {
                val locale = localeList[it]
                if (locale != null) systemLocales.add(locale)
            }
        } catch (_: Throwable) {
            systemLocales.add(context.resources.configuration.locale())
        }
        systemSubtypes.clear()
    }

    fun getSystemLocales(): List<Locale> = systemLocales.toList()

    fun getResourceSubtypesForLocale(locale: Locale): List<InputMethodSubtype> = resourceSubtypesByLocale[locale].orEmpty()

    fun getAvailableSubtypeLocales(): List<Locale> = resourceSubtypesByLocale.keys.toList()

    /**
     * Update subtypes that contain the layout. If new name is null (layout deleted) and the
     * subtype is now identical to a resource subtype, remove the subtype from additional subtypes.
     */
    @SuppressLint("UseKtx") // easier to read
    fun onRenameLayout(type: LayoutType, from: String, to: String?, context: Context) {
        val prefs = context.prefs()
        val editor = prefs.edit() // calling apply for each separate setting would result in an invalid intermediate state
        listOf(
            Settings.PREF_ADDITIONAL_SUBTYPES to Defaults.PREF_ADDITIONAL_SUBTYPES,
            Settings.PREF_ENABLED_SUBTYPES to Defaults.PREF_ENABLED_SUBTYPES,
            Settings.PREF_SELECTED_SUBTYPE to Defaults.PREF_SELECTED_SUBTYPE
        ).forEach { (key, default) ->
            val new = prefs.getString(key, default)!!.split(Separators.SETS).mapNotNullTo(mutableSetOf()) {
                if (it.isEmpty()) return@mapNotNullTo null
                val subtype = it.toSettingsSubtype()
                if (subtype.layoutName(type) == from) {
                    if (to == null) {
                        val defaultLayout = if (type !== LayoutType.MAIN) null
                            // if we just delete a main layout, we may end up with something like Hindi (QWERTY)
                            // so better replace it with a default layout for that locale
                            else resourceSubtypesByLocale[subtype.locale]?.first()?.mainLayoutName()
                        val newSubtype = if (defaultLayout == null) subtype.withoutLayout(type)
                            else subtype.withLayout(type, defaultLayout)
                        if (newSubtype.isSameAsDefault() && key == Settings.PREF_ADDITIONAL_SUBTYPES) null
                        else newSubtype.toPref()
                    }
                    else subtype.withLayout(type, to).toPref()
                }
                else subtype.toPref()
            }.joinToString(Separators.SETS)
            editor.putString(key, new)
        }
        editor.apply()
        if (Settings.readDefaultLayoutName(type, prefs) == from)
            Settings.writeDefaultLayoutName(to, type, prefs)
        reloadEnabledSubtypes(context)
    }

    fun reloadEnabledSubtypes(context: Context) {
        enabledSubtypes.clear()
        loadAdditionalSubtypes(context.prefs())
        loadEnabledSubtypes(context)
        if (RichInputMethodManager.isInitialized())
            RichInputMethodManager.getInstance().refreshSubtypeCaches()
    }

    fun createSettingsSubtypes(prefSubtypes: String): List<SettingsSubtype> =
        prefSubtypes.split(Separators.SETS).mapNotNull {
            if (it.isEmpty()) null
            else it.toSettingsSubtype()
        }

    fun createPrefSubtypes(subtypes: Collection<SettingsSubtype>): String =
        subtypes.map { it.toPref() }.toSortedSet().joinToString(Separators.SETS)

    fun init(context: Context) {
        SubtypeLocaleUtils.init(context) // necessary to get the correct getKeyboardLayoutSetName

        // necessary to set system locales at start, because for some weird reason (bug?)
        // LocaleManagerCompat.getSystemLocales(context) sometimes doesn't return all system locales
        reloadSystemLocales(context)

        loadResourceSubtypes(context.resources)
        loadAdditionalSubtypes(context.prefs())
        loadEnabledSubtypes(context)
    }

    @Suppress("SameReturnValue")
    private fun getDefaultEnabledSubtypes(): List<InputMethodSubtype> {
        if (systemSubtypes.isNotEmpty()) return systemSubtypes
        val subtypes = systemLocales.mapNotNull { locale ->
            val subtypesOfLocale = resourceSubtypesByLocale[locale]
            // get best match
                ?: LocaleUtils.getBestMatch(locale, resourceSubtypesByLocale.keys) {it}?.let { resourceSubtypesByLocale[it] }
            subtypesOfLocale?.firstOrNull()
        }
        if (subtypes.isEmpty()) {
            // hardcoded fallback to en-US for weird cases
            systemSubtypes.add(resourceSubtypesByLocale[Locale.US]!!.first())
        } else {
            systemSubtypes.addAll(subtypes)
        }
        return systemSubtypes
    }

    private fun loadResourceSubtypes(resources: Resources) {
        getResourceSubtypes(resources).forEach {
            resourceSubtypesByLocale.getOrPut(it.locale()) { ArrayList(2) }.add(it)
        }
    }

    private fun loadAdditionalSubtypes(prefs: SharedPreferences) {
        additionalSubtypes.clear()
        val additionalSubtypeString = prefs.getString(Settings.PREF_ADDITIONAL_SUBTYPES, Defaults.PREF_ADDITIONAL_SUBTYPES)!!
        val subtypes = SubtypeUtilsAdditional.createAdditionalSubtypes(additionalSubtypeString)
        additionalSubtypes.addAll(subtypes)
    }

    // requires loadResourceSubtypes to be called before
    private fun loadEnabledSubtypes(context: Context) {
        val prefs = context.prefs()
        val settingsSubtypes = createSettingsSubtypes(prefs.getString(Settings.PREF_ENABLED_SUBTYPES, Defaults.PREF_ENABLED_SUBTYPES)!!)
        for (settingsSubtype in settingsSubtypes) {
            if (settingsSubtype.isAdditionalSubtype(prefs)) {
                enabledSubtypes.add(settingsSubtype.toAdditionalSubtype())
                continue
            }
            val subtypesForLocale = resourceSubtypesByLocale[settingsSubtype.locale]
            if (subtypesForLocale == null) {
                val message = "no resource subtype for $settingsSubtype"
                Log.w(TAG, message)
                if (DebugFlags.DEBUG_ENABLED)
                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                else // don't remove in debug mode
                    removeEnabledSubtype(prefs, settingsSubtype)
                continue
            }

            val subtype = subtypesForLocale.firstOrNull { it.mainLayoutNameOrQwerty() == (settingsSubtype.mainLayoutName() ?: SubtypeLocaleUtils.QWERTY) }
            if (subtype == null) {
                val message = "subtype $settingsSubtype could not be loaded"
                Log.w(TAG, message)
                if (DebugFlags.DEBUG_ENABLED)
                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                else // don't remove in debug mode
                    removeEnabledSubtype(prefs, settingsSubtype)
                continue
            }

            enabledSubtypes.add(subtype)
        }
    }

    /** @return whether pref was changed */
    private fun removeEnabledSubtype(prefs: SharedPreferences, subtype: SettingsSubtype): Boolean {
        val oldSubtypes = createSettingsSubtypes(prefs.getString(Settings.PREF_ENABLED_SUBTYPES, Defaults.PREF_ENABLED_SUBTYPES)!!)
        val newSubtypes = oldSubtypes - subtype
        if (oldSubtypes == newSubtypes)
            return false // already removed
        prefs.edit { putString(Settings.PREF_ENABLED_SUBTYPES, createPrefSubtypes(newSubtypes)) }
        if (subtype == prefs.getString(Settings.PREF_SELECTED_SUBTYPE, Defaults.PREF_SELECTED_SUBTYPE)!!.toSettingsSubtype()) {
            // switch subtype if the currently used one has been disabled
            try {
                val nextSubtype = RichInputMethodManager.getInstance().getNextSubtypeInThisIme(true)
                if (subtype == nextSubtype?.toSettingsSubtype())
                    KeyboardSwitcher.getInstance().switchToSubtype(getDefaultEnabledSubtypes().first())
                else
                    KeyboardSwitcher.getInstance().switchToSubtype(nextSubtype)
            } catch (_: Exception) { } // do nothing if RichInputMethodManager isn't initialized
        }
        return true
    }

    private val enabledSubtypes = mutableListOf<InputMethodSubtype>()
    private val resourceSubtypesByLocale = LinkedHashMap<Locale, MutableList<InputMethodSubtype>>(100)
    private val additionalSubtypes = mutableListOf<InputMethodSubtype>()
    private val systemLocales = mutableListOf<Locale>()
    private val systemSubtypes = mutableListOf<InputMethodSubtype>()
    private val TAG = SubtypeSettings::class.simpleName
}
