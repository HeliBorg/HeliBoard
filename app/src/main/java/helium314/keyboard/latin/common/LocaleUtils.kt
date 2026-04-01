/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.common

import android.content.res.Resources
import helium314.keyboard.compat.locale
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodSubtype
import helium314.keyboard.latin.utils.ScriptUtils.script
import helium314.keyboard.latin.utils.SubtypeLocaleUtils
import helium314.keyboard.latin.utils.getSecondaryLocales
import helium314.keyboard.latin.utils.runInLocale
import java.util.Locale

/**
 * A class to help with handling Locales in string form.
 *
 *
 * This file has the same meaning and features (and shares all of its code) with the one with the
 * same name in Latin IME. They need to be kept synchronized; for any update/bugfix to
 * this file, consider also updating/fixing the version in Latin IME.
 */
object LocaleUtils {
    // Locale match level constants.
    // A higher level of match is guaranteed to have a higher numerical value.
    // Some room is left within constants to add match cases that may arise necessary
    // in the future, for example differentiating between the case where the countries
    // are both present and different, and the case where one of the locales does not
    // specify the countries. This difference is not needed now.
    // Nothing matches.
    private const val LOCALE_NO_MATCH = 0

    // The language (and maybe more) matches, but the script is different
    private const val LOCALE_MATCH_SCRIPT_DIFFER = 1

    // The languages matches, but the country are different. Or, the reference locale requires a
    // country and the tested locale does not have one.
    private const val LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER = 3

    // The languages and country match, but the variants are different. Or, the reference locale
    // requires a variant and the tested locale does not have one.
    private const val LOCALE_LANGUAGE_AND_COUNTRY_MATCH_VARIANT_DIFFER = 6

    // The required locale is null or empty so it will accept anything, and the tested locale
    // is non-null and non-empty.
    private const val LOCALE_ANY_MATCH = 10

    // The language matches, and the tested locale specifies a country but the reference locale
    // does not require one.
    private const val LOCALE_LANGUAGE_MATCH = 15

    // The language and the country match, and the tested locale specifies a variant but the
    // reference locale does not require one.
    private const val LOCALE_LANGUAGE_AND_COUNTRY_MATCH = 20

    // The compared locales are fully identical. This is the best match level.
    private const val LOCALE_FULL_MATCH = 30

    const val LOCALE_GOOD_MATCH = LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER

    /**
     * Return how well a tested locale matches a reference locale.
     *
     *
     * This will check the tested locale against the reference locale and return a measure of how
     * a well it matches the reference. The general idea is that the tested locale has to match
     * every specified part of the required locale. A full match occur when they are equal, a
     * partial match when the tested locale agrees with the reference locale but is more specific,
     * and a difference when the tested locale does not comply with all requirements from the
     * reference locale.
     * In more detail, if the reference locale specifies at least a language and the testedLocale
     * does not specify one, or specifies a different one, LOCALE_NO_MATCH is returned. If the
     * reference locale is empty or null, it will match anything - in the form of LOCALE_FULL_MATCH
     * if the tested locale is empty or null, and LOCALE_ANY_MATCH otherwise. If the reference and
     * tested locale agree on the language, but not on the country,
     * LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER is returned if the reference locale specifies a country,
     * and LOCALE_LANGUAGE_MATCH otherwise.
     * If they agree on both the language and the country, but not on the variant,
     * LOCALE_LANGUAGE_AND_COUNTRY_MATCH_VARIANT_DIFFER is returned if the reference locale
     * specifies a variant, and LOCALE_LANGUAGE_AND_COUNTRY_MATCH otherwise. If everything matches,
     * LOCALE_FULL_MATCH is returned.
     * Examples:
     * en <=> en_US  => LOCALE_LANGUAGE_MATCH
     * en_US <=> en => LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER
     * en_US_POSIX <=> en_US_Android  =>  LOCALE_LANGUAGE_AND_COUNTRY_MATCH_VARIANT_DIFFER
     * en_US <=> en_US_Android => LOCALE_LANGUAGE_AND_COUNTRY_MATCH
     * sp_US <=> en_US  =>  LOCALE_NO_MATCH
     * de <=> de  => LOCALE_FULL_MATCH
     * en_US <=> en_US => LOCALE_FULL_MATCH
     * "" <=> en_US => LOCALE_ANY_MATCH
     *
     * @param reference the reference locale to test against.
     * @param tested the locale to test.
     * @return a constant that measures how well the tested locale matches the reference locale.
     */
    fun getMatchLevel(reference: Locale, tested: Locale): Int {
        if (reference == tested) return LOCALE_FULL_MATCH
        if (reference.toString().isEmpty()) return LOCALE_ANY_MATCH
        if (reference.language != tested.language) return LOCALE_NO_MATCH
        // language matches
        if (reference.script() != tested.script()) {
            return LOCALE_MATCH_SCRIPT_DIFFER
        }
        // script matches
        if (reference.country != tested.country) {
            return if (reference.country.isEmpty()) LOCALE_LANGUAGE_MATCH
                else LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER
        }
        // country matches
        return if (reference.variant == tested.variant) LOCALE_FULL_MATCH
            else if (reference.variant.isEmpty()) LOCALE_LANGUAGE_AND_COUNTRY_MATCH
            else LOCALE_LANGUAGE_AND_COUNTRY_MATCH_VARIANT_DIFFER
    }

    @JvmStatic
    fun isGoodMatch(subtype: RichInputMethodSubtype, locale: Locale) =
        getMatchLevel(locale, subtype.locale) >= LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER
            || getSecondaryLocales(subtype.rawSubtype.extraValue)
                .any { getMatchLevel(locale, it) >= LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER }

    @JvmStatic
    fun <T> getBestMatch(locale: Locale, collection: Collection<T>, toLocale: (T) -> Locale): T? {
        var best: T? = null
        var bestLevel = 0
        collection.forEach {
            val level = getMatchLevel(locale, toLocale(it))
            if (level > bestLevel && level >= LOCALE_LANGUAGE_MATCH_COUNTRY_DIFFER) {
                bestLevel = level
                best = it
            }
        }
        return best
    }

    private val sLocaleCache = HashMap<String, Locale>()

    /**
     * Creates a locale from a string specification or language tag.
     * Ideally this works as reverse of Locale.toString and Locale.toLanguageTag
     * If a localeString contains "-" it is always interpreted as language tag.
     * localeString is a string specification of a locale, in a format of "ll_cc_variant" where
     * "ll" is a language code, "cc" is a country code.
     * The script may also be part of the locale string, e.g. "ll_cc_#script"
     * Converts "ZZ" regions that used to signal latin script into actual latin script.
     * "cc" / region should be uppercase and language should be lowercase, this is automatically converted
     */
    @JvmStatic
    fun String.constructLocale(): Locale {
        synchronized(sLocaleCache) {
            sLocaleCache[this]?.let { return it }
            if (contains("-")) {
                // looks like it's actually a language tag, and not a locale string
                val locale = Locale.forLanguageTag(this)
                sLocaleCache[this] = locale
                return locale
            }
            val elements = split("_", limit = 3)
            val language = elements[0].lowercase()
            val region = elements.getOrNull(1)?.uppercase()
            val locale = if (elements.size == 1) {
                Locale(language) // "zz" works both in constructor and forLanguageTag
            } else if (elements.size == 2) {
                if (region == "ZZ") Locale.forLanguageTag(elements[0] + "-Latn")
                else Locale(language, region!!)
            } else if (language == SubtypeLocaleUtils.NO_LANGUAGE) { // localeParams.length == 3
                Locale.Builder().setLanguage(language).setVariant(elements[2]).setScript("Latn").build()
            } else if (elements[2].startsWith("#")) {
                // best guess: elements[2] is a script, e.g. sr-Latn locale to string is sr__#Latn
                Locale.Builder().setLanguage(language).setRegion(region).setScript(elements[2].substringAfter("#")).build()
            } else {
                Locale(language, region!!, elements[2])
            }
            sLocaleCache[this] = locale
            return locale
        }
    }

    fun Locale.localizedDisplayName(resources: Resources, displayLocale: Locale? = null): String {
        val languageTag = toLanguageTag()
        if (languageTag == SubtypeLocaleUtils.NO_LANGUAGE)
            return resources.getString(R.string.subtype_no_language)

        val overrideResId = when (languageTag) {
            "en-US" -> R.string.subtype_en_US
            "en-GB" -> R.string.subtype_en_GB
            "es-US" -> R.string.subtype_es_US
            "hi-Latn" -> R.string.subtype_hi_Latn
            "sr-Latn" -> R.string.subtype_sr_Latn
            "mns" -> R.string.subtype_mns
            "xdq" -> R.string.subtype_xdq
            "dru" -> R.string.subtype_dru
            "st" -> R.string.subtype_st
            "dag" -> R.string.subtype_dag
            "mhr" -> R.string.subtype_mhr
            else -> 0
        }
        if (overrideResId != 0) {
            return if (displayLocale == null) resources.getString(overrideResId)
            else runInLocale(resources, displayLocale) { it.getString(overrideResId) }
        }

        val localeDisplayName = getDisplayName(displayLocale ?: resources.configuration.locale())
        return if (localeDisplayName == languageTag) {
            getDisplayName(Locale.US) // try fallback to English name, relevant e.g. fpr pms, see https://github.com/HeliBorg/HeliBoard/pull/748
        } else {
            localeDisplayName
        }
    }
}
