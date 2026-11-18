// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.dictionary

import android.content.Context
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.core.content.edit
import helium314.keyboard.compat.locale
import helium314.keyboard.latin.R
import helium314.keyboard.latin.common.Links
import helium314.keyboard.latin.common.LocaleUtils
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.ScriptUtils.defaultScript
import helium314.keyboard.latin.utils.ScriptUtils.script
import helium314.keyboard.latin.utils.SubtypeLocaleUtils
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.getSecondaryLocales
import helium314.keyboard.latin.utils.htmlToAnnotated
import helium314.keyboard.latin.utils.locale
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.withHtmlLink
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import java.io.File
import java.util.Locale

fun getDictionaryLocales(context: Context): MutableSet<Locale> {
    val locales = HashSet<Locale>()

    // get cached dictionaries: extracted or user-added dictionaries
    DictionaryInfoUtils.getCacheDirectories(context).forEach { directory ->
        if (!hasAnythingOtherThanExtractedMainDictionary(directory)) return@forEach
        val locale = DictionaryInfoUtils.getWordListIdFromFileName(directory.name).constructLocale()
        locales.add(locale)
    }
    // get assets dictionaries
    val assetsDictionaryList = DictionaryInfoUtils.getAssetsDictionaryList(context)
    if (assetsDictionaryList != null) {
        for (dictionary in assetsDictionaryList) {
            locales.add(DictionaryInfoUtils.extractLocaleFromAssetsDictionaryFile(dictionary))
        }
    }
    return locales
}

@Composable
fun MissingDictionaryDialog(onDismissRequest: () -> Unit, locale: Locale) {
    val prefs = LocalContext.current.prefs()
    if (prefs.getBoolean(Settings.PREF_DONT_SHOW_MISSING_DICTIONARY_DIALOG, Defaults.PREF_DONT_SHOW_MISSING_DICTIONARY_DIALOG)) {
        onDismissRequest()
        return
    }
    val availableDicts = createDictionaryTextAnnotated(locale)
    val repositoryLink = stringResource(R.string.dictionary_link_text).withHtmlLink(Links.DICTIONARY_URL)
    val dictUrl = "${Links.DICTIONARY_URL}${Links.DICTIONARY_DOWNLOAD_SUFFIX}dictionaries/main_$locale.dict"
    val dictionaryLink = stringResource(R.string.dictionary_link_text).withHtmlLink(dictUrl)
    val message = stringResource(R.string.no_dictionary_message, repositoryLink, locale.toString(), dictionaryLink)
    var annotatedString = message.htmlToAnnotated()
    if (availableDicts.isNotEmpty())
        annotatedString += AnnotatedString("\n") + availableDicts

    ConfirmationDialog(
        onDismissRequest = onDismissRequest,
        cancelButtonText = stringResource(R.string.dialog_close),
        onConfirmed = { prefs.edit { putBoolean(Settings.PREF_DONT_SHOW_MISSING_DICTIONARY_DIALOG, true) } },
        confirmButtonText = stringResource(R.string.no_dictionary_dont_show_again_button),
        content = { Text(annotatedString) }
    )
}

/** if dictionaries for [locale] or language are available returns links to them */
@Composable
fun createDictionaryTextAnnotated(locale: Locale): AnnotatedString {
    val context = LocalContext.current
    val knownDicts = getKnownDictionariesForLocale(locale, context)
    if (knownDicts.isEmpty()) return AnnotatedString("")
    val knownDictLinks = knownDicts.map { (name, link) ->
        "<li>${name.withHtmlLink(link)}</li>"
    }
    return "<ul>${knownDictLinks.joinToString("\n")}</ul>".htmlToAnnotated()
}

/** returns a pair of dictionary description and link for each dictionary  */
fun getKnownDictionariesForLocale(locale: Locale, context: Context): List<Pair<String, String>> {
    val knownDicts = mutableListOf<Pair<String, String>>()
    context.assets.open("dictionaries_in_dict_repo.txt").reader().forEachLine {
        if (it.isBlank()) return@forEachLine
        if (it.startsWith("cldr,")) {
            val localeString = it.substringAfter("cldr,")
            val dictLocale = localeString.constructLocale()
            if (LocaleUtils.getMatchLevel(locale, dictLocale) < LocaleUtils.LOCALE_GOOD_MATCH) return@forEachLine
            val dictBaseUrl = Links.DICTIONARY_URL + Links.DICTIONARY_DOWNLOAD_SUFFIX + Links.DICTIONARY_EMOJI_CLDR_SUFFIX
            val dictLink = dictBaseUrl + "emoji" + "_" + localeString + ".dict"
            val dictString = "emoji: ${dictLocale.getDisplayName(context.resources.configuration.locale())}"
            knownDicts.add(dictString to dictLink)
            return@forEachLine
        }

        val (type, rest) = it.split("_", limit = 2)
        val localeString = rest.substringBefore("__")
        val extra = if (localeString == rest) null else rest.substringAfter("__")
        // we use a locale string here because that's in the dictionaries repo
        // ideally the repo would switch to language tag, but not sure how this is handled in the dictionary header
        // further, the dicts in the dictionaries repo should be compatible with other AOSP-based keyboards
        val dictLocale = localeString.constructLocale()
        if (LocaleUtils.getMatchLevel(locale, dictLocale) < LocaleUtils.LOCALE_GOOD_MATCH) return@forEachLine
        val rawDictString = "$type: ${dictLocale.getDisplayName(context.resources.configuration.locale())}"
        val dictString = when (extra) {
            null -> rawDictString
            "experimental" -> context.getString(R.string.available_dictionary_experimental, rawDictString)
            else -> "$rawDictString ($extra)"
        }
        val dictBaseUrl = Links.DICTIONARY_URL + Links.DICTIONARY_DOWNLOAD_SUFFIX + Links.DICTIONARY_NORMAL_SUFFIX
        val dictLink = "$dictBaseUrl$it.dict"
        knownDicts.add(dictString to dictLink)
    }
    return knownDicts
}

fun cleanUnusedMainDicts(context: Context) {
    val dictionaryDir = File(DictionaryInfoUtils.getWordListCacheDirectory(context))
    val dirs = dictionaryDir.listFiles() ?: return
    val usedLocaleLanguageTags = hashSetOf<String>()
    SubtypeSettings.getEnabledSubtypes().forEach { subtype ->
        val locale = subtype.locale()
        usedLocaleLanguageTags.add(locale.toLanguageTag())
    }
    SubtypeSettings.getAdditionalSubtypes().forEach { subtype ->
        getSecondaryLocales(subtype.extraValue).forEach { usedLocaleLanguageTags.add(it.toLanguageTag()) }
    }
    for (dir in dirs) {
        if (!dir.isDirectory) continue
        if (dir.name in usedLocaleLanguageTags) continue
        if (hasAnythingOtherThanExtractedMainDictionary(dir))
            continue
        dir.deleteRecursively()
    }
}

private fun hasAnythingOtherThanExtractedMainDictionary(dir: File) =
    dir.listFiles()?.any { it.name != DictionaryInfoUtils.MAIN_DICT_FILE_NAME } != false

/** creates the request (string and args) to content resolver to get personal dictionary content */
fun createPersonalDictionaryRequest(locale: Locale, alsoUseMoreRestrictiveLocales: Boolean): Pair<String, Array<String>> {
    val request = StringBuilder("(locale is NULL)")
    if (locale.script() != locale.defaultScript) {
        // non-default script (e.g. sr-Latn / sr__#Latn) -> we want an exact match
        request.append(" or (locale=?)")
        return request.toString() to arrayOf(locale.toString())
    }
    // this really needs to be the locale string, as it interacts with system
    val localeString = if (locale.language == SubtypeLocaleUtils.NO_LANGUAGE)
        "" // If we don't have a locale, use the "all locales" user dictionary.
    else locale.toString()

    // process e.g. "en_US_foo_bar_qux" to ["en", "en_US", "en_US_foo_bar_qux"] and build the request string
    val localeSplit = localeString.split("_", limit = 3)
    var localeSoFar = ""
    val localeStrings = localeSplit.mapTo(mutableListOf()) {
        localeSoFar += if (localeSoFar.isEmpty()) it else "_$it"
        request.append(" or (locale=?)")
        localeSoFar
    }

    if (alsoUseMoreRestrictiveLocales && localeStrings.size < 3) {
        request.append(" or (locale like ?)")
        localeStrings.add("${localeString}_%")
    }

    SubtypeSettings.getSystemLocales().forEach {
        if (it.language != locale.language || localeString == it.toString()) return@forEach
        if (it.country.isNullOrEmpty() || locale.country.isNullOrEmpty() || it.country == locale.country) return@forEach
        if (SubtypeSettings.getEnabledSubtypes(true).any { enabled -> enabled.locale() == it }) return@forEach
        // same language, different (but existing) countries, no subtype specifically for that system locale
        // in that case we also want to use the personal dictionary of that locale (e.g. en_GB dict and en_IE system locale)
        request.append(" or (locale=?)")
        localeStrings.add("${it.language}_${it.country}")
        if (alsoUseMoreRestrictiveLocales) {
            request.append(" or (locale like ?)")
            localeStrings.add("${it.language}_${it.country}_%")
        }
    }
    return request.toString() to localeStrings.toTypedArray()
}
