// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.emoji

import androidx.core.content.edit
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.prefs
import kotlinx.serialization.json.Json

// manages recently used emojis
object RecentEmojis {
    private val prefs = Settings.getCurrentContext().prefs()

    fun set(emojis: List<String>) {
        Log.i("test", "set $emojis")
        prefs.edit { putString(Settings.PREF_RECENT_EMOJIS, Json.encodeToString(emojis)) }
    }

    @JvmStatic
    fun add(emoji: String) {
        Log.i("test", "add $emoji")
        if (emoji.isEmpty()) return
        val recents = get()
        recents.removeAll { it == emoji }
        recents.add(0, emoji)
        set(recents.take(MAX_COUNT))
    }

     @JvmStatic
     fun addCodepoint(emoji: Int) {
         if (emoji > 0) add(StringUtils.newSingleCodePointString(emoji))
     }

    fun get(): MutableList<String> {
        val pref = prefs.getString(Settings.PREF_RECENT_EMOJIS, Defaults.PREF_RECENT_EMOJIS)
        if (pref.isNullOrEmpty()) return mutableListOf()
        return runCatching { Json.decodeFromString<MutableList<String>>(pref) }
            .getOrNull() ?: mutableListOf()
    }

    private const val MAX_COUNT = 39 // max for config_emoji_keyboard_max_recents_key_count
}
