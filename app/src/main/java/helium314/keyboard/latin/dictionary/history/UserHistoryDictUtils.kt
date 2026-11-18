package helium314.keyboard.latin.dictionary.history

import android.content.Context
import helium314.keyboard.latin.utils.Log
import java.lang.ref.SoftReference
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

object UserHistoryDictUtils {
    private const val DEBUG = false
    private val sLangUserHistoryDictCache = ConcurrentHashMap<String, SoftReference<UserHistoryDictionary>>()

    @JvmStatic
    fun getUserHistoryDictionary(context: Context?, locale: Locale): UserHistoryDictionary {
        val lookupStr = locale.toString()
        synchronized(sLangUserHistoryDictCache) {
            if (sLangUserHistoryDictCache.containsKey(lookupStr)) {
                val dict = sLangUserHistoryDictCache[lookupStr]?.get()
                if (dict != null) {
                    if (DEBUG) {
                        Log.d("UserHistoryDictUtils", "Use cached UserHistoryDictionary with lookup: $lookupStr")
                    }
                    dict.reloadDictionaryIfRequired()
                    return dict
                }
            }
            val dict = UserHistoryDictionary(context, locale)
            sLangUserHistoryDictCache[lookupStr] = SoftReference<UserHistoryDictionary>(dict)
            return dict
        }
    }

    @JvmStatic
    fun removeAllUserHistoryDictionaries(context: Context) {
        Log.i("UserHistoryDictUtils", "remove user history dictionaries")
        synchronized(sLangUserHistoryDictCache) {
            for (entry
            in sLangUserHistoryDictCache.entries) {
                entry.value?.get()?.clear()
            }
            sLangUserHistoryDictCache.clear()
            context.filesDir?.listFiles { it.name.startsWith(UserHistoryDictionary.NAME) }?.forEach {
                if (it.isDirectory) it.deleteRecursively()
                else it.delete()
            }
        }
    }
}
