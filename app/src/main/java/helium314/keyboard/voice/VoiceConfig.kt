// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import android.content.Context
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.prefs

data class Backend(
    val name: String,
    val baseUrl: String,
    val model: String,
    val apiKey: String = "",
)

data class VoiceConfig(
    val transcriptionBackends: List<Backend>,
    val editBackend: Backend,
    val recordMaxSeconds: Int,
    val requestTimeoutSeconds: Int,
    val retries: Int,
    val streaming: Boolean,
    val caCertPem: String,
) {
    companion object {
        fun aiEnabled(context: Context): Boolean =
            context.prefs().getBoolean(Settings.PREF_VOICE_ENABLED, Defaults.PREF_VOICE_ENABLED)

        fun showKeys(context: Context): Boolean =
            aiEnabled(context) &&
                context.prefs().getBoolean(Settings.PREF_VOICE_SHOW_KEYS, Defaults.PREF_VOICE_SHOW_KEYS)

        fun from(context: Context): VoiceConfig {
            val p = context.prefs()
            fun str(key: String, default: String) = (p.getString(key, default) ?: default).trim()
            fun int(key: String, default: String) =
                (p.getString(key, default) ?: default).trim().toIntOrNull() ?: default.toInt()

            val sttUrl = str(Settings.PREF_VOICE_STT_URL, Defaults.PREF_VOICE_STT_URL).trimEnd('/')
            val fallbackUrl = str(Settings.PREF_VOICE_STT_FALLBACK_URL, Defaults.PREF_VOICE_STT_FALLBACK_URL).trimEnd('/')
            val editUrl = str(Settings.PREF_VOICE_EDIT_URL, Defaults.PREF_VOICE_EDIT_URL).trimEnd('/')

            fun key(stored: String, url: String) =
                stored.ifEmpty { if (url.contains("openai.com")) BuildConfig.OPENAI_API_KEY else "" }

            val backends = buildList {
                if (sttUrl.isNotEmpty()) add(
                    Backend("primary", sttUrl, str(Settings.PREF_VOICE_STT_MODEL, Defaults.PREF_VOICE_STT_MODEL),
                        key(str(Settings.PREF_VOICE_STT_KEY, Defaults.PREF_VOICE_STT_KEY), sttUrl))
                )
                if (fallbackUrl.isNotEmpty()) add(
                    Backend("fallback", fallbackUrl, str(Settings.PREF_VOICE_STT_FALLBACK_MODEL, Defaults.PREF_VOICE_STT_FALLBACK_MODEL),
                        key(str(Settings.PREF_VOICE_STT_FALLBACK_KEY, Defaults.PREF_VOICE_STT_FALLBACK_KEY), fallbackUrl))
                )
            }

            return VoiceConfig(
                transcriptionBackends = backends,
                editBackend = Backend("edit", editUrl, str(Settings.PREF_VOICE_EDIT_MODEL, Defaults.PREF_VOICE_EDIT_MODEL),
                    key(str(Settings.PREF_VOICE_EDIT_KEY, Defaults.PREF_VOICE_EDIT_KEY), editUrl)),
                recordMaxSeconds = int(Settings.PREF_VOICE_MAX_SECONDS, Defaults.PREF_VOICE_MAX_SECONDS),
                requestTimeoutSeconds = int(Settings.PREF_VOICE_TIMEOUT, Defaults.PREF_VOICE_TIMEOUT),
                retries = int(Settings.PREF_VOICE_RETRIES, Defaults.PREF_VOICE_RETRIES),
                streaming = p.getBoolean(Settings.PREF_VOICE_STREAMING, Defaults.PREF_VOICE_STREAMING),
                caCertPem = str(Settings.PREF_VOICE_CA_CERT, Defaults.PREF_VOICE_CA_CERT),
            )
        }
    }
}
