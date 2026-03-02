// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.event

import android.content.Context
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import java.util.ArrayList

/**
 * Bengali Khipro combiner - converts Latin to Bengali using state machine.
 *
 * Mappings: assets/khipro-mappings.json
 * Source: https://github.com/KhiproKeyboard/Khipro-Mappings
 * Note: Downloaded `khipro-mappings-heliboard.json` from source and renamed to `khipro-mappings.json`.
 */
class BnKhiproCombiner : Combiner {

    private val composingText = StringBuilder()

    enum class State {
        INIT,
        SHOR_STATE,
        REPH_STATE,
        BYANJON_STATE
    }

    companion object {
        private const val MAPPING_FILE = "khipro-mappings.json"

        @OptIn(ExperimentalSerializationApi::class)
        private val json = Json {
            isLenient = true
            ignoreUnknownKeys = true
        }

        private fun loadKhiproMappings(context: Context): Map<String, Map<String, String>> {
            return try {
                val jsonString = context.assets.open(MAPPING_FILE)
                    .bufferedReader()
                    .use { it.readText() }
                json.decodeFromString<Map<String, Map<String, String>>>(jsonString)
                    .mapKeys { it.key.lowercase() }
            } catch (e: Exception) {
                Log.e("BnKhiproCombiner", "Failed to load mappings", e)
                emptyMap()
            }
        }

        private val loadedMappings: Map<String, Map<String, String>> by lazy {
            loadKhiproMappings(Settings.getCurrentContext())
        }

        // Mapping groups
        private val SHOR: Map<String, String> by lazy { loadedMappings["shor"] ?: emptyMap() }
        private val FKAR: Map<String, String> by lazy { loadedMappings["fkar"] ?: emptyMap() }
        private val BYANJON: Map<String, String> by lazy { loadedMappings["byanjon"] ?: emptyMap() }
        private val JUKTOBORNO: Map<String, String> by lazy { loadedMappings["juktoborno"] ?: emptyMap() }
        private val REPH: Map<String, String> by lazy { loadedMappings["reph"] ?: emptyMap() }
        private val PHOLA: Map<String, String> by lazy { loadedMappings["phola"] ?: emptyMap() }
        private val KAR: Map<String, String> by lazy { loadedMappings["kar"] ?: emptyMap() }
        private val DIACRITIC: Map<String, String> by lazy { loadedMappings["diacritic"] ?: emptyMap() }
        private val BIRAM: Map<String, String> by lazy { loadedMappings["biram"] ?: emptyMap() }
        private val PRITHAYOK: Map<String, String> by lazy { loadedMappings["prithayok"] ?: emptyMap() }
        private val AE: Map<String, String> by lazy { loadedMappings["ae"] ?: emptyMap() }
        private val NG: Map<String, String> by lazy { loadedMappings["ng"] ?: emptyMap() }

        private val GROUP_MAPS = mapOf(
            "shor" to SHOR,
            "fkar" to FKAR,
            "byanjon" to BYANJON,
            "juktoborno" to JUKTOBORNO,
            "reph" to REPH,
            "phola" to PHOLA,
            "kar" to KAR,
            "diacritic" to DIACRITIC,
            "biram" to BIRAM,
            "prithayok" to PRITHAYOK,
            "ae" to AE,
            "ng" to NG
        )

        private val STATE_GROUP_ORDER = mapOf(
            State.INIT to listOf("diacritic", "ng", "shor", "fkar", "prithayok", "biram", "reph", "byanjon", "juktoborno"),
            State.SHOR_STATE to listOf("diacritic", "ng", "shor", "fkar", "biram", "prithayok", "reph", "byanjon", "juktoborno"),
            
            State.REPH_STATE to listOf("prithayok", "diacritic", "ng", "ae", "juktoborno", "byanjon", "reph", "kar"),
            State.BYANJON_STATE to listOf("diacritic", "ng", "prithayok", "biram", "kar", "phola", "byanjon", "juktoborno")
        )

        private val MAXLEN_PER_GROUP = GROUP_MAPS.mapValues { (_, map) ->
            map.keys.maxOfOrNull { it.length } ?: 0
        }

        // Find longest matching sequence from current position using greedy algorithm
        private fun findLongest(state: State, text: String, i: Int): Triple<String, String, String> {
            val allowed = STATE_GROUP_ORDER[state] ?: return Triple("", "", "")
            val maxlen = allowed.maxOfOrNull { MAXLEN_PER_GROUP[it] ?: 0 } ?: 0
            val end = minOf(text.length, i + maxlen)

            for (l in (end - i) downTo 1) {
                val chunk = text.substring(i, i + l)
                for (g in allowed) {
                    val map = GROUP_MAPS[g]
                    if (map?.containsKey(chunk) == true) {
                        return Triple(g, chunk, map[chunk]!!)
                    }
                }
            }
            return Triple("", "", "")
        }

        // Determine next state based on current state and matched group
        private fun applyTransition(state: State, group: String): State {
            return when (state) {
                State.INIT -> when (group) {
                    "diacritic", "shor", "fkar", "prithayok", "biram", "ng" -> State.SHOR_STATE
                    "reph" -> State.REPH_STATE
                    "byanjon", "juktoborno" -> State.BYANJON_STATE
                    else -> state
                }
                State.SHOR_STATE -> when (group) {
                    "diacritic", "shor", "fkar", "biram", "prithayok", "ng" -> State.SHOR_STATE
                    "reph" -> State.REPH_STATE
                    "byanjon", "juktoborno" -> State.BYANJON_STATE
                    else -> state
                }
                State.REPH_STATE -> when (group) {
                    "prithayok", "diacritic", "ng", "ae", "kar" -> State.SHOR_STATE
                    "reph" -> State.REPH_STATE
                    "juktoborno", "byanjon" -> State.BYANJON_STATE
                    else -> state
                }
                State.BYANJON_STATE -> when (group) {
                    "diacritic", "kar", "prithayok", "biram", "ng" -> State.SHOR_STATE
                    "byanjon", "juktoborno" -> State.BYANJON_STATE
                    else -> state
                }
            }
        }

        // Convert Latin text to Bengali using state machine
        fun convert(text: String): String {
            var i = 0
            val n = text.length
            var state = State.INIT
            val out = mutableListOf<String>()

            while (i < n) {
                val (group, key, value) = findLongest(state, text, i)
                if (group.isEmpty()) {
                    out.add(text[i].toString())
                    i += 1
                    state = State.INIT
                    continue
                }

                // Special case: insert hasant before phola when in byanjon state
                if (state == State.BYANJON_STATE && group == "phola") {
                    out.add("্")
                    out.add(value)
                } else {
                    out.add(value)
                }

                i += key.length
                state = applyTransition(state, group)
            }

            return out.joinToString("")
        }
    }

    override fun processEvent(previousEvents: ArrayList<Event>?, event: Event): Event {
        // Validate code point before processing - 0xFFFFFFFF is invalid
        val codePoint = event.codePoint
        val isValidCodePoint = codePoint != Integer.MAX_VALUE && Character.isValidCodePoint(codePoint)

        if (event.keyCode == KeyCode.SHIFT) return event

        if (isValidCodePoint && Character.isWhitespace(codePoint)) {
            return commitAndReset(event)
        } else if (event.isFunctionalKeyEvent) {
            return commitAndReset(event)
        } else if (!isValidCodePoint) {
            return Event.createConsumedEvent(event)
        } else {
            composingText.append(Character.toChars(codePoint))

            val text = composingText.toString()
            if (text.endsWith(".ff")) {
                return commitAndReset(event)
            }

            return Event.createConsumedEvent(event)
        }
    }

    override val combiningStateFeedback: CharSequence
        get() = convert(composingText.toString())

    override fun reset() {
        composingText.setLength(0)
    }

    private fun commitAndReset(event: Event): Event {
        val text = combiningStateFeedback
        reset()
        return createEventChainFromSequence(text, event)
    }

    private fun createEventChainFromSequence(text: CharSequence, originalEvent: Event): Event {
        return Event.createSoftwareTextEvent(text, KeyCode.MULTIPLE_CODE_POINTS, originalEvent)
    }
}
