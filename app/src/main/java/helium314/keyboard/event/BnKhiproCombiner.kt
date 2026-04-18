// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.event

import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.ArrayList
import java.util.Locale

/**
 * Bengali Khipro combiner – faithful to the published m17n spec (bn-khipro.mim).
 *
 * The engine parses the bundled m17n file at runtime into a small instruction set
 * and re-evaluates the full composing buffer after every keystroke to provide
 * deterministic, longest-match behavior.
 */
class BnKhiproCombiner(
    private val engine: KhiproEngine = Companion.engine
) : Combiner {

    private val composingText = StringBuilder()

    override fun processEvent(previousEvents: ArrayList<Event>?, event: Event): Event {
        val codePoint = event.codePoint

        // Shift does not affect composition
        if (event.keyCode == KeyCode.SHIFT) return event

        // Backspace inside composition -> remove last latin char and stay consumed
        if (event.keyCode == KeyCode.DELETE) {
            if (composingText.isNotEmpty()) {
                val cp = composingText.codePointBefore(composingText.length)
                composingText.delete(composingText.length - Character.charCount(cp), composingText.length)
                // When we delete the last composing code point, we need to exit composing cleanly.
                // Returning a SPACE keypress chained with the original DELETE mirrors HangulCombiner's
                // approach and prevents the DELETE from affecting committed text.
                if (composingText.isEmpty()) {
                    reset()
                    return Event.createHardwareKeypressEvent(0x20, Constants.CODE_SPACE, 0, event, event.isKeyRepeat)
                }
                return Event.createConsumedEvent(event)
            }
            // Match HangulCombiner behavior: if we have no composing state, let backspace
            // propagate to delete previous editor contents.
            return event
        }

        // Commit composition on whitespace or any other functional key (including enter, punctuation keys routed as functional)
        val isValidCodePoint = codePoint != Integer.MAX_VALUE && Character.isValidCodePoint(codePoint)
        val isWhitespace = isValidCodePoint && Character.isWhitespace(codePoint)

        if (event.isFunctionalKeyEvent || isWhitespace) {
            return commitAndReset(event)
        }

        if (!isValidCodePoint) return Event.createConsumedEvent(event)

        composingText.append(Character.toChars(codePoint))
        return Event.createConsumedEvent(event)
    }

    override val combiningStateFeedback: CharSequence
        get() = engine.convert(composingText.toString())

    override fun reset() {
        composingText.setLength(0)
        engine.resetState()
    }

    private fun commitAndReset(event: Event): Event {
        val converted = combiningStateFeedback
        reset()
        return Event.createSoftwareTextEvent(converted, KeyCode.MULTIPLE_CODE_POINTS, event)
    }

    companion object {
        private const val SPEC_ASSET = "bn-khipro.mim"

        /**
         * Lazy loader shared across instances. Falls back to bundled resource if assets are missing
         * (e.g. during JVM unit tests).
         */
        val engine: KhiproEngine by lazy {
            val ctx = Settings.getCurrentContext()
            val specText = try {
                ctx.assets.open(SPEC_ASSET).use { input ->
                    BufferedReader(InputStreamReader(input)).readText()
                }
            } catch (e: Exception) {
                Log.w("BnKhiproCombiner", "Could not load spec from assets, falling back to classpath", e)
                try {
                    BnKhiproCombiner::class.java.classLoader
                        ?.getResourceAsStream(SPEC_ASSET)
                        ?.bufferedReader()?.readText()
                } catch (_: Exception) {
                    null
                }
            }
            if (specText.isNullOrBlank()) {
                // Fatal: keep engine with empty spec to avoid crashes, but log loudly.
                Log.e("BnKhiproCombiner", "bn-khipro.mim could not be loaded; Khipro combiner disabled")
                KhiproEngine("")
            } else {
                KhiproEngine(specText)
            }
        }
    }
}

// ------------------------
// Mini m17n interpreter
// ------------------------

private typealias VarMap = MutableMap<String, Int>

private data class MapEntry(val key: String, val output: String, val actions: List<Action>)
private data class StateRule(val mapName: String, val actions: List<Action>)
private data class StateDef(
    val name: String,
    val entryActions: List<Action>,
    val rules: List<StateRule>
)

private sealed interface Action {
    data class Set(val variable: String, val valueExpr: String) : Action
    data class Insert(val text: String) : Action
    data class Delete(val count: Int) : Action // delete to the left of cursor
    data class Move(val delta: Int?, val toEnd: Boolean = false) : Action
    data class Shift(val state: String) : Action
    data object Commit : Action
    data class Cond(val condition: Condition, val actions: List<Action>) : Action
    /** Represents m17n (cond (test actions...) (test2 actions2...) ...) */
    data class CondBranch(val branches: List<Pair<Condition, List<Action>>>) : Action
}

private sealed interface Condition {
    data object Always : Condition
    data class Equals(val variable: String, val value: Int) : Condition
    data class And(val left: Condition, val right: Condition) : Condition
    data class Or(val left: Condition, val right: Condition) : Condition
}

/**
 * Parses the bn-khipro.mim file and performs streaming conversion with greedy longest-match
 * respecting per-state matcher ordering.
 */
class KhiproEngine(specText: String) {
    private val maps: Map<String, List<MapEntry>>
    private val states: Map<String, StateDef>
    private var currentStateName: String = "init"
    private var vars: VarMap = mutableMapOf()

    init {
        val parser = SexpParser(specText)
        val root = parser.parse()
        val (m, s) = SpecBuilder.fromSexp(root)
        maps = m
        states = s
        resetState()
    }

    fun resetState() {
        currentStateName = "init"
        vars = mutableMapOf()
        applyEntryActions()
    }

    fun convert(input: String): String {
        // Restart state machine for each full recomputation
        resetState()
        val out = StringBuilder()
        var cursor = 0 // cursor within out
        var i = 0
        while (i < input.length) {
            val state = states[currentStateName] ?: break
            val match = findMatch(state, input, i)
            if (match == null) {
                // no match -> emit raw char, move to init if not already
                out.insert(cursor, input[i])
                cursor += 1
                i += 1
                if (currentStateName != "init") {
                    currentStateName = "init"
                    applyEntryActions()
                }
                continue
            }

            val (entry, rule) = match
            executeActions(entry.actions, out, setCursorFactory(cursorRef = { cursor }, cursorSetter = { cursor = it }))
            executeActions(rule.actions, out, setCursorFactory(cursorRef = { cursor }, cursorSetter = { cursor = it }))
            cursor = cursor.coerceIn(0, out.length)

            i += entry.key.length
        }
        return out.toString()
    }

    private fun applyEntryActions() {
        val state = states[currentStateName] ?: return
        var cursor = 0
        executeActions(state.entryActions, StringBuilder(), setCursorFactory(cursorRef = { cursor }, cursorSetter = { cursor = it }))
    }

    private fun findMatch(state: StateDef, input: String, index: Int): Pair<MapEntry, StateRule>? {
        var best: Pair<MapEntry, StateRule>? = null
        var bestLen = -1
        for (rule in state.rules) {
            val entries = maps[rule.mapName] ?: continue
            for (entry in entries) {
                if (entry.key.length <= bestLen) continue
                if (input.regionMatches(index, entry.key, 0, entry.key.length, ignoreCase = false)) {
                    best = entry to rule
                    bestLen = entry.key.length
                }
            }
        }
        return best
    }

    private fun executeActions(actions: List<Action>, out: StringBuilder, cursorAccessor: CursorAccessor) {
        for (action in actions) {
            when (action) {
                is Action.Set -> {
                    val v = action.valueExpr.toIntOrNull() ?: vars[action.valueExpr] ?: 0
                    vars[action.variable] = v
                }
                is Action.Insert -> {
                    val pos = cursorAccessor.get()
                    out.insert(pos, action.text)
                    cursorAccessor.set(pos + action.text.length)
                }
                is Action.Delete -> {
                    val pos = cursorAccessor.get()
                    val start = (pos - action.count).coerceAtLeast(0)
                    if (start < pos && start < out.length) {
                        val end = pos.coerceAtMost(out.length)
                        out.delete(start, end)
                        cursorAccessor.set(start)
                    }
                }
                is Action.Move -> {
                    val newPos = if (action.toEnd) out.length else (cursorAccessor.get() + (action.delta ?: 0))
                    cursorAccessor.set(newPos.coerceIn(0, out.length))
                }
                is Action.Shift -> {
                    currentStateName = action.state
                    applyEntryActions()
                }
                is Action.Commit -> { /* no-op for recomputation model */ }
                is Action.Cond -> if (evalCond(action.condition)) executeActions(action.actions, out, cursorAccessor)
                is Action.CondBranch -> {
                    for ((cond, acts) in action.branches) {
                        if (evalCond(cond)) {
                            executeActions(acts, out, cursorAccessor)
                            break
                        }
                    }
                }
            }
        }
    }

    private fun evalCond(cond: Condition): Boolean = when (cond) {
        is Condition.Always -> true
        is Condition.Equals -> vars[cond.variable] == cond.value
        is Condition.And -> evalCond(cond.left) && evalCond(cond.right)
        is Condition.Or -> evalCond(cond.left) || evalCond(cond.right)
    }
}

// ------------ Parsing helpers ------------

private class SexpParser(private val text: String) {
    private var idx = 0

    fun parse(): List<Any> {
        val result = mutableListOf<Any>()
        while (skipWs()) {
            result.add(readExpr())
        }
        return result
    }

    private fun readExpr(): Any {
        skipWs()
        if (idx >= text.length) error("Unexpected EOF")
        return when (text[idx]) {
            '(' -> {
                idx++
                val list = mutableListOf<Any>()
                while (skipWs() && text[idx] != ')') {
                    list.add(readExpr())
                }
                if (idx >= text.length || text[idx] != ')') error("Unclosed list")
                idx++
                list
            }
            '"' -> readString()
            else -> readAtom()
        }
    }

    private fun readString(): String {
        val sb = StringBuilder()
        idx++ // skip "
        while (idx < text.length && text[idx] != '"') {
            sb.append(text[idx])
            idx++
        }
        if (idx >= text.length) error("Unterminated string")
        idx++ // closing quote
        return sb.toString()
    }

    private fun readAtom(): String {
        val start = idx
        while (idx < text.length && !text[idx].isWhitespace() && text[idx] != '(' && text[idx] != ')') idx++
        return text.substring(start, idx)
    }

    private fun skipWs(): Boolean {
        while (idx < text.length) {
            if (text[idx].isWhitespace()) { idx++; continue }
            if (text[idx] == ';' && idx + 1 < text.length && text[idx + 1] == ';') {
                // comment to end of line
                while (idx < text.length && text[idx] != '\n') idx++
                continue
            }
            return true
        }
        return false
    }
}

private object SpecBuilder {
    fun fromSexp(root: List<Any>): Pair<Map<String, List<MapEntry>>, Map<String, StateDef>> {
        var maps: Map<String, List<MapEntry>> = emptyMap()
        var states: Map<String, StateDef> = emptyMap()
        for (node in root) {
            if (node !is List<*>) continue
            val head = node.firstOrNull() as? String ?: continue
            when (head.lowercase(Locale.ROOT)) {
                "map" -> maps = parseMaps(node.drop(1))
                "state" -> states = parseStates(node.drop(1))
            }
        }
        return maps to states
    }

    private fun parseMaps(nodes: List<Any?>): Map<String, List<MapEntry>> {
        val result = mutableMapOf<String, List<MapEntry>>()
        for (n in nodes) {
            val lst = n as? List<*> ?: continue
            val name = lst.firstOrNull() as? String ?: continue
            val entries = mutableListOf<MapEntry>()
            for (rawEntry in lst.drop(1)) {
                val eList = rawEntry as? List<*> ?: continue
                val keyAtom = eList.firstOrNull()
                val key = when (keyAtom) {
                    is String -> keyAtom
                    is List<*> -> keyAtom.firstOrNull() as? String ?: continue
                    else -> continue
                }
                val actions = mutableListOf<Action>()
                var output = ""
                for (item in eList.drop(1)) {
                    when (item) {
                        is String -> output = item
                        is List<*> -> {
                            val act = parseAction(item)
                            if (act != null) actions.add(act)
                        }
                    }
                }
                if (output.isNotEmpty()) actions.add(Action.Insert(output))
                entries.add(MapEntry(key, output, actions))
            }
            result[name] = entries.sortedByDescending { it.key.length }
        }
        return result
    }

    private fun parseStates(nodes: List<Any?>): Map<String, StateDef> {
        val result = mutableMapOf<String, StateDef>()
        for (n in nodes) {
            val lst = n as? List<*> ?: continue
            val name = lst.firstOrNull() as? String ?: continue
            val entryActions = mutableListOf<Action>()
            val rules = mutableListOf<StateRule>()
            for (entry in lst.drop(1)) {
                val el = entry as? List<*> ?: continue
                val head = el.firstOrNull() as? String ?: continue
                if (head == "t") {
                    entryActions += el.drop(1).mapNotNull { parseAction(it) }
                } else {
                    val ruleActions = el.drop(1).mapNotNull { parseAction(it) }
                    rules += StateRule(head, ruleActions)
                }
            }
            result[name] = StateDef(name, entryActions, rules)
        }
        return result
    }

    private fun parseAction(node: Any?): Action? {
        if (node !is List<*>) return null
        val head = node.firstOrNull() as? String ?: return null
        return when (head) {
            "set" -> {
                val v = node.getOrNull(1) as? String ?: return null
                val valueExpr = node.getOrNull(2) as? String ?: return null
                Action.Set(v, valueExpr)
            }
            "insert" -> (node.getOrNull(1) as? String)?.let {
                val text = if (it.startsWith("?")) it.drop(1) else it
                Action.Insert(text)
            }
            "delete" -> parseDelete(node.getOrNull(1) as? String)
            "move" -> parseMove(node.getOrNull(1) as? String)
            "shift" -> (node.getOrNull(1) as? String)?.let { Action.Shift(it) }
            "commit" -> Action.Commit
            "cond" -> parseCond(node.drop(1))
            else -> null
        }
    }

    private fun parseDelete(token: String?): Action.Delete? {
        if (token == null) return null
        return if (token.startsWith("@-")) {
            val num = token.removePrefix("@-").ifBlank { "1" }.toIntOrNull() ?: 1
            Action.Delete(num)
        } else null
    }

    private fun parseMove(token: String?): Action.Move? {
        if (token == null) return null
        return when (token) {
            "@>" -> Action.Move(delta = null, toEnd = true)
            "@-" -> Action.Move(delta = -1)
            else -> if (token.startsWith("@-")) {
                val num = token.removePrefix("@-").toIntOrNull() ?: 1
                Action.Move(delta = -num)
            } else null
        }
    }

    private fun parseCond(parts: List<Any?>): Action? {
        // m17n cond supports multiple branches: (cond (test actions...) (test2 actions2...) ... )
        val branches = mutableListOf<Pair<Condition, List<Action>>>()
        for (branch in parts) {
            val bl = branch as? List<*> ?: continue
            if (bl.isEmpty()) continue
            val first = bl.firstOrNull()
            val cond = when (first) {
                is List<*> -> parseCondition(first)
                is String -> if (first == "1") Condition.Always else null
                else -> null
            } ?: continue
            val acts = bl.drop(1).mapNotNull { parseAction(it) }
            branches += cond to acts
        }
        if (branches.isEmpty()) return null
        return Action.CondBranch(branches)
    }

    private fun parseCondition(node: List<*>): Condition? {
        val head = node.firstOrNull() as? String ?: return null
        return when (head) {
            "1" -> Condition.Always
            "=" -> {
                val v = node.getOrNull(1) as? String ?: return null
                val value = (node.getOrNull(2) as? String)?.toIntOrNull() ?: return null
                Condition.Equals(v, value)
            }
            "&" -> {
                val left = parseCondition(node.getOrNull(1) as? List<*> ?: return null) ?: return null
                val right = parseCondition(node.getOrNull(2) as? List<*> ?: return null) ?: return null
                Condition.And(left, right)
            }
            "|" -> {
                val left = parseCondition(node.getOrNull(1) as? List<*> ?: return null) ?: return null
                val right = parseCondition(node.getOrNull(2) as? List<*> ?: return null) ?: return null
                Condition.Or(left, right)
            }
            else -> null
        }
    }
}

private fun setCursorFactory(cursorRef: () -> Int, cursorSetter: (Int) -> Unit): CursorAccessor = object : CursorAccessor {
    override fun get(): Int = cursorRef()
    override fun set(pos: Int) = cursorSetter(pos)
}

private interface CursorAccessor {
    fun get(): Int
    fun set(pos: Int)
}
