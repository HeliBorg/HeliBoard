package helium314.keyboard.latin.utils

object DebugLogUtils {
    @JvmStatic
    fun getStackTrace(limit: Int) = getStackTrace(limit, 1)

    @JvmStatic
    fun getSimpleStackTrace(limit: Int) = getStackTrace(limit, 1, true, ", ")

    /**
     * Get the string representation of the current stack trace, for debugging purposes.
     * @param limit the maximum number of stack frames to be returned.
     * @return a readable, carriage-return-separated string for the current stack trace.
     */
    fun getStackTrace(limit: Int = Int.MAX_VALUE / 2, dropFirst: Int = 0, simplify: Boolean = false, join: String = "\n"): String {
        val sb = StringBuilder()
        try {
            throw RuntimeException()
        } catch (e: RuntimeException) {
            val frames = e.stackTrace
            var j = dropFirst + 1 // +1 because the first frame is here
            while (j < frames.size && j < limit + 1 + dropFirst) {
                var text = frames[j].toString()
                if (simplify && text.startsWith("helium314.keyboard")) {
                    var p = false
                    for (i in text.indices.reversed()) {
                        if (!p) {
                            if (text[i] == '(')
                                p = true
                            continue
                        }
                        if (text[i] == '.') {
                            text = text.substring(i + 1)
                            break
                        }
                    }
                }
                sb.append(text).append(join)
                ++j
            }
        }
        sb.setLength(sb.length - join.length)
        return sb.toString()
    }
}
