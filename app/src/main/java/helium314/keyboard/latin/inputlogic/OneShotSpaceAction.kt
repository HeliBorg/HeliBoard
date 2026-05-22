// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.inputlogic

object OneShotSpaceAction {
    const val NONE = 0
    const val JOIN_NEXT = 1
    const val FORCE_NEXT_SPACE = 2

    private var action: Int = NONE

    @JvmStatic
    @Synchronized
    fun armJoinNext() {
        action = JOIN_NEXT
    }

    @JvmStatic
    @Synchronized
    fun armForceNextSpace() {
        action = FORCE_NEXT_SPACE
    }

    @JvmStatic
    @Synchronized
    fun isJoinNextArmed() = action == JOIN_NEXT

    @JvmStatic
    @Synchronized
    fun isForceNextSpaceArmed() = action == FORCE_NEXT_SPACE

    @JvmStatic
    @Synchronized
    fun isAnyArmed() = action != NONE

    @JvmStatic
    @Synchronized
    fun clear(): Boolean {
        if (action == NONE) return false
        action = NONE
        return true
    }

    @JvmStatic
    @Synchronized
    fun consumeAction(): Int {
        val current = action
        if (current != NONE) action = NONE
        return current
    }
}
