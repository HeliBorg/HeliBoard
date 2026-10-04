// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.internal;

import kotlin.enums.EnumEntries;

/// This "weak stack" struct works by packing enum ordinals by half-bytes in a
/// 64-bit integer. Thus, it is not compatible with enums of more than 2⁴ = 16
/// entries.
///
/// Bitwise operations are performed on these half-bytes to create stack-like
/// logic. Because they are stored in a 64-bit integer, the stack has a capacity
/// of 16. When entries are added in excess of this capacity, the oldest entry
/// is forgotten.
///
/// Conversely, the stack has no concept of "item count". Instead, when [#pop()]
/// is called in excess of the entries that were pushed and remembered, the
/// enum's first entry is returned as a fallback value. Thus, the order of
/// constants in the enum definition is significant.
///
/// This struct is intended for use where exact recall is not critical and
/// there's a reasonable default to fall back to when entries are forgotten or
/// absent. This provides a very efficient short-term memory that doesn't crash
/// on over or underflow.
final class WeakStack<E extends Enum<E>> {
    private final EnumEntries<E> entries;
    private long stack = 0x0;

    WeakStack(EnumEntries<E> entries) {
        if (entries.size() > 0x10) {
            throw new IllegalArgumentException("Enum is too large");
        } // we can be certain the ordinals will fit into half-bytes

        this.entries = entries;
    }

    public void push(E e) {
        stack <<= 0x4;
        stack |= e.ordinal();
    }

    public E pop() {
        var ord = (int) (stack & 0xF);
        stack >>>= 0x4;
        return entries.get(ord);
    }

    public void wipe() {
        stack = 0x0;
    }
}
