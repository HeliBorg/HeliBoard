/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

public final class BinaryDictIOUtils {
    private BinaryDictIOUtils() {
        // This utility class is not publicly instantiable.
    }

    /**
     * Helper method to hide the actual value of the no children address.
     */
    public static boolean hasChildrenAddress(int address) {
        return FormatSpec.NO_CHILDREN_ADDRESS != address;
    }
    /**
     * Compute the binary size of the node count
     * @param count the node count
     * @return the size of the node count, either 1 or 2 bytes.
     */
    public static int getPtNodeCountSize(int count) {
        if (FormatSpec.MAX_PTNODES_FOR_ONE_BYTE_PTNODE_COUNT >= count) {
            return 1;
        } else if (FormatSpec.MAX_PTNODES_IN_A_PT_NODE_ARRAY >= count) {
            return 2;
        } else {
            throw new RuntimeException("Can't have more than "
                + FormatSpec.MAX_PTNODES_IN_A_PT_NODE_ARRAY + " PtNode in a PtNodeArray (found "
                + count + ")");
        }
    }
}
