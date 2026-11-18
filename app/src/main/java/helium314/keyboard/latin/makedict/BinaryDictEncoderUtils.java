/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.makedict.BinaryDictDecoderUtils.CharEncoding;
import helium314.keyboard.latin.makedict.FormatSpec.FormatOptions;
import helium314.keyboard.latin.makedict.FusionDictionary.PtNode;
import helium314.keyboard.latin.makedict.FusionDictionary.PtNodeArray;
import helium314.keyboard.latin.utils.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

/**
 * Encodes binary files for a FusionDictionary.
 * <p>
 * All the methods in this class are static.
 * <p>
 * TODO: Rename this class to DictEncoderUtils.
 */
public class BinaryDictEncoderUtils {
    private static final String TAG = "BinaryDictEncoderUtils";

    private BinaryDictEncoderUtils() {
        // This utility class is not publicly instantiable.
    }
    // Arbitrary limit to how much passes we consider address size compression should
    // terminate in. At the time of this writing, our largest dictionary completes
    // compression in five passes.
    // If the number of passes exceeds this number, makedict bails with an exception on
    // suspicion that a bug might be causing an infinite loop.
    private static final int MAX_PASSES = 24;
    /**
     * Compute the binary size of the character array.
     * <p>
     * If only one character, this is the size of this character. If many, it's the sum of their
     * sizes + 1 byte for the terminator.
     *
     * @param characters the character array
     * @return the size of the char array, including the terminator if any
     */
    static int getPtNodeCharactersSize(int[] characters, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int size = CharEncoding.getCharArraySize(characters, codePointToOneByteCodeMap);
        if (characters.length > 1) size += FormatSpec.PTNODE_TERMINATOR_SIZE;
        return size;
    }
    /**
     * Compute the binary size of the character array in a PtNode
     * <p>
     * If only one character, this is the size of this character. If many, it's the sum of their
     * sizes + 1 byte for the terminator.
     *
     * @param ptNode the PtNode
     * @return the size of the char array, including the terminator if any
     */
    private static int getPtNodeCharactersSize(PtNode ptNode, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        return getPtNodeCharactersSize(ptNode.mChars, codePointToOneByteCodeMap);
    }
    /**
     * Compute the binary size of the PtNode count for a node array.
     * @param nodeArray the nodeArray
     * @return the size of the PtNode count, either 1 or 2 bytes.
     */
    private static int getPtNodeCountSize(PtNodeArray nodeArray) {
        return BinaryDictIOUtils.getPtNodeCountSize(nodeArray.mData.size());
    }
    private static int getShortcutSize(WeightedString shortcut, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int size = 1;
        String word = shortcut.mWord;
        int length = word.length();

        for (int i = 0; i < length; i = word.offsetByCodePoints(i, 1)) {
            int codePoint = word.codePointAt(i);
            size += CharEncoding.getCharSize(codePoint, codePointToOneByteCodeMap);
        }

        return size + 1;
    }
    private static int getShortcutListSize(ArrayList<WeightedString> shortcutList, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        if (shortcutList == null || shortcutList.isEmpty()) return 0;
        int size = FormatSpec.PTNODE_SHORTCUT_LIST_SIZE_SIZE;
        for (WeightedString shortcut : shortcutList) {
            size += getShortcutSize(shortcut, codePointToOneByteCodeMap);
        }
        return size;
    }
    /**
     * Compute the maximum size of a PtNode, assuming 3-byte addresses for everything.
     *
     * @param ptNode the PtNode to compute the size of.
     * @return the maximum size of the PtNode.
     */
    private static int getPtNodeMaximumSize(PtNode ptNode, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int size = getNodeHeaderSize(ptNode, codePointToOneByteCodeMap);
        if (ptNode.isTerminal()) {
            // If terminal, one byte for the frequency.
            size += FormatSpec.PTNODE_FREQUENCY_SIZE;
        }
        size += FormatSpec.PTNODE_MAX_ADDRESS_SIZE; // For children address
        if (null != ptNode.mBigrams) {
            size += (FormatSpec.PTNODE_ATTRIBUTE_FLAGS_SIZE
                + FormatSpec.PTNODE_ATTRIBUTE_MAX_ADDRESS_SIZE)
                * ptNode.mBigrams.size();
        }
        return size;
    }
    /**
     * Compute the maximum size of each PtNode of a PtNode array, assuming 3-byte addresses for
     * everything, and caches it in the `mCachedSize' member of the nodes; deduce the size of
     * the containing node array, and cache it it its 'mCachedSize' member.
     *
     * @param ptNodeArray the node array to compute the maximum size of.
     */
    private static void calculatePtNodeArrayMaximumSize(PtNodeArray ptNodeArray, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int size = getPtNodeCountSize(ptNodeArray);
        for (PtNode node : ptNodeArray.mData) {
            int nodeSize = getPtNodeMaximumSize(node, codePointToOneByteCodeMap);
            node.mCachedSize = nodeSize;
            size += nodeSize;
        }
        ptNodeArray.mCachedSize = size;
    }
    /**
     * Compute the size of the header (flag + [parent address] + characters size) of a PtNode.
     *
     * @param ptNode the PtNode of which to compute the size of the header
     */
    private static int getNodeHeaderSize(PtNode ptNode, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        return FormatSpec.PTNODE_FLAGS_SIZE + getPtNodeCharactersSize(ptNode,
            codePointToOneByteCodeMap);
    }
    /**
     * Compute the size, in bytes, that an address will occupy.
     * <p>
     * This can be used either for children addresses (which are always positive) or for
     * attribute, which may be positive or negative but
     * store their sign bit separately.
     *
     * @param address the address
     * @return the byte size.
     */
    static int getByteSize(int address) {
        assert(address <= FormatSpec.UINT24_MAX);
        if (!BinaryDictIOUtils.hasChildrenAddress(address)) {
            return 0;
        } else if (Math.abs(address) <= FormatSpec.UINT8_MAX) {
            return 1;
        } else if (Math.abs(address) <= FormatSpec.UINT16_MAX) {
            return 2;
        } else {
            return 3;
        }
    }
    static int writeUIntToBuffer(byte[] buffer, int fromPosition, int value,
                                 int size) {
        int position = fromPosition;
        switch(size) {
            case 4:
                buffer[position++] = (byte) ((value >> 24) & 0xFF);
                /* fall through */
            case 3:
                buffer[position++] = (byte) ((value >> 16) & 0xFF);
                /* fall through */
            case 2:
                buffer[position++] = (byte) ((value >> 8) & 0xFF);
                /* fall through */
            case 1:
                buffer[position++] = (byte) (value & 0xFF);
                break;
            default:
                /* nop */
        }
        return position;
    }
    static void writeUIntToStream(OutputStream stream, int value, int size)
        throws IOException {
        switch(size) {
            case 4:
                stream.write((value >> 24) & 0xFF);
                /* fall through */
            case 3:
                stream.write((value >> 16) & 0xFF);
                /* fall through */
            case 2:
                stream.write((value >> 8) & 0xFF);
                /* fall through */
            case 1:
                stream.write(value & 0xFF);
                break;
            default:
                /* nop */
        }
    }
    // End utility methods
    // This method is responsible for finding a nice ordering of the nodes that favors run-time
    // cache performance and dictionary size.
    static ArrayList<PtNodeArray> flattenTree(
        PtNodeArray rootNodeArray) {
        int treeSize = FusionDictionary.countPtNodes(rootNodeArray);
        Log.i(TAG, "Counted nodes : " + treeSize);
        ArrayList<PtNodeArray> flatTree = new ArrayList<>(treeSize);
        return flattenTreeInner(flatTree, rootNodeArray);
    }
    private static ArrayList<PtNodeArray> flattenTreeInner(ArrayList<PtNodeArray> list,
                                                           PtNodeArray ptNodeArray) {
        // Removing the node is necessary if the tails are merged, because we would then
        // add the same node several times when we only want it once. A number of places in
        // the code also depends on any node being only once in the list.
        // Merging tails can only be done if there are no attributes. Searching for attributes
        // in LatinIME code depends on a total breadth-first ordering, which merging tails
        // breaks. If there are no attributes, it should be fine (and reduce the file size)
        // to merge tails, and removing the node from the list would be necessary. However,
        // we don't merge tails because breaking the breadth-first ordering would result in
        // extreme overhead at bigram lookup time (it would make the search function O(n) instead
        // of the current O(log(n)), where n=number of nodes in the dictionary which is pretty
        // high).
        // If no nodes are ever merged, we can't have the same node twice in the list, hence
        // searching for duplicates in unnecessary. It is also very performance consuming,
        // since `list' is an ArrayList so it's an O(n) operation that runs on all nodes, making
        // this simple list.remove operation O(n*n) overall. On Android this overhead is very
        // high.
        // For future reference, the code to remove duplicate is a simple : list.remove(node);
        list.add(ptNodeArray);
        ArrayList<PtNode> branches = ptNodeArray.mData;
        for (PtNode ptNode : branches) {
            if (null != ptNode.mChildren) flattenTreeInner(list, ptNode.mChildren);
        }
        return list;
    }
    /**
     * Get the offset from a position inside a current node array to a target node array, during
     * update.
     * <p>
     * If the current node array is before the target node array, the target node array has not
     * been updated yet, so we should return the offset from the old position of the current node
     * array to the old position of the target node array. If on the other hand the target is
     * before the current node array, it already has been updated, so we should return the offset
     * from the new position in the current node array to the new position in the target node
     * array.
     *
     * @param currentNodeArray node array containing the PtNode where the offset will be written
     * @param offsetFromStartOfCurrentNodeArray offset, in bytes, from the start of currentNodeArray
     * @param targetNodeArray the target node array to get the offset to
     * @return the offset to the target node array
     */
    private static int getOffsetToTargetNodeArrayDuringUpdate(PtNodeArray currentNodeArray,
                                                              int offsetFromStartOfCurrentNodeArray, PtNodeArray targetNodeArray) {
        boolean isTargetBeforeCurrent = (targetNodeArray.mCachedAddressBeforeUpdate
            < currentNodeArray.mCachedAddressBeforeUpdate);
        if (isTargetBeforeCurrent) {
            return targetNodeArray.mCachedAddressAfterUpdate
                - (currentNodeArray.mCachedAddressAfterUpdate
                + offsetFromStartOfCurrentNodeArray);
        }
        return targetNodeArray.mCachedAddressBeforeUpdate
            - (currentNodeArray.mCachedAddressBeforeUpdate + offsetFromStartOfCurrentNodeArray);
    }
    /**
     * Get the offset from a position inside a current node array to a target PtNode, during
     * update.
     *
     * @param currentNodeArray node array containing the PtNode where the offset will be written
     * @param offsetFromStartOfCurrentNodeArray offset, in bytes, from the start of currentNodeArray
     * @param targetPtNode the target PtNode to get the offset to
     * @return the offset to the target PtNode
     */
    // TODO: is there any way to factorize this method with the one above?
    private static int getOffsetToTargetPtNodeDuringUpdate(PtNodeArray currentNodeArray,
                                                           int offsetFromStartOfCurrentNodeArray, PtNode targetPtNode) {
        int oldOffsetBasePoint = currentNodeArray.mCachedAddressBeforeUpdate
            + offsetFromStartOfCurrentNodeArray;
        boolean isTargetBeforeCurrent = (targetPtNode.mCachedAddressBeforeUpdate
            < oldOffsetBasePoint);
        // If the target is before the current node array, then its address has already been
        // updated. We can use the AfterUpdate member, and compare it to our own member after
        // update. Otherwise, the AfterUpdate member is not updated yet, so we need to use the
        // BeforeUpdate member, and of course we have to compare this to our own address before
        // update.
        if (isTargetBeforeCurrent) {
            int newOffsetBasePoint = currentNodeArray.mCachedAddressAfterUpdate
                + offsetFromStartOfCurrentNodeArray;
            return targetPtNode.mCachedAddressAfterUpdate - newOffsetBasePoint;
        }
        return targetPtNode.mCachedAddressBeforeUpdate - oldOffsetBasePoint;
    }
    /**
     * Computes the actual node array size, based on the cached addresses of the children nodes.
     * <p>
     * Each node array stores its tentative address. During dictionary address computing, these
     * are not final, but they can be used to compute the node array size (the node array size
     * depends on the address of the children because the number of bytes necessary to store an
     * address depends on its numeric value. The return value indicates whether the node array
     * contents (as in, any of the addresses stored in the cache fields) have changed with
     * respect to their previous value.
     *
     * @param ptNodeArray the node array to compute the size of.
     * @param dict the dictionary in which the word/attributes are to be found.
     * @return false if none of the cached addresses inside the node array changed, true otherwise.
     */
    private static boolean computeActualPtNodeArraySize(PtNodeArray ptNodeArray, FusionDictionary dict,
                                                        HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        boolean changed = false;
        int size = getPtNodeCountSize(ptNodeArray);
        for (PtNode ptNode : ptNodeArray.mData) {
            ptNode.mCachedAddressAfterUpdate = ptNodeArray.mCachedAddressAfterUpdate + size;
            if (ptNode.mCachedAddressAfterUpdate != ptNode.mCachedAddressBeforeUpdate) {
                changed = true;
            }
            int nodeSize = getNodeHeaderSize(ptNode, codePointToOneByteCodeMap);
            if (ptNode.isTerminal()) {
                nodeSize += FormatSpec.PTNODE_FREQUENCY_SIZE;
            }
            if (null != ptNode.mChildren) {
                nodeSize += getByteSize(getOffsetToTargetNodeArrayDuringUpdate(ptNodeArray,
                    nodeSize + size, ptNode.mChildren));
            }
            nodeSize += getShortcutListSize(ptNode.mShortcutTargets, null);
            if (null != ptNode.mBigrams) {
                for (WeightedString bigram : ptNode.mBigrams) {
                    int offset = getOffsetToTargetPtNodeDuringUpdate(ptNodeArray,
                        nodeSize + size + FormatSpec.PTNODE_ATTRIBUTE_FLAGS_SIZE,
                        FusionDictionary.findWordInTree(dict.mRootNodeArray, bigram.mWord));
                    nodeSize += getByteSize(offset) + FormatSpec.PTNODE_ATTRIBUTE_FLAGS_SIZE;
                }
            }
            ptNode.mCachedSize = nodeSize;
            size += nodeSize;
        }
        if (ptNodeArray.mCachedSize != size) {
            ptNodeArray.mCachedSize = size;
            changed = true;
        }
        return changed;
    }
    /**
     * Initializes the cached addresses of node arrays and their containing nodes from their size.
     *
     * @param flatNodes the list of node arrays.
     * @return the byte size of the entire stack.
     */
    private static int initializePtNodeArraysCachedAddresses(
        ArrayList<PtNodeArray> flatNodes) {
        int nodeArrayOffset = 0;
        for (PtNodeArray nodeArray : flatNodes) {
            nodeArray.mCachedAddressBeforeUpdate = nodeArrayOffset;
            int nodeCountSize = getPtNodeCountSize(nodeArray);
            int nodeffset = 0;
            for (PtNode ptNode : nodeArray.mData) {
                ptNode.mCachedAddressBeforeUpdate = ptNode.mCachedAddressAfterUpdate =
                    nodeCountSize + nodeArrayOffset + nodeffset;
                nodeffset += ptNode.mCachedSize;
            }
            nodeArrayOffset += nodeArray.mCachedSize;
        }
        return nodeArrayOffset;
    }
    /**
     * Updates the cached addresses of node arrays after recomputing their new positions.
     *
     * @param flatNodes the list of node arrays.
     */
    private static void updatePtNodeArraysCachedAddresses(ArrayList<PtNodeArray> flatNodes) {
        for (PtNodeArray nodeArray : flatNodes) {
            nodeArray.mCachedAddressBeforeUpdate = nodeArray.mCachedAddressAfterUpdate;
            for (PtNode ptNode : nodeArray.mData) {
                ptNode.mCachedAddressBeforeUpdate = ptNode.mCachedAddressAfterUpdate;
            }
        }
    }
    /**
     * Compute the addresses and sizes of an ordered list of PtNode arrays.
     * <p>
     * This method takes a list of PtNode arrays and will update their cached address and size
     * values so that they can be written into a file. It determines the smallest size each of the
     * PtNode arrays can be given the addresses of its children and attributes, and store that into
     * each PtNode.
     * The order of the PtNode is given by the order of the array. This method makes no effort
     * to find a good order; it only mechanically computes the size this order results in.
     *
     * @param dict the dictionary
     * @param flatNodes the ordered list of PtNode arrays
     * @return the same array it was passed. The nodes have been updated for address and size.
     */
    static ArrayList<PtNodeArray> computeAddresses(FusionDictionary dict, ArrayList<PtNodeArray> flatNodes,
                                                                 HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        // First get the worst possible sizes and offsets
        for (PtNodeArray n : flatNodes) {
            calculatePtNodeArrayMaximumSize(n, codePointToOneByteCodeMap);
        }
        int offset = initializePtNodeArraysCachedAddresses(flatNodes);
        Log.i(TAG, "Compressing the array addresses. Original size : " + offset);
        Log.i(TAG, "(Recursively seen size : " + offset + ")");
        int passes = 0;
        boolean changesDone = false;
        do {
            changesDone = false;
            int ptNodeArrayStartOffset = 0;
            for (PtNodeArray ptNodeArray : flatNodes) {
                ptNodeArray.mCachedAddressAfterUpdate = ptNodeArrayStartOffset;
                int oldNodeArraySize = ptNodeArray.mCachedSize;
                boolean changed = computeActualPtNodeArraySize(ptNodeArray, dict,
                    codePointToOneByteCodeMap);
                int newNodeArraySize = ptNodeArray.mCachedSize;
                if (oldNodeArraySize < newNodeArraySize) {
                    throw new RuntimeException("Increased size ?!");
                }
                ptNodeArrayStartOffset += newNodeArraySize;
                changesDone |= changed;
            }
            updatePtNodeArraysCachedAddresses(flatNodes);
            ++passes;
            if (passes > MAX_PASSES) throw new RuntimeException("Too many passes - probably a bug");
        } while (changesDone);
        PtNodeArray lastPtNodeArray = flatNodes.get(flatNodes.size() - 1);
        Log.i(TAG, "Compression complete in " + passes + " passes.");
        Log.i(TAG, "After address compression : "
            + (lastPtNodeArray.mCachedAddressAfterUpdate + lastPtNodeArray.mCachedSize));
        return flatNodes;
    }
    /**
     * Validity-checking method.
     * <p>
     * This method checks a list of PtNode arrays for juxtaposition, that is, it will do
     * nothing if each node array's cached address is actually the previous node array's address
     * plus the previous node's size.
     * If this is not the case, it will throw an exception.
     *
     * @param arrays the list of node arrays to check
     */
    static void checkFlatPtNodeArrayList(ArrayList<PtNodeArray> arrays) {
        int offset = 0;
        int index = 0;
        for (PtNodeArray ptNodeArray : arrays) {
            // BeforeUpdate and AfterUpdate addresses are the same here, so it does not matter
            // which we use.
            if (ptNodeArray.mCachedAddressAfterUpdate != offset) {
                throw new RuntimeException("Wrong address for node " + index
                    + " : expected " + offset + ", got " +
                    ptNodeArray.mCachedAddressAfterUpdate);
            }
            ++index;
            offset += ptNodeArray.mCachedSize;
        }
    }
    /**
     * Helper method to write a children position to a file.
     *
     * @param buffer the buffer to write to.
     * @param fromIndex the index in the buffer to write the address to.
     * @param position the position to write.
     * @return the size in bytes the address actually took.
     */
    static int writeChildrenPosition(byte[] buffer, int fromIndex, int position) {
        int index = fromIndex;
        switch (getByteSize(position)) {
            case 1:
                buffer[index++] = (byte)position;
                return 1;
            case 2:
                buffer[index++] = (byte)(0xFF & (position >> 8));
                buffer[index++] = (byte)(0xFF & position);
                return 2;
            case 3:
                buffer[index++] = (byte)(0xFF & (position >> 16));
                buffer[index++] = (byte)(0xFF & (position >> 8));
                buffer[index++] = (byte)(0xFF & position);
                return 3;
            case 0:
                return 0;
            default:
                throw new RuntimeException("Position " + position + " has a strange size");
        }
    }
    /**
     * Makes the flag value for a PtNode.
     *
     * @param hasMultipleChars whether the PtNode has multiple chars.
     * @param isTerminal whether the PtNode is terminal.
     * @param childrenAddressSize the size of a children address.
     * @param hasBigrams whether the PtNode has bigrams.
     * @param isNotAWord whether the PtNode is not a word.
     * @param isPossiblyOffensive whether the PtNode is a possibly offensive entry.
     * @return the flags
     */
    static int makePtNodeFlags(boolean hasMultipleChars, boolean isTerminal, int childrenAddressSize, boolean hasShortcuts, boolean hasBigrams,
                               boolean isNotAWord, boolean isPossiblyOffensive) {
        byte flags = 0;
        if (hasMultipleChars) flags |= FormatSpec.FLAG_HAS_MULTIPLE_CHARS;
        if (isTerminal) flags |= FormatSpec.FLAG_IS_TERMINAL;
        switch (childrenAddressSize) {
            case 1:
                flags |= FormatSpec.FLAG_CHILDREN_ADDRESS_TYPE_ONEBYTE;
                break;
            case 2:
                flags |= FormatSpec.FLAG_CHILDREN_ADDRESS_TYPE_TWOBYTES;
                break;
            case 3:
                flags |= FormatSpec.FLAG_CHILDREN_ADDRESS_TYPE_THREEBYTES;
                break;
            case 0:
                flags |= FormatSpec.FLAG_CHILDREN_ADDRESS_TYPE_NOADDRESS;
                break;
            default:
                throw new RuntimeException("Node with a strange address");
        }
        if (hasShortcuts) flags |= FormatSpec.FLAG_HAS_SHORTCUT_TARGETS;
        if (hasBigrams) flags |= FormatSpec.FLAG_HAS_BIGRAMS;
        if (isNotAWord) flags |= FormatSpec.FLAG_IS_NOT_A_WORD;
        if (isPossiblyOffensive) flags |= FormatSpec.FLAG_IS_POSSIBLY_OFFENSIVE;
        return flags;
    }
    static byte makePtNodeFlags(PtNode node, int childrenOffset) {
        return (byte) makePtNodeFlags(node.mChars.length > 1, node.isTerminal(),
            getByteSize(childrenOffset),
            node.mShortcutTargets != null && !node.mShortcutTargets.isEmpty(),
            node.mBigrams != null && !node.mBigrams.isEmpty(),
            node.mIsNotAWord, node.mIsPossiblyOffensive);
    }
    /**
     * Makes the flag value for a bigram.
     *
     * @param more whether there are more bigrams after this one.
     * @param offset the offset of the bigram.
     * @param bigramFrequency the frequency of the bigram, 0..255.
     * @param unigramFrequency the unigram frequency of the same word, 0..255.
     * @param word the second bigram, for debugging purposes
     * @return the flags
     */
    static int makeBigramFlags(boolean more, int offset,
                                             int bigramFrequency, int unigramFrequency, String word) {
        int bigramFlags = (more ? FormatSpec.FLAG_BIGRAM_SHORTCUT_ATTR_HAS_NEXT : 0)
            + (offset < 0 ? FormatSpec.FLAG_BIGRAM_ATTR_OFFSET_NEGATIVE : 0);
        switch (getByteSize(offset)) {
            case 1:
                bigramFlags |= FormatSpec.FLAG_BIGRAM_ATTR_ADDRESS_TYPE_ONEBYTE;
                break;
            case 2:
                bigramFlags |= FormatSpec.FLAG_BIGRAM_ATTR_ADDRESS_TYPE_TWOBYTES;
                break;
            case 3:
                bigramFlags |= FormatSpec.FLAG_BIGRAM_ATTR_ADDRESS_TYPE_THREEBYTES;
                break;
            default:
                throw new RuntimeException("Strange offset size");
        }
        int frequency;
        if (unigramFrequency > bigramFrequency) {
            // todo: frequencies should be adjusted in dicts
            //  Logging is disabled because spammy
//            Log.e(TAG, "Unigram freq is superior to bigram freq for \"" + word
//                + "\". Bigram freq is " + bigramFrequency + ", unigram freq for "
//                + word + " is " + unigramFrequency);
            frequency = unigramFrequency;
        } else {
            frequency = bigramFrequency;
        }
        bigramFlags += getBigramFrequencyDiff(unigramFrequency, frequency)
            & FormatSpec.FLAG_BIGRAM_SHORTCUT_ATTR_FREQUENCY;
        return bigramFlags;
    }
    public static int getBigramFrequencyDiff(int unigramFrequency,
                                             int bigramFrequency) {
        // We compute the difference between 255 (which means probability = 1) and the
        // unigram score. We split this into a number of discrete steps.
        // Now, the steps are numbered 0~15; 0 represents an increase of 1 step while 15
        // represents an increase of 16 steps: a value of 15 will be interpreted as the median
        // value of the 16th step. In all justice, if the bigram frequency is low enough to be
        // rounded below the first step (which means it is less than half a step higher than the
        // unigram frequency) then the unigram frequency itself is the best approximation of the
        // bigram freq that we could possibly supply, hence we should *not* include this bigram
        // in the file at all.
        // until this is done, we'll write 0 and slightly overestimate this case.
        // In other words, 0 means "between 0.5 step and 1.5 step", 1 means "between 1.5 step
        // and 2.5 steps", and 15 means "between 15.5 steps and 16.5 steps". So we want to
        // divide our range [unigramFreq..MAX_TERMINAL_FREQUENCY] in 16.5 steps to get the
        // step size. Then we compute the start of the first step (the one where value 0 starts)
        // by adding half-a-step to the unigramFrequency. From there, we compute the integer
        // number of steps to the bigramFrequency. One last thing: we want our steps to include
        // their lower bound and exclude their higher bound so we need to have the first step
        // start at exactly 1 unit higher than floor(unigramFreq + half a step).
        // Note : to reconstruct the score, the dictionary reader will need to divide
        // MAX_TERMINAL_FREQUENCY - unigramFreq by 16.5 likewise to get the value of the step,
        // and add (discretizedFrequency + 0.5 + 0.5) times this value to get the best
        // approximation. (0.5 to get the first step start, and 0.5 to get the middle of the
        // step pointed by the discretized frequency.
        float stepSize =
            (FormatSpec.MAX_TERMINAL_FREQUENCY - unigramFrequency)
                / (1.5f + FormatSpec.MAX_BIGRAM_FREQUENCY);
        float firstStepStart = 1 + unigramFrequency + (stepSize / 2.0f);
        int discretizedFrequency = (int)((bigramFrequency - firstStepStart) / stepSize);
        // If the bigram freq is less than half-a-step higher than the unigram freq, we get -1
        // here. The best approximation would be the unigram freq itself, so we should not
        // include this bigram in the dictionary. For now, register as 0, and live with the
        // small over-estimation that we get in this case. TODO: actually remove this bigram
        // if discretizedFrequency < 0.
        return discretizedFrequency > 0 ? discretizedFrequency : 0;
    }
    static int getChildrenPosition(PtNode ptNode,
                                                 HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int positionOfChildrenPosField = ptNode.mCachedAddressAfterUpdate
            + getNodeHeaderSize(ptNode, codePointToOneByteCodeMap);
        if (ptNode.isTerminal()) {
            // A terminal node has the frequency.
            // If positionOfChildrenPosField is incorrect, we may crash when jumping to the children
            // position.
            positionOfChildrenPosField += FormatSpec.PTNODE_FREQUENCY_SIZE;
        }
        return null == ptNode.mChildren ? FormatSpec.NO_CHILDREN_ADDRESS
                                        : ptNode.mChildren.mCachedAddressAfterUpdate - positionOfChildrenPosField;
    }
    /**
     * Write a PtNodeArray. The PtNodeArray is expected to have its final position cached.
     *
     * @param dict the dictionary the node array is a part of (for relative offsets).
     * @param dictEncoder the dictionary encoder.
     * @param ptNodeArray the node array to write.
     * @param codePointToOneByteCodeMap the map to convert the code points.
     */
    static void writePlacedPtNodeArray(FusionDictionary dict,
                                                     DictEncoder dictEncoder, PtNodeArray ptNodeArray,
                                                     HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        // TODO: Make the code in common with BinaryDictIOUtils#writePtNode
        dictEncoder.setPosition(ptNodeArray.mCachedAddressAfterUpdate);
        int ptNodeCount = ptNodeArray.mData.size();
        dictEncoder.writePtNodeCount(ptNodeCount);
        for (int i = 0; i < ptNodeCount; ++i) {
            PtNode ptNode = ptNodeArray.mData.get(i);
            if (dictEncoder.getPosition() != ptNode.mCachedAddressAfterUpdate) {
                throw new RuntimeException("Bug: write index is not the same as the cached address "
                    + "of the node : " + dictEncoder.getPosition() + " <> "
                    + ptNode.mCachedAddressAfterUpdate);
            }
            // Validity checks.
            if (DebugFlags.DEBUG_ENABLED && ptNode.getProbability() > FormatSpec.MAX_TERMINAL_FREQUENCY) {
                throw new RuntimeException("A node has a frequency > "
                    + FormatSpec.MAX_TERMINAL_FREQUENCY
                    + " : " + ptNode.mProbabilityInfo.toString());
            }
            dictEncoder.writePtNode(ptNode, dict, codePointToOneByteCodeMap);
        }
        if (dictEncoder.getPosition() != ptNodeArray.mCachedAddressAfterUpdate
            + ptNodeArray.mCachedSize) {
            throw new RuntimeException("Not the same size : written "
                + (dictEncoder.getPosition() - ptNodeArray.mCachedAddressAfterUpdate)
                + " bytes from a node that should have " + ptNodeArray.mCachedSize + " bytes");
        }
    }
    /**
     * Dumps a collection of useful statistics about a list of PtNode arrays.
     * <p>
     * This prints purely informative stuff, like the total estimated file size, the
     * number of PtNode arrays, of PtNodes, the repartition of each address size, etc
     *
     * @param ptNodeArrays the list of PtNode arrays.
     */
    static void showStatistics(ArrayList<PtNodeArray> ptNodeArrays) {
        int firstTerminalAddress = Integer.MAX_VALUE;
        int lastTerminalAddress = Integer.MIN_VALUE;
        int size = 0;
        int ptNodes = 0;
        int maxNodes = 0;
        int maxRuns = 0;
        for (PtNodeArray ptNodeArray : ptNodeArrays) {
            if (maxNodes < ptNodeArray.mData.size()) maxNodes = ptNodeArray.mData.size();
            for (PtNode ptNode : ptNodeArray.mData) {
                ++ptNodes;
                if (ptNode.mChars.length > maxRuns) maxRuns = ptNode.mChars.length;
                if (ptNode.isTerminal()) {
                    if (ptNodeArray.mCachedAddressAfterUpdate < firstTerminalAddress)
                        firstTerminalAddress = ptNodeArray.mCachedAddressAfterUpdate;
                    if (ptNodeArray.mCachedAddressAfterUpdate > lastTerminalAddress)
                        lastTerminalAddress = ptNodeArray.mCachedAddressAfterUpdate;
                }
            }
            if (ptNodeArray.mCachedAddressAfterUpdate + ptNodeArray.mCachedSize > size) {
                size = ptNodeArray.mCachedAddressAfterUpdate + ptNodeArray.mCachedSize;
            }
        }
        int[] ptNodeCounts = new int[maxNodes + 1];
        int[] runCounts = new int[maxRuns + 1];
        for (PtNodeArray ptNodeArray : ptNodeArrays) {
            ++ptNodeCounts[ptNodeArray.mData.size()];
            for (PtNode ptNode : ptNodeArray.mData) {
                ++runCounts[ptNode.mChars.length];
            }
        }
        Log.i(TAG, "Statistics:\n"
            + "  Total file size " + size + "\n"
            + "  " + ptNodeArrays.size() + " node arrays\n"
            + "  " + ptNodes + " PtNodes (" + ((float)ptNodes / ptNodeArrays.size())
            + " PtNodes per node)\n"
            + "  First terminal at " + firstTerminalAddress + "\n"
            + "  Last terminal at " + lastTerminalAddress + "\n"
            + "  PtNode stats : max = " + maxNodes);
    }
    /**
     * Writes a file header to an output stream.
     *
     * @param destination the stream to write the file header to.
     * @param dict the dictionary to write.
     * @param formatOptions file format options.
     * @param codePointOccurrenceArray code points ordered by occurrence count.
     * @return the size of the header.
     */
    static int writeDictionaryHeader(OutputStream destination, FusionDictionary dict, FormatOptions formatOptions,
                                     ArrayList<Entry<Integer, Integer>> codePointOccurrenceArray)
        throws IOException, UnsupportedFormatException {
        int version = formatOptions.mVersion;
        if ((version >= FormatSpec.MINIMUM_SUPPORTED_STATIC_VERSION &&
            version <= FormatSpec.MAXIMUM_SUPPORTED_STATIC_VERSION) || (
            version >= FormatSpec.MINIMUM_SUPPORTED_DYNAMIC_VERSION &&
                version <= FormatSpec.MAXIMUM_SUPPORTED_DYNAMIC_VERSION)) {
            // Dictionary is valid
        } else {
            throw new UnsupportedFormatException("Requested file format version " + version
                + ", but this implementation only supports static versions "
                + FormatSpec.MINIMUM_SUPPORTED_STATIC_VERSION + " through "
                + FormatSpec.MAXIMUM_SUPPORTED_STATIC_VERSION + " and dynamic versions "
                + FormatSpec.MINIMUM_SUPPORTED_DYNAMIC_VERSION + " through "
                + FormatSpec.MAXIMUM_SUPPORTED_DYNAMIC_VERSION);
        }
        ByteArrayOutputStream headerBuffer = new ByteArrayOutputStream(256);
        // The magic number in big-endian order.
        // Magic number for all versions.
        headerBuffer.write((byte) (0xFF & (FormatSpec.MAGIC_NUMBER >> 24)));
        headerBuffer.write((byte) (0xFF & (FormatSpec.MAGIC_NUMBER >> 16)));
        headerBuffer.write((byte) (0xFF & (FormatSpec.MAGIC_NUMBER >> 8)));
        headerBuffer.write((byte) (0xFF & FormatSpec.MAGIC_NUMBER));
        // Dictionary version.
        headerBuffer.write((byte) (0xFF & (version >> 8)));
        headerBuffer.write((byte) (0xFF & version));
        // Options flags
        // TODO: Remove this field.
        final int options = 0;
        headerBuffer.write((byte) (0xFF & (options >> 8)));
        headerBuffer.write((byte) (0xFF & options));
        int headerSizeOffset = headerBuffer.size();
        // Placeholder to be written later with header size.
        for (int i = 0; i < 4; ++i) {
            headerBuffer.write(0);
        }
        // Write out the options.
        for (String key : dict.mOptions.mAttributes.keySet()) {
            String value = dict.mOptions.mAttributes.get(key);
            CharEncoding.writeString(headerBuffer, key, null);
            CharEncoding.writeString(headerBuffer, value, null);
        }
        // Write out the codePointTable if there is codePointOccurrenceArray.
        if (codePointOccurrenceArray != null) {
            String codePointTableString =
                encodeCodePointTable(codePointOccurrenceArray);
            CharEncoding.writeString(headerBuffer, DictionaryHeader.CODE_POINT_TABLE_KEY, null);
            CharEncoding.writeString(headerBuffer, codePointTableString, null);
        }
        int size = headerBuffer.size();
        byte[] bytes = headerBuffer.toByteArray();
        // Write out the header size.
        bytes[headerSizeOffset] = (byte) (0xFF & (size >> 24));
        bytes[headerSizeOffset + 1] = (byte) (0xFF & (size >> 16));
        bytes[headerSizeOffset + 2] = (byte) (0xFF & (size >> 8));
        bytes[headerSizeOffset + 3] = (byte) (0xFF & (size >> 0));
        destination.write(bytes);
        headerBuffer.close();
        return size;
    }
    static final class CodePointTable {
        final HashMap<Integer, Integer> mCodePointToOneByteCodeMap;
        final ArrayList<Entry<Integer, Integer>> mCodePointOccurrenceArray;
        // Let code point table empty for version 200 dictionary which used in test
        CodePointTable() {
            mCodePointToOneByteCodeMap = null;
            mCodePointOccurrenceArray = null;
        }
        CodePointTable(HashMap<Integer, Integer> codePointToOneByteCodeMap,
                       ArrayList<Entry<Integer, Integer>> codePointOccurrenceArray) {
            mCodePointToOneByteCodeMap = codePointToOneByteCodeMap;
            mCodePointOccurrenceArray = codePointOccurrenceArray;
        }
    }
    private static String encodeCodePointTable(
        ArrayList<Entry<Integer, Integer>> codePointOccurrenceArray) {
        StringBuilder codePointTableString = new StringBuilder();
        int currentCodePointTableIndex = FormatSpec.MINIMAL_ONE_BYTE_CHARACTER_VALUE;
        for (Entry<Integer, Integer> entry : codePointOccurrenceArray) {
            // Native reads the table as a string
            codePointTableString.appendCodePoint(entry.getKey());
            if (FormatSpec.MAXIMAL_ONE_BYTE_CHARACTER_VALUE < ++currentCodePointTableIndex) {
                break;
            }
        }
        return codePointTableString.toString();
    }
}
