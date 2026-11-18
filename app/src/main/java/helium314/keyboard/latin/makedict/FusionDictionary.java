/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.define.DecoderSpecificConstants;
import helium314.keyboard.latin.makedict.FormatSpec.DictionaryOptions;
import helium314.keyboard.latin.utils.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;

/** A dictionary that can fusion heads and tails of words for more compression. */
// todo: this appears to be inefficient (a LOT of GC), and quite slow when processing bigrams (or is it actually the CombinedInputOutput?)
public final class FusionDictionary implements Iterable<WordProperty> {
    private static final int CHARACTER_NOT_FOUND_INDEX = -1;

    /**
     * A node array of the dictionary, containing several PtNodes.
     * <p>
     * A PtNodeArray is but an ordered array of PtNodes, which essentially contain all the
     * real information.
     * This class also contains fields to cache size and address, to help with binary
     * generation.
     */
    public static final class PtNodeArray {
        ArrayList<PtNode> mData;
        // To help with binary generation
        int mCachedSize = Integer.MIN_VALUE;
        // mCachedAddressBefore/AfterUpdate are helpers for binary dictionary generation. They
        // always hold the same value except between dictionary address compression, during which
        // the update process needs to know about both values at the same time. Updating will
        // update the AfterUpdate value, and the code will move them to BeforeUpdate before
        // the next update pass.
        int mCachedAddressBeforeUpdate = Integer.MIN_VALUE;
        int mCachedAddressAfterUpdate = Integer.MIN_VALUE;
        int mCachedParentAddress = 0;

        public PtNodeArray() {
            mData = new ArrayList<>();
        }
        public PtNodeArray(ArrayList<PtNode> data) {
            Collections.sort(data, PTNODE_COMPARATOR);
            mData = data;
        }
    }

    /**
     * PtNode is a group of characters, with probability information, shortcut targets, bigrams,
     * and children (Pt means Patricia Trie).
     * <p>
     * This is the central class of the in-memory representation. A PtNode is what can
     * be seen as a traditional "trie node", except it can hold several characters at the
     * same time. A PtNode essentially represents one or several characters in the middle
     * of the trie tree; as such, it can be a terminal, and it can have children.
     * In this in-memory representation, whether the PtNode is a terminal or not is represented
     * by mProbabilityInfo. The PtNode is a terminal when the mProbabilityInfo is not null and the
     * PtNode is not a terminal when the mProbabilityInfo is null. A terminal may have non-null
     * shortcuts and/or bigrams, but a non-terminal may not. Moreover, children, if present,
     * are non-null.
     */
    public static final class PtNode {
        private static final int NOT_A_TERMINAL = -1;
        final int[] mChars;
        ArrayList<WeightedString> mShortcutTargets;
        ArrayList<WeightedString> mBigrams;
        // null == mProbabilityInfo indicates this is not a terminal.
        ProbabilityInfo mProbabilityInfo;
        int mTerminalId; // NOT_A_TERMINAL == mTerminalId indicates this is not a terminal.
        PtNodeArray mChildren;
        boolean mIsNotAWord; // Only a shortcut
        boolean mIsPossiblyOffensive;
        // mCachedSize and mCachedAddressBefore/AfterUpdate are helpers for binary dictionary
        // generation. Before and After always hold the same value except during dictionary
        // address compression, where the update process needs to know about both values at the
        // same time. Updating will update the AfterUpdate value, and the code will move them
        // to BeforeUpdate before the next update pass.
        // The update process does not need two versions of mCachedSize.
        int mCachedSize; // The size, in bytes, of this PtNode.
        int mCachedAddressBeforeUpdate; // The address of this PtNode (before update)
        int mCachedAddressAfterUpdate; // The address of this PtNode (after update)

        public PtNode(int[] chars, ArrayList<WeightedString> shortcutTargets, ArrayList<WeightedString> bigrams,
                      ProbabilityInfo probabilityInfo, boolean isNotAWord, boolean isPossiblyOffensive) {
            mChars = chars;
            mProbabilityInfo = probabilityInfo;
            mTerminalId = probabilityInfo == null ? NOT_A_TERMINAL : probabilityInfo.mProbability;
            mShortcutTargets = shortcutTargets;
            mBigrams = bigrams;
            mChildren = null;
            mIsNotAWord = isNotAWord;
            mIsPossiblyOffensive = isPossiblyOffensive;
        }

        public PtNode(int[] chars, ArrayList<WeightedString> shortcutTargets,
                      ArrayList<WeightedString> bigrams, ProbabilityInfo probabilityInfo,
                      boolean isNotAWord, boolean isPossiblyOffensive,
                      PtNodeArray children) {
            mChars = chars;
            mProbabilityInfo = probabilityInfo;
            mShortcutTargets = shortcutTargets;
            mBigrams = bigrams;
            mChildren = children;
            mIsNotAWord = isNotAWord;
            mIsPossiblyOffensive = isPossiblyOffensive;
        }

        public void addChild(PtNode n) {
            if (null == mChildren) {
                mChildren = new PtNodeArray();
            }
            mChildren.mData.add(n);
        }

        public int getTerminalId() {
            return mTerminalId;
        }

        public boolean isTerminal() {
            return mProbabilityInfo != null;
        }

        public int getProbability() {
            return isTerminal() ? mProbabilityInfo.mProbability : NOT_A_TERMINAL;
        }

        public boolean getIsNotAWord() {
            return mIsNotAWord;
        }

        public boolean getIsPossiblyOffensive() {
            return mIsPossiblyOffensive;
        }

        public ArrayList<WeightedString> getShortcutTargets() {
            // We don't want write permission to escape outside the package, so we return a copy
            if (null == mShortcutTargets) return null;
            return new ArrayList<>(mShortcutTargets);
        }

        public ArrayList<WeightedString> getBigrams() {
            // We don't want write permission to escape outside the package, so we return a copy
            if (null == mBigrams) return null;
            return new ArrayList<>(mBigrams);
        }

        public boolean hasSeveralChars() {
            assert(mChars.length > 0);
            return 1 < mChars.length;
        }

        /**
         * Adds a word to the bigram list. Updates the probability information if the word already
         * exists.
         */
        public void addBigram(String word, ProbabilityInfo probabilityInfo) {
            if (mBigrams == null) {
                mBigrams = new ArrayList<>();
            }
            WeightedString bigram = getBigram(word);
            if (bigram != null) {
                bigram.mProbabilityInfo = probabilityInfo;
            } else {
                bigram = new WeightedString(word, probabilityInfo);
                mBigrams.add(bigram);
            }
        }

        /**
         * Gets the shortcut target for the given word. Returns null if the word is not in the
         * shortcut list.
         */
        public WeightedString getShortcut(String word) {
            // TODO: Don't do a linear search
            if (mShortcutTargets != null) {
                int size = mShortcutTargets.size();
                for (int i = 0; i < size; ++i) {
                    WeightedString shortcut = mShortcutTargets.get(i);
                    if (shortcut.mWord.equals(word)) {
                        return shortcut;
                    }
                }
            }
            return null;
        }

        /**
         * Gets the bigram for the given word.
         * Returns null if the word is not in the bigrams list.
         */
        public WeightedString getBigram(String word) {
            // TODO: Don't do a linear search
            if (mBigrams != null) {
                int size = mBigrams.size();
                for (int i = 0; i < size; ++i) {
                    WeightedString bigram = mBigrams.get(i);
                    if (bigram.mWord.equals(word)) {
                        return bigram;
                    }
                }
            }
            return null;
        }

        /**
         * Updates the PtNode with the given properties. Adds the shortcut and bigram lists to
         * the existing ones if any. Note: unigram, bigram, and shortcut frequencies are only
         * updated if they are higher than the existing ones.
         */
        void update(ProbabilityInfo probabilityInfo, ArrayList<WeightedString> shortcutTargets,
                    ArrayList<WeightedString> bigrams, boolean isNotAWord, boolean isPossiblyOffensive) {
            mProbabilityInfo = ProbabilityInfo.max(mProbabilityInfo, probabilityInfo);
            if (shortcutTargets != null) {
                if (mShortcutTargets == null) {
                    mShortcutTargets = shortcutTargets;
                } else {
                    int size = shortcutTargets.size();
                    for (int i = 0; i < size; ++i) {
                        WeightedString shortcut = shortcutTargets.get(i);
                        WeightedString existingShortcut = getShortcut(shortcut.mWord);
                        if (existingShortcut == null) {
                            mShortcutTargets.add(shortcut);
                        } else {
                            existingShortcut.mProbabilityInfo = ProbabilityInfo.max(
                                    existingShortcut.mProbabilityInfo, shortcut.mProbabilityInfo);
                        }
                    }
                }
            }
            if (bigrams != null) {
                if (mBigrams == null) {
                    mBigrams = bigrams;
                } else {
                    int size = bigrams.size();
                    for (int i = 0; i < size; ++i) {
                        WeightedString bigram = bigrams.get(i);
                        WeightedString existingBigram = getBigram(bigram.mWord);
                        if (existingBigram == null) {
                            mBigrams.add(bigram);
                        } else {
                            existingBigram.mProbabilityInfo = ProbabilityInfo.max(
                                    existingBigram.mProbabilityInfo, bigram.mProbabilityInfo);
                        }
                    }
                }
            }
            mIsNotAWord = isNotAWord;
            mIsPossiblyOffensive = isPossiblyOffensive;
        }
    }

    public final DictionaryOptions mOptions;
    public final PtNodeArray mRootNodeArray;

    public FusionDictionary(PtNodeArray rootNodeArray, DictionaryOptions options) {
        mRootNodeArray = rootNodeArray;
        mOptions = options;
    }

    public void addOptionAttribute(String key, String value) {
        mOptions.mAttributes.put(key, value);
    }

    /**
     * Helper method to convert a String to an int array.
     */
    static int[] getCodePoints(String word) {
        // TODO: this is a copy-paste of the old contents of StringUtils.toCodePointArray,
        // which is not visible from the makedict package. Factor this code.
        int length = word.length();
        if (length == 0) return new int[] {};
        char[] characters = word.toCharArray();
        int[] codePoints = new int[Character.codePointCount(characters, 0, length)];
        int codePoint = Character.codePointAt(characters, 0);
        int dsti = 0;
        for (int srci = Character.charCount(codePoint);
                srci < length; srci += Character.charCount(codePoint), ++dsti) {
            codePoints[dsti] = codePoint;
            codePoint = Character.codePointAt(characters, srci);
        }
        codePoints[dsti] = codePoint;
        return codePoints;
    }

    /**
     * Helper method to add a word as a string.
     * <p>
     * This method adds a word to the dictionary with the given frequency. Optional
     * lists of bigrams and shortcuts can be passed here. For each word inside,
     * they will be added to the dictionary as necessary.
     *
     * @param word the word to add.
     * @param probabilityInfo probability information of the word.
     * @param shortcutTargets a list of shortcut targets for this word, or null.
     * @param isNotAWord true if this should not be considered a word (e.g. shortcut only)
     * @param isPossiblyOffensive true if this word is possibly offensive
     */
    public void add(String word, ProbabilityInfo probabilityInfo, ArrayList<WeightedString> shortcutTargets,
                    boolean isNotAWord, boolean isPossiblyOffensive) {
        add(getCodePoints(word), probabilityInfo, shortcutTargets, isNotAWord, isPossiblyOffensive);
    }

    /**
     * Sanity check for a PtNode array.
     * <p>
     * This method checks that all PtNodes in a node array are ordered as expected.
     * If they are, nothing happens. If they aren't, an exception is thrown.
     */
    private static void checkStack(PtNodeArray ptNodeArray) {
        ArrayList<PtNode> stack = ptNodeArray.mData;
        int lastValue = -1;
        for (int i = 0; i < stack.size(); ++i) {
            int currentValue = stack.get(i).mChars[0];
            if (currentValue <= lastValue) {
                throw new RuntimeException("Invalid stack");
            }
            lastValue = currentValue;
        }
    }

    /**
     * Helper method to add a new bigram to the dictionary.
     *
     * @param word0 the previous word of the context
     * @param word1 the next word of the context
     * @param probabilityInfo the bigram probability info
     */
    public void setBigram(String word0, String word1, ProbabilityInfo probabilityInfo) {
        PtNode ptNode0 = findWordInTree(mRootNodeArray, word0);
        if (ptNode0 != null) {
            PtNode ptNode1 = findWordInTree(mRootNodeArray, word1);
            if (ptNode1 == null) {
                add(getCodePoints(word1), new ProbabilityInfo(0), null, false /* isNotAWord */,
                        false /* isPossiblyOffensive */);
                // The PtNode for the first word may have moved by the above insertion,
                // if word1 and word2 share a common stem that happens not to have been
                // a cutting point until now. In this case, we need to refresh ptNode.
                ptNode0 = findWordInTree(mRootNodeArray, word0);
            }
            ptNode0.addBigram(word1, probabilityInfo);
        } else {
            throw new RuntimeException("First word of bigram not found " + word0);
        }
    }

    /**
     * Add a word to this dictionary.
     * <p>
     * The shortcuts, if any, have to be in the dictionary already. If they aren't,
     * an exception is thrown.
     *
     * @param word the word, as an int array.
     * @param probabilityInfo the probability information of the word.
     * @param shortcutTargets an optional list of shortcut targets for this word (null if none).
     * @param isNotAWord true if this is not a word for spellcheking purposes (shortcut only or so)
     * @param isPossiblyOffensive true if this word is possibly offensive
     */
    private void add(int[] word, ProbabilityInfo probabilityInfo, ArrayList<WeightedString> shortcutTargets,
                     boolean isNotAWord, boolean isPossiblyOffensive) {
        assert(probabilityInfo.mProbability <= FormatSpec.MAX_TERMINAL_FREQUENCY);
        if (word.length >= DecoderSpecificConstants.DICTIONARY_MAX_WORD_LENGTH) {
            Log.w("FusionDictionary", "Ignoring a word that is too long: word.length = " + word.length);
            return;
        }

        PtNodeArray currentNodeArray = mRootNodeArray;
        int charIndex = 0;

        PtNode currentPtNode = null;
        int differentCharIndex = 0; // Set by the loop to the index of the char that differs
        int nodeIndex = findIndexOfChar(mRootNodeArray, word[charIndex]);
        while (CHARACTER_NOT_FOUND_INDEX != nodeIndex) {
            currentPtNode = currentNodeArray.mData.get(nodeIndex);
            differentCharIndex = compareCharArrays(currentPtNode.mChars, word, charIndex);
            if (ARRAYS_ARE_EQUAL != differentCharIndex
                    && differentCharIndex < currentPtNode.mChars.length) break;
            if (null == currentPtNode.mChildren) break;
            charIndex += currentPtNode.mChars.length;
            if (charIndex >= word.length) break;
            currentNodeArray = currentPtNode.mChildren;
            nodeIndex = findIndexOfChar(currentNodeArray, word[charIndex]);
        }

        if (CHARACTER_NOT_FOUND_INDEX == nodeIndex) {
            // No node at this point to accept the word. Create one.
            int insertionIndex = findInsertionIndex(currentNodeArray, word[charIndex]);
            PtNode newPtNode = new PtNode(Arrays.copyOfRange(word, charIndex, word.length),
                    shortcutTargets, null /* bigrams */, probabilityInfo, isNotAWord,
                    isPossiblyOffensive);
            currentNodeArray.mData.add(insertionIndex, newPtNode);
            if (DebugFlags.DEBUG_ENABLED) checkStack(currentNodeArray);
        } else {
            // There is a word with a common prefix.
            if (differentCharIndex == currentPtNode.mChars.length) {
                if (charIndex + differentCharIndex >= word.length) {
                    // The new word is a prefix of an existing word, but the node on which it
                    // should end already exists as is. Since the old PtNode was not a terminal,
                    // make it one by filling in its frequency and other attributes
                    currentPtNode.update(probabilityInfo, shortcutTargets, null, isNotAWord,
                            isPossiblyOffensive);
                } else {
                    // The new word matches the full old word and extends past it.
                    // We only have to create a new node and add it to the end of this.
                    PtNode newNode = new PtNode(
                            Arrays.copyOfRange(word, charIndex + differentCharIndex, word.length),
                                    shortcutTargets, null /* bigrams */, probabilityInfo,
                                    isNotAWord, isPossiblyOffensive);
                    currentPtNode.mChildren = new PtNodeArray();
                    currentPtNode.mChildren.mData.add(newNode);
                }
            } else {
                if (0 == differentCharIndex) {
                    // Exact same word. Update the frequency if higher. This will also add the
                    // new shortcuts to the existing shortcut list if it already exists.
                    currentPtNode.update(probabilityInfo, shortcutTargets, null,
                            currentPtNode.mIsNotAWord && isNotAWord,
                            currentPtNode.mIsPossiblyOffensive || isPossiblyOffensive);
                } else {
                    // Partial prefix match only. We have to replace the current node with a node
                    // containing the current prefix and create two new ones for the tails.
                    PtNodeArray newChildren = new PtNodeArray();
                    PtNode newOldWord = new PtNode(
                            Arrays.copyOfRange(currentPtNode.mChars, differentCharIndex,
                                    currentPtNode.mChars.length), currentPtNode.mShortcutTargets,
                            currentPtNode.mBigrams, currentPtNode.mProbabilityInfo,
                            currentPtNode.mIsNotAWord, currentPtNode.mIsPossiblyOffensive,
                            currentPtNode.mChildren);
                    newChildren.mData.add(newOldWord);

                    PtNode newParent;
                    if (charIndex + differentCharIndex >= word.length) {
                        newParent = new PtNode(
                                Arrays.copyOfRange(currentPtNode.mChars, 0, differentCharIndex),
                                shortcutTargets, null /* bigrams */, probabilityInfo,
                                isNotAWord, isPossiblyOffensive, newChildren);
                    } else {
                        newParent = new PtNode(
                                Arrays.copyOfRange(currentPtNode.mChars, 0, differentCharIndex),
                                null /* shortcutTargets */, null /* bigrams */,
                                null /* probabilityInfo */, false /* isNotAWord */,
                                false /* isPossiblyOffensive */, newChildren);
                        PtNode newWord = new PtNode(Arrays.copyOfRange(word,
                                charIndex + differentCharIndex, word.length),
                                shortcutTargets, null /* bigrams */, probabilityInfo,
                                isNotAWord, isPossiblyOffensive);
                        int addIndex = word[charIndex + differentCharIndex]
                                > currentPtNode.mChars[differentCharIndex] ? 1 : 0;
                        newChildren.mData.add(addIndex, newWord);
                    }
                    currentNodeArray.mData.set(nodeIndex, newParent);
                }
                if (DebugFlags.DEBUG_ENABLED) checkStack(currentNodeArray);
            }
        }
    }

    private static final int ARRAYS_ARE_EQUAL = 0;

    /**
     * Custom comparison of two int arrays taken to contain character codes.
     * <p>
     * This method compares the two arrays passed as an argument in a lexicographic way,
     * with an offset in the dst string.
     * This method does NOT test for the first character. It is taken to be equal.
     * I repeat: this method starts the comparison at 1 <> dstOffset + 1.
     * The index where the strings differ is returned. ARRAYS_ARE_EQUAL = 0 is returned if the
     * strings are equal. This works BECAUSE we don't look at the first character.
     *
     * @param src the left-hand side string of the comparison.
     * @param dst the right-hand side string of the comparison.
     * @param dstOffset the offset in the right-hand side string.
     * @return the index at which the strings differ, or ARRAYS_ARE_EQUAL = 0 if they don't.
     */
    private static int compareCharArrays(int[] src, int[] dst, int dstOffset) {
        // We do NOT test the first char, because we come from a method that already
        // tested it.
        for (int i = 1; i < src.length; ++i) {
            if (dstOffset + i >= dst.length) return i;
            if (src[i] != dst[dstOffset + i]) return i;
        }
        if (dst.length > src.length) return src.length;
        return ARRAYS_ARE_EQUAL;
    }

    /**
     * Helper class that compares and sorts two PtNodes according to their
     * first element only. I repeat: ONLY the first element is considered, the rest
     * is ignored.
     * This comparator imposes orderings that are inconsistent with equals.
     */
    static final class PtNodeComparator implements java.util.Comparator<PtNode> {
        @Override
        public int compare(PtNode p1, PtNode p2) {
            if (p1.mChars[0] == p2.mChars[0]) return 0;
            return p1.mChars[0] < p2.mChars[0] ? -1 : 1;
        }
    }
    final static PtNodeComparator PTNODE_COMPARATOR = new PtNodeComparator();

    /**
     * Finds the insertion index of a character within a node array.
     */
    private static int findInsertionIndex(PtNodeArray nodeArray, int character) {
        ArrayList<PtNode> data = nodeArray.mData;
        PtNode reference = new PtNode(new int[] { character }, null, null, null, false, false);
        int result = Collections.binarySearch(data, reference, PTNODE_COMPARATOR);
        return result >= 0 ? result : -result - 1;
    }

    /**
     * Find the index of a char in a node array, if it exists.
     *
     * @param nodeArray the node array to search in.
     * @param character the character to search for.
     * @return the position of the character if it's there, or CHARACTER_NOT_FOUND_INDEX = -1 else.
     */
    private static int findIndexOfChar(PtNodeArray nodeArray, int character) {
        int insertionIndex = findInsertionIndex(nodeArray, character);
        if (nodeArray.mData.size() <= insertionIndex) return CHARACTER_NOT_FOUND_INDEX;
        return character == nodeArray.mData.get(insertionIndex).mChars[0] ? insertionIndex
                : CHARACTER_NOT_FOUND_INDEX;
    }

    /**
     * Helper method to find a word in a given branch.
     */
    public static PtNode findWordInTree(PtNodeArray rootNodeArray, String string) {
        PtNodeArray nodeArray = rootNodeArray;
        int index = 0;
        StringBuilder checker = DebugFlags.DEBUG_ENABLED ? new StringBuilder() : null;
        int[] codePoints = getCodePoints(string);

        PtNode currentPtNode;
        do {
            int indexOfGroup = findIndexOfChar(nodeArray, codePoints[index]);
            if (CHARACTER_NOT_FOUND_INDEX == indexOfGroup) return null;
            currentPtNode = nodeArray.mData.get(indexOfGroup);

            if (codePoints.length - index < currentPtNode.mChars.length) return null;
            int newIndex = index;
            while (newIndex < codePoints.length && newIndex - index < currentPtNode.mChars.length) {
                if (currentPtNode.mChars[newIndex - index] != codePoints[newIndex]) return null;
                newIndex++;
            }
            index = newIndex;

            if (DebugFlags.DEBUG_ENABLED) {
                checker.append(new String(currentPtNode.mChars, 0, currentPtNode.mChars.length));
            }
            if (index < codePoints.length) {
                nodeArray = currentPtNode.mChildren;
            }
        } while (null != nodeArray && index < codePoints.length);

        if (index < codePoints.length) return null;
        if (!currentPtNode.isTerminal()) return null;
        if (DebugFlags.DEBUG_ENABLED && !string.equals(checker.toString())) return null;
        return currentPtNode;
    }

    /**
     * Helper method to find out whether a word is in the dict or not.
     */
    public boolean hasWord(String s) {
        if (null == s || s.isEmpty()) {
            throw new RuntimeException("Can't search for a null or empty string");
        }
        return null != findWordInTree(mRootNodeArray, s);
    }

    /**
     * Recursively count the number of PtNodes in a given branch of the trie.
     *
     * @param nodeArray the parent node.
     * @return the number of PtNodes in all the branch under this node.
     */
    public static int countPtNodes(PtNodeArray nodeArray) {
        int nodeSize = nodeArray.mData.size();
        int size = nodeSize;
        for (int i = nodeSize - 1; i >= 0; --i) {
            PtNode ptNode = nodeArray.mData.get(i);
            if (null != ptNode.mChildren)
                size += countPtNodes(ptNode.mChildren);
        }
        return size;
    }

    /**
     * Iterator to walk through a dictionary.
     * <p>
     * This is purely for convenience.
     */
    public static final class DictionaryIterator implements Iterator<WordProperty> {
        private static final class Position {
            public Iterator<PtNode> pos;
            public int length;
            public Position(ArrayList<PtNode> ptNodes) {
                pos = ptNodes.iterator();
                length = 0;
            }
        }
        final StringBuilder mCurrentString;
        final LinkedList<Position> mPositions;

        public DictionaryIterator(ArrayList<PtNode> ptRoot) {
            mCurrentString = new StringBuilder();
            mPositions = new LinkedList<>();
            Position rootPos = new Position(ptRoot);
            mPositions.add(rootPos);
        }

        @Override
        public boolean hasNext() {
            for (Position p : mPositions) {
                if (p.pos.hasNext()) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public WordProperty next() {
            Position currentPos = mPositions.getLast();
            mCurrentString.setLength(currentPos.length);

            do {
                if (currentPos.pos.hasNext()) {
                    PtNode currentPtNode = currentPos.pos.next();
                    currentPos.length = mCurrentString.length();
                    for (int i : currentPtNode.mChars) {
                        mCurrentString.append(Character.toChars(i));
                    }
                    if (null != currentPtNode.mChildren) {
                        currentPos = new Position(currentPtNode.mChildren.mData);
                        currentPos.length = mCurrentString.length();
                        mPositions.addLast(currentPos);
                    }
                    if (currentPtNode.isTerminal()) {
                        return new WordProperty(mCurrentString.toString(),
                                currentPtNode.mProbabilityInfo,
                                currentPtNode.mShortcutTargets, currentPtNode.mBigrams,
                                currentPtNode.mIsNotAWord, currentPtNode.mIsPossiblyOffensive);
                    }
                } else {
                    mPositions.removeLast();
                    currentPos = mPositions.getLast();
                    mCurrentString.setLength(mPositions.getLast().length);
                }
            } while (true);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Unsupported yet");
        }

    }

    /**
     * Method to return an iterator.
     * <p>
     * This method enables Java's enhanced for loop. With this you can have a FusionDictionary x
     * and say : for (Word w : x) {}
     */
    @Override
    public Iterator<WordProperty> iterator() {
        return new DictionaryIterator(mRootNodeArray.mData);
    }
}
