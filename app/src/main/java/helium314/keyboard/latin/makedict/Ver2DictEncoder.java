/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import helium314.keyboard.latin.define.DebugFlags;
import helium314.keyboard.latin.makedict.BinaryDictDecoderUtils.CharEncoding;
import helium314.keyboard.latin.makedict.BinaryDictEncoderUtils.CodePointTable;

import helium314.keyboard.latin.makedict.FormatSpec.FormatOptions;
import helium314.keyboard.latin.makedict.FusionDictionary.PtNode;
import helium314.keyboard.latin.makedict.FusionDictionary.PtNodeArray;
import helium314.keyboard.latin.utils.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Objects;

/**
 * An implementation of DictEncoder for version 2 binary dictionary.
 */
public class Ver2DictEncoder implements DictEncoder {
    private final File mDictFile;
    private OutputStream mOutStream;
    private byte[] mBuffer;
    private int mPosition;
    private final int mCodePointTableMode;
    public static final String TAG = "Ver2DictEncoder";
    public static final int CODE_POINT_TABLE_OFF = 0;
    public static final int CODE_POINT_TABLE_ON = 1;
    public Ver2DictEncoder(File dictFile, int codePointTableMode) {
        mDictFile = dictFile;
        mOutStream = null;
        mBuffer = null;
        mCodePointTableMode = codePointTableMode;
    }
    // This constructor is used only by BinaryDictOffdeviceUtilsTests.
    // If you want to use this in the production code, you should consider keeping consistency of
    // the interface of Ver3DictDecoder by using factory.
    public Ver2DictEncoder(OutputStream outStream) {
        mDictFile = null;
        mOutStream = outStream;
        mCodePointTableMode = CODE_POINT_TABLE_OFF;
    }
    private void openStream() throws FileNotFoundException {
        mOutStream = new FileOutputStream(mDictFile);
    }
    private void close() throws IOException {
        if (mOutStream != null) {
            mOutStream.close();
            mOutStream = null;
        }
    }

    // todo: what is this? makes dictionary file larger
    private static CodePointTable makeCodePointTable(FusionDictionary dict) {
        HashMap<Integer, Integer> codePointOccurrenceCounts = new HashMap<>();
        for (WordProperty word : dict) {
            // Store per code point occurrence
            String wordString = word.mWord;
            for (int i = 0; i < wordString.length(); ++i) {
                int codePoint = Character.codePointAt(wordString, i);
                if (codePointOccurrenceCounts.containsKey(codePoint)) {
                    codePointOccurrenceCounts.put(codePoint,
                        codePointOccurrenceCounts.get(codePoint) + 1);
                } else {
                    codePointOccurrenceCounts.put(codePoint, 1);
                }
            }
        }
        ArrayList<Entry<Integer, Integer>> codePointOccurrenceArray =
            new ArrayList<>(codePointOccurrenceCounts.entrySet());
        // Descending order sort by occurrence (value side)
        Collections.sort(codePointOccurrenceArray, (a, b) -> {
            if (!Objects.equals(a.getValue(), b.getValue())) {
                return b.getValue().compareTo(a.getValue());
            }
            return b.getKey().compareTo(a.getKey());
        });
        int currentCodePointTableIndex = FormatSpec.MINIMAL_ONE_BYTE_CHARACTER_VALUE;
        // Temporary map for writing of nodes
        HashMap<Integer, Integer> codePointToOneByteCodeMap = new HashMap<>();
        for (Entry<Integer, Integer> entry : codePointOccurrenceArray) {
            // Put a relation from the original code point to the one byte code.
            codePointToOneByteCodeMap.put(entry.getKey(), currentCodePointTableIndex);
            if (FormatSpec.MAXIMAL_ONE_BYTE_CHARACTER_VALUE < ++currentCodePointTableIndex) {
                break;
            }
        }
        // codePointToOneByteCodeMap for writing the trie
        // codePointOccurrenceArray for writing the header
        return new CodePointTable(codePointToOneByteCodeMap, codePointOccurrenceArray);
    }
    @Override
    public void writeDictionary(FusionDictionary dict, FormatOptions formatOptions)
        throws IOException, UnsupportedFormatException {
        // We no longer support anything but the latest version of v2.
        if (formatOptions.mVersion != FormatSpec.VERSION202) {
            throw new UnsupportedFormatException(
                "The given format options has wrong version number : "
                    + formatOptions.mVersion);
        }
        if (mOutStream == null) {
            openStream();
        }
        // Make code point conversion table ordered by occurrence of code points
        // Version 201 or later have codePointTable
        CodePointTable codePointTable;
        if (mCodePointTableMode == CODE_POINT_TABLE_OFF || formatOptions.mVersion
            < FormatSpec.MINIMUM_SUPPORTED_VERSION_OF_CODE_POINT_TABLE) {
            codePointTable = new CodePointTable();
        } else {
            codePointTable = makeCodePointTable(dict);
        }
        BinaryDictEncoderUtils.writeDictionaryHeader(mOutStream, dict, formatOptions,
            codePointTable.mCodePointOccurrenceArray);
        // Addresses are limited to 3 bytes, but since addresses can be relative to each node
        // array, the structure itself is not limited to 16MB. However, if it is over 16MB deciding
        // the order of the PtNode arrays becomes a quite complicated problem, because though the
        // dictionary itself does not have a size limit, each node array must still be within 16MB
        // of all its children and parents. As long as this is ensured, the dictionary file may
        // grow to any size.
        // Leave the choice of the optimal node order to the flattenTree function.
        Log.i(TAG, "Flattening the tree...");
        ArrayList<PtNodeArray> flatNodes = BinaryDictEncoderUtils.flattenTree(dict.mRootNodeArray);
        Log.i(TAG, "Computing addresses...");
        BinaryDictEncoderUtils.computeAddresses(dict, flatNodes, codePointTable.mCodePointToOneByteCodeMap);
        Log.i(TAG, "Checking PtNode array...");
        if (DebugFlags.DEBUG_ENABLED) BinaryDictEncoderUtils.checkFlatPtNodeArrayList(flatNodes);
        // Create a buffer that matches the final dictionary size.
        PtNodeArray lastNodeArray = flatNodes.get(flatNodes.size() - 1);
        int bufferSize = lastNodeArray.mCachedAddressAfterUpdate + lastNodeArray.mCachedSize;
        mBuffer = new byte[bufferSize];
        Log.i(TAG, "Writing file...");
        for (PtNodeArray nodeArray : flatNodes) {
            BinaryDictEncoderUtils.writePlacedPtNodeArray(dict, this, nodeArray,
                codePointTable.mCodePointToOneByteCodeMap);
        }
        if (DebugFlags.DEBUG_ENABLED) BinaryDictEncoderUtils.showStatistics(flatNodes);
        mOutStream.write(mBuffer, 0, mPosition);
        close();
    }
    @Override
    public void setPosition(int position) {
        if (mBuffer == null || position < 0 || position >= mBuffer.length) return;
        mPosition = position;
    }
    @Override
    public int getPosition() {
        return mPosition;
    }
    @Override
    public void writePtNodeCount(int ptNodeCount) {
        int countSize = BinaryDictIOUtils.getPtNodeCountSize(ptNodeCount);
        if (countSize != 1 && countSize != 2) {
            throw new RuntimeException("Strange size from getGroupCountSize : " + countSize);
        }
        int encodedPtNodeCount = (countSize == 2) ?
                                       (ptNodeCount | FormatSpec.LARGE_PTNODE_ARRAY_SIZE_FIELD_SIZE_FLAG) : ptNodeCount;
        mPosition = BinaryDictEncoderUtils.writeUIntToBuffer(mBuffer, mPosition, encodedPtNodeCount,
            countSize);
    }
    private void writePtNodeFlags(PtNode ptNode, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int childrenPos = BinaryDictEncoderUtils.getChildrenPosition(ptNode,
            codePointToOneByteCodeMap);
        mPosition = BinaryDictEncoderUtils.writeUIntToBuffer(mBuffer, mPosition,
            BinaryDictEncoderUtils.makePtNodeFlags(ptNode, childrenPos),
            FormatSpec.PTNODE_FLAGS_SIZE);
    }
    private void writeCharacters(int[] codePoints, boolean hasSeveralChars, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        mPosition = CharEncoding.writeCharArray(codePoints, mBuffer, mPosition,
            codePointToOneByteCodeMap);
        if (hasSeveralChars) {
            mBuffer[mPosition++] = FormatSpec.PTNODE_CHARACTERS_TERMINATOR;
        }
    }
    private void writeFrequency(int frequency) {
        if (frequency >= 0) {
            mPosition = BinaryDictEncoderUtils.writeUIntToBuffer(mBuffer, mPosition, frequency,
                FormatSpec.PTNODE_FREQUENCY_SIZE);
        }
    }
    private void writeChildrenPosition(PtNode ptNode, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        int childrenPos = BinaryDictEncoderUtils.getChildrenPosition(ptNode,
            codePointToOneByteCodeMap);
        mPosition += BinaryDictEncoderUtils.writeChildrenPosition(mBuffer, mPosition,
            childrenPos);
    }
    /**
     * Write a bigram attributes list to mBuffer.
     *
     * @param bigrams the bigram attributes list.
     * @param dict the dictionary the node array is a part of (for relative offsets).
     */
    private void writeBigrams(ArrayList<WeightedString> bigrams, FusionDictionary dict) {
        if (bigrams == null) return;
        Iterator<WeightedString> bigramIterator = bigrams.iterator();
        while (bigramIterator.hasNext()) {
            WeightedString bigram = bigramIterator.next();
            PtNode target = FusionDictionary.findWordInTree(dict.mRootNodeArray, bigram.mWord);
            int addressOfBigram = target.mCachedAddressAfterUpdate;
            int unigramFrequencyForThisWord = target.getProbability();
            int offset = addressOfBigram
                - (mPosition + FormatSpec.PTNODE_ATTRIBUTE_FLAGS_SIZE);
            int bigramFlags = BinaryDictEncoderUtils.makeBigramFlags(bigramIterator.hasNext(),
                offset, bigram.getProbability(), unigramFrequencyForThisWord, bigram.mWord);
            mPosition = BinaryDictEncoderUtils.writeUIntToBuffer(mBuffer, mPosition, bigramFlags,
                FormatSpec.PTNODE_ATTRIBUTE_FLAGS_SIZE);
            mPosition += BinaryDictEncoderUtils.writeChildrenPosition(mBuffer, mPosition,
                Math.abs(offset));
        }
    }
    @Override
    public void writePtNode(PtNode ptNode, FusionDictionary dict, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        writePtNodeFlags(ptNode, codePointToOneByteCodeMap);
        writeCharacters(ptNode.mChars, ptNode.hasSeveralChars(), codePointToOneByteCodeMap);
        writeFrequency(ptNode.getProbability());
        writeChildrenPosition(ptNode, codePointToOneByteCodeMap);
        writeBigrams(ptNode.mBigrams, dict);
        writeShortcuts(ptNode.mShortcutTargets, codePointToOneByteCodeMap);
    }

    private void writeShortcuts(ArrayList<WeightedString> shortcuts, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
        if (shortcuts == null || shortcuts.isEmpty()) return;
        int indexOfShortcutByteSize = mPosition;
        mPosition += 2;
        Iterator<WeightedString> shortcutIterator = shortcuts.iterator();

        while (shortcutIterator.hasNext()) {
            WeightedString target = shortcutIterator.next();
            int shortcutFlags = (shortcutIterator.hasNext() ? FormatSpec.FLAG_BIGRAM_SHORTCUT_ATTR_HAS_NEXT : 0)
                + (target.getProbability() & FormatSpec.FLAG_BIGRAM_SHORTCUT_ATTR_FREQUENCY);
            mPosition = BinaryDictEncoderUtils.writeUIntToBuffer(mBuffer, mPosition, shortcutFlags, 1);
            int shortcutShift = CharEncoding.writeString(mBuffer, mPosition, target.mWord, codePointToOneByteCodeMap);
            mPosition += shortcutShift;
        }

        int shortcutByteSize = mPosition - indexOfShortcutByteSize;
        if (shortcutByteSize > FormatSpec.MAX_SHORTCUT_LIST_SIZE_IN_A_PTNODE) {
            throw new RuntimeException("Shortcut list too large");
        }

        BinaryDictEncoderUtils.writeUIntToBuffer(mBuffer, indexOfShortcutByteSize, shortcutByteSize, 2);
    }
}
