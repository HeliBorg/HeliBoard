/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin.makedict;

import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;

/**
 * Decodes binary files for a FusionDictionary.
 * <p>
 * All the methods in this class are static.
 * <p>
 * TODO: Move this file to makedict/internal.
 * TODO: Rename this class to DictDecoderUtils.
 */
public final class BinaryDictDecoderUtils {
    private BinaryDictDecoderUtils() {
        // This utility class is not publicly instantiable.
    }

    /**
     * A class grouping utility function for our specific character encoding.
     */
    static final class CharEncoding {
        /**
         * Helper method to find out whether this code fits on one byte
         */
        private static boolean fitsOnOneByte(int character,
                                             HashMap<Integer, Integer> codePointToOneByteCodeMap) {
            int codePoint = character;
            if (codePointToOneByteCodeMap != null) {
                if (codePointToOneByteCodeMap.containsKey(character)) {
                    codePoint = codePointToOneByteCodeMap.get(character);
                }
            }
            return codePoint >= FormatSpec.MINIMAL_ONE_BYTE_CHARACTER_VALUE
                && codePoint <= FormatSpec.MAXIMAL_ONE_BYTE_CHARACTER_VALUE;
        }
        /**
         * Compute the size of a character given its character code.
         * <p>
         * Char format is:
         * 1 byte = bbbbbbbb match
         * case 000xxxxx: xxxxx << 16 + next byte << 8 + next byte
         * else: if 00011111 (= 0x1F) : this is the terminator. This is a relevant choice because
         *       unicode code points range from 0 to 0x10FFFF, so any 3-byte value starting with
         *       00011111 would be outside unicode.
         * else: iso-latin-1 code
         * This allows for the whole unicode range to be encoded, including chars outside of
         * the BMP. Also everything in the iso-latin-1 charset is only 1 byte, except control
         * characters which should never happen anyway (and still work, but take 3 bytes).
         *
         * @param character the character code.
         * @return the size in binary encoded-form, either 1 or 3 bytes.
         */
        static int getCharSize(int character, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
            // See char encoding in FusionDictionary.java
            if (fitsOnOneByte(character, codePointToOneByteCodeMap)) return 1;
            if (FormatSpec.INVALID_CHARACTER == character) return 1;
            return 3;
        }
        /**
         * Compute the byte size of a character array.
         */
        static int getCharArraySize(int[] chars, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
            int size = 0;
            for (int character : chars) size += getCharSize(character, codePointToOneByteCodeMap);
            return size;
        }
        /**
         * Writes a char array to a byte buffer.
         *
         * @param codePoints the code point array to write.
         * @param buffer the byte buffer to write to.
         * @param fromIndex the index in buffer to write the character array to.
         * @param codePointToOneByteCodeMap the map to convert the code point.
         * @return the index after the last character.
         */
        static int writeCharArray(int[] codePoints, byte[] buffer, int fromIndex, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
            int index = fromIndex;
            for (int codePoint : codePoints) {
                if (codePointToOneByteCodeMap != null && codePointToOneByteCodeMap.containsKey(codePoint)) {
                    // Convert code points
                    codePoint = codePointToOneByteCodeMap.get(codePoint);
                }
                if (1 == getCharSize(codePoint, codePointToOneByteCodeMap)) {
                    buffer[index++] = (byte)codePoint;
                } else {
                    buffer[index++] = (byte)(0xFF & (codePoint >> 16));
                    buffer[index++] = (byte)(0xFF & (codePoint >> 8));
                    buffer[index++] = (byte)(0xFF & codePoint);
                }
            }
            return index;
        }
        static int writeString(byte[] buffer, int fromIndex, String word, HashMap<Integer, Integer> codePointToOneByteCodeMap) {
            int length = word.length();
            int index = fromIndex;

            for (int i = 0; i < length; i = word.offsetByCodePoints(i, 1)) {
                int codePoint = word.codePointAt(i);
                if (codePointToOneByteCodeMap != null && codePointToOneByteCodeMap.containsKey(codePoint)) {
                    codePoint = codePointToOneByteCodeMap.get(codePoint);
                }

                if (1 == getCharSize(codePoint, codePointToOneByteCodeMap)) {
                    buffer[index++] = (byte)codePoint;
                } else {
                    buffer[index++] = (byte)(0xFF & codePoint >> 16);
                    buffer[index++] = (byte)(0xFF & codePoint >> 8);
                    buffer[index++] = (byte)(0xFF & codePoint);
                }
            }

            buffer[index++] = FormatSpec.PTNODE_CHARACTERS_TERMINATOR;
            return index - fromIndex;
        }
        /**
         * Writes a string with our character format to an OutputStream.
         * <p>
         * This will also write the terminator byte.
         *
         * @param stream the OutputStream to write to.
         * @param word the string to write.
         * @return the size written, in bytes.
         */
        static int writeString(OutputStream stream, String word, HashMap<Integer, Integer> codePointToOneByteCodeMap) throws IOException {
            int length = word.length();
            int written = 0;
            for (int i = 0; i < length; i = word.offsetByCodePoints(i, 1)) {
                int codePoint = word.codePointAt(i);
                int charSize = getCharSize(codePoint, codePointToOneByteCodeMap);
                if (1 == charSize) {
                    stream.write((byte) codePoint);
                } else {
                    stream.write((byte) (0xFF & (codePoint >> 16)));
                    stream.write((byte) (0xFF & (codePoint >> 8)));
                    stream.write((byte) (0xFF & codePoint));
                }
                written += charSize;
            }
            stream.write(FormatSpec.PTNODE_CHARACTERS_TERMINATOR);
            written += FormatSpec.PTNODE_TERMINATOR_SIZE;
            return written;
        }
    }
}
