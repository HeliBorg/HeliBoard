/*
 * Copyright (C) 2011 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import helium314.keyboard.latin.makedict.FormatSpec.DictionaryOptions;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

/**
 * Reads and writes XML files for a FusionDictionary.
 * <p>
 * All functions in this class are static.
 */
public class XmlDictInputOutput {

    private static final String ROOT_TAG = "wordlist";
    private static final String WORD_TAG = "w";
    private static final String BIGRAM_TAG = "bigram";
    private static final String SHORTCUT_TAG = "shortcut";
    private static final String PROBABILITY_ATTR = "f";
    private static final String WORD_ATTR = "word";
    private static final String NOT_A_WORD_ATTR = "not_a_word";

    /**
     * SAX handler for a unigram XML file.
     */
    static private class UnigramHandler extends DefaultHandler {
        // Parser states
        private static final int START = 1;
        private static final int WORD = 2;
        private static final int UNKNOWN = 3;
        private static final int SHORTCUT_ONLY_WORD_PROBABILITY = 1;

        FusionDictionary mDictionary;
        int mState; // the state of the parser
        int mFreq; // the currently read freq
        String mWord; // the current word
        final HashMap<String, ArrayList<WeightedString>> mShortcutsMap;

        /**
         * Create the handler.
         *
         * @param shortcuts the shortcuts as a map. This may be empty, but may not be null.
         */
        public UnigramHandler(HashMap<String, ArrayList<WeightedString>> shortcuts) {
            mDictionary = null;
            mShortcutsMap = shortcuts;
            mWord = "";
            mState = START;
            mFreq = 0;
        }

        public FusionDictionary getFinalDictionary() {
            FusionDictionary dict = mDictionary;
            for (String shortcutOnly : mShortcutsMap.keySet()) {
                if (dict.hasWord(shortcutOnly)) continue;
                dict.add(shortcutOnly, new ProbabilityInfo(SHORTCUT_ONLY_WORD_PROBABILITY),
                        mShortcutsMap.get(shortcutOnly), true /* isNotAWord */, false);
            }
            mDictionary = null;
            mShortcutsMap.clear();
            mWord = "";
            mState = START;
            mFreq = 0;
            return dict;
        }

        @Override
        public void startElement(String uri, String localName, String qName, Attributes attrs) {
            if (WORD_TAG.equals(localName)) {
                mState = WORD;
                mWord = "";
                for (int attrIndex = 0; attrIndex < attrs.getLength(); ++attrIndex) {
                    String attrName = attrs.getLocalName(attrIndex);
                    if (PROBABILITY_ATTR.equals(attrName)) {
                        mFreq = Integer.parseInt(attrs.getValue(attrIndex));
                    }
                }
            } else if (ROOT_TAG.equals(localName)) {
                HashMap<String, String> attributes = new HashMap<>();
                for (int attrIndex = 0; attrIndex < attrs.getLength(); ++attrIndex) {
                    String attrName = attrs.getLocalName(attrIndex);
                    attributes.put(attrName, attrs.getValue(attrIndex));
                }
                mDictionary = new FusionDictionary(new FusionDictionary.PtNodeArray(),
                        new DictionaryOptions(attributes));
            } else {
                mState = UNKNOWN;
            }
        }

        @Override
        public void characters(char[] ch, int start, int length) {
            if (WORD == mState) {
                // The XML parser is free to return text in arbitrary chunks one after the
                // other. In particular, this happens in some implementations when it finds
                // an escape code like "&amp;".
                mWord += String.copyValueOf(ch, start, length);
            }
        }

        @Override
        public void endElement(String uri, String localName, String qName) {
            if (WORD == mState) {
                mDictionary.add(mWord, new ProbabilityInfo(mFreq), mShortcutsMap.get(mWord),
                        false /* isNotAWord */, false);
                mState = START;
            }
        }
    }

    static private class AssociativeListHandler extends DefaultHandler {
        private final String SRC_TAG;
        private final String SRC_ATTRIBUTE;
        private final String DST_TAG;
        private final String DST_ATTRIBUTE;
        private final String DST_FREQ;

        // In this version of the XML file, the bigram frequency is given as an int 0..XML_MAX
        private final static int XML_MAX = 256;
        // In memory and in the binary dictionary the bigram frequency is 0..MEMORY_MAX
        private final static int MEMORY_MAX = 256;
        private final static int XML_TO_MEMORY_RATIO = XML_MAX / MEMORY_MAX;

        private String mSrc;
        private final HashMap<String, ArrayList<WeightedString>> mAssocMap;

        public AssociativeListHandler(String srcTag, String srcAttribute,
                                      String dstTag, String dstAttribute, String dstFreq) {
            SRC_TAG = srcTag;
            SRC_ATTRIBUTE = srcAttribute;
            DST_TAG = dstTag;
            DST_ATTRIBUTE = dstAttribute;
            DST_FREQ = dstFreq;
            mSrc = null;
            mAssocMap = new HashMap<>();
        }

        @Override
        public void startElement(String uri, String localName, String qName, Attributes attrs) {
            if (SRC_TAG.equals(localName)) {
                mSrc = attrs.getValue(uri, SRC_ATTRIBUTE);
            } else if (DST_TAG.equals(localName)) {
                String dst = attrs.getValue(uri, DST_ATTRIBUTE);
                int freq = getValueFromFreqString(attrs.getValue(uri, DST_FREQ));
                WeightedString bigram = new WeightedString(dst, freq / XML_TO_MEMORY_RATIO);
                ArrayList<WeightedString> bigramList = mAssocMap.get(mSrc);
                if (null == bigramList) bigramList = new ArrayList<>();
                bigramList.add(bigram);
                mAssocMap.put(mSrc, bigramList);
            }
        }

        protected int getValueFromFreqString(String freqString) {
            return Integer.parseInt(freqString);
        }

        // This may return an empty map, but will never return null.
        public HashMap<String, ArrayList<WeightedString>> getAssocMap() {
            return mAssocMap;
        }
    }

    /**
     * SAX handler for a bigram XML file.
     */
    static private class BigramHandler extends AssociativeListHandler {
        private final static String BIGRAM_W1_TAG = "bi";
        private final static String BIGRAM_W2_TAG = "w";
        private final static String BIGRAM_W1_ATTRIBUTE = "w1";
        private final static String BIGRAM_W2_ATTRIBUTE = "w2";
        private final static String BIGRAM_FREQ_ATTRIBUTE = "p";

        public BigramHandler() {
            super(BIGRAM_W1_TAG, BIGRAM_W1_ATTRIBUTE, BIGRAM_W2_TAG, BIGRAM_W2_ATTRIBUTE,
                    BIGRAM_FREQ_ATTRIBUTE);
        }

        // As per getAssocMap(), this never returns null.
        public HashMap<String, ArrayList<WeightedString>> getBigramMap() {
            return getAssocMap();
        }
    }

    /**
     * SAX handler for a shortcut & whitelist XML file.
     */
    static private class ShortcutAndWhitelistHandler extends AssociativeListHandler {
        private final static String ENTRY_TAG = "entry";
        private final static String ENTRY_ATTRIBUTE = "shortcut";
        private final static String TARGET_TAG = "target";
        private final static String REPLACEMENT_ATTRIBUTE = "replacement";
        private final static String TARGET_PRIORITY_ATTRIBUTE = "priority";
        private final static String WHITELIST_MARKER = "whitelist";
        private final static int WHITELIST_FREQ_VALUE = 15;
        private final static int MIN_FREQ = 0;
        private final static int MAX_FREQ = 14;

        public ShortcutAndWhitelistHandler() {
            super(ENTRY_TAG, ENTRY_ATTRIBUTE, TARGET_TAG, REPLACEMENT_ATTRIBUTE,
                    TARGET_PRIORITY_ATTRIBUTE);
        }

        @Override
        protected int getValueFromFreqString(String freqString) {
            if (WHITELIST_MARKER.equals(freqString)) {
                return WHITELIST_FREQ_VALUE;
            }
            int intValue = super.getValueFromFreqString(freqString);
            if (intValue < MIN_FREQ || intValue > MAX_FREQ) {
                throw new RuntimeException("Shortcut freq out of range. Accepted range is "
                        + MIN_FREQ + ".." + MAX_FREQ);
            }
            return intValue;
        }

        // As per getAssocMap(), this never returns null.
        public HashMap<String, ArrayList<WeightedString>> getShortcutAndWhitelistMap() {
            return getAssocMap();
        }
    }

    /**
     * Basic test to find out whether the file is in the unigram XML format or not.
     * <p>
     * Concretely this only tests the header line.
     *
     * @param filename The name of the file to test.
     * @return true if the file is in the unigram XML format, false otherwise
     */
    public static boolean isXmlUnigramDictionary(String filename) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(filename), "UTF-8"))) {
            String firstLine = reader.readLine();
            return firstLine.matches("^\\s*<wordlist .*>\\s*$");
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Reads a dictionary from an XML file.
     * <p>
     * This is the public method that will parse an XML file and return the corresponding memory
     * representation.
     *
     * @param unigrams the file to read the data from.
     * @param shortcuts the file to read the shortcuts & whitelist from, or null.
     * @param bigrams the file to read the bigrams from, or null.
     * @return the in-memory representation of the dictionary.
     */
    public static FusionDictionary readDictionaryXml(BufferedInputStream unigrams,
                                                     BufferedInputStream shortcuts, BufferedInputStream bigrams)
            throws SAXException, IOException, ParserConfigurationException {
        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setNamespaceAware(true);
        SAXParser parser = factory.newSAXParser();
        BigramHandler bigramHandler = new BigramHandler();
        if (null != bigrams) parser.parse(bigrams, bigramHandler);

        ShortcutAndWhitelistHandler shortcutAndWhitelistHandler =
                new ShortcutAndWhitelistHandler();
        if (null != shortcuts) parser.parse(shortcuts, shortcutAndWhitelistHandler);

        UnigramHandler unigramHandler =
                new UnigramHandler(shortcutAndWhitelistHandler.getShortcutAndWhitelistMap());
        parser.parse(unigrams, unigramHandler);
        FusionDictionary dict = unigramHandler.getFinalDictionary();
        HashMap<String, ArrayList<WeightedString>> bigramMap = bigramHandler.getBigramMap();
        for (String firstWord : bigramMap.keySet()) {
            if (!dict.hasWord(firstWord)) continue;
            ArrayList<WeightedString> bigramList = bigramMap.get(firstWord);
            for (WeightedString bigram : bigramList) {
                if (!dict.hasWord(bigram.mWord)) continue;
                dict.setBigram(firstWord, bigram.mWord, bigram.mProbabilityInfo);
            }
        }
        return dict;
    }

    /**
     * Writes a dictionary to an XML file.
     * <p>
     * The output format is the "second" format, which supports bigrams and shortcuts/whitelist.
     *
     * @param destination a destination stream to write to.
     * @param dict the dictionary to write.
     */
    public static void writeDictionaryXml(BufferedWriter destination,
                                          FusionDictionary dict) throws IOException {
        TreeSet<WordProperty> wordPropertiesInDict = new TreeSet<>();
        for (WordProperty wordProperty : dict) {
            wordPropertiesInDict.add(wordProperty);
        }
        // TODO: use an XMLSerializer if this gets big
        destination.write("<wordlist format=\"2\"");
        for (String key : dict.mOptions.mAttributes.keySet()) {
            String value = dict.mOptions.mAttributes.get(key);
            destination.write(" " + key + "=\"" + value + "\"");
        }
        destination.write(">\n");
        destination.write("<!-- Warning: there is no code to read this format yet. -->\n");
        for (WordProperty wordProperty : wordPropertiesInDict) {
            destination.write("  <" + WORD_TAG + " " + WORD_ATTR + "=\"" + wordProperty.mWord
                    + "\" " + PROBABILITY_ATTR + "=\"" + wordProperty.getProbability()
                    + (wordProperty.mIsNotAWord ? "\" " + NOT_A_WORD_ATTR + "=\"true" : "")
                    + "\">");
            if (null != wordProperty.mShortcutTargets) {
                destination.write("\n");
                for (WeightedString target : wordProperty.mShortcutTargets) {
                    destination.write("    <" + SHORTCUT_TAG + " " + PROBABILITY_ATTR + "=\""
                            + target.getProbability() + "\">" + target.mWord + "</" + SHORTCUT_TAG
                            + ">\n");
                }
                destination.write("  ");
            }
            if (null != wordProperty.mNgrams) {
                destination.write("\n");
                for (NgramProperty ngram : wordProperty.mNgrams) {
                    WeightedString bigram = ngram.mTargetWord;
                    destination.write("    <" + BIGRAM_TAG + " " + PROBABILITY_ATTR + "=\""
                            + bigram.getProbability() + "\">" + bigram.mWord
                            + "</" + BIGRAM_TAG + ">\n");
                }
                destination.write("  ");
            }
            destination.write("</" + WORD_TAG + ">\n");
        }
        destination.write("</wordlist>\n");
        destination.close();
    }
}
