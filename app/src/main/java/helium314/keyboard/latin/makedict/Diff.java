/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.makedict;

import helium314.keyboard.latin.makedict.FusionDictionary.PtNode;

import java.util.ArrayList;
import java.util.HashMap;

public class Diff {
    public static void diffHeaders(FusionDictionary dict0, FusionDictionary dict1) {
        boolean hasDifferences = false;
        HashMap<String, String> options1 = new HashMap<>(dict1.mOptions.mAttributes);
        for (String optionKey : dict0.mOptions.mAttributes.keySet()) {
            if (!dict0.mOptions.mAttributes.get(optionKey).equals(
                    dict1.mOptions.mAttributes.get(optionKey))) {
                System.out.println("  " + optionKey + " : "
                        + dict0.mOptions.mAttributes.get(optionKey) + " <=> "
                        + dict1.mOptions.mAttributes.get(optionKey));
                hasDifferences = true;
            }
            options1.remove(optionKey);
        }
        for (String optionKey : options1.keySet()) {
            System.out.println("  " + optionKey + " : null <=> " + options1.get(optionKey));
            hasDifferences = true;
        }
        if (!hasDifferences) {
            System.out.println("  No differences");
        }
    }

    public static void diffWords(FusionDictionary dict0, FusionDictionary dict1) {
        boolean hasDifferences = false;
        for (WordProperty word0Property : dict0) {
            PtNode word1PtNode = FusionDictionary.findWordInTree(dict1.mRootNodeArray,
                    word0Property.mWord);
            if (null == word1PtNode) {
                // This word is not in dict1
                System.out.println("Deleted: " + word0Property.mWord + " "
                        + word0Property.getProbability());
                hasDifferences = true;
            } else {
                // We found the word. Compare frequencies, shortcuts, bigrams
                if (word0Property.getProbability() != word1PtNode.getProbability()) {
                    System.out.println("Probability changed: " + word0Property.mWord + " "
                            + word0Property.getProbability() + " -> "
                            + word1PtNode.getProbability());
                    hasDifferences = true;
                }
                if (word0Property.mIsNotAWord != word1PtNode.getIsNotAWord()) {
                    System.out.println("Not a word: " + word0Property.mWord + " "
                            + word0Property.mIsNotAWord + " -> " + word1PtNode.getIsNotAWord());
                    hasDifferences = true;
                }
                if (word0Property.mIsPossiblyOffensive != word1PtNode.getIsPossiblyOffensive()) {
                    System.out.println("Possibly-offensive: " + word0Property.mWord + " "
                            + word0Property.mIsPossiblyOffensive + " -> "
                            + word1PtNode.getIsPossiblyOffensive());
                    hasDifferences = true;
                }
                hasDifferences |= hasAttributesDifferencesAndPrintThemIfAny(word0Property.mWord,
                        "Bigram", word0Property.getBigrams(), word1PtNode.getBigrams());
                hasDifferences |= hasAttributesDifferencesAndPrintThemIfAny(word0Property.mWord,
                        "Shortcut", word0Property.mShortcutTargets,
                        word1PtNode.getShortcutTargets());
            }
        }
        for (WordProperty word1Property : dict1) {
            PtNode word0PtNode = FusionDictionary.findWordInTree(dict0.mRootNodeArray,
                    word1Property.mWord);
            if (null == word0PtNode) {
                // This word is not in dict0
                System.out.println("Added: " + word1Property.mWord + " "
                        + word1Property.getProbability());
                hasDifferences = true;
            }
        }
        if (!hasDifferences) {
            System.out.println("  No differences");
        }
    }

    private static boolean hasAttributesDifferencesAndPrintThemIfAny(String word, String type, ArrayList<WeightedString> list0,
                                                                     ArrayList<WeightedString> list1) {
        if (null == list1) {
            if (null == list0) return false;
            for (WeightedString attribute0 : list0) {
                System.out.println(type + " removed: " + word + " " + attribute0.mWord + " "
                        + attribute0.getProbability());
            }
            return true;
        }
        boolean hasDifferences = false;
        if (null != list0) {
            for (WeightedString attribute0 : list0) {
                // The following tests with #equals(). The WeightedString#equals() method returns
                // true if both the string and the frequency are the same.
                if (!list1.contains(attribute0)) {
                    hasDifferences = true;
                    // Search for a word with the same string but a different frequency
                    boolean foundString = false;
                    for (WeightedString attribute1 : list1) {
                        if (attribute0.mWord.equals(attribute1.mWord)) {
                            System.out.println(type + " freq changed: " + word + " "
                                    + attribute0.mWord + " " + attribute0.getProbability() + " -> "
                                    + attribute1.getProbability());
                            list1.remove(attribute1);
                            foundString = true;
                            break;
                        }
                    }
                    if (!foundString) {
                        // We come here if we haven't found any matching string.
                        System.out.println(type + " removed: " + word + " " + attribute0.mWord + " "
                                + attribute0.getProbability());
                    }
                } else {
                    list1.remove(attribute0);
                }
            }
        }
        // We removed any matching word that we found, so now list1 only contains words that
        // are not included in list0.
        for (WeightedString attribute1 : list1) {
            hasDifferences = true;
            System.out.println(type + " added: " + word + " " + attribute1.mWord + " "
                    + attribute1.getProbability());
        }
        return hasDifferences;
    }
}
