// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import kotlin.test.Test
import helium314.keyboard.latin.makedict.DictIO
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class) // necessary because there is logging in writeBinaryDictionary
class DictionaryTest {
    // this is mostly for checking internals
    @Test fun readCombined() {
        val dict = DictIO.readCombinedDictionary("src/test/resources/somelist.combined")
        val pp = dict.first { it.mWord == "patty-pan" }
        assert(pp.mIsPossiblyOffensive)
        assertEquals(15, pp.probability)
        val i = dict.first { it.mWord == "i" }
        assert(!i.mIsPossiblyOffensive)
        assertEquals(15, i.mShortcutTargets!!.single().probability)
        assertEquals(16, i.probability)
        val ag = dict.first { it.mWord == "again" }
        assertEquals(168, ag.probability)
        assertEquals("when", ag.bigrams.first().mWord)
        assertEquals(1, ag.bigrams.first().mProbabilityInfo.mProbability)
        assertEquals("when", ag.mNgrams.first().mTargetWord.mWord)
    }

    @Test fun readAndWrite() {
        val dict = DictIO.readCombinedDictionary("src/test/resources/somelist.combined")
        val tempFile = "src/test/resources/somelist.temp"
        DictIO.writeCombinedDictionary(tempFile, dict)
        val dict2 = DictIO.readCombinedDictionary(tempFile)
        assertEquals(dict.count(), dict2.count())
        File(tempFile).delete()
        DictIO.writeBinaryDictionary(tempFile, dict)
        // can't read it without lib...
        File(tempFile).delete()
    }

    @Test fun readCombinedOutput() {
        val dict = DictIO.readCombinedDictionary("src/test/resources/somelist.combined")
        val tempFile = File("src/test/resources/somelist.temp")
        DictIO.writeCombinedDictionary(tempFile.absolutePath, dict)
        val dict2 = DictIO.readCombinedDictionary(tempFile.absolutePath)
        val tempFile2 = File("src/test/resources/somelist2.temp")
        DictIO.writeCombinedDictionary(tempFile2.absolutePath, dict2)
        assertEquals(tempFile.readText(), tempFile2.readText())
        assertTrue(tempFile.readText().contains("shortcut="))
        tempFile.delete()
        tempFile2.delete()
    }

    @Test fun readCombinedOutputForHistoryDict() {
        val dict = DictIO.readCombinedDictionary("src/test/resources/history_export.combined")
        val tempFile = File("src/test/resources/somelist.temp")
        DictIO.writeCombinedDictionary(tempFile.absolutePath, dict)
        val dict2 = DictIO.readCombinedDictionary(tempFile.absolutePath)
        val tempFile2 = File("src/test/resources/somelist2.temp")
        DictIO.writeCombinedDictionary(tempFile2.absolutePath, dict2)
        assertEquals(tempFile.readText(), tempFile2.readText())
        tempFile.delete()
        tempFile2.delete()
    }
}
