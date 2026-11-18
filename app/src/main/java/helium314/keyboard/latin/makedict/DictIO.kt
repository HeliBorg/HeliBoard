// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin.makedict

import com.android.inputmethod.latin.BinaryDictionary
import helium314.keyboard.latin.common.LocaleUtils.constructLocale
import helium314.keyboard.latin.dictionary.DictionaryInfoUtils
import helium314.keyboard.latin.makedict.FormatSpec.FormatOptions
import helium314.keyboard.latin.utils.CombinedFormatUtils
import org.xml.sax.SAXException
import java.io.BufferedWriter
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.IOException
import java.util.zip.GZIPInputStream
import javax.xml.parsers.ParserConfigurationException

object DictIO {
    // todo: files differ from dicttool, but are same size and are usable
    @Throws(FileNotFoundException::class, IOException::class, UnsupportedFormatException::class)
    fun writeBinaryDictionary(
        outputFilename: String,
        dict: FusionDictionary,
        version: Int = FormatSpec.VERSION202 // 403: dynamic (body & header files), 202: static (.dict file, much smaller)
    ) {
        val formatOptions = FormatOptions(version)
        val encoder = when (version) {
            FormatSpec.VERSION4 -> Ver4DictEncoder(File(outputFilename))
            FormatSpec.VERSION202 -> Ver2DictEncoder(FileOutputStream(outputFilename).buffered())
            else -> throw UnsupportedFormatException("only versions ${FormatSpec.VERSION4} and ${FormatSpec.VERSION202} are supported")
        }
        encoder.writeDictionary(dict, formatOptions)
    }

    @Throws(FileNotFoundException::class, IOException::class)
    fun writeCombinedDictionary(outputFilename: String, dict: FusionDictionary) {
        BufferedWriter(FileWriter(outputFilename)).use { writer ->
            CombinedInputOutput.writeDictionaryCombined(writer, dict)
        }
    }

    // write directly instead of via FusionDictionary, because FusionDictionary can't deal with empty words (from history dictionary)
    @Throws(FileNotFoundException::class, IOException::class)
    fun writeCombinedDictionary(writer: BufferedWriter, dict: BinaryDictionary) {
        writer.append(CombinedFormatUtils.formatAttributeMap(dict.header.mDictionaryOptions.mAttributes))
        var token = 0
        do {
            val next = dict.getNextWordProperty(token)
            writer.append(CombinedFormatUtils.formatWordProperty(next.mWordProperty))
            token = next.mNextToken
        } while (token != 0)
   }

    @Throws(FileNotFoundException::class, IOException::class, UnsupportedFormatException::class)
    fun readBinaryDictionary(binaryFilename: String): FusionDictionary {
        val file = File(binaryFilename)
        val header = DictionaryInfoUtils.getDictionaryFileHeaderOrNull(file) ?: throw UnsupportedFormatException("could not read header")
        val dict = BinaryDictionary(binaryFilename, 0, file.length(), false, header.mLocaleString.constructLocale(), header.mIdString, false)
        return Ver4DictDecoder().readDictionaryBinary(dict)
    }

    @Throws(FileNotFoundException::class, IOException::class)
    fun readCombinedDictionary(combinedFilename: String): FusionDictionary =
        CombinedInputOutput.readDictionaryCombined(FileInputStream(combinedFilename).bufferedReader(Charsets.UTF_8))

    @Throws(FileNotFoundException::class, IOException::class)
    fun readCompressedCombinedDictionary(combinedFilename: String): FusionDictionary {
        val fileStream = FileInputStream(combinedFilename)
        return CombinedInputOutput.readDictionaryCombined(GZIPInputStream(fileStream).bufferedReader(Charsets.UTF_8))
    }

    @Throws(FileNotFoundException::class, SAXException::class, IOException::class, ParserConfigurationException::class)
    fun readXmDictionary(unigramXmlFilename: String, shortcutXmlFilename: String?, bigramXmlFilename: String?): FusionDictionary {
        val shortcuts = shortcutXmlFilename?.let { FileInputStream(shortcutXmlFilename).buffered() }
        val bigrams = bigramXmlFilename?.let { FileInputStream(shortcutXmlFilename).buffered() }
        FileInputStream(unigramXmlFilename).buffered().use { unigrams ->
            return XmlDictInputOutput.readDictionaryXml(unigrams, shortcuts, bigrams)
        }
    }

    fun diff(dict1: FusionDictionary, dict2: FusionDictionary) {
        Diff.diffHeaders(dict1, dict2)
        Diff.diffWords(dict1, dict2)
    }
}
