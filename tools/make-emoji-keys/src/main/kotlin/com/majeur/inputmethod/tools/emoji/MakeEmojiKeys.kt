// SPDX-License-Identifier: GPL-3.0-only

package com.majeur.inputmethod.tools.emoji

import com.majeur.inputmethod.tools.emoji.model.EmojiData
import com.majeur.inputmethod.tools.emoji.model.EmojiGroup
import com.majeur.inputmethod.tools.emoji.model.EmojiSpec
import java.io.File
import java.nio.charset.Charset
import java.util.*
import kotlin.system.exitProcess

class MakeEmojiKeys {

    class Options(argsArray: Array<String>) {

        var assetPath: String? = null

        init {
            val args = listOf(*argsArray).toMutableList()
            var arg: String? = null
            try {
                while (args.isNotEmpty()) {
                    arg = args.removeFirst()
                    if (arg == OPTION_ASSETS) {
                        assetPath = args.removeFirst()
                    } else {
                        usage("Unknown option: $arg")
                    }
                }
            } catch (_: NoSuchElementException) {
                usage("Option $arg needs argument")
            }
        }

        fun usage(message: String?) {
            message?.let { System.err.println(it) }
            System.err.println("usage: make-emoji-keys $OPTION_ASSETS <res_output_dir>")
            exitProcess(1)
        }
    }

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val options = Options(args)
            val jar = JarUtils.getJarFile(Companion::class.java)

            val parser = EmojiUCDTestFileParser()
            parser.parse(JarUtils.getLatestEmojiTestResource(jar))
            val emojis = parser.getParsedData()
            emojis.appendRegionalIndicators()

            val parser2 = AndroidEmojiSupportFileParser()
            parser2.parse(JarUtils.getEmojiSupportResource(jar))
            val supportData = parser2.getParsedData()

            if (options.assetPath != null) {
                writeMinApiLevels(options.assetPath!!, emojis, supportData)
                writeEmojis(options.assetPath!!, emojis)
            }
        }

        private fun writeMinApiLevels(outDir: String, emojiData: EmojiData, supportData: Map<Int, Int>) {
            val minApiLevels = mutableMapOf<Int, MutableSet<String>>()
            fun addMinLevel(emoji: EmojiSpec) {
                val minApi = getMinApi(emoji.codes, supportData)
                if (minApi < 0)
                    throw Exception("unknown min SDK for ${emoji.name}")
                if (minApi > 21)
                    minApiLevels.getOrPut(minApi) { mutableSetOf() }.add(emoji.text)
            }

            EmojiGroup.entries.filterNot { it == EmojiGroup.COMPONENT }.forEach { group ->
                emojiData[group].forEach { emoji ->
                    addMinLevel(emoji)
                    emoji.variants.forEach { addMinLevel(it) }
                }
            }
            if (minApiLevels.any { it.value.any { it.contains(" ") } })
                throw Exception("emoji contains space")
            val text = minApiLevels.map { "${it.key} ${it.value.joinToString(" ")}" }
                .sorted().joinToString("\n")
            File(outDir, "minApi.txt").writeText(text, Charset.forName("UTF-8"))
        }

        private fun writeEmojis(outDir: String, emojiData: EmojiData) {
            // each category gets a file, one main emoji per line, followed by popups
            EmojiGroup.entries.filterNot { it == EmojiGroup.COMPONENT }
                .forEach { writeEmojiGroup(File(outDir, it.name + ".txt"), emojiData[it]) }
        }

        private fun writeEmojiGroup(outFile: File, emojis: List<EmojiSpec>) {
            val text = emojis.joinToString("\n") { emoji ->
                if (emoji.variants.isEmpty()) emoji.text
                else "${emoji.text} ${emoji.variants.joinToString(" ") { it.text }}"
            }
            outFile.writeText(text, Charset.forName("UTF-8"))
        }

        private fun getMinApi(codes: IntArray, supportData: Map<Int, Int>): Int {
            val hash = codes.joinToString("").hashCode()
            return supportData[hash] ?: -1
        }
    }
}

private const val OPTION_ASSETS = "-assets"
