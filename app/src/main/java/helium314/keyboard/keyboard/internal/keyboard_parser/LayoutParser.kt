// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.internal.keyboard_parser

import android.content.Context
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.AbstractKeyData
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.AutoTextKeyData
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.CaseSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.CharWidthSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KanaSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyData
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyboardStateSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.LayoutDirectionSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.MultiTextKeyData
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.ShiftStateSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.TextKeyData
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.VariationSelector
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.toTextKey
import helium314.keyboard.latin.common.splitOnWhitespace
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.LayoutType
import helium314.keyboard.latin.utils.LayoutUtils
import helium314.keyboard.latin.utils.LayoutUtilsCustom
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.prefs
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.polymorphic

object LayoutParser {
    private const val TAG = "LayoutParser"
    private val layoutCache = hashMapOf<String, (KeyboardParams) -> MutableList<MutableList<KeyData>>>()

    fun clearCache() = layoutCache.clear()

    fun parseLayout(layoutType: LayoutType, params: KeyboardParams, context: Context): MutableList<MutableList<KeyData>> {
        if (layoutType == LayoutType.FUNCTIONAL && !params.mId.isAlphaOrSymbolKeyboard)
            return mutableListOf(mutableListOf()) // no functional keys
        val layoutName = if (layoutType == LayoutType.MAIN) params.mId.mSubtype.mainLayoutName
            else params.mId.mSubtype.layouts[layoutType] ?: Settings.readDefaultLayoutName(layoutType, context.prefs())
        return layoutCache.getOrPut(layoutType.name + layoutName) {
            createCacheLambda(layoutType, layoutName, context)
        }(params)
    }

    /**
     *  Parse for json layout files as used in FlorisBoard, see floris directory for classes taken from FlorisBoard.
     *  Some differences to the FlorisBoard keys:
     *   (currently) only normal keys supported
     *   if label or code are missing one is created from the other
     *   auto_text_key ignored (i.e. interpreted like the default TextKey)
     *   codes of multi_text_key not used, only the label
     *   (currently) popups is always read to [number, main, relevant] layoutPopupKeys, no choice of which to use or which hint is provided
     */
    fun parseJsonString(layoutText: String, strict: Boolean = true): List<List<AbstractKeyData>> =
        if (strict) checkJsonConfig.decodeFromString(layoutText.stripCommentLines())
        else florisJsonConfig.decodeFromString(layoutText.stripCommentLines())

    /** Parse simple layouts, defined only as rows of (normal) keys with popup keys. */
    fun parseSimpleString(layoutText: String): List<List<KeyData>> {
        return LayoutUtils.getSimpleRowStrings(layoutText).map { row ->
            row.split("\n").mapNotNull { parseKey(it) }
        }
    }

    private fun parseKey(key: String): KeyData? {
        if (key.isBlank()) return null
        val split = key.splitOnWhitespace()
        return if (split.size == 1) split.first().toTextKey()
        else split.first().toTextKey(split.drop(1))
    }

    private fun createCacheLambda(layoutType: LayoutType, layoutName: String, context: Context):
                (KeyboardParams) -> MutableList<MutableList<KeyData>> {
        val layoutFileContent = getLayoutFileContent(layoutType, layoutName.substringBefore("+"), context).trimStart()
        if (layoutFileContent.startsWith("[") || (LayoutUtilsCustom.isCustomLayout(layoutName) && layoutFileContent.startsWith("//"))) {
            try {
                val florisKeyData = parseJsonString(layoutFileContent, false)
                return { params ->
                    florisKeyData.mapTo(mutableListOf()) { row ->
                        row.mapNotNullTo(mutableListOf()) { it.compute(params) }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "could not parse json layout for $layoutName, falling back to simple layout parsing", e)
            }
        }
        // not a json, or invalid json
        val simpleKeyData = parseSimpleString(layoutFileContent)
        return { params ->
            simpleKeyData.mapIndexedTo(mutableListOf()) { i, row ->
                val newRow = row.toMutableList()
                if (params.mId.isAlphabetKeyboard && layoutName.endsWith("+"))
                    params.mLocaleKeyboardInfos.getExtraKeys(i+1)?.let { newRow.addAll(it) }
                newRow
            }
        }
    }

    private fun getLayoutFileContent(layoutType: LayoutType, layoutName: String, context: Context): String {
        if (LayoutUtilsCustom.isCustomLayout(layoutName))
            LayoutUtilsCustom.getLayoutFiles(layoutType, context)
                .firstOrNull { it.name.startsWith(layoutName) }?.let { return it.readText() }
        return LayoutUtils.getContent(layoutType, layoutName, context)
    }

    // allow commenting lines by starting them with "//"
    private fun String.stripCommentLines(): String =
        split("\n").filterNot { it.startsWith("//") }.joinToString("\n")

    /*
     * Copyright (C) 2021 Patrick Goldinger
     * modified
     * SPDX-License-Identifier: Apache-2.0
     */
    @OptIn(ExperimentalSerializationApi::class)
    private val florisJsonConfig = Json {
        allowTrailingComma = true
        classDiscriminator = "$"
        encodeDefaults = true
        ignoreUnknownKeys = true
        isLenient = true
        serializersModule = SerializersModule {
            polymorphic(AbstractKeyData::class) {
                subclass(TextKeyData::class, TextKeyData.serializer())
                subclass(AutoTextKeyData::class, AutoTextKeyData.serializer())
                subclass(MultiTextKeyData::class, MultiTextKeyData.serializer())
                subclass(CaseSelector::class, CaseSelector.serializer())
                subclass(ShiftStateSelector::class, ShiftStateSelector.serializer())
                subclass(VariationSelector::class, VariationSelector.serializer())
                subclass(KeyboardStateSelector::class, KeyboardStateSelector.serializer())
                subclass(LayoutDirectionSelector::class, LayoutDirectionSelector.serializer())
                subclass(CharWidthSelector::class, CharWidthSelector.serializer())
                subclass(KanaSelector::class, KanaSelector.serializer())
                defaultDeserializer { TextKeyData.serializer() }
            }
            polymorphic(KeyData::class) {
                subclass(TextKeyData::class, TextKeyData.serializer())
                subclass(AutoTextKeyData::class, AutoTextKeyData.serializer())
                subclass(MultiTextKeyData::class, MultiTextKeyData.serializer())
                defaultDeserializer { TextKeyData.serializer() }
            }
        }
    }

    // copy of florisJsonConfig, but with ignoreUnknownKeys = false so users get warned
    // this is not default because users may have old layouts that should not stop working on app upgrade
    @OptIn(ExperimentalSerializationApi::class)
    private val checkJsonConfig = Json {
        allowTrailingComma = true
        classDiscriminator = "$"
        encodeDefaults = true
        ignoreUnknownKeys = false
        isLenient = true
        serializersModule = SerializersModule {
            polymorphic(AbstractKeyData::class) {
                subclass(TextKeyData::class, TextKeyData.serializer())
                subclass(AutoTextKeyData::class, AutoTextKeyData.serializer())
                subclass(MultiTextKeyData::class, MultiTextKeyData.serializer())
                subclass(CaseSelector::class, CaseSelector.serializer())
                subclass(ShiftStateSelector::class, ShiftStateSelector.serializer())
                subclass(VariationSelector::class, VariationSelector.serializer())
                subclass(KeyboardStateSelector::class, KeyboardStateSelector.serializer())
                subclass(LayoutDirectionSelector::class, LayoutDirectionSelector.serializer())
                subclass(CharWidthSelector::class, CharWidthSelector.serializer())
                subclass(KanaSelector::class, KanaSelector.serializer())
                defaultDeserializer { TextKeyData.serializer() }
            }
            polymorphic(KeyData::class) {
                subclass(TextKeyData::class, TextKeyData.serializer())
                subclass(AutoTextKeyData::class, AutoTextKeyData.serializer())
                subclass(MultiTextKeyData::class, MultiTextKeyData.serializer())
                defaultDeserializer { TextKeyData.serializer() }
            }
        }
    }
}
