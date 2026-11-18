// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.common

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import androidx.core.graphics.ColorUtils
import helium314.keyboard.keyboard.KeyboardTheme.Companion.STYLE_HOLO
import helium314.keyboard.keyboard.KeyboardTheme.Companion.STYLE_MATERIAL
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.EnumMap
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Guards the popup selection highlight across every colour implementation.
 *
 * #2440 reported the highlight as barely visible. A reviewer of the first fix saw "no difference"
 * because that fix only touched two of the three colour classes — user colour themes (AllColors)
 * kept their own washed-out path. These tests pin the rule for all of them: releasing a finger on
 * a popup key must produce a visible change, comparable to how the functional (delete) key stands
 * out from a letter key.
 */
@RunWith(RobolectricTestRunner::class)
class PopupSelectionColorTest {

    private fun opaque(c: Int) = c or 0xFF000000.toInt()

    /** records the tint list the colour implementation applies, which is where the states live */
    private class CapturingDrawable : GradientDrawable() {
        var captured: android.content.res.ColorStateList? = null
        override fun setTintList(tint: android.content.res.ColorStateList?) {
            captured = tint
            super.setTintList(tint)
        }
    }

    private fun pressedAndNormal(colors: Colors, type: ColorType): Pair<Int, Int> {
        val drawable = CapturingDrawable()
        colors.setColor(drawable, type)
        val tint = drawable.captured
        assertTrue(tint != null, "$type must receive a tint list")
        // read both states explicitly: defaultColor is not reliable here, it can report the first
        // entry of the list rather than the one matching the empty state
        val normal: Int = tint!!.getColorForState(IntArray(0), 0)
        val pressed: Int = tint.getColorForState(intArrayOf(android.R.attr.state_pressed), 0)
        return Pair(pressed, normal)
    }

    private fun check(name: String, colors: Colors) {
        val (pressed, normal) = pressedAndNormal(colors, ColorType.POPUP_KEYS_BACKGROUND)
        val contrast = ColorUtils.calculateContrast(opaque(pressed), opaque(normal))
        println("$name: pressed=#%06X normal=#%06X contrast=%.3f".format(pressed and 0xFFFFFF, normal and 0xFFFFFF, contrast))
        assertTrue(pressed != normal, "$name: the selected popup key must differ from the popup background")
        // 1.30 is above what deriving the highlight from the popup background alone reaches:
        // measured 1.157 (material/light) and 1.211..1.220 (dark) before this change
        assertTrue(contrast >= 1.30, "$name: selection barely visible, contrast %.3f".format(contrast))
    }

    private fun defaultColors(style: String, borders: Boolean, accent: Int, background: Int,
                              keyBackground: Int, functionalKey: Int, keyText: Int) =
        DefaultColors(
            themeStyle = style, hasKeyBorders = borders, accent = accent, background = background,
            keyBackground = keyBackground, functionalKey = functionalKey, spaceBar = keyBackground,
            keyText = keyText, keyHintText = keyText
        )

    @Test fun holoDark() = check("holo/dark", defaultColors(STYLE_HOLO, true,
        Color.rgb(0x1b, 0xa8, 0xd0), Color.rgb(0x0f, 0x0f, 0x0f),
        Color.rgb(0x2b, 0x2b, 0x2b), Color.rgb(0x40, 0x40, 0x40), Color.WHITE))

    @Test fun materialLight() = check("material/light", defaultColors(STYLE_MATERIAL, true,
        Color.rgb(0x21, 0x96, 0xf3), Color.rgb(0xee, 0xee, 0xee),
        Color.WHITE, Color.rgb(0xd6, 0xd6, 0xd6), Color.BLACK))

    @Test fun materialDark() = check("material/dark", defaultColors(STYLE_MATERIAL, true,
        Color.rgb(0x21, 0x96, 0xf3), Color.rgb(0x12, 0x12, 0x12),
        Color.rgb(0x26, 0x26, 0x26), Color.rgb(0x3a, 0x3a, 0x3a), Color.WHITE))

    @Test fun materialLightNoBorders() = check("material/light/noborders", defaultColors(STYLE_MATERIAL, false,
        Color.rgb(0x21, 0x96, 0xf3), Color.rgb(0xee, 0xee, 0xee),
        Color.WHITE, Color.rgb(0xd6, 0xd6, 0xd6), Color.BLACK))

    @Test
    fun userColorTheme() {
        // this is the path the reviewer of the earlier fix was on, where nothing changed at all
        val map = EnumMap<ColorType, Int>(ColorType::class.java)
        map[ColorType.KEY_BACKGROUND] = Color.WHITE
        map[ColorType.FUNCTIONAL_KEY_BACKGROUND] = Color.rgb(0xd0, 0xd0, 0xd0)
        map[ColorType.POPUP_KEYS_BACKGROUND] = Color.rgb(0xfa, 0xfa, 0xfa)
        map[ColorType.ACTION_KEY_BACKGROUND] = Color.rgb(0x21, 0x96, 0xf3)
        check("user colours", AllColors(map, STYLE_MATERIAL, true, null))
    }
}
