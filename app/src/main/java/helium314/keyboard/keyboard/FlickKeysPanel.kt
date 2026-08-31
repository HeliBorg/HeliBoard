// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.keyboard

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import android.view.ViewGroup
import helium314.keyboard.event.HapticEvent
import helium314.keyboard.keyboard.emoji.EmojiViewCallback
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.common.Constants
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max

/**
 * 12-key flick input
 * When flick key is pressed it shows this panel immediately
 * See [PointerTracker]'s down-event handling of [isFlickKey] keys
 *
 * two distinct behaviors:
 * 1 Hold key for [HOLD_TO_EXPAND_MS] to show kana panel. Dim rest of keyboard
 * 2 Flicking before that timer expires shows selected key briefly
 *   near the flick in the direction flicked;
 *
 * all animations are relative to the center of the key pressed
 */
class FlickKeysPanel(context: Context) : View(context), PopupKeysPanel {
    private var controller: PopupKeysPanel.Controller = PopupKeysPanel.EMPTY_CONTROLLER
    private var listener: KeyboardActionListener? = null

    // key true center -> where bubble is drawn
    private var keyLocalX = 0f
    private var keyLocalY = 0f
    // where finger actually went down
    private var downX = 0f
    private var downY = 0f
    // pressed key's own bounds, so dimming overlay can leave it visible
    private var keyLeft = 0f
    private var keyTop = 0f
    private var keyRight = 0f
    private var keyBottom = 0f

    private var centerCode = KeyCode.UNSPECIFIED
    private var centerLabel = ""
    // up, down, left, right; matches [japanese_flick.json] popup order convention
    private val directionCodes = IntArray(4) { KeyCode.UNSPECIFIED }
    private val directionLabels = Array(4) { "" }
    private var selected = -1 // -1 = center (no flick), 0..3 = up/left/right/down

    // true once the hold timer has fired
    private var expanded = false
    // true once the user has flicked away from the key before [expanded] became true
    private var flicked = false

    private val expandRunnable = Runnable {
        if (!flicked) {
            expanded = true
            invalidate()
        }
    }

    private val density = resources.displayMetrics.density
    private val guideSize = density * GUIDE_SIZE_DP
    private val deadZone = density * DEAD_ZONE_DP
    private val labelOffset = density * LABEL_OFFSET_DP
    // row-1 keys sit too close to the keyboard's top edge (suggestion box) for the bubble
    // to draw correctly; This extends this above the keyboard
    private val topExtension = guideSize / 2f

    private val scrimPaint = Paint().apply { color = 0xB3000000.toInt() }
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xEE2B2B2B.toInt() }
    private val highlightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF5B8DEF.toInt() }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        textSize = density * LABEL_TEXT_SP
    }
    private val centerLabelPaint = Paint(labelPaint).apply {
        color = 0xFFAAAAAA.toInt()
        textSize = density * CENTER_TEXT_SP
    }

    fun setKey(key: Key) {
        centerCode = key.code
        centerLabel = key.label.orEmpty()
        keyLeft = key.getX().toFloat()
        keyTop = key.getY().toFloat() + topExtension
        keyRight = keyLeft + key.getWidth()
        keyBottom = keyTop + key.getHeight()
        val popupKeys = key.popupKeys
        val offset = max(0, (popupKeys?.size ?: 0) - 4)
        for (i in 0 until 4) {
            val spec = popupKeys?.getOrNull(offset + i)
            directionCodes[i] = spec?.mCode ?: KeyCode.UNSPECIFIED
            directionLabels[i] = spec?.mLabel.orEmpty()
        }
    }

    override fun showPopupKeysPanel(
        parentView: View, controller: PopupKeysPanel.Controller, pointX: Int, pointY: Int,
        listener: KeyboardActionListener
    ) {
        this.controller = controller
        this.listener = listener
        selected = -1
        expanded = false
        flicked = false
        layoutParams = ViewGroup.LayoutParams(parentView.width, parentView.height + topExtension.toInt())
        keyLocalX = pointX.toFloat()
        keyLocalY = pointY.toFloat() + topExtension
        // fallback in case [onDownEvent] is ever not called before the first onMoveEvent
        downX = keyLocalX
        downY = keyLocalY
        controller.onShowPopupKeysPanel(this)
        val origin = IntArray(2)
        parentView.getLocationInWindow(origin)
        x = origin[0].toFloat()
        y = origin[1].toFloat() - topExtension
        postDelayed(expandRunnable, HOLD_TO_EXPAND_MS)
        invalidate()
    }

    // flick keys never show an emoji popup panel
    override fun showPopupKeysPanel(
        parentView: View, controller: PopupKeysPanel.Controller, pointX: Int, pointY: Int,
        emojiViewCallback: EmojiViewCallback
    ) {}

    override fun dismissPopupKeysPanel() {
        removeCallbacks(expandRunnable)
        controller.onDismissPopupKeysPanel()
    }

    override fun onDownEvent(x: Int, y: Int, pointerId: Int, eventTime: Long) {
        downX = x.toFloat()
        downY = y.toFloat()
        selected = -1
        invalidate()
    }

    override fun onMoveEvent(x: Int, y: Int, pointerId: Int, eventTime: Long) {
        val dx = x - downX
        val dy = y - downY
        val distance = hypot(dx, dy)
        val newSelected = if (distance < deadZone) -1 else {
            val degrees = Math.toDegrees(atan2(-dy, dx).toDouble())
            when {
                degrees in -45.0..45.0 -> DIR_RIGHT
                degrees in 45.0..135.0 -> DIR_UP
                degrees in -135.0..-45.0 -> DIR_DOWN
                else -> DIR_LEFT
            }
        }
        if (!expanded && !flicked && newSelected != -1) {
            // the user flicked away before the hold timer fired: show bubble
            flicked = true
            removeCallbacks(expandRunnable)
        }
        selected = newSelected
        invalidate()
    }

    override fun onUpEvent(x: Int, y: Int, pointerId: Int, eventTime: Long) {
        val code = if (selected in 0..3) directionCodes[selected] else centerCode
        val l = listener ?: return
        if (code == KeyCode.UNSPECIFIED) return
        l.onPressKey(code, 0, 1, HapticEvent.NO_HAPTICS)
        l.onCodeInput(code, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE, false)
        l.onReleaseKey(code, false)
    }

    override fun translateX(x: Int): Int = x
    override fun translateY(y: Int): Int = y + topExtension.toInt()

    override fun getContainerView(): View = this

    override fun onDraw(canvas: Canvas) {
        val cx = anchorX()
        val cy = anchorY()
        if (expanded) {
            drawScrim(canvas)
            val half = guideSize / 2f
            val bg = RectF(cx - half, cy - half, cx + half, cy + half)
            canvas.drawRoundRect(bg, density * 12f, density * 12f, bgPaint)
            drawDirection(canvas, DIR_UP, cx, cy - labelOffset)
            drawDirection(canvas, DIR_DOWN, cx, cy + labelOffset)
            drawDirection(canvas, DIR_LEFT, cx - labelOffset, cy)
            drawDirection(canvas, DIR_RIGHT, cx + labelOffset, cy)
            if (selected == -1 && centerLabel.isNotEmpty()) {
                canvas.drawText(centerLabel, cx, cy - (centerLabelPaint.ascent() + centerLabelPaint.descent()) / 2, centerLabelPaint)
            }
        } else if (flicked && selected != -1) {
            val bx: Float; val by: Float
            when (selected) {
                DIR_UP -> { bx = cx; by = cy - labelOffset }
                DIR_DOWN -> { bx = cx; by = cy + labelOffset }
                DIR_LEFT -> { bx = cx - labelOffset; by = cy }
                else -> { bx = cx + labelOffset; by = cy }
            }
            drawDirection(canvas, selected, bx, by)
        }
        // else: do nothing as still waiting to see whether this is a hold/flick
    }

    /**
     * Dim everything except the pressed key's own bounds, which stays visible under the guide.
     */
    private fun drawScrim(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.drawRect(0f, topExtension, w, keyTop, scrimPaint)
        canvas.drawRect(0f, keyBottom, w, h, scrimPaint)
        canvas.drawRect(0f, keyTop, keyLeft, keyBottom, scrimPaint)
        canvas.drawRect(keyRight, keyTop, w, keyBottom, scrimPaint)
    }

    // keeps the guide/bubble from being drawn off the edge of the panel
    private fun anchorX(): Float {
        val margin = guideSize / 2f
        return keyLocalX.coerceIn(margin, max(margin, width - margin))
    }

    private fun anchorY(): Float {
        val margin = guideSize / 2f
        return keyLocalY.coerceIn(margin, max(margin, height - margin))
    }

    private fun drawDirection(canvas: Canvas, dir: Int, x: Float, y: Float) {
        val label = directionLabels[dir]
        if (label.isEmpty()) return
        if (dir == selected) {
            canvas.drawCircle(x, y, density * HIGHLIGHT_RADIUS_DP, highlightPaint)
        }
        canvas.drawText(label, x, y - (labelPaint.ascent() + labelPaint.descent()) / 2, labelPaint)
    }

    companion object {
        private const val HOLD_TO_EXPAND_MS = 500L
        private const val GUIDE_SIZE_DP = 150f
        private const val DEAD_ZONE_DP = 22f
        private const val LABEL_OFFSET_DP = 46f
        private const val LABEL_TEXT_SP = 20f
        private const val CENTER_TEXT_SP = 14f
        private const val HIGHLIGHT_RADIUS_DP = 20f

        const val DIR_UP = 0
        const val DIR_LEFT = 1
        const val DIR_RIGHT = 2
        const val DIR_DOWN = 3

        /** whether [key] should show its popup keys instantly (flick) rather than after a long press */
        @JvmStatic
        fun isFlickKey(key: Key): Boolean {
            val popupKeys = key.popupKeys
            return popupKeys != null && popupKeys.isNotEmpty() &&
                    RichInputMethodManager.getInstance().combiningRulesExtraValueOfCurrentSubtype == "kana_flick"
        }
    }
}
