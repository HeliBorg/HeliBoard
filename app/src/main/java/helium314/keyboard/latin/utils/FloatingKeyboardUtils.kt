package helium314.keyboard.latin.utils

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.Resources
import android.graphics.Rect
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.core.content.edit
import androidx.core.view.isGone
import androidx.core.view.isVisible
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.R
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.setFloatingSize

// todo: add a frame around the keyboard (because other people care more about optics than I do)
object FloatingKeyboardUtils {
    private val TAG = this::class.java.simpleName
    private val windowFrame = Rect()
    private var extraHeight = 0f

    @JvmStatic
    fun setFloating(view: View?) {
        val lp = view?.layoutParams as? ViewGroup.MarginLayoutParams ?: return
        view.getWindowVisibleDisplayFrame(windowFrame)
        extraHeight = getSuggestionStripHeight(view.resources) + getFloatingHandleHeight(view.resources)
        val (x, y) = readPosition(
            view.context,
            windowFrame.right - windowFrame.left - Settings.getValues().mFloatingWidth,
            windowFrame.bottom - windowFrame.top - extraHeight.toInt() - Settings.getValues().mFloatingHeight
        )
        if (DebugFlags.DEBUG_ENABLED)
            Log.d(TAG, "place floating view at $x, $y, width ${Settings.getValues().mFloatingWidth}, height ${Settings.getValues().mFloatingHeight}")
        ViewLayoutUtils.placeViewAt(view, x, y, Settings.getValues().mFloatingWidth, ViewGroup.LayoutParams.WRAP_CONTENT)
        if (view.findViewById<View>(R.id.float_handle_container)?.isVisible == true)
            return
        view.findViewById<View>(R.id.float_handle_container)?.isVisible = true
        view.findViewById<ImageView>(R.id.drag_handle)?.setDragListener(view)
        view.findViewById<ImageView>(R.id.resize_handle)?.setResizeListener(lp)
    }

    @JvmStatic
    fun disableFloating(view: View?) {
        val lp = view?.layoutParams as? ViewGroup.MarginLayoutParams ?: return
        if (lp.width == ViewGroup.LayoutParams.MATCH_PARENT) return // not floating
        if (DebugFlags.DEBUG_ENABLED)
            Log.d(TAG, "disable floating view")
        lp.width = ViewGroup.LayoutParams.MATCH_PARENT
        lp.height = ViewGroup.LayoutParams.MATCH_PARENT
        lp.leftMargin = 0
        lp.topMargin = 0
        view.findViewById<View>(R.id.float_handle_container)?.isGone = true
    }

    @JvmStatic
    fun getFloatingHandleHeight(resources: Resources) = resources.getDimension(R.dimen.config_floating_handle_height)

    @JvmStatic
    fun readPosition(context: Context, maxX: Int, maxY: Int): Pair<Int, Int> {
        val width = context.resources.displayMetrics.widthPixels
        val x = context.prefs().getInt(Settings.PREF_FLOATING_POS_X_PREFIX + width, width / 2)
        val y = context.prefs().getInt(Settings.PREF_FLOATING_POS_Y_PREFIX + width, context.resources.displayMetrics.heightPixels / 2)
        if (x > maxX || y > maxY)
            savePosition(context, maxX, maxY)
        return x.coerceIn(0, maxX) to y.coerceIn(0, maxY)
    }

    private fun savePosition(context: Context, x: Int, y: Int) {
        val width = context.resources.displayMetrics.widthPixels
        context.prefs().edit {
            putInt(Settings.PREF_FLOATING_POS_X_PREFIX + width, x)
            putInt(Settings.PREF_FLOATING_POS_Y_PREFIX + width, y)
        }
    }

    private fun getSuggestionStripHeight(resources: Resources) =
        if (Settings.getValues().mToolbarMode == ToolbarMode.HIDDEN) 0
        else resources.getDimension(R.dimen.config_suggestions_strip_height).toInt()

    @SuppressLint("ClickableViewAccessibility")
    private fun View.setDragListener(view: View) {
        var startX = 0f
        var startY = 0f
        val lp = view.layoutParams as ViewGroup.MarginLayoutParams
        var positionX = lp.leftMargin.toFloat()
        var positionY = lp.topMargin.toFloat()
        setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    startX = event.rawX
                    startY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - startX
                    val dy = event.rawY - startY
                    startX = event.rawX
                    startY = event.rawY
                    val sv = Settings.getValues()
                    val availableWidth = windowFrame.right - windowFrame.left
                    val availableHeight = windowFrame.bottom - windowFrame.top
                    positionX = (positionX + dx).coerceIn(0f, (availableWidth - sv.mFloatingWidth).toFloat())
                    positionY = (positionY + dy).coerceIn(0f, availableHeight - extraHeight - sv.mFloatingHeight)
                    lp.leftMargin = positionX.toInt()
                    lp.topMargin = positionY.toInt()
                    savePosition(context, lp.leftMargin, lp.topMargin)
                    view.layoutParams = lp // to update immediately
                    true
                }
                else -> false
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun View.setResizeListener(lp: ViewGroup.MarginLayoutParams) {
        var startX = 0f
        var startY = 0f
        val scale = 3 / context.resources.displayMetrics.density
        setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    startX = event.rawX
                    startY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - startX
                    val dy = event.rawY - startY
                    startX = event.rawX
                    startY = event.rawY
                    val availableWidth = windowFrame.right - windowFrame.left
                    val availableHeight = windowFrame.bottom - windowFrame.top
                    val maxWidth = (availableWidth * 0.9f).toInt()
                    val maxHeight = (availableHeight * 0.9f).toInt()
                    // avoid setting window outside windowFrame, view behaves strange otherwise
                    val newWidth = (Settings.getValues().mFloatingWidth + dx / scale).toInt().coerceIn(150, maxWidth)
                        .coerceAtMost(availableWidth - lp.leftMargin)
                    val newHeight = (Settings.getValues().mFloatingHeight + dy / scale).toInt().coerceIn(100, maxHeight)
                        .coerceAtMost(availableHeight - extraHeight.toInt() - lp.topMargin)
                    setFloatingSize(context, newWidth, newHeight)
                    KeyboardSwitcher.getInstance().reloadKeyboard()
                    // updating window is done in updateFloating, called by reloadKeyboard
                    true
                }
                else -> false
            }
        }
    }
}
