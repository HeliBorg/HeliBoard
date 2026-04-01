package helium314.keyboard.latin.utils

import android.R
import android.annotation.SuppressLint
import android.content.Context
import android.content.ContextWrapper
import android.content.SharedPreferences
import android.content.res.Resources
import android.inputmethodservice.InputMethodService
import android.os.Build
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.fromHtml
import androidx.compose.ui.text.withLink
import androidx.core.util.TypedValueCompat

// generic extension functions

// adapted from Kotlin source: https://github.com/JetBrains/kotlin/blob/7a7d392b3470b38d42f80c896b7270678d0f95c3/libraries/stdlib/common/src/generated/_Collections.kt#L3004
inline fun <T> Iterable<T>.sumOf(selector: (T) -> Float): Float {
    var sum = 0f
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

@SuppressLint("DiscouragedApi")
fun CharSequence.getStringResourceOrName(prefix: String, context: Context): String {
    val resId = context.resources.getIdentifier(prefix + this, "string", context.packageName)
    return if (resId == 0) this.toString() else context.getString(resId)
}

/**
 *  Splits the collection into a pair of lists on the first match of [condition], discarding the element first matching the condition.
 *  If [condition] is not met, all elements are in the first list.
 */
fun <T> Collection<T>.splitAt(condition: (T) -> Boolean): Pair<MutableList<T>, MutableList<T>> {
    var conditionMet = false
    val first = mutableListOf<T>()
    val second = mutableListOf<T>()
    forEach {
        if (conditionMet) {
            second.add(it)
        } else {
            conditionMet = condition(it)
            if (!conditionMet)
                first.add(it)
        }
    }
    return first to second
}

// like plus, but for nullable collections
fun <T> addCollections(a: Collection<T>?, b: Collection<T>?): Collection<T>? {
    if (a.isNullOrEmpty()) return b
    if (b.isNullOrEmpty()) return a
    return a + b
}

fun <T> MutableList<T>.removeFirst(predicate: (T) -> Boolean) {
    val i = indexOfFirst(predicate)
    if (i >= 0) removeAt(i)
}

fun <T> MutableList<T>.replaceFirst(predicate: (T) -> Boolean, with: (T) -> T) {
    val i = indexOfFirst(predicate)
    if (i >= 0) this[i] = with(this[i])
}

fun Context.getActivity(): ComponentActivity? {
    val componentActivity = when (this) {
        is ComponentActivity -> this
        is ContextWrapper -> baseContext.getActivity()
        else -> null
    }
    return componentActivity
}

/** SharedPreferences from deviceProtectedContext, which are accessible even without unlocking.
 *  They should not be used to store sensitive data! */
fun Context.prefs(): SharedPreferences = DeviceProtectedUtils.getSharedPreferences(this)

/** The "default" preferences that are only accessible after the device has been unlocked. */
fun Context.protectedPrefs(): SharedPreferences = getSharedPreferences("${packageName}_preferences", Context.MODE_PRIVATE)

/**
 * Returns the context object whose resources are adjusted to match the metrics of the display.
 *
 * Note that before [Build.VERSION_CODES.KITKAT], there is no way to support
 * multi-display scenarios, so the context object will just return the IME context itself.
 *
 * With initiating multi-display APIs from [Build.VERSION_CODES.KITKAT], the
 * context object has to return with re-creating the display context according the metrics
 * of the display in runtime.
 *
 * Starts from [Build.VERSION_CODES.S_V2], the returning context object has
 * became to IME context self since it ends up capable of updating its resources internally.
 */
@Suppress("deprecation")
fun Context.getDisplayContext(): Context {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S_V2) {
        // IME context sources is now managed by WindowProviderService from Android 12L.
        return this
    }
    // An issue in Q that non-activity components Resources / DisplayMetrics in
    // Context doesn't well updated when the IME window moving to external display.
    // Currently we do a workaround is to create new display context directly and re-init
    // keyboard layout with this context.
    val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
    return createDisplayContext(wm.defaultDisplay)
}

/** Override layout parameters to expand SoftInputWindow to the entire screen, See setInputView and SoftInputWindow.updateWidthHeight */
fun InputMethodService.updateSoftInputWindowLayoutParameters(inputView: View?) {
    val window = window.window ?: return
    ViewLayoutUtils.updateLayoutHeightOf(window, ViewGroup.LayoutParams.MATCH_PARENT)

    if (inputView == null) return
    // In non-fullscreen mode, InputView and its parent inputArea should expand to the entire screen
    // and be placed at the bottom of SoftInputWindow. In fullscreen mode, these shouldn't
    // expand to the entire screen and should be coexistent with mExtractedArea above.
    // See setInputView and com.android.internal.R.layout.input_method.xml.
    val layoutHeight = if (isFullscreenMode)
        ViewGroup.LayoutParams.WRAP_CONTENT
    else
        ViewGroup.LayoutParams.MATCH_PARENT
    val inputArea = window.findViewById<View?>(R.id.inputArea)
    ViewLayoutUtils.updateLayoutHeightOf(inputArea, layoutHeight)
    ViewLayoutUtils.updateLayoutGravityOf(inputArea, Gravity.BOTTOM)
    ViewLayoutUtils.updateLayoutHeightOf(inputView, layoutHeight)
}

@Composable
fun String.htmlToAnnotated() = AnnotatedString.fromHtml(this, TextLinkStyles(style = SpanStyle(color = MaterialTheme.colorScheme.primary)))

@SuppressLint("ComposableNaming") // same goes for built-in "append"
@Composable
fun AnnotatedString.Builder.appendLink(text: String, url: String) =
    withLink(
        LinkAnnotation.Url(
            url,
            styles = TextLinkStyles(style = SpanStyle(color = MaterialTheme.colorScheme.primary))
        )) {
        append(text)
    }

fun String.withHtmlLink(link: String) = "<a href='$link'>$this</a>"

/** Convenience for converting dp to px, int -> int */
fun Int.dpToPx(resources: Resources) = TypedValueCompat.dpToPx(this.toFloat(), resources.displayMetrics).toInt()
