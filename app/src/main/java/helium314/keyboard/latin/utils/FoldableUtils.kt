// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.utils

import android.content.Context
import android.content.pm.PackageManager
import android.database.ContentObserver
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import helium314.keyboard.latin.define.DebugFlags
import java.util.regex.Pattern
import kotlin.text.split

object FoldableUtils {
    private const val TAG = "FoldableUtils"

    var isFoldable = false
        private set

    var isFolded = false
        private set(value) {
            if (field == value) return
            // we could reload the keyboard at this point, but according to a user this is not necessary
            // https://github.com/HeliBorg/HeliBoard/issues/1063#issuecomment-4178571414
            Log.v(TAG, "set isFolded to $value")
            field = value
        }

    /** set [isFoldable] */
    fun init(context: Context) {
        isFoldable = getFeatureString(context) != null || hasFoldSensor(context)
        Log.i(TAG, if (isFoldable) "foldable" else "not foldable")
    }

    private fun hasFoldSensor(context: Context): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
                && context.packageManager.hasSystemFeature(PackageManager.FEATURE_SENSOR_HINGE_ANGLE))
            return true
        if (DebugFlags.DEBUG_ENABLED) {
            val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            sm.getSensorList(Sensor.TYPE_ALL).forEach {
                if (it.name.contains("hinge", true) || it.name.contains("fold", true))
                    Log.v(TAG, "no default hinge sensor, but found ${it.name} with range ${it.maximumRange}")
            }
        }
        return false
    }

    /*
     * much of the code related to display_features is modified from https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/libs/WindowManager/Jetpack/src/androidx/window
     * apparently there is some information encoded in undocumented "display_features" setting in Settings.Global
     * found values
     *  null (we assume this means not foldable, and otherwise the device is foldable -> would need more testing)
     *  empty (when folded, at least according to the user who posted the logs)
     *   ca 40° hinge both directions
     *  fold-[1124,0,1124,2480]-half-opened -> AFTER configuration change (regex no match, but why?)
     *   at ca 40° hinge when opening, 140° when closing
     *  fold-[1124,0,1124,2480]-flat -> no configuration change
     *   at ca 160° hinge
     */
    private const val DISPLAY_FEATURES = "display_features"
    private val displayFeaturesUri = Settings.Global.getUriFor(DISPLAY_FEATURES)
    private val FEATURE_PATTERN = Pattern.compile("([a-z]+)-\\[(\\d+),(\\d+),(\\d+),(\\d+)]-?(flat|half-opened)?")
    private const val FEATURE_TYPE_FOLD = "fold"
    private const val FEATURE_TYPE_HINGE = "hinge"
    private const val PATTERN_STATE_FLAT = "flat"
    private const val PATTERN_STATE_HALF_OPENED = "half-opened"

    fun getFeatureString(context: Context): String? = Settings.Global.getString(context.contentResolver, DISPLAY_FEATURES)

    private fun extractFoldedState(displayFeatures: String): Boolean {
        if (displayFeatures.isEmpty()) return true
        displayFeatures.split(";").forEach {
            try {
                val matcher = FEATURE_PATTERN.matcher(it)
                if (!matcher.matches()) return@forEach
                val featureType = matcher.group(1) // should be FEATURE_TYPE_FOLD or FEATURE_TYPE_HINGE
                val state = matcher.group(6)

                // do we have use for anything other than state? featureType might be useful for debugging
                if (DebugFlags.DEBUG_ENABLED)
                    Log.d(TAG, "found: type $featureType, state $state")
                return (state != PATTERN_STATE_FLAT && state != PATTERN_STATE_HALF_OPENED) // or go for FEATURE_TYPE_FOLD/HINGE?
            } catch (e: Exception) {
                Log.w(TAG, "error when checking $it", e)
            }
        }

        return false
    }

    /** Observes changes to [DISPLAY_FEATURES] or hinge angle, and updates [isFolded] on changes */
    class FoldableObserver(context: Context) {
        var sensorForDebug = false

        private val featureStringObserver = object : ContentObserver(Handler(Looper.getMainLooper())) {
            override fun onChange(selfChange: Boolean, uri: Uri?) {
                if (uri != displayFeaturesUri) return
                val featuresString = getFeatureString(context)
                if (featuresString == null) {
                    Log.w(TAG, "$DISPLAY_FEATURES are unexpectedly null")
                    return
                }
                if (DebugFlags.DEBUG_ENABLED)
                    Log.v(TAG, "$DISPLAY_FEATURES changed: $featuresString")
                isFolded = extractFoldedState(featuresString)
            }
        }

        private val sensorListener = object : SensorEventListener {
            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
            override fun onSensorChanged(event: SensorEvent) {
                val angle = event.values?.getOrNull(0)
                // logs from a user showed that
                // * 40° is the change between folded and half-open
                // * 160° is the change between half-open and flat
                // maybe we should use the sensor range? wait for bug reports + logs
                if (!sensorForDebug)
                    isFolded = (angle ?: 180f) < 40
                if (DebugFlags.DEBUG_ENABLED)
                    Log.v(TAG, "sensor changed: ${event.values?.toList()}")
            }
        }

        init {
            // is one of the methods clearly better? wait for bug reports + logs
            val featureString = getFeatureString(context)
            if (featureString != null) {
                context.contentResolver.registerContentObserver(displayFeaturesUri, false, featureStringObserver)
                isFolded = extractFoldedState(featureString)
                Log.v(TAG, "using $DISPLAY_FEATURES, folded: $isFolded")
            }
            if (hasFoldSensor(context) && (featureString == null || DebugFlags.DEBUG_ENABLED)) {
                sensorForDebug = featureString != null
                // see https://github.com/ryosoftware/folds/blob/master/app/src/main/java/com/ryosoftware/unfolds/UnfoldsCounterService.kt#L67-L83
                // -> we could try other sensors
                val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
                sm.registerListener(sensorListener, sm.getDefaultSensor(Sensor.TYPE_HINGE_ANGLE), SensorManager.SENSOR_DELAY_UI)
                Log.v(TAG, "using sensor, for debugging only: $sensorForDebug")
            }
        }

        fun unregister(context: Context) {
            context.contentResolver.unregisterContentObserver(featureStringObserver)
            val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            sm.unregisterListener(sensorListener)
        }
    }
}
