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
import java.util.regex.Pattern
import kotlin.text.split

object FoldableUtils {
    private const val TAG = "FoldableUtils"

    var isFoldable = false
        private set

    var isFolded = false
        private set

    fun init(context: Context) {
        isFoldable = hasDisplayFeatureString(context) || hasFoldSensor(context)
        Log.i(TAG, "isFoldable: $isFoldable")
    }

    class FoldableObserver(context: Context) {
        private val featureStringObserver = object : ContentObserver(Handler(Looper.getMainLooper())) {
            override fun onChange(selfChange: Boolean, uri: Uri?) {
                if (uri != displayFeaturesUri) return
                val featuresString = getFeatureString(context)
                if (featuresString == null) {
                    Log.w(TAG, "$DISPLAY_FEATURES should not be null")
                    return
                }
                isFolded = extractFoldedState(featuresString)
                Log.i(TAG, "$DISPLAY_FEATURES changed: $featuresString, setting to $isFolded")
            }
        }

        private val sensorListener = object : SensorEventListener {
            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
            override fun onSensorChanged(event: SensorEvent) {
                val angle = event.values?.getOrNull(0)
                isFolded == (angle ?: 180f) < 90
                Log.i(TAG, "sensor changed: ${event.values?.toList()}, setting to $isFolded")
            }
        }

        init {
            // which method is better?
            val featureString = getFeatureString(context)
            if (featureString != null) {
                context.contentResolver.registerContentObserver(displayFeaturesUri, false, featureStringObserver)
                isFolded = extractFoldedState(featureString)
                Log.v(TAG, "using $DISPLAY_FEATURES, folded: $isFolded")
            } else if (hasFoldSensor(context)) {
                // see https://github.com/ryosoftware/folds/blob/master/app/src/main/java/com/ryosoftware/unfolds/UnfoldsCounterService.kt#L67-L83
                // -> we could try other sensors
                val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
                sm.registerListener(sensorListener, sm.getDefaultSensor(Sensor.TYPE_HINGE_ANGLE), SensorManager.SENSOR_DELAY_UI)
                Log.v(TAG, "using sensor")
            }
        }

        fun unregister(context: Context) {
            context.contentResolver.unregisterContentObserver(featureStringObserver)
            val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            sm.unregisterListener(sensorListener)
        }
    }

    private fun hasFoldSensor(context: Context): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
                && context.packageManager.hasSystemFeature(PackageManager.FEATURE_SENSOR_HINGE_ANGLE))
            return true
        // maybe we have a differently named sensor, later try https://github.com/ryosoftware/folds/blob/a9c974046298a94e23733bb57d5d42aeaff424b2/app/src/main/java/com/ryosoftware/unfolds/UnfoldsCounterService.kt#L63-L83
        return false
    }

    // using code from https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/libs/WindowManager/Jetpack/src/androidx/window
    // apparently there is some information encoded in undocumented "display_features" setting in Settings.Global that requires a whole library to parse?
    private const val DISPLAY_FEATURES = "display_features"
    private val FEATURE_PATTERN = Pattern.compile("([a-z]+)-\\[(\\d+),(\\d+),(\\d+),(\\d+)]-?(flat|half-opened)?")
    private val FEATURE_TYPE_FOLD = "fold"
    private val FEATURE_TYPE_HINGE = "hinge"
    private val PATTERN_STATE_FLAT = "flat"
    private val PATTERN_STATE_HALF_OPENED = "half-opened"

    // not sure if this is correct
    private fun hasDisplayFeatureString(context: Context) = getFeatureString(context) != null

    private val displayFeaturesUri = Settings.Global.getUriFor(DISPLAY_FEATURES)

    fun getFeatureString(context: Context): String? = Settings.Global.getString(context.contentResolver, DISPLAY_FEATURES)

    // found values
    //  null (not foldable?)
    //  empty (when folded it seems)
    //   ca 40° hinge both directions
    //  fold-[1124,0,1124,2480]-half-opened -> AFTER configuration change (regex no match, but why?)
    //   ca 40° hinge when opening, 140° when closing
    //  fold-[1124,0,1124,2480]-flat -> no configuration change
    //   ca 160° hinge
    private fun extractFoldedState(displayFeatures: String): Boolean {
        if (displayFeatures.isEmpty()) return false
        displayFeatures.split(";").forEach {
            try {
                val matcher = FEATURE_PATTERN.matcher(it)
                if (!matcher.matches()) return@forEach
                val featureType = matcher.group(1) // should be FEATURE_TYPE_FOLD or FEATURE_TYPE_HINGE
                // screen dimensions? or what is it?
                val left = matcher.group(2)
                val top = matcher.group(3)
                val right = matcher.group(4)
                val bottom = matcher.group(5)
                val state = matcher.group(6)

                // todo: do we have use for anything other than state?
                Log.d(TAG, "found: type $featureType, state $state, featureRect $left, $right, $top, $bottom")
                return (state != PATTERN_STATE_FLAT && state != PATTERN_STATE_HALF_OPENED)
            } catch (e: Exception) {
                Log.w(TAG, "error when checking $it", e)
            }
        }

        return false
    }
}
