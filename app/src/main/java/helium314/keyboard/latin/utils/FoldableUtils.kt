package helium314.keyboard.latin.utils

import android.annotation.SuppressLint
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


@SuppressLint("StaticFieldLeak")
object FoldableUtils {
    fun isFoldable(context: Context) = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
        && context.packageManager.hasSystemFeature(PackageManager.FEATURE_SENSOR_HINGE_ANGLE)

    val displayFeaturesUri = Settings.Global.getUriFor("display_features")

    lateinit var context: Context

    val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean, uri: Uri?) {
            if (uri != displayFeaturesUri) return
            val f = getFeatureString(context)
            Log.i("FOLD", "features changed: $f")
            checkIt(f)
        }
    }

    fun getFeatureString(context: Context): String? = Settings.Global.getString(context.contentResolver, "display_features")

    fun registerObserver(ctx: Context) {
        context = ctx
        checkIt(getFeatureString(context))
        context.contentResolver.registerContentObserver(displayFeaturesUri, false, observer)
    }

    fun checkIt(value: String?) {
        if (value == null) {
            Log.i("FOLD", "feature string is null")
            return
        }
        try {
            value.split(";").forEach {
                val matcher = FEATURE_PATTERN.matcher(it)
                val type = when (val featureType = matcher.group(1)) {
                    FEATURE_TYPE_FOLD -> "type fold (continuous screen)"
                    FEATURE_TYPE_HINGE -> "type hinge (gap)"
                    else -> "unknown type $featureType"
                }
                val left = matcher.group(2)
                val top = matcher.group(3)
                val right = matcher.group(4)
                val bottom = matcher.group(5)
                val state = when (val stateString = matcher.group(6)) {
                    PATTERN_STATE_FLAT -> "state flat"
                    PATTERN_STATE_HALF_OPENED -> "state half opened"
                    else -> "state other: $stateString"
                }
                Log.i("FOLD", "found: type $type, state $state, featureRect $left, $right, $top, $bottom")
            }
        } catch (e: Exception) {
            Log.i("FOLD", "error when checking $value", e)
        }
    }

    val FEATURE_PATTERN = Pattern.compile("([a-z]+)-\\[(\\d+),(\\d+),(\\d+),(\\d+)]-?(flat|half-opened)?")
    val FEATURE_TYPE_FOLD = "fold"
    val FEATURE_TYPE_HINGE = "hinge"
    val PATTERN_STATE_FLAT = "flat"
    val PATTERN_STATE_HALF_OPENED = "half-opened"


    // todo: needs to be tested, folded state detection seems to be pretty insane
    //  Apparently we can't read the sensor directly, that would be too simple.
    //  So we register a listener and unregister it on first event, and hope it triggers quickly0
    // could also try using code from https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/libs/WindowManager/Jetpack/src/androidx/window
    // apparently there is some information encoded in undocumented "display_features" setting in Settings.Global that requires a whole library to parse?
    // todo: consider alternative by using smallest width >= 600dp (i.e. is tablet)
    fun isFolded(context: Context): Boolean {
        if (!isFoldable(context)) {
            Log.i("FOLD", "no hinge sensor")
            return false
        }
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        var angle: Float? = null
        val listener = object : SensorEventListener {
            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
            override fun onSensorChanged(event: SensorEvent) {
                angle = event.values?.getOrNull(0)
                Log.i("FOLD", "sensor changed: ${event.values?.toList()}")
                sm.unregisterListener(this)
            }
        }
        val hingeSensor = sm.getDefaultSensor(Sensor.TYPE_HINGE_ANGLE)
        if (hingeSensor == null) Log.i("FOLD", "no hinge sensor")
        else Log.i("FOLD", "hinge sensor: ${hingeSensor.name}, mode: ${hingeSensor.reportingMode}")
        sm.registerListener(listener, sm.getDefaultSensor(Sensor.TYPE_HINGE_ANGLE), SensorManager.SENSOR_DELAY_FASTEST)
        var wait = 0
        while (angle == null && wait < 50) {
            Thread.sleep(0, 100000)
            wait++
        }
        Log.i("FOLD", "return $angle after $wait")
        return (angle ?: 180f) < 90
    }

    @JvmField
    var isFolded = false
}

val listener = object : SensorEventListener {
    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
    override fun onSensorChanged(event: SensorEvent) {
        val angle = event.values?.getOrNull(0)
        Log.i("FOLD", "sensor changed: ${event.values?.toList()}")
        FoldableUtils.isFolded == (angle ?: 180f) < 90
    }
}
