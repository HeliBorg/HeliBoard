// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.voice

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.inputmethod.ExtractedTextRequest
import android.view.inputmethod.InputConnection
import androidx.core.content.ContextCompat
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.voice.audio.AudioRecorder
import helium314.keyboard.voice.net.EditClient
import helium314.keyboard.voice.net.RealtimeTranscriptionClient
import helium314.keyboard.voice.net.TranscriptionService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

object VoiceController {

    private const val TAG = "VoxBoard"

    enum class State { IDLE, RECORDING, TRANSCRIBING, EDITING, ERROR }

    @Volatile
    var state: State = State.IDLE
        private set

    var onStateChanged: ((State, Boolean) -> Unit)? = null

    private val main = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(Dispatchers.Main)
    private var recorder: AudioRecorder? = null
    private var editTarget: String? = null
    private var editWholeText = false
    private var activeIsEdit = false
    private var rtSession: RealtimeTranscriptionClient.Session? = null
    private var liveStreaming = false
    private var liveInsertedAny = false
    private var liveFailed = false

    private fun setState(newState: State) {
        state = newState
        main.post { onStateChanged?.invoke(newState, activeIsEdit) }
    }

    private fun toast(message: String) = KeyboardSwitcher.getInstance().showToast(message, true)

    @JvmStatic
    fun toggle(ime: LatinIME) = handleTap(ime, editWholeTextIfNoSelection = false)

    @JvmStatic
    fun toggleEdit(ime: LatinIME) = handleTap(ime, editWholeTextIfNoSelection = true)

    private fun handleTap(ime: LatinIME, editWholeTextIfNoSelection: Boolean) {
        when (state) {
            State.RECORDING -> finishRecording(ime)
            State.IDLE, State.ERROR -> {
                activeIsEdit = editWholeTextIfNoSelection
                beginRecording(ime, editWholeTextIfNoSelection)
            }
            else -> {}
        }
    }

    private fun beginRecording(ime: LatinIME, editWholeTextIfNoSelection: Boolean) {
        if (ContextCompat.checkSelfPermission(ime, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            toast("VoxBoard needs microphone permission")
            openAppSettings(ime)
            return
        }
        val ic = ime.currentInputConnection
        val selection = ic?.getSelectedText(0)?.toString().orEmpty()
        when {
            selection.isNotEmpty() -> { editTarget = selection; editWholeText = false }
            editWholeTextIfNoSelection -> {
                editTarget = ic?.getExtractedText(ExtractedTextRequest(), 0)?.text?.toString().orEmpty()
                editWholeText = true
            }
            else -> { editTarget = null; editWholeText = false }
        }
        val config = VoiceConfig.from(ime)
        if (config.streaming && editTarget == null && config.transcriptionBackends.isNotEmpty()) {
            startLiveRecording(ime, config)
            return
        }
        recorder = AudioRecorder(config.recordMaxSeconds).also { it.start() }
        setState(State.RECORDING)
        Log.i(TAG, "recording started (editMode=${editTarget != null}, wholeText=$editWholeText)")
        toast(if (editTarget != null) "VoxBoard: editing" else "VoxBoard: listening")
    }

    private fun startLiveRecording(ime: LatinIME, config: VoiceConfig) {
        liveStreaming = true
        liveInsertedAny = false
        liveFailed = false
        val backend = config.transcriptionBackends.first()
        val client = RealtimeTranscriptionClient(config.requestTimeoutSeconds, config.caCertPem)
        rtSession = client.open(backend, object : RealtimeTranscriptionClient.Listener {
            override fun onSegment(text: String, final: Boolean) {
                main.post {
                    val ic = ime.currentInputConnection ?: return@post
                    if (final) {
                        ic.finishComposingText()
                        ic.commitText("$text ", 1)
                        liveInsertedAny = true
                    } else {
                        ic.setComposingText(text, 1)
                    }
                }
            }

            override fun onClosed(cause: Throwable?) {
                if (cause != null) {
                    liveFailed = true
                    Log.w(TAG, "realtime ws closed with error", cause)
                }
            }
        })
        recorder = AudioRecorder(config.recordMaxSeconds) { chunk -> rtSession?.appendAudio(chunk) }.also { it.start() }
        setState(State.RECORDING)
        Log.i(TAG, "live recording started")
        toast("VoxBoard: listening (live)")
    }

    private fun finishRecording(ime: LatinIME) {
        if (liveStreaming) {
            finishLiveRecording(ime)
            return
        }
        val active = recorder ?: run { setState(State.IDLE); return }
        recorder = null
        val config = VoiceConfig.from(ime)
        val target = editTarget
        val whole = editWholeText
        editTarget = null
        editWholeText = false
        setState(if (target != null) State.EDITING else State.TRANSCRIBING)
        Log.i(TAG, "recording stopped, transcribing (editMode=${target != null})")
        scope.launch {
            runCatching {
                val wav = withContext(Dispatchers.IO) { active.stop() }
                Log.i(TAG, "captured wav bytes=${wav.size}")
                if (target != null) {
                    val instruction = withContext(Dispatchers.IO) {
                        TranscriptionService(config).transcribe(wav, false) {}
                    }
                    withContext(Dispatchers.IO) {
                        EditClient(config.editBackend, config.requestTimeoutSeconds, config.caCertPem).edit(target, instruction)
                    }
                } else {
                    withContext(Dispatchers.IO) {
                        TranscriptionService(config).transcribe(wav, config.streaming) { partial ->
                            main.post { ime.currentInputConnection?.setComposingText(partial, 1) }
                        }
                    }
                }
            }.onSuccess { result ->
                Log.i(TAG, "success, result length=${result.length}: \"$result\"")
                val ic = ime.currentInputConnection
                if (target != null) {
                    if (result.isNotEmpty()) {
                        if (whole) ic?.performContextMenuAction(android.R.id.selectAll)
                        ic?.commitText(result, 1)
                    }
                } else {
                    ic?.commitText(if (result.isNotEmpty()) "$result " else "", 1)
                }
                setState(State.IDLE)
            }.onFailure { error ->
                Log.e(TAG, "dictation failed", error)
                ime.currentInputConnection?.finishComposingText()
                setState(State.ERROR)
                toast("VoxBoard error: ${error.message}")
            }
        }
    }

    private fun finishLiveRecording(ime: LatinIME) {
        val active = recorder
        recorder = null
        val session = rtSession
        rtSession = null
        val config = VoiceConfig.from(ime)
        setState(State.TRANSCRIBING)
        Log.i(TAG, "live recording stopped, finalizing")
        scope.launch {
            runCatching {
                val wav = withContext(Dispatchers.IO) { active?.stop() ?: ByteArray(0) }
                session?.commit()
                delay(1500)
                session?.close()
                if (liveFailed && !liveInsertedAny) {
                    Log.w(TAG, "live transcription failed, falling back to batch")
                    val result = withContext(Dispatchers.IO) {
                        TranscriptionService(config).transcribe(wav, false) {}
                    }
                    if (result.isNotEmpty()) ime.currentInputConnection?.commitText("$result ", 1)
                }
            }.onFailure { error ->
                Log.e(TAG, "live finalize failed", error)
                toast("VoxBoard error: ${error.message}")
            }
            ime.currentInputConnection?.finishComposingText()
            liveStreaming = false
            liveInsertedAny = false
            liveFailed = false
            setState(State.IDLE)
            Log.i(TAG, "live done")
        }
    }

    private fun openAppSettings(context: Context) {
        val intent = Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.fromParts("package", context.packageName, null)
        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
    }
}
