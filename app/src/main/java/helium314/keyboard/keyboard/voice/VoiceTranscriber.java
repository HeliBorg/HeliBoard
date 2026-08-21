package helium314.keyboard.keyboard.voice;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;

import java.util.ArrayList;

public final class VoiceTranscriber {
    private static final String TAG = VoiceTranscriber.class.getSimpleName();
    private static final Intent INTENT = new Intent(
        RecognizerIntent.ACTION_RECOGNIZE_SPEECH
    ).putExtra(
        RecognizerIntent.EXTRA_LANGUAGE_MODEL,
        RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
    ).putExtra(
        // this is "best effort" at preventing the service from cutting the user
        // off. in practice, they can ignore this or set a hard upper limit on
        // the-timeout-that-shouldn't-be-there-to-begin-with. with the Google
        // service, this at least affords the user a fairer amount of time than
        // their IME does.
        RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS,
        Integer.MAX_VALUE
    );

    private final SpeechRecognizer mSpeech;
    private ListenState mListening = ListenState.NOT_LISTENING;

    public VoiceTranscriber(Context ctx) {
        ctx = ctx.getApplicationContext();
        // todo: user preference to use createOnDeviceSpeechRecognizer(). note
        //  that the Google service doesn't support this despite working without
        //  internet. ??
        //  for the older SDK versions, there's also some intent parameter that
        //  asks nicely for that. ultimately, i'm 80% sure whether it's actually
        //  on-device and private is up to whether you trust the developer. this
        //  should be communicated to the user.
        mSpeech = SpeechRecognizer.createSpeechRecognizer(ctx);
        mSpeech.setRecognitionListener(new SpeechListener());
    }

    public void toggleListening() {
        switch (mListening) {
            case NOT_LISTENING -> {
                // TODO LOL: permission request flow
                mListening = ListenState.WAITING;
                mSpeech.startListening(INTENT);
            }
            case WAITING -> {
                // todo: have some sort of "toggle buffer" mechanism here so we
                //  don't just eat inputs
            }
            case LISTENING -> {
                mListening = ListenState.WAITING;
                mSpeech.stopListening();
            }
        }
    }

    private void typeOut(String text) {
        Log.d(TAG, text);
    }

    public void stopListening() {
        mSpeech.stopListening();
    }

    public void destroy() {
        mSpeech.destroy();
    }

    private final class SpeechListener implements RecognitionListener {
        @Override
        public void onBeginningOfSpeech() {
        }

        @Override
        public void onBufferReceived(byte[] buffer) {
        }

        @Override
        public void onEndOfSpeech() {
            // this means nothing, we don't decide when the user stops speaking.
        }

        @Override
        public void onError(int error) {
            mListening = ListenState.NOT_LISTENING;
        }

        @Override
        public void onEvent(int eventType, Bundle params) {
        }

        @Override
        public void onPartialResults(Bundle partialResults) {
            ArrayList<String> recognitions
                = partialResults.getStringArrayList(
                    SpeechRecognizer.RESULTS_RECOGNITION
                )
            ;
            if (recognitions == null || recognitions.isEmpty()) {
                return;
            }

            // todo: the Google recognizer will put a leading space if it's
            //  not the first transcription in the recording session. instead,
            //  we should trim that and ourselves be the judge of whether the
            //  transcription should be space-padded based on caret/selection
            //  position and language properties.
            typeOut(recognitions.get(0));
            // so in theory, RESULTS_RECOGNITION is a list of different guesses
            // for what the user said. in practice with the Google recognizer, I
            // have only ever seen this be a singleton list. maybe to get
            // additional hypotheses we need to pass some additional parameters
            // in the recognizer intent? other results from a fully-fledged
            // recognizer that would be helpful for correction suggestions
            // include RESULTS_ALTERNATIVES and CONFIDENCE_SCORES.
        }

        @Override
        public void onReadyForSpeech(Bundle params) {
            mListening = ListenState.LISTENING;
        }

        @Override
        public void onResults(Bundle results) {
            mListening = ListenState.NOT_LISTENING;
        }

        @Override
        public void onRmsChanged(float rmsdB) {
            // todo: this could be used for a noise gauge graphic
        }
    }
}
