package com.example.visionassist;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSIONS_REQUEST_CODE = 1;
    private static final String[] REQUIRED_PERMISSIONS = {
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA
    };

    private static final long DETECTION_INTERVAL_MS = 800;
    // Minimum gap between any two spoken announcements.
    private static final long ANNOUNCEMENT_INTERVAL_MS = 4500;
    // If the SAME message would repeat, only re-speak it after this much longer
    // gap - long enough to not spam, short enough that the user knows it's still there.
    private static final long REPEAT_MESSAGE_INTERVAL_MS = 9000;

    private SpeechRecognizer speechRecognizer;
    private TextToSpeech textToSpeech;
    private Button micButton;
    private PreviewView cameraPreview;
    private ObjectDetectorHelper objectDetector;
    private ExecutorService cameraExecutor;
    private android.widget.TextView debugText;
    private boolean isDetecting = false;
    private boolean isListening = false;
    private long lastAnalyzedTime = 0;
    private long lastAnnouncementTime = 0;
    private long lastSpokenTime = 0;
    private String lastSpokenMessage = "";
    private int frameCounter = 0;
    private static final long URGENT_ANNOUNCEMENT_INTERVAL_MS = 2000;
    private List<ObjectDetectorHelper.DetectionResult> latestDetections =
            new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        debugText = findViewById(R.id.debugText);

        micButton = findViewById(R.id.micButton);
        cameraPreview = findViewById(R.id.cameraPreview);
        cameraExecutor = Executors.newSingleThreadExecutor();

        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
                textToSpeech.speak("Voice assistant ready", TextToSpeech.QUEUE_FLUSH, null, null);
            }
        });

        textToSpeech.setOnUtteranceProgressListener(new android.speech.tts.UtteranceProgressListener() {
            @Override public void onStart(String utteranceId) {}
            @Override
            public void onDone(String utteranceId) {

                if ("start_utterance".equals(utteranceId)) {
                    isDetecting = true;
                    isListening = false;
                }

                if ("stop_utterance".equals(utteranceId)) {
                    isListening = false;
                }

                if ("ahead_answer".equals(utteranceId)) {
                    isListening = false;
                }

                if ("generic_utterance".equals(utteranceId)) {
                    isListening = false;
                }

                if ("retry_utterance".equals(utteranceId)) {
                    isListening = false;
                }

                // Only start actually listening AFTER the "Listening" prompt
                // finishes speaking, so the mic doesn't pick up our own TTS.
                if ("listening_prompt".equals(utteranceId)) {
                    runOnUiThread(MainActivity.this::beginSpeechRecognition);
                }
            }
            @Override public void onError(String utteranceId) {}
        });

        setupObjectDetector();

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSIONS_REQUEST_CODE);
        } else {
            startCamera();
        }

        micButton.setOnClickListener(v -> startListening());
    }

    private void setupObjectDetector() {
        try {
            objectDetector = new ObjectDetectorHelper(this, "yolov5n.tflite", "labelmap.txt");
        } catch (Exception e) {
            Toast.makeText(this, "Failed to load detection model: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (Exception e) {
                Toast.makeText(this, "Camera failed to start: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeFrame(ImageProxy imageProxy) {
        frameCounter++;

        long currentTime = System.currentTimeMillis();

        if (!isDetecting || objectDetector == null
                || frameCounter % 5 != 0
                || (currentTime - lastAnalyzedTime) < DETECTION_INTERVAL_MS) {
            imageProxy.close();
            return;
        }
        lastAnalyzedTime = currentTime;

        try {
            Bitmap bitmap = ImageUtils.imageProxyToBitmap(imageProxy);
            List<ObjectDetectorHelper.DetectionResult> results = objectDetector.detect(bitmap);
            latestDetections = results;
            handleDetections(results, bitmap.getWidth(), bitmap.getHeight());
        } catch (Exception e) {
            e.printStackTrace();
            runOnUiThread(() -> debugText.setText("Detection error: " + e.getMessage()));
        } finally {
            imageProxy.close();
        }
    }

    private void handleDetections(List<ObjectDetectorHelper.DetectionResult> results, int imageWidth, int imageHeight) {
        if (results == null || results.isEmpty()) return;
        if (isListening) return;

        // Announce the most URGENT object (closest), not just the most confident one -
        // this is what makes the safe-path suggestion line up with what's actually said.
        ObjectDetectorHelper.DetectionResult top = pickPriorityDetection(results);
        RectF box = top.boundingBox;

        // React faster when the priority object is genuinely close, instead of
        // waiting on the same fixed interval used for far-away objects.
        boolean topIsClose = box.height() > 0.5f;
        long requiredGap = topIsClose ? URGENT_ANNOUNCEMENT_INTERVAL_MS : ANNOUNCEMENT_INTERVAL_MS;

        long currentTime = System.currentTimeMillis();
        if ((currentTime - lastAnnouncementTime) < requiredGap) return;

        String direction = estimateDirection(box);
        String distance = estimateDistance(box);
        String message = distance + " " + top.label + " " + direction;

        boolean anyCloseOrNearby = false;
        for (ObjectDetectorHelper.DetectionResult d : results) {
            if (d.boundingBox.height() > 0.25f) {
                anyCloseOrNearby = true;
                break;
            }
        }

        if (anyCloseOrNearby) {
            String safePath = computeSafePath(results);
            if (safePath != null) {
                message = message + ". " + safePath;
            }
        }

        boolean sameAsLast = message.equals(lastSpokenMessage);
        boolean repeatWindowElapsed = (currentTime - lastSpokenTime) >= REPEAT_MESSAGE_INTERVAL_MS;

        lastAnnouncementTime = currentTime;

        if (sameAsLast && !repeatWindowElapsed) {
            return;
        }

        lastSpokenTime = currentTime;
        lastSpokenMessage = message;

        textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, null);

        final String debugMessage = message;
        runOnUiThread(() -> debugText.setText(debugMessage));
    }

    private String estimateDirection(RectF box) {
        float centerX = box.centerX();
        if (centerX < 0.33) return "on your left";
        if (centerX > 0.66) return "on your right";
        return "ahead";
    }

    private String estimateDistance(RectF box) {
        float boxHeightFraction = box.height();
        if (boxHeightFraction > 0.5) return "Close,";
        if (boxHeightFraction > 0.25) return "Nearby,";
        return "Far,";
    }

    // Looks at every detected object this frame (not just the closest one) and figures out
// which third of the view - left, center, right - is safest to move toward.
// Returns null if the center is actually clear and no path suggestion is needed.
    private String computeSafePath(List<ObjectDetectorHelper.DetectionResult> results) {
        float leftMaxHeight = 0f;
        float centerMaxHeight = 0f;
        float rightMaxHeight = 0f;

        for (ObjectDetectorHelper.DetectionResult d : results) {
            float cx = d.boundingBox.centerX();
            float h = d.boundingBox.height();

            if (cx < 0.33f) {
                leftMaxHeight = Math.max(leftMaxHeight, h);
            } else if (cx > 0.66f) {
                rightMaxHeight = Math.max(rightMaxHeight, h);
            } else {
                centerMaxHeight = Math.max(centerMaxHeight, h);
            }
        }

        boolean centerBlocked = centerMaxHeight > 0.25f;
        if (!centerBlocked) {
            return "Path ahead is clear"; // CHANGED: was `return null;`
        }

        boolean leftBlocked = leftMaxHeight > 0.25f;
        boolean rightBlocked = rightMaxHeight > 0.25f;

        if (!leftBlocked && !rightBlocked) {
            return (leftMaxHeight <= rightMaxHeight) ? "Move left" : "Move right";
        } else if (!leftBlocked) {
            return "Move left";
        } else if (!rightBlocked) {
            return "Move right";
        } else {
            return "Stop, path blocked on all sides";
        }
    }

    // Picks the object to announce based on urgency (how close it is), not raw
// model confidence. NMS already sorted 'results' by confidence, but the most
// confident detection isn't necessarily the one that's about to be walked into.
    private ObjectDetectorHelper.DetectionResult pickPriorityDetection(List<ObjectDetectorHelper.DetectionResult> results) {
        ObjectDetectorHelper.DetectionResult best = null;
        for (ObjectDetectorHelper.DetectionResult d : results) {
            if (best == null || d.boundingBox.height() > best.boundingBox.height()) {
                best = d;
            }
        }
        return best;
    }

    private void startListening() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Microphone permission not granted yet", Toast.LENGTH_SHORT).show();
            return;
        }

        isListening = true;

        if (textToSpeech.isSpeaking()) {
            textToSpeech.stop();
        }

        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(
                        SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    handleCommand(matches.get(0));
                } else {
                    speakRetry();
                }
            }

            @Override public void onReadyForSpeech(Bundle params) {}
            @Override public void onBeginningOfSpeech() {}
            @Override public void onRmsChanged(float rmsdB) {}
            @Override public void onBufferReceived(byte[] buffer) {}
            @Override public void onEndOfSpeech() {}
            @Override public void onError(int error) {
                speakRetry();
            }
            @Override public void onPartialResults(Bundle partialResults) {}
            @Override public void onEvent(int eventType, Bundle params) {}
        });

        // Say "Listening" first; the actual mic opens once this finishes
        // (see the "listening_prompt" case in onDone above).
        textToSpeech.speak("Listening", TextToSpeech.QUEUE_FLUSH, null, "listening_prompt");
    }

    private void beginSpeechRecognition() {
        if (speechRecognizer == null) return;
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US);
        speechRecognizer.startListening(intent);
    }

    private void speakRetry() {
        textToSpeech.speak("Sorry, I didn't catch that. Please try again.",
                TextToSpeech.QUEUE_FLUSH, null, "retry_utterance");
        runOnUiThread(() -> debugText.setText("Didn't catch that - try again"));
    }

    private void handleCommand(String spokenText) {
        String command = spokenText.toLowerCase(Locale.US).trim();

        android.util.Log.d("VOICE_CMD", "Recognized: '" + command + "'");

        if (command.contains("start")) {
            isListening = false;
            isDetecting = false;
            textToSpeech.speak("Detection started", TextToSpeech.QUEUE_FLUSH, null, "start_utterance");

        } else if (command.contains("stop")) {
            isListening = false;
            isDetecting = false;
            textToSpeech.speak("Detection stopped", TextToSpeech.QUEUE_FLUSH, null, "stop_utterance");

        } else if (command.contains("ahead")
                || command.contains("in front")
                || command.contains("front of me")
                || (command.contains("what") && command.contains("head"))) {

            answerWhatIsAhead();

        } else {
            textToSpeech.speak("You said: " + spokenText, TextToSpeech.QUEUE_FLUSH, null, "generic_utterance");
            isListening = false;
        }
    }

    private void answerWhatIsAhead() {

        if (latestDetections == null || latestDetections.isEmpty()) {
            textToSpeech.speak("I don't see anything ahead.", TextToSpeech.QUEUE_FLUSH, null, "ahead_answer");
            return;
        }

        ObjectDetectorHelper.DetectionResult closestAhead = null;

        for (ObjectDetectorHelper.DetectionResult detection : latestDetections) {
            RectF box = detection.boundingBox;
            float centerX = box.centerX();

            if (centerX >= 0.33f && centerX <= 0.66f) {
                if (closestAhead == null || box.height() > closestAhead.boundingBox.height()) {
                    closestAhead = detection;
                }
            }
        }

        if (closestAhead == null) {
            textToSpeech.speak("I don't see anything directly ahead.", TextToSpeech.QUEUE_FLUSH, null, "ahead_answer");
        } else {
            String distance = estimateDistance(closestAhead.boundingBox);
            String message = "There is a " + closestAhead.label + " " + distance + " ahead.";
            textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, "ahead_answer");
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera and microphone permissions are required", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (speechRecognizer != null) speechRecognizer.destroy();
        if (textToSpeech != null) textToSpeech.shutdown();
        if (objectDetector != null) objectDetector.close();
        cameraExecutor.shutdown();
    }
}