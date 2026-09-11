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

    // Detection will run at roughly this interval - 400ms = ~2.5 frames per second.
    // Fast enough to catch obstacles in time, light enough on the battery.
    private static final long DETECTION_INTERVAL_MS = 400;
    // Don't speak a new announcement more often than this, so TTS doesn't spam.
    private static final long ANNOUNCEMENT_INTERVAL_MS = 4500;

    private SpeechRecognizer speechRecognizer;
    private TextToSpeech textToSpeech;
    private Button micButton;
    private PreviewView cameraPreview;
    private ObjectDetectorHelper objectDetector;
    private ExecutorService cameraExecutor;
    private android.widget.TextView debugText;
    private boolean isDetecting = false;
    private long lastAnalyzedTime = 0;
    private long lastAnnouncementTime = 0;

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
            @Override public void onDone(String utteranceId) {
                if ("start_utterance".equals(utteranceId)) {
                    isDetecting = true; // only actually begin detecting AFTER "Detection started" finishes speaking
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
            objectDetector = new ObjectDetectorHelper(this, "detection_model.tflite", "labelmap.txt");
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

    // This runs on every camera frame - but we throttle it so we only actually
    // process a frame every DETECTION_INTERVAL_MS, to save battery and CPU.
    private void analyzeFrame(ImageProxy imageProxy) {
        long currentTime = System.currentTimeMillis();

        if (!isDetecting || objectDetector == null
                || (currentTime - lastAnalyzedTime) < DETECTION_INTERVAL_MS) {
            imageProxy.close(); // must always close, or the camera pipeline stalls
            return;
        }
        lastAnalyzedTime = currentTime;

        try {
            Bitmap bitmap = ImageUtils.imageProxyToBitmap(imageProxy);
            List<ObjectDetectorHelper.DetectionResult> results = objectDetector.detect(bitmap);
            handleDetections(results, bitmap.getWidth(), bitmap.getHeight());
        } catch (Exception e) {
            // Don't crash the app on a bad frame - just skip it.
        } finally {
            imageProxy.close();
        }
    }

    private void handleDetections(List<ObjectDetectorHelper.DetectionResult> results, int imageWidth, int imageHeight) {
        if (results == null || results.isEmpty()) return;

        long currentTime = System.currentTimeMillis();
        if ((currentTime - lastAnnouncementTime) < ANNOUNCEMENT_INTERVAL_MS) return;

        ObjectDetectorHelper.DetectionResult top = results.get(0);
        RectF box = top.boundingBox;

        String direction = estimateDirection(box);
        String distance = estimateDistance(box);

        String message = distance + " " + top.label + " " + direction;
        textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, null);
        lastAnnouncementTime = currentTime;

        final String debugMessage = message;
        runOnUiThread(() -> debugText.setText(debugMessage));
    }

    private String estimateDirection(RectF box) {
        float centerX = box.centerX(); // already normalized 0-1
        if (centerX < 0.33) return "on your left";
        if (centerX > 0.66) return "on your right";
        return "ahead";
    }

    private String estimateDistance(RectF box) {
        float boxHeightFraction = box.height(); // already normalized 0-1
        if (boxHeightFraction > 0.5) return "Close,";
        if (boxHeightFraction > 0.25) return "Nearby,";
        return "Far,";
    }

    private void startListening() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Microphone permission not granted yet", Toast.LENGTH_SHORT).show();
            return;
        }

        if (textToSpeech.isSpeaking()) {
            textToSpeech.stop(); // clear the speaker immediately so it doesn't bleed into the mic
        }

        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);

        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US);

        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(
                        SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    handleCommand(matches.get(0));
                }
            }

            @Override public void onReadyForSpeech(Bundle params) {}
            @Override public void onBeginningOfSpeech() {}
            @Override public void onRmsChanged(float rmsdB) {}
            @Override public void onBufferReceived(byte[] buffer) {}
            @Override public void onEndOfSpeech() {}
            @Override public void onError(int error) {
                Toast.makeText(MainActivity.this, "Didn't catch that, try again", Toast.LENGTH_SHORT).show();
            }
            @Override public void onPartialResults(Bundle partialResults) {}
            @Override public void onEvent(int eventType, Bundle params) {}
        });

        speechRecognizer.startListening(intent);
    }

    private void handleCommand(String spokenText) {
        String command = spokenText.toLowerCase(Locale.US);

        if (command.contains("start")) {
            isDetecting = false; // make sure no detection runs during the announcement itself
            textToSpeech.speak("Detection started", TextToSpeech.QUEUE_FLUSH, null, "start_utterance");
        } else if (command.contains("stop")) {
            isDetecting = false;
            textToSpeech.speak("Detection stopped", TextToSpeech.QUEUE_FLUSH, null, "stop_utterance");
        } else {
            textToSpeech.speak("You said: " + spokenText, TextToSpeech.QUEUE_FLUSH, null, "generic_utterance");
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