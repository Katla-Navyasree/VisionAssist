package com.example.visionassist;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ObjectDetectorHelper {

    private static final int MODEL_INPUT_SIZE = 320;
    private static final int NUM_CLASSES = 80;
    private static final int NUM_PREDICTIONS = 6300;
    private static final float CONFIDENCE_THRESHOLD = 0.4f;
    private static final float IOU_THRESHOLD = 0.45f;
    // Standard YOLOv5 letterbox pad color (114,114,114)
    private static final int PAD_COLOR = 0xFF727272;

    private Interpreter interpreter;
    private List<String> labels;

    public static class DetectionResult {
        public final String label;
        public final float confidence;
        public final RectF boundingBox; // normalized 0-1 relative to the ORIGINAL bitmap passed to detect()

        DetectionResult(String label, float confidence, RectF boundingBox) {
            this.label = label;
            this.confidence = confidence;
            this.boundingBox = boundingBox;
        }
    }

    // Info needed to map a box detected in the padded 320x320 model space
    // back onto the original (non-square) camera frame.
    private static class LetterboxInfo {
        float scale;
        float padX;
        float padY;
        int origWidth;
        int origHeight;
    }

    public ObjectDetectorHelper(Context context, String modelFileName, String labelFileName) throws IOException {
        interpreter = new Interpreter(loadModelFile(context, modelFileName));
        labels = loadLabels(context, labelFileName);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabels(Context context, String labelFileName) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(labelFileName)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // Resizes into the 320x320 model input while PRESERVING aspect ratio, padding
    // the leftover space with solid gray, instead of stretching 4:3 into a square.
    // This is what was causing the direction/distance bias.
    private Bitmap letterbox(Bitmap src, LetterboxInfo outInfo) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();

        float scale = Math.min((float) MODEL_INPUT_SIZE / srcW, (float) MODEL_INPUT_SIZE / srcH);
        int scaledW = Math.round(srcW * scale);
        int scaledH = Math.round(srcH * scale);

        float padX = (MODEL_INPUT_SIZE - scaledW) / 2f;
        float padY = (MODEL_INPUT_SIZE - scaledH) / 2f;

        outInfo.scale = scale;
        outInfo.padX = padX;
        outInfo.padY = padY;
        outInfo.origWidth = srcW;
        outInfo.origHeight = srcH;

        Bitmap canvasBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(canvasBitmap);
        canvas.drawColor(PAD_COLOR);

        Bitmap scaled = Bitmap.createScaledBitmap(src, scaledW, scaledH, true);
        canvas.drawBitmap(scaled, padX, padY, new Paint(Paint.FILTER_BITMAP_FLAG));

        return canvasBitmap;
    }

    private ByteBuffer bitmapToByteBuffer(Bitmap letterboxed) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
        letterboxed.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

        for (int pixel : pixels) {
            byteBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
            byteBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
            byteBuffer.putFloat((pixel & 0xFF) / 255.0f);         // B
        }

        byteBuffer.rewind();
        return byteBuffer;
    }

    public List<DetectionResult> detect(Bitmap bitmap) {
        LetterboxInfo info = new LetterboxInfo();
        Bitmap letterboxed = letterbox(bitmap, info);
        ByteBuffer inputBuffer = bitmapToByteBuffer(letterboxed);

        float[][][] output = new float[1][NUM_PREDICTIONS][NUM_CLASSES + 5];
        interpreter.run(inputBuffer, output);

        List<DetectionResult> candidates = new ArrayList<>();
        boolean loggedThisFrame = false;

        for (int i = 0; i < NUM_PREDICTIONS; i++) {
            float[] pred = output[0][i];
            float objectness = pred[4];
            if (objectness < CONFIDENCE_THRESHOLD) continue;

            int bestClass = -1;
            float bestScore = 0f;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float score = pred[5 + c];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }

            float confidence = objectness * bestScore;
            if (confidence < CONFIDENCE_THRESHOLD || bestClass == -1) continue;

            // Some YOLOv5 tflite exports output coords already normalized 0-1,
            // others output raw pixel coords in the 320x320 model space.
            // Detect which one this model uses so it works either way: pixel-space
            // values are almost always well above 1.0, normalized ones almost never are.
            boolean isPixelSpace = pred[0] > 1.5f || pred[1] > 1.5f || pred[2] > 1.5f || pred[3] > 1.5f;

            if (!loggedThisFrame) {
                android.util.Log.d("YOLO_COORDS", "raw cx=" + pred[0] + " cy=" + pred[1]
                        + " w=" + pred[2] + " h=" + pred[3] + " pixelSpace=" + isPixelSpace);
                loggedThisFrame = true;
            }

            float cxModel, cyModel, wModel, hModel; // pixel units of the 320x320 letterboxed canvas
            if (isPixelSpace) {
                cxModel = pred[0];
                cyModel = pred[1];
                wModel = pred[2];
                hModel = pred[3];
            } else {
                cxModel = pred[0] * MODEL_INPUT_SIZE;
                cyModel = pred[1] * MODEL_INPUT_SIZE;
                wModel = pred[2] * MODEL_INPUT_SIZE;
                hModel = pred[3] * MODEL_INPUT_SIZE;
            }

            float leftModel = cxModel - wModel / 2f;
            float topModel = cyModel - hModel / 2f;
            float rightModel = cxModel + wModel / 2f;
            float bottomModel = cyModel + hModel / 2f;

            // Undo the letterbox transform to map back onto the original camera frame.
            float left = (leftModel - info.padX) / info.scale;
            float top = (topModel - info.padY) / info.scale;
            float right = (rightModel - info.padX) / info.scale;
            float bottom = (bottomModel - info.padY) / info.scale;

            RectF box = new RectF(
                    left / info.origWidth,
                    top / info.origHeight,
                    right / info.origWidth,
                    bottom / info.origHeight
            );

            String label = (bestClass < labels.size()) ? labels.get(bestClass) : "object";
            candidates.add(new DetectionResult(label, confidence, box));
        }

        return nonMaxSuppression(candidates);
    }

    private List<DetectionResult> nonMaxSuppression(List<DetectionResult> candidates) {
        List<DetectionResult> results = new ArrayList<>();
        candidates.sort(Comparator.comparingDouble((DetectionResult d) -> d.confidence).reversed());

        boolean[] suppressed = new boolean[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) continue;
            DetectionResult a = candidates.get(i);
            results.add(a);
            for (int j = i + 1; j < candidates.size(); j++) {
                if (suppressed[j]) continue;
                DetectionResult b = candidates.get(j);
                if (iou(a.boundingBox, b.boundingBox) > IOU_THRESHOLD) {
                    suppressed[j] = true;
                }
            }
        }
        return results;
    }

    private float iou(RectF a, RectF b) {
        float interLeft = Math.max(a.left, b.left);
        float interTop = Math.max(a.top, b.top);
        float interRight = Math.min(a.right, b.right);
        float interBottom = Math.min(a.bottom, b.bottom);

        float interArea = Math.max(0, interRight - interLeft) * Math.max(0, interBottom - interTop);
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);

        return interArea / (areaA + areaB - interArea + 1e-6f);
    }

    public void close() {
        if (interpreter != null) interpreter.close();
    }
}