package com.example.visionassist;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ObjectDetectorHelper {

    // EfficientDet-Lite0 expects a fixed 320x320 input image.
    private static final int MODEL_INPUT_SIZE = 320;
    private static final int MAX_DETECTIONS = 25;
    private static final float SCORE_THRESHOLD = 0.5f;

    private Interpreter interpreter;
    private List<String> labels;

    public static class DetectionResult {
        public final String label;
        public final float confidence;
        public final RectF boundingBox; // normalized 0-1 coordinates

        DetectionResult(String label, float confidence, RectF boundingBox) {
            this.label = label;
            this.confidence = confidence;
            this.boundingBox = boundingBox;
        }
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

    public List<DetectionResult> detect(Bitmap bitmap) {
        // Resize the incoming camera frame to exactly what the model expects.
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build();
        TensorImage tensorImage = new TensorImage(org.tensorflow.lite.DataType.UINT8);
        tensorImage.load(bitmap);
        tensorImage = imageProcessor.process(tensorImage);

        // These 4 output arrays are the standard SSD/EfficientDet detection format.
        float[][][] outputBoxes = new float[1][MAX_DETECTIONS][4];
        float[][] outputClasses = new float[1][MAX_DETECTIONS];
        float[][] outputScores = new float[1][MAX_DETECTIONS];
        float[] numDetections = new float[1];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputBoxes);
        outputs.put(1, outputClasses);
        outputs.put(2, outputScores);
        outputs.put(3, numDetections);

        interpreter.runForMultipleInputsOutputs(new Object[]{tensorImage.getBuffer()}, outputs);

        List<DetectionResult> results = new ArrayList<>();
        int detectedCount = (int) numDetections[0];

        for (int i = 0; i < Math.min(detectedCount, MAX_DETECTIONS); i++) {
            float score = outputScores[0][i];
            if (score < SCORE_THRESHOLD) continue;

            int classIndex = (int) outputClasses[0][i];
            String label = (classIndex >= 0 && classIndex < labels.size())
                    ? labels.get(classIndex) : "object";

            // Box format from the model is [top, left, bottom, right], normalized 0-1.
            float top = outputBoxes[0][i][0];
            float left = outputBoxes[0][i][1];
            float bottom = outputBoxes[0][i][2];
            float right = outputBoxes[0][i][3];

            RectF box = new RectF(left, top, right, bottom);
            results.add(new DetectionResult(label, score, box));
        }
        return results;
    }

    public void close() {
        if (interpreter != null) interpreter.close();
    }
}