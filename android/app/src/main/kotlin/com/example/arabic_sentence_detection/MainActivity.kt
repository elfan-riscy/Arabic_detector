package com.example.arabic_sentence_detection

import android.os.Bundle
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "arabic_sentence_model"
    private lateinit var textInterpreter: Interpreter
    private lateinit var imageInterpreter: Interpreter

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        try {
            // Load TFLite models
            val textModelBuffer = loadModelFile("final_model.tflite")
            textInterpreter = Interpreter(textModelBuffer)

            val imageModelBuffer = loadModelFile("kalimat_model.tflite")
            imageInterpreter = Interpreter(imageModelBuffer)
        } catch (e: Exception) {
            throw RuntimeException("Error loading TensorFlow Lite models: ${e.message}")
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "predictText" -> {
                    val inputText = call.argument<String>("text")
                    if (inputText != null) {
                        try {
                            val predictionResult = predictText(inputText)
                            result.success(predictionResult)
                        } catch (e: Exception) {
                            result.error("PREDICTION_ERROR", "Error during text prediction: ${e.message}", null)
                        }
                    } else {
                        result.error("INVALID_INPUT", "Input text is missing", null)
                    }
                }
                "predictFromImage" -> {
                    val imagePath = call.argument<String>("imagePath")
                    if (imagePath != null) {
                        try {
                            val predictionResult = predictImage(imagePath)
                            result.success(predictionResult)
                        } catch (e: Exception) {
                            result.error("PREDICTION_ERROR", "Error during image prediction: ${e.message}", null)
                        }
                    } else {
                        result.error("INVALID_INPUT", "Image path is missing", null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val assetFileDescriptor = assets.openFd(modelName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun predictText(inputText: String): Map<String, String> {
        val inputBuffer = ByteBuffer.allocateDirect(4 * 100).apply {
            order(ByteOrder.nativeOrder())
            val textAsNumbers = inputText.toCharArray().map { it.code.toFloat() }
            for (i in 0 until 100) {
                putFloat(if (i < textAsNumbers.size) textAsNumbers[i] else 0f)
            }
        }

        val outputBuffer = ByteBuffer.allocateDirect(4 * 3).apply { order(ByteOrder.nativeOrder()) }
        textInterpreter.run(inputBuffer, outputBuffer)

        return parseOutput(outputBuffer)
    }

    private fun predictImage(imagePath: String): Map<String, String> {
        val imageFile = File(imagePath)
        if (!imageFile.exists()) throw RuntimeException("Image file not found at $imagePath")

        // Preprocess image
        val bitmap = BitmapFactory.decodeFile(imagePath)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        val outputBuffer = ByteBuffer.allocateDirect(4 * 3).apply { order(ByteOrder.nativeOrder()) }
        imageInterpreter.run(inputBuffer, outputBuffer)

        return parseOutput(outputBuffer)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3).apply {
            order(ByteOrder.nativeOrder())
        }

        val intValues = IntArray(224 * 224)
        bitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        for (pixelValue in intValues) {
            val r = (pixelValue shr 16 and 0xFF) / 255.0f
            val g = (pixelValue shr 8 and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        return inputBuffer
    }

    private fun parseOutput(outputBuffer: ByteBuffer): Map<String, String> {
        outputBuffer.rewind()
        val output = FloatArray(3)
        outputBuffer.asFloatBuffer().get(output)

        val labels = listOf("Isim", "Fi'il", "Huruf")
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
        val resultType = labels.getOrElse(maxIndex) { "Tidak Diketahui" }
        val resultFeatures = when (resultType) {
            "Isim" -> "Ciri-ciri Isim: Kata benda, nama, atau keterangan."
            "Fi'il" -> "Ciri-ciri Fi'il: Kata kerja atau tindakan."
            "Huruf" -> "Ciri-ciri Huruf: Kata sambung atau partikel."
            else -> "Tidak diketahui."
        }

        return mapOf("type" to resultType, "features" to resultFeatures)
    }

    override fun onDestroy() {
        super.onDestroy()
        textInterpreter.close()
        imageInterpreter.close()
    }
}
