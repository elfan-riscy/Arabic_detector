import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteOrder

class KalimatModel(context: Context) {

    private var interpreter: Interpreter

    init {
        // Memuat model TensorFlow Lite dari assets
        val modelBuffer = loadModelFile(context, "kalimat_model.tflite")
        interpreter = Interpreter(modelBuffer)
    }

    // Fungsi untuk memuat file model dari assets
    private fun loadModelFile(context: Context, modelPath: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Fungsi untuk prediksi
    fun predict(inputData: FloatArray): FloatArray {
        // Ubah input data menjadi ByteBuffer
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputData.size)
            .order(ByteOrder.nativeOrder())
        inputData.forEach { inputBuffer.putFloat(it) }

        // Buat array untuk output
        val outputData = FloatArray(10) // Sesuaikan dengan jumlah output model

        // Menjalankan inferensi
        interpreter.run(inputBuffer, outputData)
        return outputData
    }

    // Fungsi untuk menutup interpreter ketika selesai
    fun close() {
        interpreter.close()
    }
}
