import tensorflow as tf
import numpy as np

# 1. Muat model TFLite
model_path = "arabic_sentence_model.tflite"  # Path ke model TFLite Anda
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 2. Cek detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:", input_details)
print("Output Details:", output_details)

# 3. Definisikan tokenizer sederhana
# Tokenisasi ini harus sesuai dengan yang digunakan saat melatih model
tokenizer = {
    "ينصر": 1,       # Contoh mapping teks ke token
    "الكتاب": 2,
    "من البيت": 3
}
max_length = 100

def preprocess_input(text, batch_size=32):
    """
    Preproses teks menjadi array numerik dengan padding dan mendukung batch size.
    """
    sequence = [tokenizer.get(text, 0)]  # Ambil token dari teks; gunakan 0 jika tidak dikenal
    padded_sequence = np.zeros((batch_size, max_length), dtype=np.float32)  # Batch size 32
    padded_sequence[:, :len(sequence)] = sequence  # Ulangi sequence untuk batch size 32
    return padded_sequence

# 4. Input untuk prediksi
input_text = "الكتاب"  # Masukkan teks Arab yang ingin diuji
input_data = preprocess_input(input_text)
print("Input Data:", input_data)

# 5. Set tensor input
interpreter.set_tensor(input_details[0]['index'], input_data)

# 6. Jalankan prediksi
interpreter.invoke()

# 7. Ambil hasil prediksi
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output Data:", output_data)

# 8. Interpretasi hasil
# Pastikan label sesuai dengan urutan output model Anda
labels = ["Fi'il", "Isim", "Huruf"]
predicted_index = np.argmax(output_data[0])  # Ambil indeks kelas dengan probabilitas tertinggi
predicted_label = labels[predicted_index]

print(f"Prediksi untuk '{input_text}': {predicted_label}")
