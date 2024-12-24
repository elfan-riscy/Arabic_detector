import tensorflow as tf

# Load the TFLite model
model_path = "assets/arabic_sentence_model.tflite"  # Ganti dengan lokasi model Anda
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Inspect input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:", input_details)
print("Output Details:", output_details)
