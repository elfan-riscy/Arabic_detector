import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="final_model.tflite")
interpreter.allocate_tensors()

# Prepare input
input_text = "كتاب"  # Contoh input
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Convert input to a compatible format (e.g., ASCII, padding to size 100)
input_array = np.zeros((1, 100), dtype=np.float32)
for i, char in enumerate(input_text):
    if i < 100:
        input_array[0, i] = ord(char)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_array)

# Run inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output)
