import numpy as np
import tensorflow as tf
from PIL import Image

def load_model(model_path):
    """Load TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for the model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0).astype(np.float32)

def predict_image(model_path, image_path):
    """Predict image classification using the TFLite model."""
    interpreter = load_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    input_data = preprocess_image(image_path, tuple(input_details[0]['shape'][1:3]))

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Load and test the model
model_path = "assets/kalimat_model.tflite"
image_path = "path/to/gambar1.png"
output = predict_image(model_path, image_path)
labels = ["Isim", "Fi'il", "Huruf"]
predicted_label = labels[np.argmax(output)]
print(f"Predicted Label: {predicted_label}")
print(f"Raw Output: {output}")
