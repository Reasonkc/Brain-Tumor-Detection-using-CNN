import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="./model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image_path, input_shape):
    """
    Preprocess an image to match the model's input requirements.

    Args:
        image_path (str): Path to the image.
        input_shape (tuple): Model's expected input shape (batch_size, height, width, channels).

    Returns:
        np.ndarray: Preprocessed image with shape (1, height, width, channels).
    """
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    # Resize the image to match the model's input size
    image = image.resize((input_shape[1], input_shape[2]))  # Resize to (224x224)
    # Normalize pixel values to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    # Add batch and channel dimensions to match input shape
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Shape: (1, height, width, 1)
    return image_array

# Path to the image
image_path = "image.jpg"

# Preprocess the image
input_shape = input_details[0]['shape']  # (1, height, width, channels)
print("Expected input shape:", input_shape)  # Debugging input shape
test_input = preprocess_image(image_path, input_shape)
print("Actual input shape:", test_input.shape)  # Debugging actual shape

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], test_input)

# Run inference
interpreter.invoke()

# Get the output tensor
output = interpreter.get_tensor(output_details[0]['index'])

# Print the raw output
print("Raw Output (Probabilities):", output)

# Map indices to class names (Update with your actual class names)
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
predicted_class_index = np.argmax(output)  # Get the index of the max probability
predicted_class = class_labels[predicted_class_index]  # Get the class name

# Print the prediction
print(f"Predicted Class Index: {predicted_class_index}")
print(f"Predicted Class: {predicted_class}")
