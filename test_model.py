from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

# Path to the image you want to predict
image_path = './image.jpg'

# Load the image and convert to grayscale
image = load_img(image_path, color_mode='grayscale', target_size=(224, 224))  # Grayscale
image_array = img_to_array(image) / 255.0  # Rescale pixel values to [0, 1]

# Expand dimensions to match the model input (batch_size, height, width, channels)
image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension for grayscale
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Load your pre-trained model
model = load_model('./cnn_model_grayscale.keras')

# Predict the class probabilities
prediction = model.predict(image_array)
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Get the predicted class
predicted_class_idx = np.argmax(prediction)
predicted_label = class_labels[predicted_class_idx]

# Output the predicted class and probabilities
print(f"Predicted Class: {predicted_label}")
print(f"Class Probabilities: {prediction}")
