from tensorflow.keras.models import load_model

# Load the model
model = load_model('cnn_model_grayscale.keras')  # Update path if necessary

model.export('saved_model')