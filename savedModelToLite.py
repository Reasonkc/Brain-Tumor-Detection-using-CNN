import tensorflow as tf

# Path to your SavedModel directory
saved_model_dir = "./saved_model"

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable optimization for quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_path = "./model_quantized.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Quantized model saved to {tflite_model_path}")
