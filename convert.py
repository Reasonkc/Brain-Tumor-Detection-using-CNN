from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('cnn_model_grayscale.keras')  # Update path if necessary

# Scale factor for fixed-point representation (e.g., Q8.8 format)
scale_factor = 2**8

# Output file path
output_file_path = "weights_for_verilog.txt"

# Helper function to convert a value to 16-bit two's complement hexadecimal
def to_fixed_point_hex(value, scale_factor):
    fixed_point = int(float(value) * scale_factor)
    if fixed_point < 0:
        fixed_point = (1 << 16) + fixed_point  # Convert to 16-bit two's complement
    return f"{fixed_point:04X}"  # Format as 4-digit hexadecimal

# Extract weights and biases and write to the output file
with open(output_file_path, "w") as f:
    for layer in model.layers:
        weights = layer.get_weights()  # Get weights and biases as a list
        if len(weights) > 0:  # Skip layers without weights or biases
            layer_weights = weights[0]  # First element: weights
            layer_biases = weights[1] if len(weights) > 1 else None  # Second element: biases, if present

            # Process weights
            flat_weights = np.ravel(layer_weights)  # Flatten weights for iteration
            num_outputs = layer_weights.shape[-1]  # Number of outputs (last dimension)

            for output_idx in range(num_outputs):
                start_idx = output_idx * flat_weights.size // num_outputs
                end_idx = (output_idx + 1) * flat_weights.size // num_outputs
                chunk = flat_weights[start_idx:end_idx]
                for weight_idx, value in enumerate(chunk):
                    hex_value = to_fixed_point_hex(value, scale_factor)
                    f.write(f"{hex_value}\n")

            # Process biases, if present
            if layer_biases is not None:
                for bias_idx, bias in enumerate(np.ravel(layer_biases)):
                    hex_value = to_fixed_point_hex(bias, scale_factor)
                    f.write(f"{hex_value}\n")

print(f"Weights and biases have been saved to {output_file_path}")
