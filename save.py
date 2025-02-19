from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('./cnn_model_grayscale.keras')  # Update path as necessary

# Scale factor for fixed-point representation (e.g., Q8.8 format)
scale_factor = 2**8

# Open the output file
output_file_path = "weights_for_verilog.txt"
with open(output_file_path, "w") as f:
    # Loop through each layer to extract weights and biases
    for layer in model.layers:
        if len(layer.get_weights()) > 0:  # Only process layers with weights
            layer_weights, layer_biases = layer.get_weights()

            # Convert weights to fixed-point and write to file
            for row in layer_weights:
                if row.ndim > 1:  # Multi-dimensional weights
                    for value in row:
                        fixed_point = int(value * scale_factor)
                        if fixed_point < 0:
                            fixed_point = (1 << 16) + fixed_point  # Two's complement for 16 bits
                        f.write(f"{fixed_point:04X}\n")  # Write as 4-digit hex
                else:  # Single-dimensional weights (e.g., Conv2D filters)
                    fixed_point = int(row * scale_factor)
                    if fixed_point < 0:
                        fixed_point = (1 << 16) + fixed_point  # Two's complement for 16 bits
                    f.write(f"{fixed_point:04X}\n")  # Write as 4-digit hex

            # Convert biases to fixed-point and write to file
            for bias in layer_biases:
                fixed_point = int(bias * scale_factor)
                if fixed_point < 0:
                    fixed_point = (1 << 16) + fixed_point  # Two's complement for 16 bits
                f.write(f"{fixed_point:04X}\n")  # Write as 4-digit hex

print(f"Weights and biases saved to {output_file_path}")
