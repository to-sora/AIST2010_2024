import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def save_tensor_as_png(file_path):
    # Load the .pt file
    tensor = torch.load(file_path)
    
    # Get the base filename without extension
    base_filename = os.path.splitext(file_path)[0]

    # Loop through each slice in the tensor
    for i in range(tensor.shape[0]):
        # Convert tensor slice to numpy for plotting
        slice_data = tensor[i][0:300].numpy()

        # Set up plot with enhanced contrast
        plt.figure()
        plt.imshow(slice_data, aspect='auto', origin='lower', cmap='inferno')  # Use a high-contrast color map like 'inferno'

        # Normalize the color range for better contrast visualization
        plt.colorbar()
        plt.title(f"Slice {i}")
        plt.xlabel("Time Steps")
        plt.ylabel("Mel Bins")

        # Save the figure as PNG
        output_path = f"{base_filename}_slice_{i}.png"
        plt.savefig(output_path, dpi=300)  # Higher DPI for clearer image
        plt.close()
        print(f"Saved slice {i} as {output_path}")
import os
import torch
import glob
import re

# Example usage:
# save_tensor_as_png("path_to_your_tensor.pt")
def get_latest_checkpoint(model_save_dir, version):
    """
    Finds the latest checkpoint file for the given version.

    Args:
        model_save_dir (str): Directory where model checkpoints are saved.
        version (str): Current version identifier.

    Returns:
        tuple: (latest_checkpoint_path (str or None), latest_epoch (int))
    """
    pattern = os.path.join(model_save_dir, f"{version}_model_e*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None, 0  # No checkpoint found
    
    # Extract epoch numbers using regex
    epoch_pattern = re.compile(rf"{re.escape(version)}_model_e(\d+)\.pth$")
    epochs = []
    for file in checkpoint_files:
        basename = os.path.basename(file)
        match = epoch_pattern.match(basename)
        if match:
            epochs.append(int(match.group(1)))
    
    if not epochs:
        return None, 0
    
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(model_save_dir, f"{version}_model_e{latest_epoch}.pth")
    return latest_checkpoint, latest_epoch
