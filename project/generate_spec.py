import os
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
INPUT_DIR = "output"  # Directory containing the .wav files
OUTPUT_DIR = os.path.join(INPUT_DIR, "spectrograms")  # Directory to save spectrogram images
SAMPLE_RATE = 22050  # Sample rate for loading .wav files

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_spectrogram(wav_path, spectrogram_path):
    """
    Generates and saves a spectrogram image from a .wav file without using Matplotlib.

    Parameters:
    - wav_path (str): Path to the input .wav file.
    - spectrogram_path (str): Path to save the output spectrogram image.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        
        # Compute the Short-Time Fourier Transform (STFT)
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        
        # Convert amplitude to decibels
        DB = librosa.amplitude_to_db(D, ref=np.max)
        
        # Normalize the spectrogram to 0-255 for image representation
        # Ensures no loss of pixel data
        DB_scaled = 255 * (DB - DB.min()) / (DB.max() - DB.min())
        DB_scaled = DB_scaled.astype(np.uint8)
        
        # Optionally, you can transpose the spectrogram to have time on the x-axis
        # and frequency on the y-axis
        DB_scaled = np.flip(DB_scaled, axis=0)  # Flip vertically for correct orientation
        
        # Convert the numpy array to an image
        image = Image.fromarray(DB_scaled)
        
        # Save the image in grayscale mode ('L' mode for 8-bit pixels)
        image.save(spectrogram_path, format='PNG')
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

def main():
    """
    Main function to process all .wav files in the input directory and generate spectrograms.
    """
    # List all files in the input directory
    files = os.listdir(INPUT_DIR)
    
    # Filter out .wav files
    wav_files = [f for f in files if f.lower().endswith('.wav')]
    
    if not wav_files:
        print("No .wav files found in the input directory.")
        return
    
    print(f"Found {len(wav_files)} .wav files. Starting spectrogram generation...")
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        wav_path = os.path.join(INPUT_DIR, wav_file)
        
        # Define the spectrogram filename (same base name with .png extension)
        base_name = os.path.splitext(wav_file)[0]
        spectrogram_filename = f"{base_name}.png"
        spectrogram_path = os.path.join(OUTPUT_DIR, spectrogram_filename)
        
        # Generate and save the spectrogram
        generate_spectrogram(wav_path, spectrogram_path)
    
    print("All spectrograms generated successfully.")

if __name__ == "__main__":
    main()
