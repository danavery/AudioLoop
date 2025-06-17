import csv
import os

import torch
import torchaudio
from torch import nn
from tqdm import tqdm

from .utils.log_normalize import LogNormalize

# Your transform
spec_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128
    ),
    LogNormalize(top_db=80),
)

# Fixed length for all spectrograms (in time frames)
FIXED_LENGTH = 993  # Max length from training data analysis

def fix_spectrogram_length(spec, target_length=FIXED_LENGTH):
    """Fix spectrogram to target length by padding or cropping."""
    current_length = spec.shape[-1]  # Time dimension is last

    if current_length < target_length:
        # Pad with zeros on the right
        pad_size = target_length - current_length
        spec = torch.nn.functional.pad(spec, (0, pad_size), mode='constant', value=0)
    elif current_length > target_length:
        # Crop from the center
        start_idx = (current_length - target_length) // 2
        spec = spec[..., start_idx:start_idx + target_length]

    return spec

def create_specs_from_urbansound8k(
    metadata_csv="data/urbansound8k/UrbanSound8K.csv",
    audio_root="data/urbansound8k",
    output_dir="data/all_specs"
):
    """
    Create spectrograms for the entire UrbanSound8K dataset.

    Args:
        metadata_csv: Path to UrbanSound8K.csv metadata file
        audio_root: Root directory containing fold1/, fold2/, etc.
        output_dir: Directory to save .pt spectrogram files
    """

    os.makedirs(output_dir, exist_ok=True)

    # Read the metadata CSV
    audio_files = []
    with open(metadata_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            filename = row['slice_file_name']
            fold = row['fold']
            class_id = int(row['classID'])
            class_name = row['class']

            # Construct full path: audio_root/fold{fold}/{filename}
            audio_path = os.path.join(audio_root, f"fold{fold}", filename)

            audio_files.append({
                'filename': filename,
                'audio_path': audio_path,
                'fold': fold,
                'class_id': class_id,
                'class_name': class_name
            })

    print(f"Found {len(audio_files)} audio files in UrbanSound8K dataset")

    # Process each file
    successful = 0
    failed = 0

    for i, file_info in enumerate(tqdm(audio_files, desc="Creating spectrograms")):
        try:
            audio_path = file_info['audio_path']
            filename = file_info['filename']

            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                failed += 1
                continue

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Create spectrogram
            spec = spec_transform(waveform)

            # Fix spectrogram length
            spec = fix_spectrogram_length(spec, FIXED_LENGTH)

            # Save spectrogram
            output_filename = filename.replace(".wav", ".pt")
            output_path = os.path.join(output_dir, output_filename)
            torch.save(spec, output_path)

            if i == 0:
                print(f"Sample audio shape: {waveform.shape}")
                print(f"Sample spectrogram shape (before fixing): {spec_transform(waveform).shape}")
                print(f"Sample spectrogram shape (after fixing): {spec.shape}")

            successful += 1

        except Exception as e:
            print(f"Error processing {file_info['filename']}: {e}")
            failed += 1
            continue

    print("\nSpectrogram creation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")

    return successful, failed

def create_inference_csv(
    metadata_csv="data/urbansound8k/UrbanSound8K.csv",
    output_csv="outputs/urbansound8k_files.csv"
):
    """
    Create a CSV file listing all UrbanSound8K files for inference.
    Format: filename,class_id,fold
    """

    files_data = []
    with open(metadata_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            filename = row['slice_file_name']
            class_id = int(row['classID'])
            fold = row['fold']

            files_data.append({
                'filename': filename,
                'class_id': class_id,
                'fold': fold
            })

    # Write to new CSV
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['filename', 'class_id', 'fold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files_data)

    print(f"Created inference CSV with {len(files_data)} files: {output_csv}")
    return output_csv

if __name__ == "__main__":
    # Create spectrograms for all UrbanSound8K files
    successful, failed = create_specs_from_urbansound8k()

    # Create CSV for inference
    if successful > 0:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        inference_csv = create_inference_csv()
        print(f"Ready for inference! Use: {inference_csv}")
