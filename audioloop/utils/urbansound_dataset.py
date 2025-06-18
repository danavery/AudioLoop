import csv
import os

import torch


class UrbanSoundDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, specs_dir="data/all_specs"):
        """
        Dataset for loading UrbanSound8K precomputed spectrograms.

        Args:
            csv_file: Path to CSV file with format: filename,is_positive,original_class OR filename,class_id
            specs_dir: Directory containing precomputed .pt spectrogram files
        """
        self.specs_dir = specs_dir
        self.data = []

        with open(csv_file, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                filename = row['filename']

                # Handle both binary format and original format
                if 'is_positive' in row:
                    # Binary labels format (generalized)
                    label = int(row['is_positive'])
                    original_class = int(row['original_class'])
                elif 'is_siren' in row:
                    # Legacy binary siren format (for backward compatibility)
                    label = int(row['is_siren'])
                    original_class = int(row['original_class'])
                else:
                    # Original UrbanSound8K format
                    class_id = int(row['class_id'])
                    # Convert to binary: siren (class 8) = 1, everything else = 0
                    label = 1 if class_id == 8 else 0
                    original_class = class_id

                # Convert audio filename to spectrogram filepath
                spec_filename = filename.replace(".wav", ".pt")
                spec_filepath = os.path.join(specs_dir, spec_filename)

                self.data.append({
                    'filename': filename,
                    'spec_filepath': spec_filepath,
                    'label': label,  # Binary label (0 or 1)
                    'original_class': original_class  # Original UrbanSound8K class
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        spec_filepath = item['spec_filepath']

        # Load precomputed spectrogram
        if not os.path.exists(spec_filepath):
            raise FileNotFoundError(f"Spectrogram file not found: {spec_filepath}")

        data = torch.load(spec_filepath)

        # Return as dictionary for compatibility
        return {
            "data": data,
            "label": item['label'],  # Binary label (0 or 1)
            "filename": item['filename'],
            "filepath": spec_filepath,
            "original_class": item['original_class']  # Original UrbanSound8K class
        }


if __name__ == "__main__":
    # Test with urbansound8k_files.csv
    dataset = UrbanSoundDataset(csv_file="../../urbansound8k_files.csv")
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Spectrogram shape: {sample['data'].shape}")
            print(f"Binary label: {sample['label']}")
            print(f"Original class: {sample['original_class']}")
            print(f"Filename: {sample['filename']}")
        except Exception as e:
            print(f"Error loading spectrogram: {e}")
