import csv
import os

import torch


class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, labels_file, specs_dir="data/specs"):
        """
        Dataset for loading precomputed spectrograms.

        Args:
            labels_file: Path to CSV file with format: filepath,label,run
            specs_dir: Directory containing precomputed .pt spectrogram files
        """
        self.specs_dir = specs_dir
        self.labels = []

        with open(labels_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, row in enumerate(csv_reader):
                # Skip header row if it exists
                if i == 0 and len(row) > 0 and row[0].lower() in ['filepath', 'filename']:
                    continue

                if len(row) >= 2:  # At minimum need filepath and label
                    filepath, label = row[0], row[1]
                    run = row[2] if len(row) > 2 else "1"

                    # Convert audio filepath to spectrogram filepath
                    audio_filename = os.path.basename(filepath)
                    spec_filename = audio_filename.replace(".wav", ".pt")
                    spec_filepath = os.path.join(specs_dir, spec_filename)

                    self.labels.append((spec_filepath, int(label), run))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec_filepath, label, run = self.labels[idx]

        # Load precomputed spectrogram
        if not os.path.exists(spec_filepath):
            raise FileNotFoundError(f"Spectrogram file not found: {spec_filepath}")

        data = torch.load(spec_filepath)
        filename = os.path.basename(spec_filepath)

        # Return as dictionary for compatibility
        return {
            "data": data,
            "label": label,
            "filename": filename,
            "filepath": spec_filepath,
            "run": run
        }


if __name__ == "__main__":
    # Test with labels.csv
    dataset = SpectrogramDataset(labels_file="../../labels.csv", specs_dir="../../data/specs")
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Spectrogram shape: {sample['data'].shape}")
            print(f"Label: {sample['label']}")
            print(f"Filename: {sample['filename']}")
        except Exception as e:
            print(f"Error loading spectrogram: {e}")
