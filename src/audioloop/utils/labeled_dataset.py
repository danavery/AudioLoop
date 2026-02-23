import csv
import os

import torch
from torchcodec.decoders import AudioDecoder


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, labels_file=None, root_dir=None, meta_csv=None, transform=None):
        """
        Dataset for loading labeled audio files.

        Args:
            labels_file: Path to CSV file with headers. Required columns: filename (or filepath) and label
            root_dir: Root directory for audio files (legacy parameter for compatibility)
            meta_csv: Metadata CSV file (legacy parameter for compatibility)
            transform: Transform to apply to audio data
        """
        self.transform = transform
        self.labels = []

        # Use labels_file if provided, otherwise fall back to meta_csv for compatibility
        csv_file = labels_file or meta_csv

        if csv_file is None:
            raise ValueError("Either labels_file or meta_csv must be provided")

        with open(csv_file) as f:
            # Validate that file has headers
            first_line = f.readline().strip()
            f.seek(0)  # Reset to beginning

            required_headers = ["filename", "filepath", "label"]
            has_valid_header = any(header in first_line.lower() for header in required_headers)

            if not has_valid_header:
                raise ValueError(
                    f"CSV file must have headers. Expected columns: filename (or filepath) and label. "
                    f"First line was: {first_line[:100]}"
                )

            # Use DictReader for files with headers
            reader = csv.DictReader(f)
            for row in reader:
                # Get filepath - handle both 'filename' and 'filepath' columns
                if "filename" in row:
                    filepath = row["filename"]
                elif "filepath" in row:
                    filepath = row["filepath"]
                else:
                    continue  # Skip rows without filepath

                # Get label
                if "label" in row:
                    label = int(row["label"])
                else:
                    continue  # Skip rows without label

                self.labels.append((filepath, label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath, label = self.labels[idx]

        # Load audio file
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        decoder = AudioDecoder(filepath, num_channels=1)
        data = decoder.get_all_samples().data
        filename = os.path.basename(filepath)

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        # Return as dictionary for compatibility with spectrogram creation
        return {
            "data": data,
            "label": label,
            "filename": filename,
            "filepath": filepath,
        }


if __name__ == "__main__":
    # Test with labels.csv
    dataset = LabeledDataset(labels_file="../../labels.csv")
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Audio shape: {sample['data'].shape}")
            print(f"Label: {sample['label']}")
            print(f"Filename: {sample['filename']}")
        except Exception as e:
            print(f"Error loading audio: {e}")
