import csv
import os

import torch
import torchaudio


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, labels_file=None, root_dir=None, meta_csv=None, transform=None):
        """
        Dataset for loading labeled audio files.

        Args:
            labels_file: Path to CSV file with format: filepath,label,run
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
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) >= 2:  # At minimum need filepath and label
                    filepath, label = row[0], row[1]
                    run = row[2] if len(row) > 2 else "1"
                    self.labels.append((filepath, int(label), run))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath, label, run = self.labels[idx]

        # Load audio file
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        waveform, sample_rate = torchaudio.load(filepath)
        data = waveform
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
            "run": run,
        }


if __name__ == "__main__":
    # Check available backends
    print("Available torchaudio backends:")
    print(f"  Current backend: {torchaudio.get_audio_backend()}")
    print(f"  Available backends: {torchaudio.list_audio_backends()}")

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
            print("Try setting a different backend:")
            print("  torchaudio.set_audio_backend('sox_io')")
            print("  or install soundfile: pip install soundfile")
