import csv
import os

import torch


class SpectrogramDataset(torch.utils.data.Dataset):
    """
    Unified dataset for loading precomputed spectrograms from CSV files.

    Handles multiple CSV formats:
    - Training format: filepath,label,run
    - Binary labels format: filename,label,original_class
    - Extended format: filename,label,run,original_class (with optional fields)

    The dataset automatically detects the CSV format and adapts accordingly.
    """

    def __init__(self, csv_file, specs_dir="data/specs"):
        """
        Initialize the dataset.

        Args:
            csv_file: Path to CSV file containing labels
            specs_dir: Directory containing precomputed .pt spectrogram files
        """
        self.specs_dir = specs_dir
        self.samples = []

        with open(csv_file, 'r') as f:
            # Peek at first line to detect format
            first_line = f.readline().strip()
            f.seek(0)  # Reset to beginning

            # Check if file has headers
            has_headers = any(header in first_line.lower() for header in
                            ['filename', 'filepath', 'label', 'run', 'original_class', 'is_positive'])

            if has_headers:
                # Use DictReader for files with headers
                reader = csv.DictReader(f)
                for row in reader:
                    sample = self._parse_dict_row(row)
                    if sample:
                        self.samples.append(sample)
            else:
                # Use regular reader for files without headers (legacy training sets)
                reader = csv.reader(f)
                for row in reader:
                    sample = self._parse_list_row(row)
                    if sample:
                        self.samples.append(sample)

    def _parse_dict_row(self, row):
        """Parse a row from a CSV with headers."""
        # Get filename - handle both 'filename' and 'filepath' columns
        if 'filename' in row:
            filename = os.path.basename(row['filename'])
        elif 'filepath' in row:
            filename = os.path.basename(row['filepath'])
        else:
            return None

        # Get label - handle both 'label' and 'is_positive' columns
        if 'label' in row:
            label = int(row['label'])
        elif 'is_positive' in row:
            label = int(row['is_positive'])
        else:
            return None

        # Optional fields
        run = row.get('run', '1')
        original_class = int(row['original_class']) if 'original_class' in row else None

        # Build spectrogram path
        spec_filename = filename.replace('.wav', '.pt')
        spec_filepath = os.path.join(self.specs_dir, spec_filename)

        return {
            'filename': filename,
            'spec_filepath': spec_filepath,
            'label': label,
            'run': run,
            'original_class': original_class
        }

    def _parse_list_row(self, row):
        """Parse a row from a CSV without headers (legacy format)."""
        if len(row) < 2:
            return None

        # Legacy format: filepath,label,run
        filepath = row[0]
        label = int(row[1])
        run = row[2] if len(row) > 2 else '1'

        filename = os.path.basename(filepath)
        spec_filename = filename.replace('.wav', '.pt')
        spec_filepath = os.path.join(self.specs_dir, spec_filename)

        return {
            'filename': filename,
            'spec_filepath': spec_filepath,
            'label': label,
            'run': run,
            'original_class': None
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec_filepath = sample['spec_filepath']

        # Load precomputed spectrogram
        if not os.path.exists(spec_filepath):
            raise FileNotFoundError(f"Spectrogram file not found: {spec_filepath}")

        data = torch.load(spec_filepath)

        # Build return dictionary with all available fields
        result = {
            'data': data,
            'label': sample['label'],
            'filename': sample['filename'],
            'filepath': spec_filepath,
            'run': sample['run']
        }

        # Only include original_class if it exists
        if sample['original_class'] is not None:
            result['original_class'] = sample['original_class']

        return result


if __name__ == "__main__":
    # Test with different CSV formats
    import sys

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        specs_dir = sys.argv[2] if len(sys.argv) > 2 else "data/specs"

        dataset = SpectrogramDataset(csv_file=csv_file, specs_dir=specs_dir)
        print(f"Dataset loaded from: {csv_file}")
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            # Show first sample
            sample = dataset[0]
            print("\nFirst sample:")
            for key, value in sample.items():
                if key != 'data':
                    print(f"  {key}: {value}")
                else:
                    print(f"  data: tensor with shape {value.shape}")

            # Check consistency
            print(f"All samples have consistent keys: {all(set(dataset[i].keys()) == set(sample.keys()) for i in range(min(5, len(dataset))))}")
    else:
        print("Usage: python spectrogram_dataset.py <csv_file> [specs_dir]")
