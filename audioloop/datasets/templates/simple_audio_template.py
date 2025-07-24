"""
TEMPLATE: Simple audio dataset configuration for folder + CSV pattern.

⚠️  THIS IS A TEMPLATE FILE - DO NOT MODIFY DIRECTLY!

Instead:
1. Copy this file to: audioloop/datasets/your_dataset_name_config.py
2. Rename the class from TemplateAudioConfig to YourDatasetNameConfig
3. Customize the paths and settings for your dataset
4. Use with: --dataset your_dataset_name

This template supports the common pattern of:
- Audio files in a directory (WAV, MP3, etc.)
- Labels in a CSV file with columns: filename, label

Example CSV format:
    filename,label
    audio1.wav,speech
    audio2.wav,music
    audio3.wav,noise

Or with additional columns:
    filename,label,speaker_id,duration
    audio1.wav,speech,speaker_001,2.5
    audio2.wav,music,speaker_002,3.1

Copy example:
    cp audioloop/datasets/templates/simple_audio_template.py \\
       audioloop/datasets/my_speech_config.py
"""

import csv
from pathlib import Path
from typing import Any, ClassVar

import torch.nn as nn
import torchaudio

from .dataset_config import DatasetConfig


class TemplateAudioConfig(DatasetConfig):
    """
    TEMPLATE: Configuration for simple folder + CSV audio datasets.
    
    ⚠️  RENAME THIS CLASS when you copy this file!
    Example: MyDatasetConfig, SpeechDatasetConfig, etc.
    
    Then customize the paths and settings below for your specific dataset.
    """
    
    # =============================================================================
    # 🚨 CUSTOMIZE THESE PATHS FOR YOUR DATASET 🚨
    # =============================================================================
    
    # Path to your CSV file with labels
    _dataset_csv_path = Path("data/YOUR_DATASET_NAME/labels.csv")
    
    # Directory containing your audio files
    _audio_root_path = Path("data/YOUR_DATASET_NAME/clips")
    
    # Audio file extension (will try common extensions if file not found)
    _audio_extension = ".wav"
    
    # CSV column names (customize if your CSV has different column names)
    _filename_column = "filename"
    _label_column = "label"
    
    # Define your class vocabulary (customize for your classes)
    _class_vocabulary: ClassVar[dict[int, str]] = {
        0: "class_one",
        1: "class_two", 
        2: "class_three",
    }
    
    # Audio processing parameters (customize if needed)
    _sample_rate = 22050
    _n_fft = 1024
    _hop_length = 512
    _n_mels = 128
    _top_db = 80.0
    _fixed_length = 993  # Standard AudioLoop spectrogram length
    
    # =============================================================================
    # INTERFACE IMPLEMENTATION (usually no need to modify below this line)
    # =============================================================================
    
    @property
    def dataset_csv(self) -> Path:
        """Path to the main dataset CSV file."""
        return self._dataset_csv_path
    
    @property
    def audio_root(self) -> Path:
        """Root directory containing audio files."""
        return self._audio_root_path
    
    @property
    def name_to_id(self) -> dict[str, int]:
        """Mapping from class names to class IDs."""
        return {name: class_id for class_id, name in self._class_vocabulary.items()}
    
    @property
    def vocabulary(self) -> dict[int, str]:
        """Mapping from class IDs to class names."""
        return self._class_vocabulary.copy()
    
    def get_metadata_entries(self) -> list[dict[str, Any]]:
        """Get list of metadata entries for active learning."""
        return self.load_metadata()
    
    def load_metadata(self, split: str = "dev") -> list[dict[str, Any]]:
        """
        Load metadata entries from the CSV file.
        
        Args:
            split: Ignored for simple datasets (no train/dev/test splits)
            
        Returns:
            List of metadata dictionaries
        """
        if not self.dataset_csv.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.dataset_csv}\n"
                f"Create a CSV file with columns: {self._filename_column}, {self._label_column}\n"
                f"Example:\n"
                f"  {self._filename_column},{self._label_column}\n"
                f"  audio1.wav,speech\n"
                f"  audio2.wav,music"
            )
        
        entries = []
        with self.dataset_csv.open("r") as f:
            reader = csv.DictReader(f)
            
            # Validate required columns
            fieldnames = reader.fieldnames or []
            if self._filename_column not in fieldnames:
                raise ValueError(f"CSV missing required column: '{self._filename_column}'")
            if self._label_column not in fieldnames:
                raise ValueError(f"CSV missing required column: '{self._label_column}'")
            
            for row in reader:
                filename = row[self._filename_column].strip()
                label = row[self._label_column].strip()
                
                # Skip empty rows
                if not filename or not label:
                    continue
                
                # Get audio path (try multiple extensions if needed)
                audio_path = self.get_audio_path(filename)
                
                entry = {
                    "filename": filename,
                    "labels": [label],  # AudioLoop expects list of labels
                    "audio_path": audio_path,
                    "fold": None,  # Simple datasets don't have folds
                }
                
                # Add any additional columns from CSV
                for key, value in row.items():
                    if key not in [self._filename_column, self._label_column]:
                        entry[key] = value
                
                entries.append(entry)
        
        return entries
    
    def list_classes(self) -> None:
        """Print available classes for this dataset."""
        print(f"Simple Audio Dataset Classes ({len(self.vocabulary)} total):")
        print("=" * 50)
        for class_id in sorted(self.vocabulary.keys()):
            name = self.vocabulary[class_id]
            print(f"{class_id:3d}: {name}")
        print(f"\nDataset CSV: {self.dataset_csv}")
        print(f"Audio directory: {self.audio_root}")
    
    def get_audio_path(self, filename: str, fold: int | None = None) -> Path:
        """
        Get full path to audio file, trying multiple extensions if needed.
        
        Args:
            filename: Audio filename from metadata
            fold: Ignored for simple datasets
            
        Returns:
            Full path to the audio file
        """
        # If filename already has extension, try it first
        if "." in filename:
            audio_path = self.audio_root / filename
            if audio_path.exists():
                return audio_path
        
        # Try with the configured extension
        base_name = filename.split(".")[0]  # Remove extension if present
        audio_path = self.audio_root / f"{base_name}{self._audio_extension}"
        if audio_path.exists():
            return audio_path
        
        # Try common audio extensions
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
            audio_path = self.audio_root / f"{base_name}{ext}"
            if audio_path.exists():
                return audio_path
        
        # Return the preferred path even if file doesn't exist (for error messages)
        return self.audio_root / f"{base_name}{self._audio_extension}"
    
    def get_audio_processing_params(self) -> dict[str, Any]:
        """Get audio processing parameters for spectrogram generation."""
        return {
            "sample_rate": self._sample_rate,
            "n_fft": self._n_fft,
            "hop_length": self._hop_length,
            "n_mels": self._n_mels,
            "top_db": self._top_db,
            "fixed_length": self._fixed_length,
        }
    
    def is_positive_class(self, class_name: str, positive_class: str | int) -> bool:
        """Determine if a class matches the positive class for binary classification."""
        if isinstance(positive_class, str):
            return class_name == positive_class
        if isinstance(positive_class, int):
            return (
                positive_class in self.name_to_id
                and self.name_to_id.get(class_name) == positive_class
            )
        return False
    
    def get_spectrogram_path(self, filename: str, specs_dir: Path) -> Path:
        """Get path where spectrogram should be stored."""
        # Remove extension and add .pt
        base_filename = filename.split(".")[0]
        return specs_dir / f"{base_filename}.pt"
    
    def create_spectrogram_transform(self) -> nn.Sequential:
        """Create PyTorch transform pipeline for generating spectrograms."""
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                n_mels=self._n_mels,
            ),
            torchaudio.transforms.AmplitudeToDB(top_db=self._top_db),
        )
    
    def parse_metadata_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format."""
        filename = row[self._filename_column].strip()
        label = row[self._label_column].strip()
        
        return {
            "filename": filename,
            "labels": [label],
            "audio_path": self.get_audio_path(filename),
            "fold": None,
        }
    
    def get_binary_label(self, item: dict[str, Any], positive_class_id: int, positive_class_name: str) -> bool:
        """Get binary label for an item based on positive class."""
        # item["labels"] is a list of label names
        item_labels = item.get("labels", [])
        
        # Check if any of the item's labels match the positive class
        return any(self.is_positive_class(label, positive_class_name) for label in item_labels)