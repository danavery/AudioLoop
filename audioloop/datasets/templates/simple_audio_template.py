"""
TEMPLATE: Simple audio dataset configuration for folder + CSV pattern.

⚠️  THIS IS A TEMPLATE FILE - DO NOT MODIFY DIRECTLY!

Instead:
1. Copy this file to: audioloop/datasets/your_dataset_name_config.py
2. Rename the class from TemplateAudioConfig to YourDatasetNameConfig  
3. Customize the paths and settings for your dataset
4. Use with: --dataset your_dataset_name

This template provides a complete DatasetConfig implementation with:
- Audio file loading from directory structures
- CSV metadata parsing with flexible column names
- Spectrogram preprocessing pipeline
- Binary classification support
- Automatic file extension detection

Supported CSV formats:
    filename,label
    audio1.wav,speech
    audio2.wav,music

Or with additional metadata:
    filename,label,speaker_id,duration
    audio1.wav,speech,speaker_001,2.5
    audio2.wav,music,speaker_002,3.1

Copy and customize example:
    cp audioloop/datasets/templates/simple_audio_template.py \\
       audioloop/datasets/my_dataset_config.py
    
Then edit: class name, paths, vocabulary, and audio parameters.
"""

import csv
import logging
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torchaudio

from audioloop.utils.log_normalize import LogNormalize

from ..dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class TemplateAudioConfig(DatasetConfig):
    """
    TEMPLATE: Configuration for simple folder + CSV audio datasets.
    
    ⚠️  RENAME THIS CLASS when you copy this file!
    Example: MyDatasetConfig, SpeechDatasetConfig, etc.
    
    Then customize the paths, vocabulary, and audio parameters below.
    This template provides all required DatasetConfig methods for full AudioLoop compatibility.
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
            LogNormalize(top_db=self._top_db),
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
    
    def get_binary_label(self, item: dict[str, Any], positive_class_id: int, positive_class_name: str) -> int:
        """Get binary label for an item based on positive class criteria."""
        # item["labels"] is a list of label names
        item_labels = item.get("labels", [])
        
        # Check if any of the item's labels match the positive class
        return 1 if any(self.is_positive_class(label, positive_class_name) for label in item_labels) else 0
    
    def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Fix spectrogram to target length by padding or cropping."""
        current_length = spec.shape[-1]  # Time dimension is last
        target_length = self._fixed_length
        
        if current_length < target_length:
            # Pad with zeros on the right
            pad_size = target_length - current_length
            spec = torch.nn.functional.pad(spec, (0, pad_size), mode="constant", value=0)
        elif current_length > target_length:
            # Crop from the center
            start_idx = (current_length - target_length) // 2
            spec = spec[..., start_idx : start_idx + target_length]
        
        return spec
    
    def process_single_file(self, file_info: dict, output_dir: Path) -> tuple[bool, int | None]:
        """Process a single audio file and save its spectrogram."""
        try:
            audio_path = file_info["audio_path"]
            filename = file_info["filename"]
            
            # Check if audio file exists
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return False, None
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert stereo to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Create spectrogram
            spec_transform = self.create_spectrogram_transform()
            spec = spec_transform(waveform)
            
            # Store original length before fixing
            original_length = spec.shape[-1]
            
            # Fix spectrogram length
            spec = self.fix_spectrogram_length(spec)
            
            # Save spectrogram
            base_filename = filename.split(".")[0]  # Remove extension
            output_filename = f"{base_filename}.pt"
            output_path = output_dir / output_filename
            torch.save(spec, output_path)
            
            return True, original_length
            
        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            return False, None