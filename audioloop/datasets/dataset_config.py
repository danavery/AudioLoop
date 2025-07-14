"""
Base dataset configuration interface for AudioLoop.

Provides a common interface for all datasets used in active learning,
allowing new datasets to be added without modifying existing code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class DatasetConfig(ABC):
    """Common interface for dataset configurations used in active learning."""
    
    @property
    @abstractmethod
    def dataset_csv(self) -> Path:
        """Path to the main dataset CSV file."""
        pass
    
    @property
    @abstractmethod
    def audio_root(self) -> Path:
        """Root directory containing audio files."""
        pass
    
    @abstractmethod
    def get_metadata_entries(self) -> List[Dict[str, Any]]:
        """Get list of metadata entries for active learning.
        
        Returns:
            List of dicts with consistent keys:
            - filename: Audio filename (str)
            - class_name: Human-readable class name (str) 
            - fold: Fold number if applicable (int or None)
        """
        pass
    
    @abstractmethod
    def get_audio_path(self, filename: str, fold: int | None = None) -> Path:
        """Get full path to audio file.
        
        Args:
            filename: Audio filename from metadata
            fold: Fold number (used by some datasets like UrbanSound8K)
            
        Returns:
            Full path to the audio file
        """
        pass