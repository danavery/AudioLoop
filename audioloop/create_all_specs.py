import argparse
import csv
import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torchaudio
from tqdm import tqdm

from .datasets import UrbanSound8KConfig, UrbanSound8KProcessor
from .datasets.fsd50k import FSD50KConfig, FSD50KProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStats:
    """Track processing statistics."""

    def __init__(self):
        self.successful = 0
        self.failed = 0
        self.errors_by_type = defaultdict(int)
        self.processing_times = []
        self.start_time = time.time()

    def record_success(self, processing_time: float | None = None):
        """Record a successful processing."""
        self.successful += 1
        if processing_time:
            self.processing_times.append(processing_time)

    def record_failure(self, error_type: str = "unknown"):
        """Record a failed processing."""
        self.failed += 1
        self.errors_by_type[error_type] += 1

    def summary(self) -> str:
        """Get processing summary."""
        total_time = time.time() - self.start_time
        avg_time = (
            sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        )

        summary = [
            f"Processing complete in {total_time:.2f}s",
            f"Successful: {self.successful}",
            f"Failed: {self.failed}",
        ]

        if self.processing_times:
            summary.append(f"Average processing time: {avg_time:.3f}s per file")

        if self.errors_by_type:
            summary.append("Error breakdown:")
            for error_type, count in self.errors_by_type.items():
                summary.append(f"  {error_type}: {count}")

        return "\n".join(summary)


def create_specs(processor, config=None, clear_output=True) -> tuple[int, int]:
    """
    Create spectrograms for any dataset using the provided processor.

    Args:
        processor: Dataset processor that handles dataset-specific operations
        config: Dataset configuration. If None, uses processor's config.
        clear_output: Whether to clear existing spectrograms before processing

    Returns:
        Tuple of (successful_count, failed_count)
    """
    if config is None:
        config = processor.config

    # Validate inputs
    # Check for metadata file - different datasets have different attribute names
    metadata_file = None
    if hasattr(config, "metadata_csv"):
        metadata_file = config.metadata_csv
    elif hasattr(config, "dev_csv"):
        metadata_file = config.dev_csv

    if metadata_file and not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    if not config.audio_root.exists():
        raise FileNotFoundError(f"Audio root directory not found: {config.audio_root}")

    # Clear and create output directory
    if clear_output and config.output_dir.exists():
        logger.info(f"Clearing existing spectrograms in {config.output_dir}")
        shutil.rmtree(config.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    logger.info(f"Loading {processor.__class__.__name__} metadata...")
    audio_files = processor.load_metadata()
    logger.info(f"Found {len(audio_files)} audio files in dataset")

    # Process files
    stats = ProcessingStats()

    for i, file_info in enumerate(tqdm(audio_files, desc="Creating spectrograms")):
        start_time = time.time()

        success = processor.process_single_file(file_info, config.output_dir)

        processing_time = time.time() - start_time

        if success:
            stats.record_success(processing_time)

            # Log sample info for first file
            if i == 0:
                try:
                    waveform, _ = torchaudio.load(file_info["audio_path"])
                    sample_spec = processor.spec_transform(waveform)
                    fixed_spec = processor.fix_spectrogram_length(sample_spec)
                    logger.info(f"Sample audio_path: {file_info['audio_path']}")
                    logger.info(f"Sample audio shape: {waveform.shape}")
                    logger.info(f"Sample spectrogram shape (before fixing): {sample_spec.shape}")
                    logger.info(f"Sample spectrogram shape (after fixing): {fixed_spec.shape}")
                except Exception as e:
                    logger.warning(f"Could not log sample info: {e}")
        else:
            stats.record_failure()

    # Print summary
    logger.info("\n" + stats.summary())
    logger.info(f"Output directory: {config.output_dir}")

    return stats.successful, stats.failed


def create_inference_csv(processor, config=None) -> Path:
    """
    Create a CSV file listing all dataset files for inference.
    Format: filename,labels (labels as comma-separated string)

    Args:
        processor: Dataset processor that handles dataset-specific operations
        config: Dataset configuration. If None, uses processor's config.

    Returns:
        Path to created inference CSV
    """
    if config is None:
        config = processor.config

    # Load metadata
    audio_files = processor.load_metadata()

    # Prepare data for CSV - use labels arrays consistently
    files_data = []
    for file_info in audio_files:
        files_data.append({
            "filename": file_info["filename"],
            "labels": ",".join(file_info["labels"])
        })

    # Ensure output directory exists
    config.inference_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    with config.inference_csv.open("w", newline="") as f:
        fieldnames = ["filename", "labels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files_data)

    logger.info(f"Created inference CSV with {len(files_data)} files: {config.inference_csv}")
    return config.inference_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create spectrograms for audio datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process UrbanSound8K dataset (default)
  python -m audioloop.create_all_specs

  # Process FSD50K dataset
  python -m audioloop.create_all_specs --dataset fsd50k

  # Process without clearing existing spectrograms
  python -m audioloop.create_all_specs --no-clear
        """,
    )

    parser.add_argument(
        "--dataset",
        choices=["urbansound8k", "fsd50k"],
        default="urbansound8k",
        help="Dataset to process (default: urbansound8k)",
    )

    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear existing spectrograms before processing",
    )

    args = parser.parse_args()

    # Set up dataset processor
    if args.dataset == "fsd50k":
        config = FSD50KConfig()
        processor = FSD50KProcessor(config)
        logger.info("Processing FSD50K dataset (dev split)")
    else:
        config = UrbanSound8KConfig()
        processor = UrbanSound8KProcessor(config)
        logger.info("Processing UrbanSound8K dataset")

    # Create spectrograms
    successful, failed = create_specs(processor, clear_output=not args.no_clear)

    # Create CSV for inference
    if successful > 0:
        inference_csv = create_inference_csv(processor)
        logger.info(f"Ready for inference! Use: {inference_csv}")
    else:
        logger.error("No spectrograms were created successfully")
