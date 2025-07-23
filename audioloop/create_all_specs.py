import argparse
import csv
import logging
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torchaudio
from tqdm import tqdm

from .config import AudioLoopConfig
from .datasets import UrbanSound8KConfig
from .datasets.fsd50k_config import FSD50KConfig

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
        self.spectrogram_lengths = []
        self.start_time = time.time()

    def record_success(self, processing_time: float | None = None, spec_length: int | None = None):
        """Record a successful processing."""
        self.successful += 1
        if processing_time:
            self.processing_times.append(processing_time)
        if spec_length:
            self.spectrogram_lengths.append(spec_length)

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

        if self.spectrogram_lengths:
            summary.append("")
            summary.append("Spectrogram length statistics (before fixing to 993):")
            summary.append(f"  Count: {len(self.spectrogram_lengths)}")
            summary.append(f"  Min: {min(self.spectrogram_lengths)}")
            summary.append(f"  Max: {max(self.spectrogram_lengths)}")
            summary.append(f"  Mean: {statistics.mean(self.spectrogram_lengths):.1f}")
            summary.append(f"  Median: {statistics.median(self.spectrogram_lengths):.1f}")
            summary.append(f"  Std Dev: {statistics.stdev(self.spectrogram_lengths):.1f}")

            # Add histogram
            summary.extend(self._create_length_histogram())

        if self.errors_by_type:
            summary.append("Error breakdown:")
            for error_type, count in self.errors_by_type.items():
                summary.append(f"  {error_type}: {count}")

        return "\n".join(summary)

    def _create_length_histogram(self) -> list[str]:
        """Create a text-based histogram of spectrogram lengths."""
        if not self.spectrogram_lengths:
            return []

        lengths = sorted(self.spectrogram_lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        # Calculate percentiles to handle outliers
        p95 = lengths[int(0.95 * len(lengths))]
        p99 = lengths[int(0.99 * len(lengths))]
        p999 = lengths[int(0.999 * len(lengths))]

        # Use 99th percentile as histogram max to avoid extreme outliers
        hist_max = p99
        outliers = [length for length in lengths if length > hist_max]

        # Create 20 bins up to 99th percentile
        num_bins = 20
        bin_width = (hist_max - min_len) / num_bins
        bins = [0] * num_bins

        # Count items in each bin
        for length in lengths:
            if length <= hist_max:
                bin_idx = min(int((length - min_len) / bin_width), num_bins - 1)
                bins[bin_idx] += 1

        # Find max count for scaling
        max_count = max(bins)

        # Create histogram display
        histogram = ["", "Length distribution (up to 99th percentile):"]

        for i, count in enumerate(bins):
            bin_start = min_len + i * bin_width
            bin_end = min_len + (i + 1) * bin_width

            # Scale bar length (max 40 chars)
            bar_length = int((count / max_count) * 40) if max_count > 0 else 0
            bar = "█" * bar_length

            # Format bin range
            if bin_width >= 1:
                range_str = f"{bin_start:6.0f}-{bin_end:6.0f}"
            else:
                range_str = f"{bin_start:6.1f}-{bin_end:6.1f}"

            histogram.append(f"  {range_str}: {bar} ({count})")

        # Add outlier summary
        if outliers:
            histogram.append("")
            histogram.append(f"Outliers beyond 99th percentile ({p99:.0f}): {len(outliers)} files")
            histogram.append(f"  95th percentile: {p95:.0f}")
            histogram.append(f"  99th percentile: {p99:.0f}")
            histogram.append(f"  99.9th percentile: {p999:.0f}")
            histogram.append(f"  Maximum: {max_len:.0f}")

        return histogram


def create_specs(config, dataset_config, clear_output=True) -> tuple[int, int]:
    """
    Create spectrograms for any dataset using the provided configurations.

    Args:
        config: Central AudioLoopConfig for output directory (specs_dir)
        dataset_config: Dataset configuration that handles dataset-specific operations
        clear_output: Whether to clear existing spectrograms before processing

    Returns:
        Tuple of (successful_count, failed_count)
    """

    # Validate inputs
    # Check for metadata file - different datasets have different attribute names
    metadata_file = None
    if hasattr(dataset_config, "metadata_csv"):
        metadata_file = dataset_config.metadata_csv
    elif hasattr(dataset_config, "dev_csv"):
        metadata_file = dataset_config.dev_csv

    if metadata_file and not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    if not dataset_config.audio_root.exists():
        raise FileNotFoundError(f"Audio root directory not found: {dataset_config.audio_root}")

    # Clear and create output directory
    if clear_output and config.specs_dir.exists():
        logger.info(f"Clearing existing spectrograms in {config.specs_dir}")
        shutil.rmtree(config.specs_dir)

    config.specs_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    logger.info(f"Loading {dataset_config.__class__.__name__} metadata...")
    audio_files = dataset_config.load_metadata(split="dev")
    logger.info(f"Found {len(audio_files)} audio files in dataset")

    # Process files
    stats = ProcessingStats()

    for i, file_info in enumerate(tqdm(audio_files, desc="Creating spectrograms")):
        start_time = time.time()

        success, spec_length = dataset_config.process_single_file(file_info, config.specs_dir)

        processing_time = time.time() - start_time

        if success:
            stats.record_success(processing_time, spec_length)

            # Log sample info for first file
            if i == 0:
                try:
                    waveform, _ = torchaudio.load(file_info["audio_path"])
                    sample_spec = dataset_config.spec_transform(waveform)
                    fixed_spec = dataset_config.fix_spectrogram_length(sample_spec)
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
    logger.info(f"Output directory: {config.specs_dir}")

    return stats.successful, stats.failed


def create_inference_csv(config, dataset_config) -> Path:
    """
    Create a CSV file listing all dataset files for inference.
    Format: filename,labels (labels as comma-separated string)

    Args:
        config: Central AudioLoopConfig for output path
        dataset_config: Dataset configuration that handles dataset-specific operations

    Returns:
        Path to created inference CSV
    """
    # Load metadata
    audio_files = dataset_config.load_metadata(split="dev")

    # Prepare data for CSV - use labels arrays consistently
    files_data = []
    for file_info in audio_files:
        files_data.append(
            {"filename": file_info["filename"], "labels": ",".join(file_info["labels"])}
        )

    # Get inference CSV path from central config
    inference_csv_path = config.get_inference_csv_path(config.dataset)

    # Ensure output directory exists
    inference_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    with inference_csv_path.open("w", newline="") as f:
        fieldnames = ["filename", "labels"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files_data)

    logger.info(f"Created inference CSV with {len(files_data)} files: {inference_csv_path}")
    return inference_csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create spectrograms for audio datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process FSD50K dataset (default)
  python -m audioloop.create_all_specs

  # Process UrbanSound8K dataset
  python -m audioloop.create_all_specs --dataset urbansound8k

  # Process without clearing existing spectrograms
  python -m audioloop.create_all_specs --no-clear
        """,
    )

    parser.add_argument(
        "--dataset",
        choices=["urbansound8k", "fsd50k"],
        default=None,
        help="Dataset to process ('fsd50k' or 'urbansound8k'). Can also be set via AUDIOLOOP_DATASET environment variable.",
    )

    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear existing spectrograms before processing",
    )

    args = parser.parse_args()

    # Get unified config
    try:
        config = AudioLoopConfig(dataset=args.dataset)
        dataset_name = config.dataset
    except ValueError as e:
        logger.error(f"Dataset error: {e}")
        exit(1)

    # Set up dataset config (processor functionality now merged into config)
    if dataset_name == "fsd50k":
        dataset_config = FSD50KConfig()
        logger.info("Processing FSD50K dataset")
    else:
        dataset_config = UrbanSound8KConfig()
        logger.info("Processing UrbanSound8K dataset")

    # Create spectrograms
    successful, failed = create_specs(config, dataset_config, clear_output=not args.no_clear)

    # Create CSV for inference
    if successful > 0:
        inference_csv = create_inference_csv(config, dataset_config)
        logger.info(f"Ready for inference! Use: {inference_csv}")
    else:
        logger.error("No spectrograms were created successfully")
