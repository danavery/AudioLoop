import argparse
import csv
import logging
import os
from pathlib import Path

from .config import AudioLoopConfig
from .utils.paths import extract_version_from_filename

# Set up logger for this module
logger = logging.getLogger(__name__)


def compute_and_log_candidate_metrics(new_labels_csv, config, log_level=logging.INFO):
    """
    Compute candidate metrics for the labeled candidates and save to history.

    Args:
        new_labels_csv: Path to candidates CSV file
        config: AudioLoopConfig instance (or None)
        log_level: Logging level for output

    Returns:
        Dict of metrics if successful, empty dict otherwise
    """
    if config is None:
        return {}

    from .utils.candidate_metrics import calculate_and_save_candidate_metrics

    # Extract cycle number from candidates filename
    cycle = extract_version_from_filename(Path(new_labels_csv), "candidates")

    if cycle is None:
        logger.log(
            log_level,
            f"Could not extract cycle number from {Path(new_labels_csv).name}, skipping metrics",
        )
        return {}

    # Calculate and save metrics
    metrics = calculate_and_save_candidate_metrics(Path(new_labels_csv), config.output_dir, cycle)

    if metrics:
        logger.log(
            log_level,
            f"Candidate metrics (cycle {cycle}): "
            f"F1={metrics['f1_score']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f} "
            f"({metrics['num_candidates']} candidates)",
        )

    return metrics


def merge_training_sets(
    original_csv, new_labels_csv, output_csv=None, config=None, log_level=logging.INFO
):
    """
    Merge original training set with newly labeled samples.

    Samples are keyed by filename: a newly labeled candidate that already exists in the
    original training set replaces it (keeping its position), so re-labeling a file or
    re-running a merge never duplicates rows.

    Args:
        original_csv: Path to existing training set (e.g., training_set_v1.csv)
        new_labels_csv: Path to newly labeled samples (candidates CSV with human labels)
        output_csv: Path for merged training set (auto-generated if None)
        config: AudioLoopConfig instance (required for auto-generating output path)
        log_level: Logging level (logging.INFO for normal output, logging.WARNING for quiet)

    Returns:
        str: Path to the created output file
    """
    # Set up logging for this merge run
    logger.setLevel(log_level)
    # Auto-generate output filename if not provided
    if output_csv is None:
        if config is None:
            raise ValueError("config is required when output_csv is not provided")

        # Get version from original CSV and increment
        from pathlib import Path

        current_version = extract_version_from_filename(Path(original_csv), "training_set")
        version = (current_version or 1) + 1
        output_csv = str(config.get_training_set_path(version))

    # Merged samples keyed by filename: a newly labeled candidate REPLACES an existing
    # entry (the human's latest label is the freshest truth), and duplicates within
    # either input collapse to the last occurrence. Insertion order is preserved, so
    # replaced entries keep their original position.
    merged: dict[str, int] = {}

    # Read original training set (requires headers)
    if os.path.exists(original_csv):
        with open(original_csv) as f:
            # Validate headers
            first_line = f.readline().strip()
            f.seek(0)

            required_headers = ["filename", "filepath", "label"]
            has_valid_header = any(header in first_line.lower() for header in required_headers)

            if not has_valid_header:
                raise ValueError(
                    f"Original CSV must have headers. Expected columns: filename (or filepath) and label. "
                    f"First line was: {first_line[:100]}"
                )

            reader = csv.DictReader(f)
            for row in reader:
                # Handle both 'filename' and 'filepath' columns
                if "filename" in row:
                    filepath = row["filename"]
                elif "filepath" in row:
                    filepath = row["filepath"]
                else:
                    continue

                # Get label
                if "label" not in row:
                    continue

                # Handle both filename and full filepath
                filename = os.path.basename(filepath) if filepath.startswith("/") else filepath

                merged[filename] = int(row["label"])
        logger.info(f"Loaded {len(merged)} samples from {original_csv}")
    else:
        logger.warning(f"Warning: {original_csv} not found, starting fresh")

    # Read new labels from candidates CSV format
    new_count = 0
    new_positive = 0
    new_negative = 0
    replaced_count = 0
    with open(new_labels_csv) as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Handle candidates CSV format (with needs_human_label column)
            if "needs_human_label" not in row or "filename" not in row:
                logger.error(
                    "Error: Expected candidates CSV format with 'needs_human_label' and 'filename' columns"
                )
                logger.error(f"Found columns: {list(row.keys())}")
                continue

            filename = os.path.basename(row["filename"]) if row["filename"] else ""
            label = row["needs_human_label"].strip()

            if not filename or label == "":
                continue  # Skip unlabeled samples

            try:
                label_int = int(label)
                if label_int not in [0, 1]:
                    logger.warning(f"Warning: Invalid label '{label}' for {filename}, skipping")
                    continue

                if filename in merged:
                    replaced_count += 1
                else:
                    new_count += 1
                    if label_int == 1:
                        new_positive += 1
                    else:
                        new_negative += 1
                merged[filename] = label_int
            except ValueError:
                logger.warning(f"Warning: Invalid label '{label}' for {filename}, skipping")
                continue

    logger.info(
        f"Added {new_count} new labeled samples ({new_positive} positive, {new_negative} negative)"
    )
    if replaced_count:
        logger.info(f"Updated {replaced_count} existing samples with new human labels")

    # Compute and save candidate metrics
    compute_and_log_candidate_metrics(new_labels_csv, config, log_level)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    # Write merged training set
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["filename", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({"filename": filename, "label": label} for filename, label in merged.items())

    logger.info(f"Created merged training set: {output_csv}")
    logger.info(
        f"Total samples: {len(merged)} (original: {len(merged) - new_count}, new: {new_count})"
    )

    # Show label distribution
    label_counts = {0: 0, 1: 0}
    for label in merged.values():
        label_counts[label] += 1

    logger.info(
        f"Label distribution: {label_counts[0]} negative class, {label_counts[1]} positive class"
    )

    # Display final training set composition (always visible, not just at INFO level)
    print(
        f"New training set: {len(merged)} total ({label_counts[1]} positive, {label_counts[0]} negative)"
    )

    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Merge human labels from active learning candidates into training sets"
    )
    parser.add_argument("original_csv", help="Original training set CSV")
    parser.add_argument(
        "candidates_csv", help="Candidates CSV with filled needs_human_label column"
    )
    parser.add_argument("-o", "--output", help="Output merged training set CSV")
    parser.add_argument("--experiment", help="Experiment name for output directory")

    args = parser.parse_args()

    # Create config to ensure consistent path handling
    config = AudioLoopConfig.from_project(experiment_name=args.experiment)
    merge_training_sets(args.original_csv, args.candidates_csv, args.output, config=config)


if __name__ == "__main__":
    # Example usage
    print("Active Learning Label Management")
    print("=" * 40)
    print("\nWorkflow:")
    print("1. Run active learning cycle to generate candidates:")
    print("   python -m audioloop.active_learning --class-name siren --run-number 1")
    print("   → Creates outputs/labeling_candidates_v1.csv")
    print()
    print("2. Human fills in 'needs_human_label' column in candidates file:")
    print("   - Open outputs/labeling_candidates_v1.csv")
    print("   - Fill 'needs_human_label' column with 0 or 1")
    print("   - Save the file")
    print()
    print("3. Merge human labels back into training set:")
    print(
        "   python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv"
    )
    print("   → Creates training_sets/training_set_v2.csv with combined data")
    print()
    print("4. Train model and run next cycle:")
    print("   python -m audioloop.train training_sets/training_set_v2.csv -v 2")
    print("   python -m audioloop.active_learning --class-name siren --run-number 2")
    print()
    print("For experiments, use matching directories:")
    print(
        "   python -m audioloop.merge_labels training_sets/exp/training_set_v1.csv outputs/exp/labeling_candidates_v1.csv --experiment exp"
    )
    print()

    main()
