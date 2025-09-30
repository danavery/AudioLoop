import argparse
import csv
import logging
import os

from .config import AudioLoopConfig
from .utils.paths import extract_version_from_filename

# Set up logger for this module
logger = logging.getLogger(__name__)


def merge_training_sets(
    original_csv, new_labels_csv, output_csv=None, config=None, log_level=logging.INFO
):
    """
    Merge original training set with newly labeled samples.

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

    all_data = []

    # Read original training set
    if os.path.exists(original_csv):
        with open(original_csv) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0 and row[0].lower() in ["filename", "filepath"]:
                    continue
                if len(row) >= 2:
                    # Handle both filename and full filepath
                    filepath = row[0]
                    filename = os.path.basename(filepath) if filepath.startswith("/") else filepath

                    all_data.append(
                        {
                            "filename": filename,
                            "label": int(row[1]),
                        }
                    )
        logger.info(f"Loaded {len(all_data)} samples from {original_csv}")
    else:
        logger.warning(f"Warning: {original_csv} not found, starting fresh")

    # Read new labels from candidates CSV format
    new_count = 0
    new_positive = 0
    new_negative = 0
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

                all_data.append({"filename": filename, "label": label_int})
                new_count += 1
                if label_int == 1:
                    new_positive += 1
                else:
                    new_negative += 1
            except ValueError:
                logger.warning(f"Warning: Invalid label '{label}' for {filename}, skipping")
                continue

    logger.info(
        f"Added {new_count} new labeled samples ({new_positive} positive, {new_negative} negative)"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    # Write merged training set
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["filename", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    logger.info(f"Created merged training set: {output_csv}")
    logger.info(
        f"Total samples: {len(all_data)} (original: {len(all_data) - new_count}, new: {new_count})"
    )

    # Show label distribution
    label_counts = {0: 0, 1: 0}
    for item in all_data:
        label_counts[item["label"]] += 1

    logger.info(
        f"Label distribution: {label_counts[0]} negative class, {label_counts[1]} positive class"
    )

    # Display final training set composition (always visible, not just at INFO level)
    print(
        f"New training set: {len(all_data)} total ({label_counts[1]} positive, {label_counts[0]} negative)"
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
    config = AudioLoopConfig(experiment_name=args.experiment)
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
