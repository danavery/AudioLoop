import argparse
import csv
import os
from pathlib import Path


def merge_training_sets(original_csv, new_labels_csv, output_csv):
    """
    Merge original training set with newly labeled samples.

    Args:
        original_csv: Path to existing training set (e.g., training_set_v1.csv)
        new_labels_csv: Path to newly labeled samples (candidates CSV with human labels)
        output_csv: Path for merged training set (e.g., training_set_v2.csv)
    """
    all_data = []

    # Read original training set
    if os.path.exists(original_csv):
        with open(original_csv, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:  # Skip header if it exists
                    if row[0].lower() in ['filename', 'filepath']:
                        continue
                if len(row) >= 2:
                    # Handle both filename and full filepath
                    filepath = row[0]
                    filename = os.path.basename(filepath) if filepath.startswith('/') else filepath

                    all_data.append({
                        'filename': filename,
                        'label': int(row[1]),
                        'run': row[2] if len(row) > 2 else '1'
                    })
        print(f"Loaded {len(all_data)} samples from {original_csv}")
    else:
        print(f"Warning: {original_csv} not found, starting fresh")

    # Read new labels from candidates CSV format
    new_count = 0
    with open(new_labels_csv, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Handle candidates CSV format (with needs_human_label column)
            if 'needs_human_label' not in row or 'filename' not in row:
                print("Error: Expected candidates CSV format with 'needs_human_label' and 'filename' columns")
                print(f"Found columns: {list(row.keys())}")
                continue

            filename = os.path.basename(row['filename']) if row['filename'] else ''
            label = row['needs_human_label'].strip()

            if not filename or label == '':
                continue  # Skip unlabeled samples

            try:
                label_int = int(label)
                if label_int not in [0, 1]:
                    print(f"Warning: Invalid label '{label}' for {filename}, skipping")
                    continue

                # Get run number from row if available
                run = row.get('run', '1')

                all_data.append({
                    'filename': filename,
                    'label': label_int,
                    'run': run
                })
                new_count += 1
            except ValueError:
                print(f"Warning: Invalid label '{label}' for {filename}, skipping")
                continue

    print(f"Added {new_count} new labeled samples")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    # Write merged training set
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['filename', 'label', 'run']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"Created merged training set: {output_csv}")
    print(f"Total samples: {len(all_data)} (original: {len(all_data)-new_count}, new: {new_count})")

    # Show label distribution
    label_counts = {0: 0, 1: 0}
    for item in all_data:
        label_counts[item['label']] += 1

    print(f"Label distribution: {label_counts[0]} negative class, {label_counts[1]} positive class")

    return output_csv


def main():
    parser = argparse.ArgumentParser(description='Merge human labels from active learning candidates into training sets')
    parser.add_argument('original_csv', help='Original training set CSV')
    parser.add_argument('candidates_csv', help='Candidates CSV with filled needs_human_label column')
    parser.add_argument('-o', '--output', help='Output merged training set CSV')

    args = parser.parse_args()

    if not args.output:
        # Auto-generate version number
        base = Path(args.original_csv).stem
        if base.startswith('training_set_v'):
            version = int(base.split('_v')[1]) if '_v' in base else 1
            new_version = version + 1
        else:
            new_version = 2
        output_dir = "training_sets"
        os.makedirs(output_dir, exist_ok=True)
        args.output = f"{output_dir}/training_set_v{new_version}.csv"

    merge_training_sets(args.original_csv, args.candidates_csv, args.output)


if __name__ == "__main__":
    # Example usage
    print("Active Learning Label Management")
    print("="*40)
    print("\nWorkflow:")
    print("1. Run active learning cycle to generate candidates:")
    print("   python -m audioloop.run_active_learning --class-name siren")
    print("   → Creates outputs/labeling_candidates_v1.csv")
    print()
    print("2. Human fills in 'needs_human_label' column in candidates file:")
    print("   - Open outputs/labeling_candidates_v1.csv")
    print("   - Fill 'needs_human_label' column with 0 or 1")
    print("   - Save the file")
    print()
    print("3. Merge human labels back into training set:")
    print("   python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv")
    print("   → Creates training_sets/training_set_v2.csv with combined data")
    print()
    print("4. Use new training set for next cycle:")
    print("   python -m audioloop.run_active_learning --class-name siren --model outputs/model_v2.pt")
    print()

    main()
