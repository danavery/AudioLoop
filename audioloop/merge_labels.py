import argparse
import csv
import os
from pathlib import Path


def candidates_to_labeling_format(candidates_csv, output_csv):
    """
    Convert candidates CSV to simple labeling format for human review.

    Input: candidates CSV with all prediction data
    Output: simple CSV with just filename,label,run for human to fill in
    """
    labeling_data = []

    with open(candidates_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract just the essentials for human labeling
            # Convert from spec filename back to audio filename
            audio_filename = row['filename']

            # Create empty label entry for human to fill
            labeling_data.append({
                'filename': audio_filename,
                'label': '',  # Human will fill this in (0 or 1)
                'run': '1'  # Default run number
            })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['filename', 'label', 'run']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeling_data)

    print(f"Created labeling format file: {output_csv}")
    print(f"Contains {len(labeling_data)} samples for human labeling")
    print("Instructions: Fill in the 'label' column with 0 (negative class) or 1 (positive class)")

    return output_csv


def merge_training_sets(original_csv, new_labels_csv, output_csv):
    """
    Merge original training set with newly labeled samples.

    Args:
        original_csv: Path to existing training set (e.g., training_set_v1.csv)
        new_labels_csv: Path to newly labeled samples from human
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

    # Read new labels
    new_count = 0
    with open(new_labels_csv, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:  # Skip header if it exists
                if row[0].lower() in ['filename', 'filepath']:
                    continue

            if len(row) >= 2:
                filename = row[0]
                label = row[1].strip()

                if label == '':
                    print(f"Warning: Empty label for {filename}, skipping")
                    continue

                try:
                    label_int = int(label)
                    if label_int not in [0, 1]:
                        print(f"Warning: Invalid label '{label}' for {filename}, skipping")
                        continue

                    all_data.append({
                        'filename': filename,
                        'label': label_int,
                        'run': row[2] if len(row) > 2 else '1'
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
    parser = argparse.ArgumentParser(description='Manage active learning label files')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Command to convert candidates to labeling format
    convert_parser = subparsers.add_parser('convert', help='Convert candidates to labeling format')
    convert_parser.add_argument('candidates_csv', help='Input candidates CSV file')
    convert_parser.add_argument('-o', '--output', default='outputs/for_labeling.csv',
                              help='Output CSV for human labeling')

    # Command to merge training sets
    merge_parser = subparsers.add_parser('merge', help='Merge original and new labels')
    merge_parser.add_argument('original_csv', help='Original training set CSV')
    merge_parser.add_argument('new_labels_csv', help='New labels CSV from human')
    merge_parser.add_argument('-o', '--output', help='Output merged training set CSV')

    args = parser.parse_args()

    if args.command == 'convert':
        candidates_to_labeling_format(args.candidates_csv, args.output)

    elif args.command == 'merge':
        if not args.output:
            # Auto-generate output filename
            base = Path(args.original_csv).stem
            if base.startswith('training_set_v'):
                version = int(base.split('_v')[1]) if '_v' in base else 1
                new_version = version + 1
            else:
                new_version = 2
            args.output = f"training_set_v{new_version}.csv"

        merge_training_sets(args.original_csv, args.new_labels_csv, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    # Example usage
    print("Active Learning Label Management")
    print("="*40)
    print("\nExample workflow:")
    print("1. python -m audioloop.merge_labels convert outputs/labeling_candidates.csv")
    print("   → Creates outputs/for_labeling.csv for human to fill in")
    print()
    print("2. Human fills in labels in outputs/for_labeling.csv")
    print()
    print("3. python -m audioloop.merge_labels merge training_set_v1.csv outputs/for_labeling.csv")
    print("   → Creates training_set_v2.csv with combined data")
    print()

    main()
