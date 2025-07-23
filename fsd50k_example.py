#!/usr/bin/env python3
# type: ignore
"""
Example script showing how to use FSD50K dataset with AudioLoop's binary classification workflow.

This demonstrates the key steps:
1. Create binary labels from FSD50K multi-label data
2. Generate spectrograms for audio files
3. Use with existing training and active learning pipeline

Usage examples:
    # List available classes and semantic groups
    python fsd50k_example.py --list-classes
    python fsd50k_example.py --list-groups

    # Create binary labels for specific class
    python fsd50k_example.py --class Guitar --output outputs/fsd50k_guitar_labels.csv

    # Create binary labels for semantic group
    python fsd50k_example.py --group musical_instruments --output outputs/fsd50k_music_labels.csv

    # Process audio files to spectrograms
    python fsd50k_example.py --create-specs --class Guitar

    # Full workflow example
    python fsd50k_example.py --workflow --class Piano
"""

import argparse

from audioloop.datasets.fsd50k_config import (
    FSD50KConfig,
    list_semantic_groups,
)


def create_binary_labels_example(class_name=None, group_name=None, output_csv=None, split="dev"):
    """Example of creating binary labels from FSD50K."""
    print("Creating FSD50K binary labels...")
    print("-" * 40)

    dataset_config = FSD50KConfig()

    if class_name:
        print(f"Strategy: One-vs-all for class '{class_name}'")
        result_path = dataset_config.create_binary_labels_one_vs_all(
            positive_class=class_name,
            split=split,
            output_csv=output_csv or f"outputs/fsd50k_{class_name.lower()}_binary.csv",
        )
    elif group_name:
        print(f"Strategy: Semantic group '{group_name}'")
        result_path = dataset_config.create_binary_labels_semantic_group(
            group_name=group_name,
            split=split,
            output_csv=output_csv or f"outputs/fsd50k_{group_name}_binary.csv",
        )
    else:
        raise ValueError("Must specify either class_name or group_name")

    print(f"✅ Binary labels created: {result_path}")
    return result_path


def create_spectrograms_example(labels_csv, max_files=100):
    """Example of creating spectrograms from FSD50K audio files."""
    print("Creating spectrograms from FSD50K audio...")
    print("-" * 40)

    dataset_config = FSD50KConfig()

    # Create output directory
    dataset_config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load binary labels to get list of files to process
    import csv

    files_to_process = []

    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_files:
                break

            filename = row["filename"]
            audio_path = dataset_config.get_audio_path(filename)

            files_to_process.append(
                {
                    "filename": filename,
                    "audio_path": audio_path,
                    "label": row["label"],
                    "original_labels": row["original_labels"],
                }
            )

    # Process files
    processed_count = 0
    for file_info in files_to_process:
        if file_info["audio_path"].exists():
            success = dataset_config.process_single_file(file_info, dataset_config.output_dir)
            if success:
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"  Processed {processed_count}/{len(files_to_process)} files...")
        else:
            print(f"  ⚠️ Audio file not found: {file_info['audio_path']}")

    print(f"✅ Successfully processed {processed_count}/{len(files_to_process)} spectrograms")
    print(f"   Saved to: {dataset_config.output_dir}")

    return processed_count


def integration_example(class_name):
    """Example showing integration with existing AudioLoop workflow."""
    print(f"FSD50K Integration Example: {class_name}")
    print("=" * 60)

    # Step 1: Create binary labels
    print("\n1. Creating binary labels...")
    labels_csv = create_binary_labels_example(class_name=class_name)

    # Step 2: Create spectrograms (limited sample)
    print("\n2. Creating spectrograms (sample)...")
    processed_count = create_spectrograms_example(labels_csv, max_files=50)

    if processed_count > 0:
        print("\n3. Next steps for full AudioLoop workflow:")
        print(f"   • Train model: python -m audioloop.train {labels_csv}")
        print(
            f"   • Run active learning: python -m audioloop.active_learning --model outputs/model_v1.pt --class-name {class_name}"
        )
        print(
            "   • Label candidates: python -m audioloop.label_audio outputs/labeling_candidates_v1.csv"
        )
        print(
            f"   • Merge labels: python -m audioloop.merge_labels {labels_csv} outputs/labeling_candidates_v1.csv"
        )
    else:
        print("\n⚠️ No audio files found. Make sure FSD50K audio is available at:")
        dataset_config = FSD50KConfig()
        print(f"   {dataset_config.audio_root}")


def comparison_example():
    """Compare different binary classification strategies."""
    print("FSD50K Binary Classification Strategy Comparison")
    print("=" * 60)

    dataset_config = FSD50KConfig()

    # Example 1: One-vs-all for specific instrument
    print("\n1. One-vs-all: Piano vs everything else")
    piano_csv = dataset_config.create_binary_labels_one_vs_all(
        positive_class="Piano", output_csv="outputs/fsd50k_piano_onevsall.csv"
    )

    # Example 2: Semantic group: All musical instruments
    print("\n2. Semantic group: Musical instruments vs everything else")
    music_csv = dataset_config.create_binary_labels_semantic_group(
        group_name="musical_instruments", output_csv="outputs/fsd50k_music_semantic.csv"
    )

    # Example 3: Custom semantic group
    print("\n3. Custom group: String instruments vs everything else")
    string_instruments = {
        "Guitar",
        "Acoustic_guitar",
        "Electric_guitar",
        "Bass_guitar",
        "Plucked_string_instrument",
        "Bowed_string_instrument",
    }
    strings_csv = dataset_config.create_binary_labels_semantic_group(
        group_name="string_instruments",
        positive_classes=string_instruments,
        output_csv="outputs/fsd50k_strings_custom.csv",
    )

    print("\n📊 Results:")
    print(f"   Piano-only: {piano_csv}")
    print(f"   All music: {music_csv}")
    print(f"   String instruments: {strings_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="FSD50K integration examples for AudioLoop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Information commands
    parser.add_argument("--list-classes", action="store_true", help="List all FSD50K classes")
    parser.add_argument(
        "--list-groups", action="store_true", help="List predefined semantic groups"
    )

    # Binary label creation
    parser.add_argument(
        "--class",
        type=str,
        dest="class_name",
        help="Create binary labels for specific class (one-vs-all)",
    )
    parser.add_argument(
        "--group", type=str, dest="group_name", help="Create binary labels for semantic group"
    )
    parser.add_argument("--output", type=str, help="Output CSV file for binary labels")
    parser.add_argument(
        "--split",
        choices=["dev", "eval"],
        default="dev",
        help="Dataset split to use (default: dev)",
    )

    # Spectrogram creation
    parser.add_argument(
        "--create-specs", action="store_true", help="Create spectrograms from binary labels"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum files to process for spectrograms (default: 100)",
    )

    # Workflow examples
    parser.add_argument(
        "--workflow", action="store_true", help="Run full integration workflow example"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare different binary classification strategies"
    )

    args = parser.parse_args()

    # Handle information commands
    if args.list_classes:
        dataset_config = FSD50KConfig()
        dataset_config.list_classes()
        return None

    if args.list_groups:
        list_semantic_groups()
        return None

    # Handle workflow examples
    if args.compare:
        comparison_example()
        return None

    if args.workflow:
        if not args.class_name:
            print("Error: --workflow requires --class")
            return 1
        integration_example(args.class_name)
        return None

    # Handle binary label creation
    if args.class_name or args.group_name:
        try:
            labels_csv = create_binary_labels_example(
                class_name=args.class_name,
                group_name=args.group_name,
                output_csv=args.output,
                split=args.split,
            )

            # Create spectrograms if requested
            if args.create_specs:
                create_spectrograms_example(labels_csv, args.max_files)

        except Exception as e:
            print(f"Error: {e}")
            return 1
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    exit(main())
