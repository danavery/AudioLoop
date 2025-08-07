#!/usr/bin/env python3
"""
Create targeted subsets of AudioSet unbalanced dataset.

Usage:
    python -m audioloop.create_audioset_subset "Brass instrument" 100000 0.05
    python -m audioloop.create_audioset_subset "Speech" 200000 0.03
"""

import argparse
import random
from pathlib import Path


def create_subset(class_name: str, total_samples: int, positive_ratio: float = 0.05):
    """Create AudioSet subset with controlled positive/negative ratio."""

    # Load AudioSet paths
    from .config import AudioLoopConfig
    from .datasets.audioset_config import AudiosetConfig

    config = AudioLoopConfig(dataset="audioset")
    audioset_config = config.get_dataset_config()

    # Type check - ensure we have the right config type
    if not isinstance(audioset_config, AudiosetConfig):
        raise TypeError(f"Expected AudiosetConfig, got {type(audioset_config)}")

    # Note: ontology loading removed as MID->name conversion is handled by dataset config

    print(
        f"Creating subset for '{class_name}' ({positive_ratio:.1%} positive, {total_samples:,} total)"
    )

    # Parse metadata and split by target class
    positives, negatives = [], []
    debug_count = 0

    # Use the exact same parsing logic as audioset_config.py
    audioset_config = config.get_dataset_config()
    audio_files = audioset_config.load_metadata(split="unbalanced_train")

    print(f"Loaded {len(audio_files)} samples from AudioSet unbalanced")

    for file_info in audio_files:  # Process all available files to find positives
        label_names = file_info.get("labels", [])

        if debug_count < 3:
            print(f"Debug sample {debug_count}: {file_info['ytid']}")
            print(f"  Labels: {label_names}")
            print(f"  Looking for: '{class_name}'")
            print(f"  Found: {class_name in label_names}")
            debug_count += 1

        sample = {
            "ytid": file_info["ytid"],
            "start": file_info["start_seconds"],
            "end": file_info["end_seconds"],
            "labels": ",".join(file_info.get("mids", [])),
        }

        if class_name in label_names:
            positives.append(sample)
        else:
            negatives.append(sample)

    print(f"Found {len(positives):,} positive, {len(negatives):,} negative samples")

    # Calculate target counts
    target_pos = int(total_samples * positive_ratio)
    target_neg = total_samples - target_pos

    # Sample subset
    if len(positives) < target_pos:
        print(f"Warning: Only {len(positives)} positives available, using all")
        target_pos = len(positives)
        target_neg = total_samples - target_pos

    selected_pos = random.sample(positives, target_pos)
    selected_neg = random.sample(negatives, target_neg)
    subset = selected_pos + selected_neg
    random.shuffle(subset)

    # Save subset CSV
    safe_name = class_name.replace(" ", "_").lower()
    subsets_dir = Path("subsets")
    subsets_dir.mkdir(exist_ok=True)
    output_path = subsets_dir / f"audioset_subset_{safe_name}_{total_samples}.csv"

    with output_path.open("w") as f:
        f.write(f"# AudioSet subset: {class_name}, {len(subset)} samples (from unbalanced_train)\n")
        f.write("# SPLIT: unbalanced_train\n")
        for sample in subset:
            f.write(f'{sample["ytid"]}, {sample["start"]}, {sample["end"]}, "{sample["labels"]}"\n')

    print(
        f"Saved {len(subset):,} samples ({len(selected_pos)} pos, {len(selected_neg)} neg) to {output_path}"
    )
    print(
        f"Next: python -m audioloop.create_all_specs --dataset audioset --metadata-file {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Create AudioSet subset")
    parser.add_argument("class_name", help="Target class (e.g., 'Brass instrument')")
    parser.add_argument("total_samples", type=int, help="Total samples in subset")
    parser.add_argument(
        "positive_ratio", type=float, nargs="?", default=0.05, help="Positive ratio (default: 0.05)"
    )

    args = parser.parse_args()

    if not (0 < args.positive_ratio < 1):
        print("Error: positive_ratio must be between 0 and 1")
        return 1

    create_subset(args.class_name, args.total_samples, args.positive_ratio)
    return None


if __name__ == "__main__":
    exit(main())
