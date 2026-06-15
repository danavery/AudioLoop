#!/usr/bin/env python3
"""
Prepare subset-specific feature-cache directories for efficient remote deployment.

Creates a directory containing only the cached features needed for a specific subset,
using hard links (zero storage overhead) or copies as fallback. This enables
efficient syncing to remote training environments without transferring feature_cache.

Workflow:
    1. Local: Create subset directory with hard links from feature_cache
    2. Sync: rsync subset directory to remote pod
    3. Remote: Train using subset directory as feature_cache_dir

Usage:
    python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv
    python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv --output-dir data/subset_features/dog_100k
    python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv --use-copy
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

from .config import AudioLoopConfig


def prepare_subset_features(
    subset_csv: Path,
    output_dir: Path,
    source_feature_cache_dir: Path,
    cache_subdir: str,
    use_copy: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Prepare a subset-specific feature-cache directory.

    Cached features live under a per-extractor subdir (`cache_subdir`) of the cache root, so
    both the source and the destination are namespaced by it: files are read from
    `source_feature_cache_dir/<cache_subdir>/` and written to `output_dir/<cache_subdir>/`. This keeps
    the destination a valid cache root — after syncing `output_dir/` to the remote's feature_cache_dir,
    the remote dataset finds features at `feature_cache_dir/<cache_subdir>/`. The extractor.json manifest
    is copied alongside so the remote's `ensure_cache_dir` validation passes.

    Args:
        subset_csv: Path to subset CSV file (with 'filename' column)
        output_dir: Cache root to create subset specs in (the per-extractor subdir is created under it)
        source_feature_cache_dir: Source cache root containing per-extractor subdirs (e.g., feature_cache)
        cache_subdir: The extractor's cache subdir name (e.g. "spectrogram", "embed_wav2vec2")
        use_copy: If True, copy files instead of hard linking
        verbose: Print progress information

    Returns:
        dict: Statistics about the operation (linked, copied, missing, errors)
    """
    stats = {
        "total": 0,
        "linked": 0,
        "copied": 0,
        "missing": 0,
        "errors": 0,
        "skipped_existing": 0,
    }

    # Namespace both source and destination by the extractor's cache subdir.
    source_root = source_feature_cache_dir / cache_subdir
    dest_root = output_dir / cache_subdir
    dest_root.mkdir(parents=True, exist_ok=True)

    # Copy the manifest so the remote's ensure_cache_dir validation passes.
    source_manifest = source_root / "extractor.json"
    if source_manifest.exists():
        shutil.copy2(source_manifest, dest_root / "extractor.json")

    # Read subset CSV to get filenames
    filenames = []
    with open(subset_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            # Convert audio extensions to .pt
            for ext in [".wav", ".flac", ".mp3", ".ogg"]:
                filename = filename.replace(ext, ".pt")
            filenames.append(filename)

    stats["total"] = len(filenames)

    if verbose:
        print(f"Preparing subset specs for {len(filenames)} files...")
        print(f"Source: {source_root}")
        print(f"Output: {dest_root}")
        print(f"Method: {'copy' if use_copy else 'hard link'}")
        print()

    # Process each file
    for i, filename in enumerate(filenames, 1):
        source_path = source_root / filename
        dest_path = dest_root / filename

        # Skip if destination already exists
        if dest_path.exists():
            stats["skipped_existing"] += 1
            if verbose and i % 1000 == 0:
                print(f"Progress: {i}/{len(filenames)} ({stats['skipped_existing']} existing)")
            continue

        # Check if source exists
        if not source_path.exists():
            stats["missing"] += 1
            if verbose:
                print(f"Missing: {filename}")
            continue

        try:
            if use_copy:
                # Copy file
                shutil.copy2(source_path, dest_path)
                stats["copied"] += 1
            else:
                # Try hard link first
                try:
                    os.link(source_path, dest_path)
                    stats["linked"] += 1
                except OSError as e:
                    # Hard link failed (different filesystem?), fall back to copy
                    if verbose:
                        print(f"Hard link failed for {filename}, falling back to copy: {e}")
                    shutil.copy2(source_path, dest_path)
                    stats["copied"] += 1

            # Progress update every 1000 files
            if verbose and i % 1000 == 0:
                print(f"Progress: {i}/{len(filenames)}")

        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"Error processing {filename}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare subset-specific feature-cache directory for efficient remote deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create subset specs directory with hard links (zero storage overhead)
  python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv

  # Custom output directory
  python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv \\
      --output-dir data/subset_features/dog_100k

  # Use copying instead of hard links (for different filesystems)
  python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv --use-copy

  # Specify custom source specs directory
  python -m audioloop.prepare_subset_features subsets/audioset_dog_100k.csv \\
      --feature-cache-dir /path/to/feature_cache

Deployment Workflow:
  1. Create subset-specific specs directory (this tool)
  2. Sync to remote: rsync -avz --no-o --no-g data/subset_features/dog_100k/ pod:/workspace/data/feature_cache/
  3. Train on remote: python -m audioloop.train subset.csv --feature-cache-dir /workspace/data/feature_cache

Note: --no-o --no-g prevents chown permission errors on remote systems
        """,
    )

    parser.add_argument(
        "subset_csv",
        type=Path,
        help="Path to subset CSV file (must have 'filename' column)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for subset specs (default: data/subset_features/SUBSET_NAME)",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Source specs directory (default: from AudioLoopConfig, usually data/feature_cache)",
    )
    parser.add_argument(
        "--use-copy",
        action="store_true",
        help="Copy files instead of hard linking (use if source/dest on different filesystems)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate subset CSV exists
    if not args.subset_csv.exists():
        parser.error(f"Subset CSV not found: {args.subset_csv}")

    # Config is needed both for the default source dir and for the extractor's cache subdir.
    config = AudioLoopConfig()

    # Get source specs directory / cache root (from config if not specified)
    source_feature_cache_dir = args.feature_cache_dir or config.feature_cache_dir

    # The extractor owns the cache subdir name; source/dest are namespaced by it.
    cache_subdir = config.get_feature_extractor().cache_subdir

    # Validate source directory exists
    if not source_feature_cache_dir.exists():
        parser.error(f"Source specs directory not found: {source_feature_cache_dir}")

    # Determine output directory (from subset CSV name if not specified)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: data/subset_features/SUBSET_NAME (without .csv extension)
        subset_name = args.subset_csv.stem
        output_dir = Path("data/subset_features") / subset_name

    # Prepare subset specs
    stats = prepare_subset_features(
        subset_csv=args.subset_csv,
        output_dir=output_dir,
        source_feature_cache_dir=source_feature_cache_dir,
        cache_subdir=cache_subdir,
        use_copy=args.use_copy,
        verbose=not args.quiet,
    )

    # Print summary
    if not args.quiet:
        print()
        print("=" * 60)
        print("Summary:")
        print(f"  Total files in subset: {stats['total']}")
        print(f"  Hard linked: {stats['linked']}")
        print(f"  Copied: {stats['copied']}")
        print(f"  Already existed (skipped): {stats['skipped_existing']}")
        print(f"  Missing in source: {stats['missing']}")
        print(f"  Errors: {stats['errors']}")
        print()

        successful = stats["linked"] + stats["copied"] + stats["skipped_existing"]
        print(f"  ✓ Successfully prepared: {successful}/{stats['total']} files")

        if stats["missing"] > 0:
            print(f"  ⚠ Warning: {stats['missing']} files missing in source directory")
        if stats["errors"] > 0:
            print(f"  ✗ Errors: {stats['errors']} files failed to process")

        print()
        print(f"Output directory: {output_dir.absolute()}")

        # Storage efficiency note
        if stats["linked"] > 0:
            print()
            print("Note: Hard links use zero additional storage (same inode as source).")
            print("      Deleting this directory won't affect the source specs.")


if __name__ == "__main__":
    main()
