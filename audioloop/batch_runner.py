#!/usr/bin/env python3
"""
Batch runner for running multiple YAML configurations sequentially.

Outputs are organized into batch directories:
    outputs/batch_20251215_143022/
        search_mode/
        label_mode/
        minimal/

Usage:
    # Basic usage
    python -m audioloop.batch_runner config1.yaml config2.yaml config3.yaml

    # With shared initial training set
    python -m audioloop.batch_runner \
        --initial-training-set training_sets/shared/bootstrap.csv \
        configs/*.yaml
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from audioloop.automated_workflow import load_workflow_params, run_automated_workflow
from audioloop.config import AudioLoopConfig


def run_single_config(config_path: Path, batch_id: str, initial_training_set: Path | None = None) -> dict:
    """
    Run a single config file within a batch.

    Args:
        config_path: Path to YAML config file
        batch_id: Batch identifier (e.g., "batch_20251215_143022")
        initial_training_set: Optional path to shared initial training set to copy
                             into each experiment's training_sets directory as
                             training_set_v1.csv (skipped if v1 already exists)

    Returns:
        dict with keys: success (bool), error (str|None), runtime_seconds (float),
                       experiment_name (str), num_cycles (int)
    """
    start_time = time.time()

    try:
        # Load workflow params
        workflow_params = load_workflow_params(config_path)

        # Load config and override experiment_name to nest under batch
        config = AudioLoopConfig.from_yaml(config_path)
        original_experiment_name = config.experiment_name

        # Nest experiment under batch directory
        batched_experiment_name = f"{batch_id}/{original_experiment_name}"
        config = AudioLoopConfig.from_yaml(config_path, experiment_name=batched_experiment_name)

        # Copy initial training set if provided
        if initial_training_set:
            training_set_v1 = config.get_training_set_path(1)
            training_set_v1.parent.mkdir(parents=True, exist_ok=True)

            if not training_set_v1.exists():
                print(f"   Copying initial training set: {initial_training_set} -> {training_set_v1}")
                shutil.copy2(initial_training_set, training_set_v1)
            else:
                print(f"   Using existing training set: {training_set_v1}")

        # Save batch run metadata to output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        run_metadata = {
            "batch_id": batch_id,
            "config_file": str(config_path.absolute()),
            "original_experiment_name": original_experiment_name,
            "batched_experiment_name": batched_experiment_name,
            "initial_training_set": str(initial_training_set.absolute()) if initial_training_set else None,
            "workflow_params": workflow_params,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save JSON with full metadata
        metadata_file = config.output_dir / "batch_run_info.json"
        with open(metadata_file, "w") as f:
            json.dump(run_metadata, f, indent=2)

        # Also copy the original config file for reproducibility
        config_copy = config.output_dir / "config.yaml"
        shutil.copy2(config_path, config_copy)

        # Extract workflow params with defaults
        class_name = workflow_params.get("class_name")
        if not class_name:
            raise ValueError(
                f"Config {config_path} missing required 'class_name' in workflow section"
            )

        num_cycles = workflow_params.get("num_cycles", 3)
        start_cycle = workflow_params.get("start_cycle", 1)
        auto_label = workflow_params.get("auto_label", False)
        evaluation_mode = workflow_params.get("evaluation_mode", False)
        audio_dir = workflow_params.get("audio_dir")

        # Run workflow
        print(f"\n{'='*80}")
        print(f"Running: {config_path}")
        print(f"Experiment: {original_experiment_name} (in batch {batch_id})")
        print(f"Class: {class_name}, Cycles: {num_cycles}")
        print(f"{'='*80}\n")

        run_automated_workflow(
            config=config,
            class_name=class_name,
            num_cycles=num_cycles,
            start_cycle=start_cycle,
            auto_label=auto_label,
            evaluation_mode=evaluation_mode,
            audio_dir=audio_dir,
            seed=config.seed,
        )

        runtime = time.time() - start_time

        return {
            "success": True,
            "error": None,
            "runtime_seconds": runtime,
            "experiment_name": original_experiment_name,
            "num_cycles": num_cycles,
        }

    except Exception as e:
        runtime = time.time() - start_time
        error_msg = f"{type(e).__name__}: {e!s}"

        print(f"\n❌ Error in {config_path}:")
        print(f"   {error_msg}")
        print("\nTraceback:")
        traceback.print_exc()
        print()

        return {
            "success": False,
            "error": error_msg,
            "runtime_seconds": runtime,
            "experiment_name": "unknown",
            "num_cycles": 0,
        }


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple YAML configurations sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run specific configs with shared initial training set
    python -m audioloop.batch_runner \
        --initial-training-set training_sets/shared/bootstrap_dog_40.csv \
        config1.yaml config2.yaml

    # Run all configs in a directory
    python -m audioloop.batch_runner \
        --initial-training-set training_sets/shared/bootstrap.csv \
        configs/sweep/*.yaml

Each config file should contain both 'workflow' and 'config' sections.
The initial training set is copied to each experiment's directory as training_set_v1.csv.
Outputs are organized into timestamped batch directories:
    outputs/batch_20251215_143022/{experiment_name}/
    training_sets/batch_20251215_143022/{experiment_name}/
        """,
    )

    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="YAML configuration files to run",
    )

    parser.add_argument(
        "--initial-training-set",
        type=Path,
        help="Path to initial training_set_v1.csv to copy into each experiment directory",
    )

    args = parser.parse_args()

    # Validate all files exist
    missing_files = [f for f in args.configs if not f.exists()]
    if missing_files:
        print("❌ Config files not found:")
        for f in missing_files:
            print(f"   {f}")
        return 1

    # Validate initial training set if provided
    if args.initial_training_set and not args.initial_training_set.exists():
        print(f"❌ Initial training set not found: {args.initial_training_set}")
        return 1

    # Generate batch ID from timestamp
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run batch
    print(f"\n{'='*80}")
    print(f"BATCH RUNNER - {len(args.configs)} configurations")
    print(f"Batch ID: {batch_id}")
    print(f"{'='*80}")

    batch_start = time.time()
    results = []

    for i, config_path in enumerate(args.configs, 1):
        print(f"\n[{i}/{len(args.configs)}] {config_path.name}")
        result = run_single_config(config_path, batch_id, args.initial_training_set)
        result["config_file"] = str(config_path)
        results.append(result)

    batch_runtime = time.time() - batch_start

    # Print summary
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nBatch ID: {batch_id}")
    print(f"Output directory: outputs/{batch_id}/")
    print(f"\nCompleted: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    print(f"Total runtime: {format_duration(batch_runtime)}")

    if successful:
        print("\n✅ Successful runs:")
        for r in successful:
            print(f"   {Path(r['config_file']).name}")
            print(f"      Experiment: {r['experiment_name']}")
            print(f"      Runtime: {format_duration(r['runtime_seconds'])}")

    if failed:
        print("\n❌ Failed runs:")
        for r in failed:
            print(f"   {Path(r['config_file']).name}")
            print(f"      Error: {r['error']}")

    print(f"\n{'='*80}\n")

    # Exit with error code if any failed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
