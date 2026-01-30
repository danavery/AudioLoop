"""
Initialize a new AudioLoop project.

Creates the directory structure and configuration file needed for an AudioLoop
project. This allows users to work from their own directory without writing
data to the installed package location.

Usage:
    python -m audioloop.init_project
    python -m audioloop.init_project /path/to/project
    python -m audioloop.init_project --force

Examples:
    # Initialize in current directory
    cd ~/my-audio-project
    python -m audioloop.init_project

    # Initialize in a new directory
    python -m audioloop.init_project ~/projects/dog-classifier

    # Reinitialize (overwrites audioloop.yaml)
    python -m audioloop.init_project --force
"""

import argparse
from pathlib import Path

DIRECTORIES = [
    "data/all_specs",
    "outputs",
    "training_sets",
    "subsets",
    "configs",
]

DEFAULT_CONFIG = """\
# AudioLoop Project Configuration
# ================================
# All settings here override code defaults. CLI arguments override these.
# Uncomment and modify any setting to change the default for this project.
#
# See AudioLoopConfig in src/audioloop/config.py for full documentation.

# ------------------------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------------------------
# dataset: fsd50k                  # Options: fsd50k, urbansound8k, audioset, or your custom dataset
# subset_csv:                      # Restrict to subset CSV (created by create_subset)

# ------------------------------------------------------------------------------
# Training Parameters
# ------------------------------------------------------------------------------
# max_epochs: 1000
# seed: 42
# batch_size: 32
# num_workers: 2                   # Parallel data loading workers
# learning_rate: 0.001
# weight_decay: 0.00001            # L2 regularization

# Model
# model_type: cnn5layer            # Available: cnn5layer, simple_cnn
# class_weighting:                 # null, "adaptive", or float (0.0-1.0)

# ------------------------------------------------------------------------------
# Learning Rate Scheduler
# ------------------------------------------------------------------------------
# use_lr_scheduler: true           # Enable ReduceLROnPlateau
# lr_scheduler_factor: 0.5         # Multiply LR by this on plateau
# lr_scheduler_patience: 5         # Epochs to wait before reducing LR
# lr_scheduler_min_lr: 0.000001    # Minimum learning rate

# ------------------------------------------------------------------------------
# Training Stopping Criteria
# ------------------------------------------------------------------------------
# stopping_criterion_type: plateau # Options: plateau, accuracy, hybrid
# patience: 20                     # Epochs without improvement before stopping
# min_delta: 0.01                  # Minimum change to qualify as improvement
# accuracy_floor:                  # Min accuracy before patience counts (null = auto)

# ------------------------------------------------------------------------------
# Active Learning Parameters
# ------------------------------------------------------------------------------
# total_candidates: 50             # Samples to label per cycle
# selection_mode: entropy          # Options: entropy, confidence, mixed_entropy,
#                                  #          basic_transition, stratified_uncertainty, random
# positive_percentage:             # null = no stratification (pure selection_mode)
# min_confidence: 0.8              # Minimum confidence for candidate selection
# candidate_pool_multiplier: 5     # Select from pool of N * total_candidates

# ------------------------------------------------------------------------------
# Cycle Stopping Criteria (for automated active learning)
# ------------------------------------------------------------------------------
# cycle_stopping_strategy: none    # Options: none, label, search
# cycle_patience: 5                # Cycles without improvement before stopping
# cycle_min_delta: 0.02            # Minimum improvement to reset patience
# cycle_min_cycles: 10             # Minimum cycles before stopping allowed
# cycle_window: 3                  # Cycles to average for rolling metrics
# cycle_std_threshold: 0.08        # Max std dev for "stable" performance
# precision_floor: auto            # Min precision for 'search' mode ("auto" or 0.0-1.0)

# ------------------------------------------------------------------------------
# Evaluation Mode (requires ground truth labels)
# ------------------------------------------------------------------------------
# with_ground_truth: false         # Add ground_truth/correct columns to predictions
"""


def init_project(target_dir: Path | None = None, force: bool = False) -> Path:
    """Initialize an AudioLoop project directory.

    Args:
        target_dir: Directory to initialize. Defaults to current working directory.
        force: If True, overwrite existing audioloop.yaml.

    Returns:
        Path to the initialized project directory.
    """
    project_dir = (target_dir or Path.cwd()).resolve()
    config_file = project_dir / "audioloop.yaml"

    if config_file.exists() and not force:
        print(f"Project already exists: {project_dir}")
        print("Use --force to reinitialize.")
        return project_dir

    # Create directories
    for subdir in DIRECTORIES:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)
        print(f"  Created {subdir}/")

    # Write config file
    config_file.write_text(DEFAULT_CONFIG)
    print("  Created audioloop.yaml")

    print(f"\nInitialized AudioLoop project in {project_dir}")
    return project_dir


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new AudioLoop project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize in current directory
  python -m audioloop.init_project

  # Initialize in specific directory
  python -m audioloop.init_project /path/to/my-project

  # Reinitialize existing project
  python -m audioloop.init_project --force
        """,
    )

    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Target directory (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinitialize existing project (overwrites audioloop.yaml)",
    )

    args = parser.parse_args()
    init_project(target_dir=args.directory, force=args.force)


if __name__ == "__main__":
    main()
