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
import shutil
from pathlib import Path

DIRECTORIES = [
    "data/feature_cache",
    "datasets",
    "outputs",
    "training_sets",
    "subsets",
    "configs",
]

DEFAULT_CONFIG_PATH = Path(__file__).parent / "templates" / "audioloop.yaml"


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

    # Copy dataset templates
    templates_src = Path(__file__).parent / "datasets" / "templates"
    templates_dst = project_dir / "datasets" / "templates"
    if templates_src.is_dir() and not templates_dst.exists():
        shutil.copytree(templates_src, templates_dst, ignore=shutil.ignore_patterns("__pycache__"))
        print("  Copied datasets/templates/")

    # Write config file
    config_file.write_text(DEFAULT_CONFIG_PATH.read_text())
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
