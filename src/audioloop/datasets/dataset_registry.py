"""
Simple dataset discovery for AudioLoop.

Convention:
- Filename determines dataset name: my_dataset_config.py -> "my_dataset"
- Class name follows pattern: my_dataset_config.py -> MyDatasetConfig
- Direct import approach - no complex discovery needed

Discovery locations (checked in order):
1. Project-level: {project_root}/datasets/*_config.py (takes precedence)
2. Built-in: audioloop/datasets/*_config.py
"""

import importlib.util
from pathlib import Path

from .dataset_config import DatasetConfig


def _get_project_datasets_dir() -> Path | None:
    """Get the project-level datasets directory, if it exists."""
    try:
        from audioloop.utils.paths import get_project_root

        project_dir = get_project_root() / "datasets"
        if project_dir.is_dir():
            return project_dir
    except (RuntimeError, ImportError):
        pass
    return None


def _derive_class_name(name: str) -> str:
    """Derive expected class name from dataset name.

    Handles special cases for built-in datasets with non-standard casing.
    """
    if name == "fsd50k":
        return "FSD50KConfig"
    if name == "urbansound8k":
        return "UrbanSound8KConfig"
    return f"{''.join(word.capitalize() for word in name.split('_'))}Config"


def _load_class_from_file(file_path: Path, name: str) -> type[DatasetConfig] | None:
    """Dynamically load a DatasetConfig subclass from an arbitrary file path."""
    class_name = _derive_class_name(name)
    module_name = f"audioloop_project_datasets.{name}_config"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, class_name):
        return getattr(module, class_name)
    return None


def get_dataset_config_class(name: str) -> type[DatasetConfig]:
    """Get dataset config class by name.

    Checks project-level datasets/ directory first, then built-in package datasets.
    """
    # 1. Try project-level datasets/ directory first
    project_dir = _get_project_datasets_dir()
    if project_dir is not None:
        project_file = project_dir / f"{name}_config.py"
        if project_file.is_file():
            config_class = _load_class_from_file(project_file, name)
            if config_class is not None:
                return config_class

    # 2. Try built-in package datasets
    try:
        module_name = f"{name}_config"
        module = __import__(f"audioloop.datasets.{module_name}", fromlist=[module_name])

        class_name = _derive_class_name(name)
        if hasattr(module, class_name):
            return getattr(module, class_name)
        raise ValueError(f"Expected class '{class_name}' not found in {module_name}.py")

    except ImportError:
        available = list_available_datasets()
        raise ValueError(
            f"Dataset '{name}' not found. Available: {', '.join(sorted(available))}\n"
            f"To add '{name}': Create datasets/{name}_config.py with class "
            f"{''.join(word.capitalize() for word in name.split('_'))}Config"
        ) from None


def list_available_datasets() -> list[str]:
    """List available datasets by scanning both project and built-in directories."""
    available = set()

    # 1. Scan built-in package directory
    builtin_dir = Path(__file__).parent
    for py_file in builtin_dir.glob("*_config.py"):
        if py_file.name == "dataset_config.py" or "templates/" in str(py_file):
            continue
        available.add(py_file.stem.replace("_config", ""))

    # 2. Scan project-level datasets/ directory
    project_dir = _get_project_datasets_dir()
    if project_dir is not None:
        for py_file in project_dir.glob("*_config.py"):
            available.add(py_file.stem.replace("_config", ""))

    return sorted(available)
