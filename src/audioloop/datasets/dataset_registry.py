"""
Simple dataset discovery for AudioLoop.

Convention:
- Filename determines dataset name: my_dataset_config.py -> "my_dataset"
- Class name can be anything — the system finds the DatasetConfig subclass automatically
- Direct import approach - no complex discovery needed

Discovery locations (checked in order):
1. Project-level: {project_root}/datasets/*_config.py (takes precedence)
2. Built-in: audioloop/datasets/*_config.py
"""

import importlib.util
from pathlib import Path
from types import ModuleType

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


def _load_dataset_module(name: str) -> ModuleType:
    """Load the module for a dataset by name.

    Project-level datasets/{name}_config.py (loaded by file path) takes precedence
    over the built-in audioloop.datasets.{name}_config package module. This is the
    only part of discovery that legitimately differs between project and built-in
    datasets: project files are loose on disk (loaded by path), built-ins are
    package members (imported so their relative imports resolve).

    Raises:
        ValueError: If no dataset module can be found for the given name.
    """
    # 1. Project-level datasets/ directory first (loaded by file path)
    project_dir = _get_project_datasets_dir()
    if project_dir is not None:
        project_file = project_dir / f"{name}_config.py"
        if project_file.is_file():
            module_name = f"audioloop_project_datasets.{name}_config"
            spec = importlib.util.spec_from_file_location(module_name, project_file)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    # 2. Built-in package datasets (imported by dotted name)
    try:
        module_name = f"{name}_config"
        return __import__(f"audioloop.datasets.{module_name}", fromlist=[module_name])
    except ImportError:
        available = list_available_datasets()
        raise ValueError(
            f"Dataset '{name}' not found. Available: {', '.join(sorted(available))}\n"
            f"To add '{name}': Create datasets/{name}_config.py with a class "
            f"that inherits from DatasetConfig"
        ) from None


def get_dataset_config_class(name: str) -> type[DatasetConfig]:
    """Get dataset config class by name.

    Checks project-level datasets/ directory first, then built-in package datasets.
    The resolved module is scanned for any DatasetConfig subclass, so the class can
    be named anything.

    Raises:
        ValueError: If the module has no DatasetConfig subclass.
    """
    module = _load_dataset_module(name)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, DatasetConfig) and attr is not DatasetConfig:
            return attr

    raise ValueError(f"No DatasetConfig subclass found in {name}_config.py")


def create_dataset(name: str, subset_csv: Path | None = None) -> DatasetConfig:
    """Create a dataset config instance by name.

    The construction counterpart to discovery: resolves the DatasetConfig subclass
    for `name` and instantiates it. If `subset_csv` is given, the dataset is
    restricted to that custom CSV. This is the single construction point for
    datasets, mirroring `create_model` for models.

    Args:
        name: Dataset name (e.g. "fsd50k").
        subset_csv: Optional CSV restricting the dataset to a subset of files.

    Raises:
        ValueError: If no dataset/subclass is found for `name`.
        FileNotFoundError: If `subset_csv` is given but does not exist.
        NotImplementedError: If the dataset does not support custom CSV files.
    """
    config_class = get_dataset_config_class(name)
    dataset_config = config_class()

    if subset_csv is not None:
        if not subset_csv.exists():
            raise FileNotFoundError(f"Subset CSV not found: {subset_csv}")
        dataset_config.set_custom_csv(subset_csv)

    return dataset_config


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
