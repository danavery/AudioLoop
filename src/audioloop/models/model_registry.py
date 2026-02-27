"""
Simple model discovery for AudioLoop.

Convention:
- Filename determines model name: my_model.py -> "my_model"
- Class name can be anything — the system finds the AudioLoopModel subclass automatically
- Direct import approach - no complex discovery needed

Discovery locations (checked in order):
1. Project-level: {project_root}/models/*.py (takes precedence)
2. Built-in: audioloop/models/*.py
"""

import importlib.util
from pathlib import Path

from .audio_loop_model import AudioLoopModel

# Files to skip when scanning for models
_SKIP_FILES = {"__init__.py", "audio_loop_model.py", "model_registry.py"}


def _get_project_models_dir() -> Path | None:
    """Get the project-level models directory, if it exists."""
    try:
        from audioloop.utils.paths import get_project_root

        project_dir = get_project_root() / "models"
        if project_dir.is_dir():
            return project_dir
    except (RuntimeError, ImportError):
        pass
    return None


def _load_class_from_file(file_path: Path) -> type[AudioLoopModel] | None:
    """Dynamically load an AudioLoopModel subclass from an arbitrary file path."""
    module_name = f"audioloop_project_models.{file_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find any class that inherits from AudioLoopModel (excluding the base class)
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, AudioLoopModel)
            and attr != AudioLoopModel
        ):
            return attr
    return None


def get_model_class(name: str) -> type[AudioLoopModel]:
    """Get model class by name.

    Checks project-level models/ directory first, then built-in package models.
    """
    # 1. Try project-level models/ directory first
    project_dir = _get_project_models_dir()
    if project_dir is not None:
        project_file = project_dir / f"{name}.py"
        if project_file.is_file():
            model_class = _load_class_from_file(project_file)
            if model_class is not None:
                return model_class

    # 2. Try built-in package models
    try:
        module = __import__(f"audioloop.models.{name}", fromlist=[name])

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, AudioLoopModel)
                and attr != AudioLoopModel
            ):
                return attr

        raise ValueError(f"No AudioLoopModel subclass found in {name}.py")

    except ImportError:
        available = list_available_models()
        raise ValueError(
            f"Model '{name}' not found. Available: {', '.join(sorted(available))}\n"
            f"To add '{name}': Create models/{name}.py with a class that inherits from AudioLoopModel"
        ) from None


def list_available_models() -> list[str]:
    """List available models by scanning both project and built-in directories."""
    available = set()

    # 1. Scan built-in package directory
    builtin_dir = Path(__file__).parent
    for py_file in builtin_dir.glob("*.py"):
        if py_file.name in _SKIP_FILES or "templates/" in str(py_file):
            continue
        available.add(py_file.stem)

    # 2. Scan project-level models/ directory
    project_dir = _get_project_models_dir()
    if project_dir is not None:
        for py_file in project_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            available.add(py_file.stem)

    return sorted(available)


def get_model_descriptions() -> dict[str, str]:
    """Get descriptions for all available models.

    Returns:
        Dict mapping model names to their descriptions.

    Notes:
        - Accesses class-level description attribute without instantiation
        - Falls back to generic description if attribute missing or import fails
        - Safe to call even if some models have errors
    """
    descriptions = {}
    for model_name in list_available_models():
        try:
            model_class = get_model_class(model_name)
            # Access class-level description attribute
            description = getattr(model_class, "description", f"{model_name} model")
            descriptions[model_name] = description
        except (ImportError, ValueError):
            # If model can't be imported, use generic description
            descriptions[model_name] = f"{model_name} model"

    return descriptions
