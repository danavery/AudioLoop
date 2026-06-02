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
from types import ModuleType

from .audio_loop_model import AudioLoopModel


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


def _load_model_module(name: str) -> ModuleType | None:
    """Load the module for a model by name, or None if it cannot be imported.

    Project-level models/{name}.py (loaded by file path) takes precedence over the
    built-in audioloop.models.{name} package module. This is the only part of
    discovery that legitimately differs between project and built-in models:
    project files are loose on disk (loaded by path), built-ins are package members
    (imported so their relative imports resolve).

    A present-but-broken project file lets its import error propagate, so the user
    sees the real problem. An absent or unimportable built-in module returns None.
    """
    # 1. Project-level models/ directory first (loaded by file path)
    project_dir = _get_project_models_dir()
    if project_dir is not None:
        project_file = project_dir / f"{name}.py"
        if project_file.is_file():
            module_name = f"audioloop_project_models.{name}"
            spec = importlib.util.spec_from_file_location(module_name, project_file)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    # 2. Built-in package models (imported by dotted name)
    try:
        return __import__(f"audioloop.models.{name}", fromlist=[name])
    except ImportError:
        return None


def _find_model_subclass(module: ModuleType) -> type[AudioLoopModel] | None:
    """Return the first AudioLoopModel subclass in a module, or None if there is none."""
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, AudioLoopModel)
            and attr is not AudioLoopModel
        ):
            return attr
    return None


def get_model_class(name: str) -> type[AudioLoopModel]:
    """Get model class by name.

    Checks project-level models/ directory first, then built-in package models.
    The resolved module is scanned for any AudioLoopModel subclass, so the class
    can be named anything.

    Raises:
        ValueError: If no module is found for `name`, or it has no AudioLoopModel
            subclass.
    """
    module = _load_model_module(name)
    if module is None:
        available = list_available_models()
        raise ValueError(
            f"Model '{name}' not found. Available: {', '.join(available)}\n"
            f"To add '{name}': Create models/{name}.py with a class that inherits from AudioLoopModel"
        )

    model_class = _find_model_subclass(module)
    if model_class is not None:
        return model_class

    raise ValueError(f"No AudioLoopModel subclass found in {name}.py")


def list_available_models() -> list[str]:
    """List available models: every file that actually contains an AudioLoopModel subclass.

    Uses the same predicate as get_model_class (load the module, look for a
    subclass), so a name is listed if and only if it resolves. Infrastructure
    files (the base class, this registry, __init__.py) are excluded automatically
    because they contain no concrete subclass; dunder files are skipped outright as
    language infrastructure that can never be a model name.
    """
    candidate_stems = set()
    builtin_dir = Path(__file__).parent
    candidate_stems.update(py_file.stem for py_file in builtin_dir.glob("*.py"))
    project_dir = _get_project_models_dir()
    if project_dir is not None:
        candidate_stems.update(py_file.stem for py_file in project_dir.glob("*.py"))

    available = set()
    for stem in candidate_stems:
        if stem.startswith("__"):
            continue
        try:
            module = _load_model_module(stem)
        except Exception:
            continue  # present-but-broken file: skip from the listing
        if module is not None and _find_model_subclass(module) is not None:
            available.add(stem)

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
