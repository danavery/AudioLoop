"""
Simple model discovery for AudioLoop.

Convention:
- Filename determines model name: my_model.py -> "my_model"
- Class name follows pattern: my_model.py -> MyModel
- Direct import approach - no complex discovery needed
"""

from pathlib import Path

from .audio_loop_model import AudioLoopModel


def get_model_class(name: str) -> type[AudioLoopModel]:
    """Get model class by importing the module and finding AudioLoopModel subclass."""
    try:
        # Direct import: "resnet50" -> import audioloop.models.resnet50
        module = __import__(f"audioloop.models.{name}", fromlist=[name])

        # Find any class that inherits from AudioLoopModel (excluding the base class itself)
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
        # Show available options by scanning directory
        available = list_available_models()
        raise ValueError(
            f"Model '{name}' not found. Available: {', '.join(sorted(available))}\n"
            f"To add '{name}': Create audioloop/models/{name}.py with a class that inherits from AudioLoopModel"
        ) from None


def list_available_models() -> list[str]:
    """List available models by scanning the models directory."""
    models_dir = Path(__file__).parent
    available = []

    for py_file in models_dir.glob("*.py"):
        # Skip special files and templates
        if py_file.name in (
            "__init__.py",
            "audio_loop_model.py",
            "model_registry.py",
        ) or "templates/" in str(py_file):
            continue

        # Extract model name: my_model.py -> my_model
        model_name = py_file.stem
        available.append(model_name)

    return available


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
