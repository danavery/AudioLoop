"""
Tests for model discovery in model_registry.

Mirrors tests/datasets/test_dataset_registry.py. Focused on content-based
discovery: a file is a model iff it contains an AudioLoopModel subclass, so there
is no skip-list of infrastructure filenames to maintain.
"""

import pytest

from audioloop.models import model_registry
from audioloop.models.audio_loop_model import AudioLoopModel

# A project model whose class name is deliberately arbitrary.
_MODEL = """\
from audioloop.models.audio_loop_model import AudioLoopModel


class MyCoolModel(AudioLoopModel):
    pass
"""

# A file with no AudioLoopModel subclass.
_NOT_A_MODEL = """\
class JustAHelper:
    pass
"""


def test_builtin_models_listed_without_skip_list(monkeypatch):
    """Built-ins resolve; infra files are excluded by content, not a denylist."""
    monkeypatch.setattr(model_registry, "_get_project_models_dir", lambda: None)

    names = model_registry.list_available_models()

    assert {"cnn5layer", "cnn7layer", "simplecnn"} <= set(names)
    for infra in ("model_registry", "audio_loop_model", "__init__"):
        assert infra not in names


def test_project_model_listed_only_with_subclass(tmp_path, monkeypatch):
    """A project .py file is listed iff it actually contains an AudioLoopModel subclass."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "cool.py").write_text(_MODEL)
    (models_dir / "helper.py").write_text(_NOT_A_MODEL)
    monkeypatch.setattr(model_registry, "_get_project_models_dir", lambda: models_dir)

    names = model_registry.list_available_models()

    assert "cool" in names
    assert "helper" not in names


def test_resolves_model_with_nonstandard_name(tmp_path, monkeypatch):
    """An AudioLoopModel subclass is found regardless of its class name."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "cool.py").write_text(_MODEL)
    monkeypatch.setattr(model_registry, "_get_project_models_dir", lambda: models_dir)

    cls = model_registry.get_model_class("cool")

    assert cls.__name__ == "MyCoolModel"
    assert issubclass(cls, AudioLoopModel)


def test_unknown_model_raises_with_available_list(monkeypatch):
    """An unknown name raises a helpful error listing available models."""
    monkeypatch.setattr(model_registry, "_get_project_models_dir", lambda: None)

    with pytest.raises(ValueError, match="not found"):
        model_registry.get_model_class("does_not_exist")
