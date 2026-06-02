"""
Tests for dataset discovery in dataset_registry.

Focused on the scan-for-subclass behavior: the dataset name selects the file,
and the loaded module is scanned for any DatasetConfig subclass (the class name
no longer has to match a derived pattern).
"""

from pathlib import Path

import pytest

from audioloop.datasets import dataset_registry
from audioloop.datasets.dataset_config import DatasetConfig

# A project dataset file whose class is deliberately NOT named "<Name>Config",
# proving discovery resolves by subclass rather than by a derived class name.
_NONSTANDARD_CONFIG = """\
from audioloop.datasets.dataset_config import DatasetConfig


class TotallyUnrelatedName(DatasetConfig):
    pass
"""

# A file with no DatasetConfig subclass at all.
_EMPTY_CONFIG = """\
class NotADatasetConfig:
    pass
"""


def _write_project_dataset(tmp_path, monkeypatch, filename, contents):
    """Create a project datasets/ dir with one config file and point discovery at it."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / filename).write_text(contents)
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: datasets_dir)
    return datasets_dir


def test_resolves_class_with_nonstandard_name(tmp_path, monkeypatch):
    """A DatasetConfig subclass is found regardless of its class name."""
    _write_project_dataset(tmp_path, monkeypatch, "weird_config.py", _NONSTANDARD_CONFIG)

    cls = dataset_registry.get_dataset_config_class("weird")

    assert cls.__name__ == "TotallyUnrelatedName"
    assert issubclass(cls, DatasetConfig)


def test_missing_subclass_fails_loud(tmp_path, monkeypatch):
    """A project file with no DatasetConfig subclass raises instead of silently falling through."""
    _write_project_dataset(tmp_path, monkeypatch, "empty_config.py", _EMPTY_CONFIG)

    with pytest.raises(ValueError, match="No DatasetConfig subclass found"):
        dataset_registry.get_dataset_config_class("empty")


def test_builtin_datasets_still_resolve(monkeypatch):
    """Built-in datasets resolve via the package-import path (no project dir)."""
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: None)

    for name in ("fsd50k", "urbansound8k", "audioset"):
        cls = dataset_registry.get_dataset_config_class(name)
        assert issubclass(cls, DatasetConfig)
        assert cls is not DatasetConfig


def test_unknown_dataset_raises_with_available_list(monkeypatch):
    """An unknown name raises a helpful error listing available datasets."""
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: None)

    with pytest.raises(ValueError, match="not found"):
        dataset_registry.get_dataset_config_class("does_not_exist")


def test_create_dataset_builds_instance(monkeypatch):
    """create_dataset returns a ready DatasetConfig instance for a built-in dataset."""
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: None)

    dataset = dataset_registry.create_dataset("fsd50k")

    assert isinstance(dataset, DatasetConfig)


def test_create_dataset_missing_subset_csv_raises(monkeypatch):
    """A subset_csv that does not exist raises FileNotFoundError at construction."""
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: None)

    with pytest.raises(FileNotFoundError, match="Subset CSV not found"):
        dataset_registry.create_dataset("fsd50k", subset_csv=Path("/no/such/subset.csv"))


def test_listing_includes_only_files_with_subclass(tmp_path, monkeypatch):
    """A *_config.py file is listed iff it actually contains a DatasetConfig subclass."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "good_config.py").write_text(_NONSTANDARD_CONFIG)  # has a subclass
    (datasets_dir / "broken_config.py").write_text(_EMPTY_CONFIG)  # no subclass
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: datasets_dir)

    names = dataset_registry.list_available_datasets()

    assert "good" in names
    assert "broken" not in names
    # built-ins are still discovered alongside project datasets
    assert {"fsd50k", "urbansound8k", "audioset"} <= set(names)


def test_listing_matches_resolvability(tmp_path, monkeypatch):
    """Listed datasets all resolve; an unlisted broken file does not (one predicate)."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "broken_config.py").write_text(_EMPTY_CONFIG)
    monkeypatch.setattr(dataset_registry, "_get_project_datasets_dir", lambda: datasets_dir)

    names = dataset_registry.list_available_datasets()

    for name in names:
        assert issubclass(dataset_registry.get_dataset_config_class(name), DatasetConfig)

    assert "broken" not in names
    with pytest.raises(ValueError, match="No DatasetConfig subclass found"):
        dataset_registry.get_dataset_config_class("broken")
