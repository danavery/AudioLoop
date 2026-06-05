"""
Pytest configuration and fixtures for AudioLoop tests.
"""

import csv

import pytest


@pytest.fixture
def candidates_csv(tmp_path):
    """Factory that writes a candidates CSV under tmp_path and returns its Path.

    Replaces the per-file `create_test_candidates_csv` helpers (previously duplicated across
    the auto-labeling / metrics / candidate-metrics suites, each leaking a delete=False temp
    file). Callers pass explicit `fieldnames` because the column sets differ per function
    under test; each row dict is padded with empty strings for any unspecified columns.
    """

    def _write(rows, fieldnames, *, name="candidates.csv"):
        path = tmp_path / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                full_row = dict.fromkeys(fieldnames, "")
                full_row.update(row)
                writer.writerow(full_row)
        return path

    return _write


@pytest.fixture(autouse=True)
def setup_project_root(tmp_path, monkeypatch):
    """Set up a temporary project root for all tests.

    This fixture runs automatically for every test, ensuring that
    get_project_root() works without requiring an audioloop.yaml file
    in the actual working directory.
    """
    # Clear any cached project root from previous tests
    from audioloop.utils.paths import clear_project_root_cache

    clear_project_root_cache()

    # Set the project root to the temp directory
    monkeypatch.setenv("AUDIOLOOP_PROJECT_ROOT", str(tmp_path))

    # Create the basic directory structure
    (tmp_path / "data" / "all_specs").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "training_sets").mkdir()

    yield tmp_path

    # Clear cache after test completes
    clear_project_root_cache()
