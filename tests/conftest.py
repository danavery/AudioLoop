"""
Pytest configuration and fixtures for AudioLoop tests.
"""

import pytest


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
