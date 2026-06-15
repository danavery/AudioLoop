"""
Tests for AudioLoop path utilities.

Focused on environment variable handling and key path generation behavior.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from audioloop.utils.paths import (
    clear_project_root_cache,
    create_output_directories,
    get_data_root,
    get_feature_cache_dir,
    get_output_dir,
    get_output_root,
    get_project_root,
    get_training_sets_dir,
)


class TestProjectRoot:
    """Test project root detection."""

    def test_project_root_from_env_var(self, tmp_path, monkeypatch):
        """Test that AUDIOLOOP_PROJECT_ROOT env var is used."""
        clear_project_root_cache()
        monkeypatch.setenv("AUDIOLOOP_PROJECT_ROOT", str(tmp_path))
        assert get_project_root() == tmp_path

    def test_project_root_from_yaml(self, tmp_path, monkeypatch):
        """Test that audioloop.yaml is detected."""
        clear_project_root_cache()
        monkeypatch.delenv("AUDIOLOOP_PROJECT_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "audioloop.yaml").write_text("dataset: fsd50k\n")
        assert get_project_root() == tmp_path

    def test_project_root_missing_raises_error(self, tmp_path, monkeypatch):
        """Test that missing project raises RuntimeError."""
        clear_project_root_cache()
        monkeypatch.delenv("AUDIOLOOP_PROJECT_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="No AudioLoop project found"):
            get_project_root()


class TestEnvironmentVariables:
    """Test environment variable handling for paths."""

    def test_data_root_env_override(self, monkeypatch):
        """Test that AUDIOLOOP_DATA_ROOT overrides project-based default."""
        monkeypatch.setenv("AUDIOLOOP_DATA_ROOT", "/custom/data")
        assert get_data_root() == Path("/custom/data")

    def test_output_root_env_override(self, monkeypatch):
        """Test that AUDIOLOOP_OUTPUT_ROOT overrides project-based default."""
        monkeypatch.setenv("AUDIOLOOP_OUTPUT_ROOT", "/custom/output")
        assert get_output_root() == Path("/custom/output")

    def test_data_root_uses_project_root(self, tmp_path, monkeypatch):
        """Test that data root uses project root when no env var set."""
        clear_project_root_cache()
        monkeypatch.delenv("AUDIOLOOP_DATA_ROOT", raising=False)
        monkeypatch.setenv("AUDIOLOOP_PROJECT_ROOT", str(tmp_path))
        assert get_data_root() == tmp_path / "data"

    def test_output_root_uses_project_root(self, tmp_path, monkeypatch):
        """Test that output root uses project root when no env var set."""
        clear_project_root_cache()
        monkeypatch.delenv("AUDIOLOOP_OUTPUT_ROOT", raising=False)
        monkeypatch.setenv("AUDIOLOOP_PROJECT_ROOT", str(tmp_path))
        assert get_output_root() == tmp_path

    def test_feature_cache_dir_environment_combinations(self, tmp_path, monkeypatch):
        """Test specs directory with different environment combinations."""
        clear_project_root_cache()
        monkeypatch.setenv("AUDIOLOOP_PROJECT_ROOT", str(tmp_path))

        # Default (project-relative)
        monkeypatch.delenv("AUDIOLOOP_DATA_ROOT", raising=False)
        monkeypatch.delenv("AUDIOLOOP_SPECS_DIR", raising=False)
        assert get_feature_cache_dir() == tmp_path / "data" / "feature_cache"

        # Custom data root only
        monkeypatch.setenv("AUDIOLOOP_DATA_ROOT", "/custom")
        assert get_feature_cache_dir() == Path("/custom/feature_cache")

        # Custom specs subdir only
        monkeypatch.delenv("AUDIOLOOP_DATA_ROOT", raising=False)
        monkeypatch.setenv("AUDIOLOOP_SPECS_DIR", "spectrograms")
        assert get_feature_cache_dir() == tmp_path / "data" / "spectrograms"

        # Both custom
        monkeypatch.setenv("AUDIOLOOP_DATA_ROOT", "/custom")
        monkeypatch.setenv("AUDIOLOOP_SPECS_DIR", "spectrograms")
        assert get_feature_cache_dir() == Path("/custom/spectrograms")


class TestDirectoryGeneration:
    """Test directory path generation."""

    @pytest.mark.parametrize(
        "experiment,expected_path",
        [
            (None, "outputs"),
            ("test_exp", "outputs/test_exp"),
        ],
    )
    def test_output_directory_names(self, experiment, expected_path):
        """Test output directory generation with different experiment names."""
        result = get_output_dir(experiment)
        if experiment is None:
            assert result.name == expected_path
        else:
            # For experiments, check nested structure
            assert str(result).endswith(expected_path)
            assert result.parent.name == "outputs"
            assert result.name == experiment

    @pytest.mark.parametrize(
        "experiment,expected_path",
        [
            (None, "training_sets"),
            ("test_exp", "training_sets/test_exp"),
        ],
    )
    def test_training_sets_directory_names(self, experiment, expected_path):
        """Test training sets directory generation."""
        result = get_training_sets_dir(experiment)
        if experiment is None:
            assert result.name == expected_path
        else:
            # For experiments, check nested structure
            assert str(result).endswith(expected_path)
            assert result.parent.name == "training_sets"
            assert result.name == experiment

    def test_directory_consistency(self):
        """Test that output and training directories are consistent."""
        # Same experiment should produce consistent naming pattern
        exp_name = "test_experiment"
        output_dir = get_output_dir(exp_name)
        training_dir = get_training_sets_dir(exp_name)

        assert "test_experiment" in str(output_dir)
        assert "test_experiment" in str(training_dir)
        # Both should be nested under the same root, but in different base directories
        assert output_dir.parent.parent == training_dir.parent.parent
        assert output_dir.parent.name == "outputs"
        assert training_dir.parent.name == "training_sets"
        assert output_dir.name == exp_name
        assert training_dir.name == exp_name


class TestDirectoryCreation:
    """Test directory creation functionality."""

    def test_create_directories(self):
        """Test that directories are created correctly."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)),
        ):
            # Create directories
            create_output_directories("test_exp")

            # Verify they exist
            expected_output = Path(temp_dir) / "outputs" / "test_exp"
            expected_training = Path(temp_dir) / "training_sets" / "test_exp"

            assert expected_output.exists()
            assert expected_training.exists()
            assert expected_output.is_dir()
            assert expected_training.is_dir()

    def test_create_directories_idempotent(self):
        """Test that creating directories multiple times doesn't fail."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)),
        ):
            # Create twice - should not raise error
            create_output_directories()
            create_output_directories()

            # Should still exist
            assert (Path(temp_dir) / "outputs").exists()
            assert (Path(temp_dir) / "training_sets").exists()


class TestPathIntegration:
    """Test integration between different path functions."""

    def test_paths_use_same_root(self):
        """Test that related paths use the same root directory."""
        with patch("audioloop.utils.paths.get_output_root", return_value=Path("/test/root")):
            output_dir = get_output_dir("exp")
            training_dir = get_training_sets_dir("exp")

            # Both should have nested structure under the same root.
            assert output_dir.name == "exp"
            assert training_dir.name == "exp"
            assert output_dir.parent.parent == Path("/test/root")
            assert training_dir.parent.parent == Path("/test/root")
            assert output_dir.parent.name == "outputs"
            assert training_dir.parent.name == "training_sets"

    def test_create_matches_get_functions(self):
        """Test that create_output_directories creates paths matching get functions."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)),
        ):
            exp_name = "integration_test"

            # Get expected paths
            expected_output = get_output_dir(exp_name)
            expected_training = get_training_sets_dir(exp_name)

            # Create directories
            create_output_directories(exp_name)

            # Verify the expected paths exist
            assert expected_output.exists()
            assert expected_training.exists()
