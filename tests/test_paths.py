"""
Tests for AudioLoop path utilities.

Focused on environment variable handling and key path generation behavior.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from audioloop.utils.paths import (
    get_data_root,
    get_output_root,
    get_specs_dir,
    get_output_dir,
    get_training_sets_dir,
    create_output_directories,
)


class TestEnvironmentVariables:
    """Test environment variable handling for paths."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Clear environment for consistent tests."""
        with patch.dict(os.environ, {}, clear=True):
            yield

    @pytest.mark.parametrize("env_var,func,default", [
        ("AUDIOLOOP_DATA_ROOT", get_data_root, "data"),
        ("AUDIOLOOP_OUTPUT_ROOT", get_output_root, "."),
    ])
    def test_root_environment_override(self, env_var, func, default):
        """Test that environment variables override defaults."""
        # Test default
        assert func() == Path(default)
        
        # Test override
        with patch.dict(os.environ, {env_var: "/custom/path"}):
            assert func() == Path("/custom/path")

    def test_specs_dir_environment_combinations(self):
        """Test specs directory with different environment combinations."""
        # Default
        assert get_specs_dir() == Path("data/all_specs")
        
        # Custom data root only
        with patch.dict(os.environ, {"AUDIOLOOP_DATA_ROOT": "/custom"}):
            assert get_specs_dir() == Path("/custom/all_specs")
        
        # Custom specs subdir only
        with patch.dict(os.environ, {"AUDIOLOOP_SPECS_DIR": "spectrograms"}):
            assert get_specs_dir() == Path("data/spectrograms")
        
        # Both custom
        with patch.dict(os.environ, {
            "AUDIOLOOP_DATA_ROOT": "/custom",
            "AUDIOLOOP_SPECS_DIR": "spectrograms"
        }):
            assert get_specs_dir() == Path("/custom/spectrograms")


class TestDirectoryGeneration:
    """Test directory path generation."""

    @pytest.mark.parametrize("experiment,expected_suffix", [
        (None, "outputs"),
        ("test_exp", "outputs_test_exp"),
    ])
    def test_output_directory_names(self, experiment, expected_suffix):
        """Test output directory generation with different experiment names."""
        result = get_output_dir(experiment)
        assert result.name == expected_suffix

    @pytest.mark.parametrize("experiment,expected_suffix", [
        (None, "training_sets"),
        ("test_exp", "training_sets_test_exp"),
    ])
    def test_training_sets_directory_names(self, experiment, expected_suffix):
        """Test training sets directory generation."""
        result = get_training_sets_dir(experiment)
        assert result.name == expected_suffix

    def test_directory_consistency(self):
        """Test that output and training directories are consistent."""
        # Same experiment should produce consistent naming pattern
        exp_name = "test_experiment"
        output_dir = get_output_dir(exp_name)
        training_dir = get_training_sets_dir(exp_name)
        
        assert "test_experiment" in str(output_dir)
        assert "test_experiment" in str(training_dir)
        assert output_dir.parent == training_dir.parent


class TestDirectoryCreation:
    """Test directory creation functionality."""

    def test_create_directories(self):
        """Test that directories are created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)):
                # Create directories
                create_output_directories("test_exp")
                
                # Verify they exist
                expected_output = Path(temp_dir) / "outputs_test_exp"
                expected_training = Path(temp_dir) / "training_sets_test_exp"
                
                assert expected_output.exists()
                assert expected_training.exists()
                assert expected_output.is_dir()
                assert expected_training.is_dir()

    def test_create_directories_idempotent(self):
        """Test that creating directories multiple times doesn't fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)):
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
            
            # Both should use the same root
            assert output_dir.parent == Path("/test/root")
            assert training_dir.parent == Path("/test/root")

    def test_create_matches_get_functions(self):
        """Test that create_output_directories creates paths matching get functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)):
                exp_name = "integration_test"
                
                # Get expected paths
                expected_output = get_output_dir(exp_name)
                expected_training = get_training_sets_dir(exp_name)
                
                # Create directories
                create_output_directories(exp_name)
                
                # Verify the expected paths exist
                assert expected_output.exists()
                assert expected_training.exists()