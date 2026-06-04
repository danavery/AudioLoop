"""
Integration tests for the unified configuration system.

Focused on essential cross-component interactions and realistic usage patterns.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from audioloop.config import AudioLoopConfig
from audioloop.utils.paths import get_output_dir, get_training_sets_dir


class TestConfigPathIntegration:
    """Test integration between config and path utilities."""

    def test_config_paths_match_utilities(self):
        """Test that config paths match corresponding utility functions."""
        config = AudioLoopConfig(experiment_name="test_exp")

        # Config paths should match utility functions
        assert config.output_dir == get_output_dir("test_exp")
        assert config.training_sets_dir == get_training_sets_dir("test_exp")

    def test_directory_creation_integration(self):
        """Test that config can create directories that match its paths."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("audioloop.utils.paths.get_output_root", return_value=Path(temp_dir)),
        ):
            config = AudioLoopConfig(experiment_name="integration_test")

            # Create directories through config
            config.create_directories()

            # Verify config paths exist
            assert config.output_dir.exists()
            assert config.training_sets_dir.exists()


class TestEnvironmentIntegration:
    """Test environment variable integration across components."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Clean environment for consistent tests."""
        with patch.dict(os.environ, {}, clear=True):
            yield

    def test_dataset_environment_override(self):
        """Test proper precedence: explicit parameters override environment variables."""
        with patch.dict(os.environ, {"AUDIOLOOP_DATASET": "urbansound8k"}):
            # Environment used when no explicit dataset provided
            config1 = AudioLoopConfig()
            config2 = AudioLoopConfig()
            # Explicit parameter overrides environment variable
            config3 = AudioLoopConfig(dataset="fsd50k")

            assert config1.dataset == "urbansound8k"
            assert config2.dataset == "urbansound8k"
            assert config3.dataset == "fsd50k"  # Explicit parameter takes precedence

    def test_path_environment_integration(self):
        """Test that path environment variables work with config."""
        with patch.dict(
            os.environ,
            {"AUDIOLOOP_OUTPUT_ROOT": "/custom/output", "AUDIOLOOP_DATA_ROOT": "/custom/data"},
        ):
            config = AudioLoopConfig()

            # Paths should reflect environment variables
            assert str(config.output_dir).startswith("/custom/output")
            assert str(config.specs_dir).startswith("/custom/data")


class TestDatasetIntegration:
    """Test dataset configuration integration."""

    @pytest.mark.parametrize("dataset", ["fsd50k", "urbansound8k"])
    def test_dataset_config_retrieval(self, dataset):
        """Test that dataset configs can be retrieved for both datasets."""
        config = AudioLoopConfig(dataset=dataset)

        # Should be able to get dataset config
        dataset_config = config.get_dataset_config()
        assert dataset_config is not None

        # Should have required interface
        assert hasattr(dataset_config, "get_audio_path")
        assert hasattr(dataset_config, "load_metadata")
        assert hasattr(dataset_config, "process_single_file")
        # Audio->tensor production moved to the feature extractor.
        assert hasattr(dataset_config, "feature_extractor")


class TestVersionedWorkflow:
    """Test versioned file workflow integration."""

    def test_consistent_versioning(self):
        """Test that versioned paths are consistent across different file types."""
        config = AudioLoopConfig(experiment_name="workflow_test")
        version = 5

        # All versioned files should use same version pattern
        model_path = config.get_model_path(version)
        predictions_path = config.get_predictions_path(version)
        candidates_path = config.get_candidates_path(version)
        training_path = config.get_training_set_path(version)

        # All should contain version number
        for path in [model_path, predictions_path, candidates_path, training_path]:
            assert f"_v{version}" in str(path)

        # Output files should be in output dir, training in training dir
        assert model_path.parent == config.output_dir
        assert predictions_path.parent == config.output_dir
        assert candidates_path.parent == config.output_dir
        assert training_path.parent == config.training_sets_dir

    def test_different_experiments_isolated(self):
        """Test that different experiments don't interfere with each other."""
        config1 = AudioLoopConfig(experiment_name="exp1")
        config2 = AudioLoopConfig(experiment_name="exp2")

        # Should have different paths
        assert config1.output_dir != config2.output_dir
        assert config1.training_sets_dir != config2.training_sets_dir

        # But same version should work for both
        version = 1
        model1 = config1.get_model_path(version)
        model2 = config2.get_model_path(version)

        assert model1 != model2
        assert "exp1" in str(model1)
        assert "exp2" in str(model2)


class TestSpecsDirPathConfig:
    """Test specs_dir_path YAML configuration."""

    def test_specs_dir_path_relative(self):
        """Test that specs_dir_path in yaml resolves relative to project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create audioloop.yaml so get_project_root() works
            yaml_path = Path(temp_dir) / "audioloop.yaml"
            yaml_path.write_text(
                "config:\n  specs_dir_path: data/my_specs\n  experiment_name: test\n"
            )
            with patch(
                "audioloop.utils.paths.get_project_root", return_value=Path(temp_dir)
            ), patch(
                "audioloop.config.get_project_root", return_value=Path(temp_dir)
            ):
                config = AudioLoopConfig.from_yaml(yaml_path)
                assert config.specs_dir == Path(temp_dir) / "data" / "my_specs"

    def test_specs_dir_path_absolute(self):
        """Test that absolute specs_dir_path is used as-is."""
        config = AudioLoopConfig(specs_dir_path="/mnt/fast/specs")
        assert config.specs_dir == Path("/mnt/fast/specs")

    def test_specs_dir_path_none_falls_back(self):
        """Test that None specs_dir_path falls back to get_specs_dir()."""
        config = AudioLoopConfig()
        assert config.specs_dir_path is None
        # Should delegate to get_specs_dir() which uses env var / defaults
        from audioloop.utils.paths import get_specs_dir

        assert config.specs_dir == get_specs_dir()


class TestRealUsagePatterns:
    """Test realistic usage patterns."""

    def test_typical_cli_workflow(self):
        """Test typical CLI-like usage pattern."""
        # Simulate CLI argument processing
        config = AudioLoopConfig(experiment_name="cli_test", dataset="urbansound8k")

        # Should work without errors
        assert config.experiment_name == "cli_test"
        assert config.dataset == "urbansound8k"

        # Should be able to generate paths
        model_path = config.get_model_path(1)
        assert "cli_test" in str(model_path)
        assert model_path.name == "model_v1.pt"

    def test_experiment_switching(self):
        """Test switching between experiments in same session."""
        # Different experiments should work independently
        experiments = ["baseline", "improved", "final"]
        configs = [AudioLoopConfig(experiment_name=exp) for exp in experiments]

        # All should be valid and independent
        for i, config in enumerate(configs):
            assert config.experiment_name == experiments[i]
            assert experiments[i] in str(config.output_dir)

            # Each should have unique paths
            for j, other_config in enumerate(configs):
                if i != j:
                    assert config.output_dir != other_config.output_dir

    def test_configuration_persistence(self):
        """Test that configuration behaves consistently across multiple calls."""
        config = AudioLoopConfig(experiment_name="persistent")

        # Multiple calls should return identical results
        assert config.output_dir == config.output_dir
        assert config.get_model_path(1) == config.get_model_path(1)

        # Dataset config should be retrievable multiple times
        dataset_config1 = config.get_dataset_config()
        dataset_config2 = config.get_dataset_config()
        assert isinstance(dataset_config1, type(dataset_config2))
