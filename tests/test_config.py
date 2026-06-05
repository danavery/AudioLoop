"""
Tests for the unified AudioLoop configuration system.

Focused on behavior testing with minimal mocking and good use of parameterization.
"""

import os
from unittest.mock import patch

import pytest

from audioloop.config import AudioLoopConfig


class TestAudioLoopConfig:
    """Core configuration functionality."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Clear environment variables for consistent tests."""
        with patch.dict(os.environ, {}, clear=True):
            yield

    def test_default_behavior(self):
        """Test default configuration."""
        config = AudioLoopConfig()
        assert config.dataset == "fsd50k"
        assert config.experiment_name is None

    @pytest.mark.parametrize("dataset", ["fsd50k", "urbansound8k"])
    def test_dataset_selection(self, dataset):
        """Test configuration with different datasets."""
        config = AudioLoopConfig(dataset=dataset)
        assert config.dataset == dataset

    def test_experiment_name(self):
        """Test experiment name handling."""
        config = AudioLoopConfig(experiment_name="test_exp")
        assert config.experiment_name == "test_exp"

    @pytest.mark.parametrize(
        "env_dataset,expected", [("fsd50k", "fsd50k"), ("urbansound8k", "urbansound8k")]
    )
    def test_environment_override(self, env_dataset, expected):
        """Test environment variable override."""
        with patch.dict(os.environ, {"AUDIOLOOP_DATASET": env_dataset}):
            config = AudioLoopConfig()
            assert config.dataset == expected

    def test_invalid_dataset_error(self):
        """Test error handling for invalid dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            AudioLoopConfig(dataset="invalid")

    def test_default_dataset_when_not_specified(self):
        """Test that default dataset is used when not specified."""
        config = AudioLoopConfig()
        assert config.dataset == "fsd50k"


class TestConfigPaths:
    """Path generation functionality."""

    @pytest.mark.parametrize(
        "experiment,expected_output,expected_training",
        [(None, "outputs", "training_sets"), ("test", "test", "test")],
    )
    def test_directory_paths(self, experiment, expected_output, expected_training):
        """Test output and training directory generation."""
        config = AudioLoopConfig(experiment_name=experiment)

        if experiment is None:
            assert config.output_dir.name == expected_output
            assert config.training_sets_dir.name == expected_training
        else:
            # For experiments, check the nested structure: outputs/experiment/
            assert config.output_dir.parent.name == "outputs"
            assert config.output_dir.name == expected_output
            assert config.training_sets_dir.parent.name == "training_sets"
            assert config.training_sets_dir.name == expected_training

    @pytest.mark.parametrize(
        "version,expected_file",
        [
            (1, "model_v1.pt"),
            (42, "predictions_v42.csv"),
        ],
    )
    def test_versioned_paths(self, version, expected_file):
        """Test versioned file path generation."""
        config = AudioLoopConfig()

        if "model" in expected_file:
            path = config.get_model_path(version)
        else:
            path = config.get_predictions_path(version)

        assert path.name == expected_file

    def test_path_consistency(self):
        """Test that paths are deterministic: two configs for the same experiment agree."""
        config_a = AudioLoopConfig(experiment_name="test")
        config_b = AudioLoopConfig(experiment_name="test")

        # Same inputs -> same paths, across independent instances (not self-comparison).
        assert config_a.output_dir == config_b.output_dir
        assert config_a.get_model_path(1) == config_b.get_model_path(1)
        assert config_a.get_model_path(1).name == "model_v1.pt"


class TestConfigConstructor:
    """Configuration constructor functionality."""

    def test_default_constructor(self):
        """Test default constructor."""
        config = AudioLoopConfig()
        assert isinstance(config, AudioLoopConfig)
        assert config.experiment_name is None

    def test_constructor_with_args(self):
        """Test constructor with arguments."""
        config = AudioLoopConfig(experiment_name="test", dataset="urbansound8k")
        assert config.experiment_name == "test"
        assert config.dataset == "urbansound8k"

    def test_constructor_respects_environment(self):
        """Test constructor respects environment variables."""
        with patch.dict(os.environ, {"AUDIOLOOP_DATASET": "urbansound8k"}):
            config = AudioLoopConfig()
            assert config.dataset == "urbansound8k"


class TestDatasetIntegration:
    """Dataset configuration integration."""

    @pytest.mark.parametrize("dataset", ["fsd50k", "urbansound8k"])
    def test_can_get_dataset_components(self, dataset):
        """Test that dataset components can be retrieved."""
        config = AudioLoopConfig(dataset=dataset)

        # Should be able to get dataset config
        dataset_config = config.get_dataset_config()
        assert dataset_config is not None
        assert hasattr(dataset_config, "get_audio_path")

    def test_dataset_config_interface(self):
        """Test dataset config implements required interface."""
        config = AudioLoopConfig(dataset="fsd50k")
        dataset_config = config.get_dataset_config()

        # Test required methods exist
        assert callable(dataset_config.get_audio_path)
        assert callable(dataset_config.load_metadata)


class TestTrainingParameters:
    """Test training parameter configuration."""

    def test_training_parameter_defaults(self):
        """Test that training parameters have correct defaults."""
        config = AudioLoopConfig()
        assert config.max_epochs == 1000
        assert config.seed == 42
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.model_type == "cnn5layer"
        assert config.stopping_criterion_type == "plateau"
        assert config.patience == 20
        assert config.min_delta == 0.01
        assert config.accuracy_floor is None

    def test_training_parameter_overrides(self):
        """Test training parameter constructor overrides."""
        config = AudioLoopConfig(
            max_epochs=500,
            seed=123,
            batch_size=64,
            learning_rate=0.01,
            model_type="simplecnn",
            stopping_criterion_type="accuracy",
            patience=30,
            min_delta=0.05,
            accuracy_floor=0.9,
        )
        assert config.max_epochs == 500
        assert config.seed == 123
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.model_type == "simplecnn"
        assert config.stopping_criterion_type == "accuracy"
        assert config.patience == 30
        assert config.min_delta == 0.05
        assert config.accuracy_floor == 0.9

    def test_training_parameter_validation_positive_values(self):
        """Test validation of positive training parameters."""
        with pytest.raises(ValueError, match="max_epochs must be positive"):
            AudioLoopConfig(max_epochs=-1)

        with pytest.raises(ValueError, match="max_epochs must be positive"):
            AudioLoopConfig(max_epochs=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            AudioLoopConfig(batch_size=-1)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            AudioLoopConfig(learning_rate=-0.1)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            AudioLoopConfig(learning_rate=0.0)

    def test_stopping_criterion_validation(self):
        """Test validation of stopping criterion parameters."""
        # Note: stopping_criterion_type validation moved to create_stopping_criterion()
        # Only patience validation remains in config

        with pytest.raises(ValueError, match="patience must be positive"):
            AudioLoopConfig(stopping_criterion_type="plateau", patience=-1)

        with pytest.raises(ValueError, match="patience must be positive"):
            AudioLoopConfig(stopping_criterion_type="plateau", patience=0)

    def test_valid_stopping_criterion_types(self):
        """Test that valid stopping criteria work."""
        # Plateau should work
        config1 = AudioLoopConfig(stopping_criterion_type="plateau")
        assert config1.stopping_criterion_type == "plateau"

        # Accuracy should work
        config2 = AudioLoopConfig(stopping_criterion_type="accuracy")
        assert config2.stopping_criterion_type == "accuracy"


class TestSubsetCSVIntegration:
    """Test subset_csv parameter integration with dataset configs."""

    def test_subset_csv_parameter_default_none(self):
        """Test that subset_csv defaults to None."""
        config = AudioLoopConfig()
        assert config.subset_csv is None

    def test_subset_csv_parameter_can_be_set(self, tmp_path):
        """Test that subset_csv can be set via constructor."""
        subset_path = tmp_path / "subset.csv"
        subset_path.write_text("filename,label\ntest.wav,1\n")

        config = AudioLoopConfig(subset_csv=subset_path)
        assert config.subset_csv == subset_path

    def test_get_dataset_config_applies_subset_csv(self, tmp_path):
        """Test that get_dataset_config() applies subset_csv automatically."""
        # Create a fake subset CSV
        subset_path = tmp_path / "subset.csv"
        subset_path.write_text("filename,label\ntest.wav,1\n")

        config = AudioLoopConfig(dataset="audioset", subset_csv=subset_path)
        dataset_config = config.get_dataset_config()

        # The dataset config should have the custom CSV set
        assert dataset_config._custom_csv_path == subset_path  # type: ignore[attr-defined]

    def test_get_dataset_config_raises_error_if_subset_csv_missing(self, tmp_path):
        """Test that error is raised if subset_csv file doesn't exist."""
        missing_path = tmp_path / "nonexistent.csv"

        config = AudioLoopConfig(dataset="audioset", subset_csv=missing_path)

        with pytest.raises(FileNotFoundError, match="Subset CSV not found"):
            config.get_dataset_config()

    def test_get_dataset_config_without_subset_csv_works_normally(self):
        """Test that get_dataset_config() works normally without subset_csv."""
        config = AudioLoopConfig(dataset="fsd50k")
        dataset_config = config.get_dataset_config()

        # Should work fine without subset_csv
        assert dataset_config is not None
        assert hasattr(dataset_config, "load_metadata")


class TestActiveLearningParameters:
    """Test active learning parameter configuration."""

    def test_active_learning_parameter_defaults(self):
        """Test that active learning parameters have correct defaults."""
        config = AudioLoopConfig()
        assert config.subset_csv is None
        assert config.total_candidates == 50
        assert config.positive_percentage is None  # Default is no stratification
        assert config.min_confidence == 0.8
        assert config.selection_mode == "entropy"
        assert config.basic_transition_f1_threshold == 0.2
        assert config.basic_transition_confidence_threshold == 0.9
        assert config.basic_transition_variance_threshold == 0.12
        assert config.auto_thresholds is False
        assert config.estimated_positive_pct is None

    def test_active_learning_parameter_overrides(self):
        """Test active learning parameter constructor overrides."""
        config = AudioLoopConfig(
            total_candidates=100,
            positive_percentage=0.6,
            min_confidence=0.9,
            selection_mode="entropy",
            basic_transition_f1_threshold=0.3,
            basic_transition_confidence_threshold=0.85,
            basic_transition_variance_threshold=0.15,
            auto_thresholds=True,
            estimated_positive_pct=0.1,
        )
        assert config.total_candidates == 100
        assert config.positive_percentage == 0.6
        assert config.min_confidence == 0.9
        assert config.selection_mode == "entropy"
        assert config.basic_transition_f1_threshold == 0.3
        assert config.basic_transition_confidence_threshold == 0.85
        assert config.basic_transition_variance_threshold == 0.15
        assert config.auto_thresholds is True
        assert config.estimated_positive_pct == 0.1

    def test_active_learning_parameter_validation_positive_values(self):
        """Test validation of positive active learning parameters."""
        with pytest.raises(ValueError, match="total_candidates must be positive"):
            AudioLoopConfig(total_candidates=-1)

        with pytest.raises(ValueError, match="total_candidates must be positive"):
            AudioLoopConfig(total_candidates=0)

    def test_active_learning_parameter_validation_ranges(self):
        """Test validation of range-bound active learning parameters."""
        # positive_percentage validation
        with pytest.raises(
            ValueError, match=r"positive_percentage must be None or between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(positive_percentage=-0.1)

        with pytest.raises(
            ValueError, match=r"positive_percentage must be None or between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(positive_percentage=1.1)

        # min_confidence validation
        with pytest.raises(ValueError, match=r"min_confidence must be between 0\.0 and 1\.0"):
            AudioLoopConfig(min_confidence=-0.1)

        with pytest.raises(ValueError, match=r"min_confidence must be between 0\.0 and 1\.0"):
            AudioLoopConfig(min_confidence=1.1)

        # estimated_positive_pct validation
        with pytest.raises(
            ValueError, match=r"estimated_positive_pct must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(estimated_positive_pct=-0.1)

        with pytest.raises(
            ValueError, match=r"estimated_positive_pct must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(estimated_positive_pct=1.1)

    def test_selection_mode_validation(self):
        """Test validation of selection mode parameter."""
        with pytest.raises(ValueError, match="Unknown selection mode"):
            AudioLoopConfig(selection_mode="invalid")

        # Valid selection modes should work
        config1 = AudioLoopConfig(selection_mode="confidence")
        assert config1.selection_mode == "confidence"

        config2 = AudioLoopConfig(selection_mode="entropy")
        assert config2.selection_mode == "entropy"

        config3 = AudioLoopConfig(selection_mode="basic_transition")
        assert config3.selection_mode == "basic_transition"

    def test_basic_transition_parameter_validation(self):
        """Test validation of basic transition parameters."""
        # f1_threshold validation
        with pytest.raises(
            ValueError, match=r"basic_transition_f1_threshold must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(basic_transition_f1_threshold=-0.1)

        with pytest.raises(
            ValueError, match=r"basic_transition_f1_threshold must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(basic_transition_f1_threshold=1.1)

        # confidence_threshold validation
        with pytest.raises(
            ValueError, match=r"basic_transition_confidence_threshold must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(basic_transition_confidence_threshold=-0.1)

        with pytest.raises(
            ValueError, match=r"basic_transition_confidence_threshold must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(basic_transition_confidence_threshold=1.1)

        # variance_threshold validation
        with pytest.raises(
            ValueError, match=r"basic_transition_variance_threshold must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(basic_transition_variance_threshold=-0.1)

        with pytest.raises(
            ValueError, match=r"basic_transition_variance_threshold must be between 0\.0 and 1\.0"
        ):
            AudioLoopConfig(basic_transition_variance_threshold=1.1)

    def test_cycle_stopping_strategy_validation(self):
        """Test validation of cycle_stopping_strategy parameter."""
        # Note: cycle_stopping_strategy validation moved to create_cycle_stopping_criterion()
        # Config accepts any string value

        # Valid strategies should work
        config1 = AudioLoopConfig(cycle_stopping_strategy="none")
        assert config1.cycle_stopping_strategy == "none"

        config2 = AudioLoopConfig(cycle_stopping_strategy="label")
        assert config2.cycle_stopping_strategy == "label"

        config3 = AudioLoopConfig(cycle_stopping_strategy="search")
        assert config3.cycle_stopping_strategy == "search"

    def test_precision_floor_validation(self):
        """Test validation of precision_floor parameter."""
        # Valid: "auto" string
        config1 = AudioLoopConfig(precision_floor="auto")
        assert config1.precision_floor == "auto"

        # Valid: float between 0.0 and 1.0
        config2 = AudioLoopConfig(precision_floor=0.5)
        assert config2.precision_floor == 0.5

        config3 = AudioLoopConfig(precision_floor=0.0)
        assert config3.precision_floor == 0.0

        config4 = AudioLoopConfig(precision_floor=1.0)
        assert config4.precision_floor == 1.0

        # Invalid: string other than "auto"
        with pytest.raises(ValueError, match="precision_floor must be 'auto' or a float"):
            AudioLoopConfig(precision_floor="invalid")

        # Invalid: float out of range
        with pytest.raises(ValueError, match=r"precision_floor must be between 0\.0 and 1\.0"):
            AudioLoopConfig(precision_floor=-0.1)

        with pytest.raises(ValueError, match=r"precision_floor must be between 0\.0 and 1\.0"):
            AudioLoopConfig(precision_floor=1.5)

        # Invalid: wrong type
        with pytest.raises(ValueError, match="precision_floor must be 'auto' or a float"):
            AudioLoopConfig(precision_floor=None)  # type: ignore[arg-type]
