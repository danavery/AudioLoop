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
        [(None, "outputs", "training_sets"), ("test", "outputs_test", "training_sets_test")],
    )
    def test_directory_paths(self, experiment, expected_output, expected_training):
        """Test output and training directory generation."""
        config = AudioLoopConfig(experiment_name=experiment)

        assert config.output_dir.name == expected_output
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
        """Test that paths are consistent across calls."""
        config = AudioLoopConfig(experiment_name="test")

        # Multiple calls should return same paths
        assert config.output_dir == config.output_dir
        assert config.get_model_path(1) == config.get_model_path(1)


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
        assert callable(dataset_config.get_metadata_entries)


class TestTrainingParameters:
    """Test training parameter configuration."""

    def test_training_parameter_defaults(self):
        """Test that training parameters have correct defaults."""
        config = AudioLoopConfig()
        assert config.max_epochs == 1000
        assert config.seed == 42
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.model_type == "soundcnn"
        assert config.use_batchnorm is None  # Auto-detect
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
            use_batchnorm=False,
            stopping_criterion_type="accuracy",
            patience=30,
            min_delta=0.05,
            accuracy_floor=0.9
        )
        assert config.max_epochs == 500
        assert config.seed == 123
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.model_type == "simplecnn"
        assert config.use_batchnorm is False
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
        with pytest.raises(ValueError, match="Unknown stopping criterion"):
            AudioLoopConfig(stopping_criterion_type="invalid")
        
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


class TestActiveLearningParameters:
    """Test active learning parameter configuration."""

    def test_active_learning_parameter_defaults(self):
        """Test that active learning parameters have correct defaults."""
        config = AudioLoopConfig()
        assert config.total_candidates == 50
        assert config.positive_percentage == 0.75
        assert config.min_confidence == 0.8
        assert config.selection_mode == "confidence"
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
            estimated_positive_pct=0.1
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
        with pytest.raises(ValueError, match="positive_percentage must be between 0.0 and 1.0"):
            AudioLoopConfig(positive_percentage=-0.1)
        
        with pytest.raises(ValueError, match="positive_percentage must be between 0.0 and 1.0"):
            AudioLoopConfig(positive_percentage=1.1)
        
        # min_confidence validation
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            AudioLoopConfig(min_confidence=-0.1)
        
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            AudioLoopConfig(min_confidence=1.1)
        
        # estimated_positive_pct validation
        with pytest.raises(ValueError, match="estimated_positive_pct must be between 0.0 and 1.0"):
            AudioLoopConfig(estimated_positive_pct=-0.1)
        
        with pytest.raises(ValueError, match="estimated_positive_pct must be between 0.0 and 1.0"):
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
        with pytest.raises(ValueError, match="basic_transition_f1_threshold must be between 0.0 and 1.0"):
            AudioLoopConfig(basic_transition_f1_threshold=-0.1)
        
        with pytest.raises(ValueError, match="basic_transition_f1_threshold must be between 0.0 and 1.0"):
            AudioLoopConfig(basic_transition_f1_threshold=1.1)
        
        # confidence_threshold validation
        with pytest.raises(ValueError, match="basic_transition_confidence_threshold must be between 0.0 and 1.0"):
            AudioLoopConfig(basic_transition_confidence_threshold=-0.1)
        
        with pytest.raises(ValueError, match="basic_transition_confidence_threshold must be between 0.0 and 1.0"):
            AudioLoopConfig(basic_transition_confidence_threshold=1.1)
        
        # variance_threshold validation
        with pytest.raises(ValueError, match="basic_transition_variance_threshold must be between 0.0 and 1.0"):
            AudioLoopConfig(basic_transition_variance_threshold=-0.1)
        
        with pytest.raises(ValueError, match="basic_transition_variance_threshold must be between 0.0 and 1.0"):
            AudioLoopConfig(basic_transition_variance_threshold=1.1)
